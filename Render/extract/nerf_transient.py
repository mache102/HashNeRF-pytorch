import torch 
import pdb 
from einops import rearrange, reduce
from torch.distributions import Categorical
from collections import OrderedDict
from itertools import dropwhile

from math_util import alpha_composite 


class Extractor:
    def __init__(self, usage: dict, white_bkgd: bool = False, 
                 raw_noise_std: float = 0.):
        
        self.usage = usage 

        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std # (train, test)

        # unpack order: 
        # static color(3), static sigma(1), gradient(3), 
        # transient color(3), transient sigma(1), transient beta(1)
        self.unpack_lookup = OrderedDict({
            "static_color": [0, 3], # (start, step)
            "static_sigma": [3, 1],
            "gradient": [4, 3],
            "transient_color": [7, 3],
            "transient_sigma": [10, 1],
            "transient_beta": [11, 1],
        })
        # now for the usage cases 
        # if a return is unused, then remove it from the lookup
        # and adjust the starts of subsequent returns
        if not self.usage["gradient"]:
            offset = self.unpack_lookup["gradient"][1]
            for key in list(dropwhile(lambda x: x != "gradient", 
                                      self.unpack_lookup))[1:]:
                self.unpack_lookup[key][0] -= offset
            self.unpack_lookup.pop("gradient")

        if not self.usage["transient"]:
            offset = self.unpack_lookup["transient_color"][1] + \
                     self.unpack_lookup["transient_sigma"][1] + \
                     self.unpack_lookup["transient_beta"][1]
            for key in list(dropwhile(lambda x: x != "transient_beta", 
                                      self.unpack_lookup))[1:]:
                self.unpack_lookup[key][0] -= offset
            self.unpack_lookup.pop("transient_color")
            self.unpack_lookup.pop("transient_sigma")
            self.unpack_lookup.pop("transient_beta")

    def get_delta(self, samples, rays_d):
        """Retrieve deltas from samples"""
        # differences of z vals = deltas. last value is 1e10 (astronomically large in the context)
        deltas = samples[:, 1:] - samples[:, :-1]
        deltas = torch.cat([deltas, torch.Tensor([1e10]).expand(deltas[:,:1].shape)], -1)
        deltas = deltas * torch.norm(rearrange(rays_d, 'r c -> r () c'), dim=-1) # (render_bsz, samples)
        return deltas

    def __call__(self, raw: torch.Tensor, samples: torch.Tensor, 
                 rays_d: torch.Tensor, test_mode: bool = False):
        """
        Transforms model's predictions to semantically meaningful values.
        """
        def unpack(name):
            """Unpacks a return from the raw output"""
            start, step = self.unpack_lookup[name]
            if step == 1:
                return raw[..., start]
            return raw[..., start:start+step]
        raw_noise_std = self.raw_noise_std[0] if test_mode is False else self.raw_noise_std[1]  
        results = {}

        delta = self.get_delta(samples, rays_d) # (render_bsz, samples)

        static_sigma = unpack("static_sigma")   
        if raw_noise_std > 0:
            static_sigma += torch.randn(static_sigma.shape) * raw_noise_std
        static_alpha = alpha_composite(static_sigma, delta) # (render_bsz, samples)

        if self.usage["transient"]:
            transient_color = unpack("transient_color")
            transient_sigma = unpack("transient_sigma")
            transient_beta = unpack("transient_beta")

            transient_alpha = alpha_composite(transient_sigma, delta) 
            dual_alpha = alpha_composite(static_sigma + transient_sigma, delta) 
        else:
            # dual degenerates to static 
            dual_alpha = alpha_composite(static_sigma, delta) # (render_bsz, samples)
        dual_alpha_shifted = torch.cat([torch.ones_like(dual_alpha[:, :1]), 
                                        1 - dual_alpha + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
        dual_transmittance = torch.cumprod(dual_alpha_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        weights = dual_alpha * dual_transmittance
        opacity = reduce(weights, 'n1 n2 -> n1', 'sum')

        static_color = torch.sigmoid(unpack("static_color")) # (render_bsz, samples, 3)
        static_color = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * static_color,
                            'n1 n2 c -> n1 c', 'sum') # (render_bsz, 3)        
        if self.white_bkgd:
            static_color += (1. - rearrange(opacity, 'n -> n 1'))

        depth = reduce(weights * samples, 'n1 n2 -> n1', 'sum')

        results["color"] = static_color  
        results["opacity"] = opacity
        results["depth"] = depth
        results["weights"] = weights

        if self.usage["transient"]:
            results["transient_sigma"] = transient_sigma
            transient_weights = transient_alpha * dual_transmittance

            transient_color = \
                reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1') * transient_color,
                       'n1 n2 c -> n1 c', 'sum')
            results["beta"] = reduce(transient_weights * transient_beta, 'n1 n2 -> n1', 'sum')

            results['color_transient'] = transient_color
            results['color_dual'] = static_color + transient_color

            if test_mode is True:
                # Compute also static and transient rgbs when only one field exists.
                # The result is different from when both fields exist, since the transimttance
                # will change.
                static_alpha_shifted = torch.cat([torch.ones_like(static_alpha[:, :1]), 
                                                  1-static_alpha], -1)
                static_transmittance = torch.cumprod(static_alpha_shifted[:, :-1], -1)
                static_weights_ = static_alpha * static_transmittance
                static_color_ = \
                    reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1') * static_color,
                           'n1 n2 c -> n1 c', 'sum')
                if self.white_bkgd:
                    static_color_ += 1 - rearrange(opacity, 'n -> n 1')

                results['color_test'] = static_color_
                results['depth_test'] = reduce(static_weights_ * samples, 'n1 n2 -> n1', 'sum')

                transient_alpha_shifted = \
                    torch.cat([torch.ones_like(transient_alpha[:, :1]), 1-transient_alpha], -1)
                transient_transmittance = torch.cumprod(transient_alpha_shifted[:, :-1], -1)
                transient_weights_ = transient_alpha * transient_transmittance

                results['color_transient_test'] = \
                    reduce(rearrange(transient_weights_, 'n1 n2 -> n1 n2 1') * transient_color,
                           'n1 n2 c -> n1 c', 'sum')
                results['depth_transient_test'] = \
                    reduce(transient_weights_ * samples, 'n1 n2 -> n1', 'sum')

        if self.usage["gradient"]:
            gradient = torch.tanh(unpack("gradient")) # (render_bsz, samples, 3))
            gradient = reduce(weights * gradient, 'n1 n2 c -> n1 c', 'sum')
            results["gradient"] = gradient

        # Calculate weights sparsity loss
        try:
            temp = 1.0 - reduce(weights, 'n1 n2 -> n1 1', 'sum') + 1e-6
            entropy = Categorical(probs = torch.cat([weights, temp], dim=-1)).entropy()
        except:
            print("Something occured in calculating weights sparsity loss. Begin debugging...")
            pdb.set_trace()
        results["weights"] = weights
        results["sparsity_loss"] = entropy 

        return results 