import torch 
import pdb 
from einops import rearrange, reduce
from torch.distributions import Categorical
from collections import OrderedDict
from itertools import dropwhile

from math_util import alpha_composite 


class Extractor:
    def __init__(self, usage: dict, rays_d: torch.Tensor,
                 white_bkgd: bool = False, raw_noise_std: float = 0.):
        self.usage = usage 
        self.rays_d = rays_d 

        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std

        # unpack order: 
        # static color(3), static sigma(1), gradient(3), 
        # transient color(3), transient sigma(1), transient beta(1)
        self.unpack_lookup = OrderedDict({
            "static_color": [0, 3], # (start, step)
            "static_sigma": [3, 1],
            "gradient": [4, 3],
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

    def get_delta(self, samples):
        """Retrieve deltas from samples"""
        # differences of z vals = deltas. last value is 1e10 (astronomically large in the context)
        deltas = samples[:, 1:] - samples[:, :-1]
        deltas = torch.cat([deltas, torch.Tensor([1e10]).expand(deltas[:,:1].shape)], -1)
        deltas = deltas * torch.norm(rearrange(self.rays_d, 'r c -> r () c'), dim=-1) # (render_bsz, samples)
        return deltas

    def __call__(self, raw: torch.Tensor, samples: torch.Tensor, test_mode=False):
        """
        Transforms model's predictions to semantically meaningful values.
        """
        def unpack(name):
            """Unpacks a return from the raw output"""
            start, step = self.unpack_lookup[name]
            return raw[..., start:start+step]
        
        results = {}

        deltas = self.get_delta(samples) # (render_bsz, samples)

        static_sigmas = unpack("static_sigmas")   
        if self.raw_noise_std > 0:
            static_sigmas += torch.randn(static_sigmas.shape) * self.raw_noise_std
        alphas = alpha_composite(static_sigmas, deltas) # (render_bsz, samples)

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
        weights = alphas * transmittance
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

        if self.usage["gradient"] and not test_mode:
            gradient = torch.tanh(unpack("gradient")) # (render_bsz, samples, 3))
            gradient = reduce(weights * gradient, 'n1 n2 c -> n1 c', 'sum')
            results["gradient"] = gradient

        # Calculate weights sparsity loss
        try:
            entropy = Categorical(probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
        except:
            print("Something occured in calculating weights sparsity loss. Begin debugging...")
            pdb.set_trace()
        results["weights"] = weights
        results["sparsity_loss"] = entropy 

        return results 