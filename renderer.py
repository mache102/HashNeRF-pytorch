import argparse 
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from radam import RAdam

from load.load_data import CameraConfig
from util import debug_dict

from einops import rearrange, reduce

from dataclasses import dataclass 

@dataclass 
class RayPredictions:
    rgb: torch.Tensor
    depth: torch.Tensor
    disparity: torch.Tensor
    accumulation: torch.Tensor
    weights: torch.Tensor
    sparsity_loss: torch.Tensor

class VolumetricRenderer:
    def __init__(self, cc: CameraConfig, 
                 models: dict, embedders: dict,
                 args: argparse.Namespace):
        """
        NeRF volumetric rendering

        models:
            coarse and fine models for predicting RGB and density

        embedders:
            embedders for positional ennc. & viewdir ennc.

        ----
        forward:
            rays, retraw
        """     
        self.models = models
        self.embedders = embedders

        self.unpack_cc(cc)
        self.unpack_args(args)

    def unpack_cc(self, cc):
        """
        Unpack camera config
        (height, width, focal,
        intrinsic matrix, near, far)
        """
        self.h = cc.height
        self.w = cc.width
        self.focal = cc.focal
        self.k = cc.k
        self.near = cc.near
        self.far = cc.far

    def unpack_args(self, args):
        """
        Unpack arguments for rendering.

        Originally passed to render methods
        through **render_kwargs.

        Special case (testing):
            perturb = False 
            raw_noise_std = 0

        seed: random seed 
        dataset_type: dataset type (equirect, blender, etc)

        proc_bsz: num of rays processed in parallel, adjust to prevent OOM
        net_bsz: num of rays sent to model in parallel (proc_bsz % net_bsz == 0)
        perturb: 0 for uniform sampling, 1 for jittered (stratified rand points) sampling
        N_samples: num of coarse samples per ray
        N_importance: num of additional fine samples per ray
        use_viewdirs: whether to use full 5D input (+dirs) instead of 3D
        white_bkgd: whether to assume white background
        raw_noise_std: std dev of noise added to regularize sigma_a output, 1e0 recommended
        lindisp: If True, sample linearly in inverse depth rather than in depth.
        """
        self.seed = args.seed 
        self.dataset_type = args.dataset_type

        self.proc_bsz = args.chunk
        self.net_bsz = args.net_chunk
        self.perturb = args.perturb
        self.N_samples = args.N_samples 
        self.N_importance = args.N_importance 
        self.use_viewdirs = args.use_viewdirs
        self.white_bkgd = args.white_bkgd
        self.raw_noise_std = args.raw_noise_std 
        self.lindisp = args.lindisp

        self.perturb_test = False 
        self.raw_noise_std_test = 0

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def render(self, rays, reshape_to, **kwargs):
        """
        Rendering entry function

        Args:
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        reshape_to: shape to reshape outputs to.
        
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disparity_map: [batch_size]. Disparity map. Inverse of depth.
        accumulation_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_batch().
        """
        total_rays = rays.shape[0]

        outputs = {}
        for i in range(0, total_rays, self.proc_bsz):
            batch_output = self.render_batch(rays[i:i + self.proc_bsz], **kwargs)
            for k, v in batch_output.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)             

            if self.N_importance > 0:
                for k, v in batch_output["fine"].items():
                    if k not in outputs:
                        outputs[k] = []
                    outputs[k].append(v)                      
            
            if "raw" in batch_output:
                outputs["raw"].append(batch_output["raw"])
    
        outputs = {k: torch.cat(v, 0) for k, v in outputs.items()}
        outputs = {k: torch.reshape(v, reshape_to + list(v.shape[1:])) for k, v in outputs.items()}
        # for k in outputs: 
        #     try: 
        #         print(k, outputs[k].shape, outputs[k][:10])
        #     except AttributeError:
        #         continue

        priority = ['rgb_map', 'depth_map', 'accumulation_map']

        ret_main = {k: outputs[k] for k in priority}
        ret_extra = {k: outputs[k] for k in outputs if k not in priority}
        return ret_main, ret_extra

    def run_network(self, inputs, model_name):
        """
        Prepare inputs (embed, concat) and apply network
        """
        pos_ = inputs["pos"] # use this for reshaping (second to last line)
        pos = pos_.reshape(-1, pos_.shape[-1])
        pos_embed, keep_mask = self.embedders["pos"](pos)

        if inputs.get("dir") is not None:
            dir = inputs["dir"]
            # expand with the original pos input shape!
            # mistakenly used pos instead of pos_
            dir = dir[:, None].expand(pos_.shape) 
            dir = dir.reshape(-1, dir.shape[-1])
            dir_embed = self.embedders["dir"](dir)

            embeds = torch.cat([pos_embed, dir_embed], -1)
        else:
            embeds = pos_embed

        outputs = []
        for i in range(0, embeds.shape[0], self.net_bsz):
            outputs.append(self.models[model_name](embeds[i:i + self.net_bsz]))

        outputs = torch.cat(outputs, 0)
        outputs[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
        outputs = torch.reshape(outputs, list(pos_.shape[:-1]) + [outputs.shape[-1]])
        return outputs
    
    def render_batch(self, rays, verbose=False, 
                     retraw=False, test=False):
        """
        Volumetric rendering for a single batch.

        rays: All information necessary for sampling along a ray.
            Shape: (N_rays, 8) or (N_rays, 11) (if use_viewdirs)
            0:3 - ray origin (N_rays, 3)
            3:6 - ray direction (N_rays, 3)
            6:8 - near and far bounds (N_rays, 2)
            8:11 (if use_viewdirs) - viewing direction (N_rays, 3)

        retraw: bool. If True, include model's raw, unprocessed predictions.
        verbose: bool. If True, print more debugging info.

        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disparity_map: [num_rays]. Disparity map. 1 / depth.
        accumulation_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disparity_map. Output for coarse model.
        acc0: See accumulation_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        if test is True:
            perturb = self.perturb_test
            raw_noise_std = self.raw_noise_std_test
        else:
            perturb = self.perturb
            raw_noise_std = self.raw_noise_std

        # unpack batch of rays
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:,0:3], rays[:,3:6]
        viewdirs = rays[:,-3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[...,6:8], [-1,1,2]) # [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1] # [N_rays, 1]

        # prepare for sampling
        t_vals = torch.linspace(0., 1., steps=self.N_samples)
        if not self.lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        # concatenate positional samples
        # [N_rays, N_samples, 3]
        positional = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
        raw = self.run_network(inputs={"pos": positional, "dir": viewdirs},
                               model_name="coarse")
        coarse_preds = self.raw2outputs(raw, z_vals, rays_d, 
                                        raw_noise_std=raw_noise_std)
        weights = coarse_preds.weights

        if self.N_importance > 0:
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], 
                                det=(perturb==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            # concatenate positional samples
            # [N_rays, N_samples + N_importance, 3]
            positional = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
            model_name = "fine" if self.models["fine"] is not None else "coarse"
            raw = self.run_network(inputs={"pos": positional, "dir": viewdirs},
                                   model_name=model_name)

            fine_preds = self.raw2outputs(raw, z_vals, rays_d, 
                                            raw_noise_std=raw_noise_std)

        if self.N_importance > 0:
            ret = {
                "rgb_map_0": coarse_preds.rgb,
                "depth_map_0": coarse_preds.depth,
                "disparity_map_0": coarse_preds.disparity,
                "accumulation_map_0": coarse_preds.accumulation,
                "sparsity_loss_0": coarse_preds.sparsity_loss,     

                "rgb_map": fine_preds.rgb,
                "depth_map": fine_preds.depth,
                "disparity_map": fine_preds.disparity,
                "accumulation_map": fine_preds.accumulation,
                "sparsity_loss": fine_preds.sparsity_loss,
                "z_std": torch.std(z_samples, dim=-1, unbiased=False) # (N_rays,)
            }
        else: 
            ret = {
                "rgb_map": coarse_preds.rgb,
                "depth_map": coarse_preds.depth,
                "disparity_map": coarse_preds.disparity,
                "accumulation_map": coarse_preds.accumulation,
                "sparsity_loss": coarse_preds.sparsity_loss,
            }
        if retraw:
            ret['raw'] = raw
    
        # debug_dict(ret)
        return ret

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disparity_map: [num_rays]. Disparity map. Inverse of depth map.
            accumulation_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        
        Section 4, pg. 6. Volume Rendering with Radiance Fields
        C(r) = sum (i = 1 to N) T_i * (1-exp(sigma*dist))c_i

        volumetric rendering of a color c with density sigma. c(t) and sigma(t) are the color & density at point r(t)
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        # differences of z vals = dists. last value is 1e10 (astronomically large in the context)
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        noise = 0.
        if self.raw_noise_std > 0.: # sigma
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # sigma_loss = sigma_sparsity_loss(raw[...,3])
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

        ones = torch.ones((alpha.shape[0], 1))
        weights = alpha * torch.cumprod(torch.cat([ones, 1.-alpha + 1e-10], -1), -1)[:, :-1]
        accumulation_map = torch.sum(weights, -1)
        depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        disparity_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        if self.white_bkgd:
            rgb_map = rgb_map + (1. - accumulation_map[...,None])

        # Calculate weights sparsity loss
        try:
            entropy = Categorical(probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
        except:
            print("Something occured in calculating weights sparsity loss. Begin debugging...")
            pdb.set_trace()
        sparsity_loss = entropy

        return RayPredictions(rgb_map, depth_map, disparity_map, 
                                accumulation_map, weights, sparsity_loss)


    # Hierarchical sampling (section 5.2)
    def sample_pdf(self, bins, weights, det=False, eps=1e-5):
        """
        Sample N_samples samples from bins with distribution defined by weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_samples: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_samples = self.N_importance 

        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero
        pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples) (keep dims)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                                # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_samples, device=bins.device)
            u = u.expand(N_rays, N_samples)
        else:
            u = torch.rand(N_rays, N_samples, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
        cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
        bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

if __name__ == '__main__':
    """
    testing purposes only
    """
    N_rays = 4
    N_samples = 101

    n, f = 0.2, 0.6
    t = np.linspace(0, 1, N_samples)

    z = 1 / ((1 - t) / n + t / f)

    z = np.broadcast_to(z, (N_rays, N_samples))

    rays_d = np.random.rand(N_rays, 3)

    pts = rays_d[...,None,:] * z[...,:,None] # [N_rays, N_samples, 3]

    # plot all pts 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[...,0], pts[...,1], pts[...,2])
    plt.show()