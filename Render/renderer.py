import argparse 
import matplotlib.pyplot as plt
import numpy as np
import pdb

from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dataclasses import dataclass 
from einops import rearrange, reduce

from ray_util import *
from load.load_data import CameraConfig
from Optimizer.radam import RAdam
from math_util import alpha_composite

class VolumetricRenderer:
    def __init__(self, cc: CameraConfig, 
                 models: dict, embedders: dict,
                 use: dict, args: argparse.Namespace):
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
        self.use = use

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

        render_bsz: num of rays processed/rendered in parallel, adjust to prevent OOM
        net_bsz: num of rays sent to model in parallel (render_bsz % net_bsz == 0)
        perturb: 0 for uniform sampling, 1 for jittered (stratified rand points) sampling
        N_coarse: num of coarse samples per ray
        N_fine: num of additional fine samples per ray
        white_bkgd: whether to assume white background
        raw_noise_std: std dev of noise added to regularize sigma_a output, 1e0 recommended
        lindisp: If True, sample linearly in inverse depth rather than in depth.
        """
        self.seed = args.seed 
        self.dataset_type = args.dataset_type

        self.render_bsz = args.render_bsz
        self.net_bsz = args.net_bsz
        self.N_coarse = args.N_coarse 
        self.N_fine = args.N_fine 
        self.perturb = args.perturb
        self.white_bkgd = args.white_bkgd
        self.raw_noise_std = args.raw_noise_std 
        self.lindisp = args.lindisp

        self.perturb_test = False 
        self.raw_noise_std_test = 0

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def render(self, rays, **kwargs):
        """
        Rendering entry function

        Args:
        rays: concatenated rays of shape (train_bsz, ?)
        og_shape: shape to reshape outputs to.
        
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disparity_map: [batch_size]. Disparity map. Inverse of depth.
        accumulation_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_batch().
        """
        total_rays = rays.shape[0]

        outputs = defaultdict(list)
        for i in range(0, total_rays, self.render_bsz):
            # (render_bsz, ?)
            batch_output = self.render_batch(rays[i:i + self.render_bsz], **kwargs)
            for k, v in batch_output.items():
                outputs[k] += [v]           
    
        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0) 

        return outputs

    def run_network(self, inputs, model):
        """
        Prepare inputs (embed, concat) and apply network

        inputs
            xyz: (render_bsz, N_coarse, 3)
            dir: (render_bsz, 3)
        
        model: coarse or fine
        """
        xyz_shape = inputs["xyz"].shape
        xyz = rearrange(inputs["xyz"], 'r cs c -> (r cs) c') # (render_bsz * N_coarse, 3)
        em_xyz, keep_mask = self.embedders["xyz"](xyz) #TODO: no keep mask for non hashenc
        embeds = em_xyz

        if self.use["dirs"]:
            dir = inputs["dir"]
            dir = rearrange(dir, 'r c -> r () c').expand(xyz_shape)
            dir = rearrange(dir, 'r cs c -> (r cs) c') # (render_bsz * N_coarse, 3)
            em_dir = self.embedders["dir"](dir)
            embeds = torch.cat([em_xyz, em_dir], -1)

        if self.use["appearance"]:


        outputs = []
        for i in range(0, embeds.shape[0], self.net_bsz):
            outputs.append(self.models[model](embeds[i:i + self.net_bsz]))

        # (net_bsz, ?)
        outputs = torch.cat(outputs, 0)
        outputs[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
        # (render_bsz, N_coarse, ?)
        outputs = torch.reshape(outputs, list(xyz_shape[:-1]) + [outputs.shape[-1]])
        return outputs
    
    def render_batch(self, rays, verbose=False, 
                     retraw=False, test=False):
        """
        Volumetric rendering for a single batch.

        rays: All information necessary for sampling along a ray.
        retraw: return model's raw, unprocessed predictions.
        verbose: print more debugging info [unused].

        Returns:
        rgb_map: (render_bsz, 3). Estimated RGB color of a ray. Comes from fine model.
        disparity_map: (render_bsz,). Disparity map. 1 / depth.
        accumulation_map: (render_bsz,). Accumulated opacity along each ray. Comes from fine model.
        raw: (render_bsz, N_coarse, ?). Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disparity_map. Output for coarse model.
        acc0: See accumulation_map. Output for coarse model.
        z_std: (render_bsz,). Standard deviation of distances along ray for each sample.
        """
        if test is True:
            perturb = self.perturb_test
            raw_noise_std = self.raw_noise_std_test
        else:
            perturb = self.perturb
            raw_noise_std = self.raw_noise_std

        # unpack batch of rays
        rays_o = rays[:, 0:3] # (render_bsz, 3)
        rays_d = rays[:, 3:6] # (render_bsz, 3)
        near = rays[:, 6].unsqueeze(-1) # (render_bsz, 1)
        far = rays[:, 7].unsqueeze(-1) # (render_bsz, 1)
        dir = rays[:, 8:11] if rays.shape[-1] > 8 else None # (render_bsz, 3)

        # prepare for sampling
        t_vals = torch.linspace(0., 1., steps=self.N_coarse)
        # (render_bsz, N_coarse)
        if not self.lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        if perturb > 0.:
            # get intervals between samples
            # mid: (render_bsz, N_coarse - 1)
            # upper, lower: same as z_vals
            mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand # same shape

        # concatenate positional samples
        # (render_bsz, N_coarse, 3)
        rearrange(rays_o, 'r c -> r () c')
        xyz = rearrange(rays_o, 'r c -> r () c') \
            + rearrange(rays_d, 'r c -> r () c') \
            * rearrange(z_vals, 'r cs -> r cs ()')
        raw = self.run_network(inputs={"xyz": xyz, "dir": dir},
                               model="coarse")
        coarse_preds = self.raw2outputs(raw, z_vals, rays_d, 
                                        raw_noise_std=raw_noise_std)
        weights = coarse_preds.weights

        if self.N_fine > 0:
            z_vals_mid = .5 * (z_vals[:,1:] + z_vals[:,:-1]) # (render_bsz, N_coarse - 1)
            z_samples = self.sample_pdf(z_vals_mid, weights[:,1:-1], 
                                        perturb=perturb)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            # concatenate positional samples
            # (render_bsz, N_coarse + N_fine, 3)
            xyz = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[:, :, None] 
            model_name = "fine" if self.models["fine"] is not None else "coarse"
            raw = self.run_network(inputs={"xyz": xyz, "dir": dir},
                                   model=model_name)

            fine_preds = self.raw2outputs(raw, z_vals, rays_d, 
                                            raw_noise_std=raw_noise_std)

        ret = {}
        for k in coarse_preds:
            ret[f"coarse_{k}"] = coarse_preds[k]
        if self.N_fine > 0:
            for k in fine_preds:
                ret[f"fine_{k}"] = fine_preds[k]
            ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False) # (render_bsz,)
        if retraw:
            ret['raw'] = raw

        # debug_dict(ret)
        return ret

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: (render_bsz, samples, ?). Prediction from model.
            z_vals: (render_bsz, samples). Integration time.
            rays_d: (render_bsz, 3). Direction of each ray.
        Returns:
            rgb_map: (render_bsz, 3). Estimated RGB color of a ray.
            disparity_map: (render_bsz,). Disparity map. Inverse of depth map.
            accumulation_map: (render_bsz,). Sum of weights along each ray.
            weights: (render_bsz, samples). Weights assigned to each sampled color.
            depth_map: (render_bsz,). Estimated distance to object.
        
        Section 4, pg. 6. Volume Rendering with Radiance Fields
        C(r) = sum (i = 1 to N) T_i * (1-exp(sigma*dist))c_i

        volumetric rendering of a color c with density sigma. c(t) and sigma(t) are the color & density at point r(t)
        """
        results = {}

        # differences of z vals = dists. last value is 1e10 (astronomically large in the context)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[:,:1].shape)], -1)
        dists = dists * torch.norm(rearrange(rays_d, 'r c -> r () c'), dim=-1) # (render_bsz, samples)

        static_sigmas = raw[...,3]    
        if raw_noise_std > 0:
            static_sigmas += torch.randn(static_sigmas.shape) * raw_noise_std

        static_alphas = alpha_composite(static_sigmas, dists) # (render_bsz, N_coarse)
        if self.use["transient"]:
            transient_rgbs = raw[..., 4:7]
            transient_sigmas = raw[..., 7]
            transient_betas = raw[..., 8]

            transient_alphas = alpha_composite(transient_sigmas, dists) 
            alphas = alpha_composite(static_sigmas + transient_sigmas, dists) 

        ones = torch.ones((static_alphas.shape[0], 1)) # for the prod start with 1s
        weights = static_alphas * torch.cumprod(torch.cat([ones, 1. - static_alphas + 1e-10], -1), -1)[:, :-1] 

        static_rgbs = torch.sigmoid(raw[...,:3]) # (render_bsz, samples, 3)
        static_rgbs = torch.sum(weights[...,None] * rgb, -2)  # (render_bsz, 3)  
        accumulation = torch.sum(weights, -1)
        if self.white_bkgd:
            static_rgbs += (1. - accumulation[...,None])
        results["static_rgbs"] = static_rgbs    
        results["accumulation"] = accumulation

        depth = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        disparity = 1./torch.max(1e-10 * torch.ones_like(depth), depth)
        results["depth"] = depth
        results["disparity"] = disparity

        results['transient_sigmas'] = transient_sigmas

        # Calculate weights sparsity loss
        try:
            entropy = Categorical(probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
        except:
            print("Something occured in calculating weights sparsity loss. Begin debugging...")
            pdb.set_trace()
        results["weights"] = weights
        results["sparsity_loss"] = entropy 

        return results 

    def sample_pdf(self, bins, weights, perturb=0, eps=1e-5):
        """
        Sample a pdf

        bins: (render_bsz, N_coarse - 1)
        weights: (render_bsz, N_coarse - 2)
        perturb: 0 for uniform sampling, 1 for jittered (stratified rand points) sampling

        create a cdf from the weights obtained from the coarse model
        """
        N_fine = self.N_fine
        # Get pdf
        weights = weights + eps # prevent nans
        pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum')
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (render_bsz, N_coarse-1)

        # Take uniform samples
        if not perturb:
            u = torch.linspace(0, 1, N_fine)
            u = u.expand((cdf.shape[0], N_fine))
        else:
            u = torch.rand((cdf.shape[0], N_fine))

        # Invert CDF
        u = u.contiguous()
        idxs = torch.searchsorted(cdf, u, right=True) # (render_bsz, N_fine)
        below = torch.clamp_min(idxs - 1, 0)
        above = torch.clamp_max(idxs, cdf.shape[-1] - 1)

        # (render_bsz, N_fine * 2)
        idxs_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2) 
        cdf_g = rearrange(torch.gather(cdf, 1, idxs_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
        bins_g = rearrange(torch.gather(bins, 1, idxs_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

        denom = cdf_g[...,1] - cdf_g[...,0] # (render_bsz, N_fine)
        denom[denom < eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)
        samples = bins_g[...,0] + (u - cdf_g[...,0]) / denom * (bins_g[...,1] - bins_g[...,0])
        return samples # (render_bsz, N_fine)

def prepare_rays(cc: CameraConfig, use: dict,
                rays = None, c2w = None,
                c2w_staticcam = None, ndc: bool = False, ):
    """
    Prepare rays for rendering in batches

    returns:
        rays: (N, 8) or (N, 11) if use_viewdirs
        og_shape: shape of rays_d before flattening: (train_bsz,) or (h, w)
    """
    if c2w is not None:
        # special case to render full image
        # both (h, w, 3)
        rays_o, rays_d = get_rays(cc.height, cc.width, c2w, 
                                  focal=cc.focal, K=cc.k)
    else:
        # use provided ray batch
        rays_o, rays_d = rays["o"], rays["d"]

    # we take note of the rays' shape 
    # prior to flattening 
    # for reshaping later
    # (train_bsz,) or (h, w) 
    og_shape = list(rays_d.shape[:-1])

    # let us refer to h*w as train_bsz

    if use["dirs"]:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(cc.height, cc.width, 
                                      c2w_staticcam, focal=cc.focal, K=cc.k)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = rearrange(viewdirs, 'h w c -> (h w) c') # (train_bsz, 3)

    # normalized device coordinates, for forward facing scenes
    # shape unchanged
    if ndc:
        rays_o, rays_d = get_ndc_rays(cc.height, cc.width, focal=cc.k[0][0], 
                                        near=1., rays_o=rays_o, rays_d=rays_d)

    # Create ray batch
    # (train_bsz, 3)
    if len(rays_o.shape) == 3:
        rays_o = rearrange(rays_o, 'h w c -> (h w) c')
        rays_d = rearrange(rays_d, 'h w c -> (h w) c')  

    # (train_bsz, 1)
    bounds_shape = (rays_d.shape[0], 1)
    near = torch.full(bounds_shape, cc.near)
    far = torch.full(bounds_shape, cc.far)

    # (train_bsz, 3 + 3 + 1 + 1) (+3 if use_viewdirs)
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use["dirs"]:
        rays = torch.cat([rays, viewdirs], -1)
    if use["appearance"]:
        appearance = rays["t"]
        if len(appearance) == 3:
            appearance = rearrange(appearance, 'h w c -> (h w) c')  
        rays = torch.cat(rays[rays, appearance], -1)

    # (train_bsz, ?) and ((train_bsz,) or (h, w))
    return rays, og_shape

if __name__ == '__main__':
    """
    testing purposes only
    """
    render_bsz = 4
    N_coarse = 101

    n, f = 0.2, 0.6
    t = np.linspace(0, 1, N_coarse)

    z = 1 / ((1 - t) / n + t / f)

    z = np.broadcast_to(z, (render_bsz, N_coarse))

    rays_d = np.random.rand(render_bsz, 3)

    pts = rays_d[...,None,:] * z[...,:,None] # [render_bsz, N_coarse, 3]

    # plot all pts 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[...,0], pts[...,1], pts[...,2])
    plt.show()
