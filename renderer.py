import argparse 
import matplotlib.pyplot as plt
import numpy as np
import pdb

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import defaultdict

from load_data import CameraConfig
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
    def __init__(self, models: dict, embedders: dict,
                 usage: dict, args: argparse.Namespace):
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
        self.usage = usage

        self.unpack_args(args)

    def unpack_args(self, args):
        """
        Unpack arguments for rendering.

        Originally passed to render methods
        through **render_kwargs.

        Special case (testing):
            perturb = False 
            raw_noise_std = 0

        seed: random seed 

        render_bsz: num of rays processed in parallel, adjust to prevent OOM
        net_bsz: num of rays sent to model in parallel (render_bsz % net_bsz == 0)
        perturb: 0 for uniform sampling, 1 for jittered (stratified rand points) sampling
        N_coarse: num of coarse samples per ray
        N_fine: num of additional fine samples per ray
        raw_noise_std: std dev of noise added to regularize sigma_a output, 1e0 recommended
        """
        self.seed = args.seed 
        self.render_bsz = args.render_bsz
        self.net_bsz = args.net_bsz
        self.perturb = args.perturb
        self.N_coarse = args.N_coarse 
        self.N_fine = args.N_fine 
        self.raw_noise_std = args.raw_noise_std 

        self.bbox = args.bbox

        self.perturb_test = False 
        self.raw_noise_std_test = 0

        if self.seed is not None:
            torch.manual_seed(self.seed)

    def get_mask(self, x):
        """
        hash encoding mask 
        """
        # move this from hash encoding func to here
        mask = x == torch.max(torch.min(x, self.bbox[1]), self.bbox[0])
        mask = mask.sum(dim=-1) == mask.shape[-1]
        return mask 

    def render(self, rays, **kwargs):
        """
        Rendering entry function
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

    def run_network(self, inputs, model_name):
        """
        Prepare inputs (embed, concat) and apply network
        """
        pos_ = inputs["pos"] # use this for reshaping (second to last line)
        pos = pos_.reshape(-1, pos_.shape[-1])
        pos_embed = self.embedders["xyz"](pos)

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

        mask = self.get_mask(pos)
        outputs = torch.cat(outputs, 0)
        outputs[~mask, -1] = 0 # set sigma to 0 for invalid points
        outputs = torch.reshape(outputs, list(pos_.shape[:-1]) + [outputs.shape[-1]])
        return outputs
    
    def render_batch(self, rays, test=False):
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
        color: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disparity_map: [num_rays]. Disparity map. 1 / depth.
        opacity: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See color. Output for coarse model.
        disp0: See disparity_map. Output for coarse model.
        acc0: See opacity. Output for coarse model.
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
        viewdirs = rays[:,8:11] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[...,6:8], [-1,1,2]) # [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1] # [N_rays, 1]

        # prepare for sampling
        t_vals = torch.linspace(0., 1., steps=self.N_coarse)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, self.N_coarse])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        # concatenate positional samples
        # [N_rays, N_coarse, 3]
        positional = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
        raw = self.run_network(inputs={"pos": positional, "dir": viewdirs},
                               model_name="coarse")
        coarse_out = self.raw2outputs(raw, z_vals, rays_d, 
                                        raw_noise_std=raw_noise_std)
        weights = coarse_out["weights"]

        if self.N_fine > 0:
            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], 
                                det=(perturb==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            # concatenate positional samples
            # [N_rays, N_coarse + N_fine, 3]
            positional = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
            model_name = "fine" if self.models["fine"] is not None else "coarse"
            raw = self.run_network(inputs={"pos": positional, "dir": viewdirs},
                                   model_name=model_name)

            fine_out = self.raw2outputs(raw, z_vals, rays_d, 
                                        raw_noise_std=raw_noise_std)

        ret = {}
        for k in coarse_out:
            ret[f"coarse_{k}"] = coarse_out[k]
        if self.N_fine > 0:
            for k in fine_out:
                ret[f"fine_{k}"] = fine_out[k]
        return ret

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            color: [num_rays, 3]. Estimated RGB color of a ray.
            disparity: [num_rays]. Disparity map. Inverse of depth map.
            opacity: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth: [num_rays]. Estimated distance to object.
        
        Section 4, pg. 6. Volume Rendering with Radiance Fields
        C(r) = sum (i = 1 to N) T_i * (1-exp(sigma*dist))c_i

        volumetric rendering of a color c with density sigma. c(t) and sigma(t) are the color & density at point r(t)
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        # differences of z vals = dists. last value is 1e10 (astronomically large in the context)
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_coarse]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_coarse, 3]
        noise = 0.
        if raw_noise_std > 0.: # sigma
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # sigma_loss = sigma_sparsity_loss(raw[...,3])
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_coarse]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

        ones = torch.ones((alpha.shape[0], 1))
        transmittance = torch.cumprod(torch.cat([ones, 1.-alpha + 1e-10], -1), -1)[:, :-1]
        weights = alpha * transmittance
        color = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]      

        depth = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        # disparity = 1./torch.max(1e-10 * torch.ones_like(depth), depth)
        opacity = torch.sum(weights, -1)


        # Calculate weights sparsity loss
        try:
            entropy = Categorical(probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
        except:
            print("Something occured in calculating weights sparsity loss. Begin debugging...")
            pdb.set_trace()
        sparsity_loss = entropy

        return {
            "color": color,
            "depth": depth,
            "opacity": opacity,
            # "disparity": disparity,
            "weights": weights,
            "sparsity_loss": sparsity_loss
        }

    def sample_pdf(self, bins, weights, det=False):
        N_coarse = self.N_fine
        # Get pdf
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_coarse)
            u = u.expand(list(cdf.shape[:-1]) + [N_coarse])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_coarse])

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_coarse, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

if __name__ == '__main__':
    """
    testing purposes only
    """
    N_rays = 4
    N_coarse = 101

    n, f = 0.2, 0.6
    t = np.linspace(0, 1, N_coarse)

    z = 1 / ((1 - t) / n + t / f)

    z = np.broadcast_to(z, (N_rays, N_coarse))

    rays_d = np.random.rand(N_rays, 3)

    pts = rays_d[...,None,:] * z[...,:,None] # [N_rays, N_coarse, 3]

    # plot all pts 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[...,0], pts[...,1], pts[...,2])
    plt.show()

