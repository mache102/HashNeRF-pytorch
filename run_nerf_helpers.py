import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import imageio

from tqdm import trange

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ray_util import get_rays, get_ndc_rays
from util import to_8b, psnr

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

DEBUG = False
class VolRenderer:
    """
    Volumetric rendering (nerf evaulation)
    """
    def __init__(self, dataset_type: str, 
                 h: int, w: int, focal: float = None, k: np.ndarray = None,
                 proc_chunk: int = 1024*32, near: float = 0., far: float = 1.,
                 use_viewdirs: bool = False):
        """
        h: height of image in pixels
        w: width of image in pixels
        focal: focal length of pinhole camera

        proc_chunk: max number of rays to process simultaneously. 
            prevent OOM, does not affect final results
        near, far: ray distances
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.

        args: arguments
        """     
        self.dataset_type = dataset_type

        self.h = h
        self.w = w
        self.focal = focal

        if k is None:
            self.k = np.array([
                [focal, 0, 0.5 * w],
                [0, focal, 0.5 * h],
                [0, 0, 1]
            ])
        else:
            self.k = k

        self.proc_chunk = proc_chunk
        self.near = near
        self.far = far
        self.use_viewdirs = use_viewdirs

    def save_imgs(self, rgb, depth, savedir, i,
                  method="imageio"):
        """
        save rgb and depth as a figure
        """
        if method == "imageio":
            fn = os.path.join(savedir, f'rgb_{i:03d}.png')
            imageio.imwrite(fn, to_8b(rgb))
        
            fn = os.path.join(savedir, f'd_{i:03d}.png')
            imageio.imwrite(fn, to_8b(depth))     
        
        elif method == "matplotlib":
            # save rgb and depth as a figure
            fig = plt.figure(figsize=(25, 15))
            ax = fig.add_subplot(1, 2, 1)
            rgb = to_8b(rgb)
            ax.imshow(rgb)
            ax.axis('off')
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(depth, cmap='plasma', vmin=0, vmax=1)
            ax.axis('off')

            fn = os.path.join(savedir, f'{i:03d}.png')
            plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        else:
            raise NotImplementedError

    def render_path(self, poses, render_kwargs,
                    gt_imgs=None, savedir=None, render_factor = 1):
        """
        Rendering for test set
        """
        h = self.h * render_factor
        w = self.w * render_factor

        rgbs = []
        depths = []

        """
        rendering pipeline for equirect and standard rectilinear options

        (most operations take place in render() and its called funcs)
        """
        if self.dataset_type == "equirect":
            """
            equirectangular datasets
            """
            rays_o, rays_d = poses
            batch = h * w    
            for i in trange(rays_o.shape[0] // batch):
                start = i * batch
                end = (i + 1) * batch
                rays, reshape_to = self.prepare_rays(rays=[rays_o[start:end], 
                                                           rays_d[start:end]], 
                                                     ndc=False)
                rgb, depth, _, _ = \
                    self.render(rays=rays, reshape_to=reshape_to,
                                **render_kwargs)
                
                if i == 0:
                    print(rgb.shape, depth.shape)
                rgb = rgb.reshape(h, w, 3).cpu().numpy()
                rgbs.append(rgb)

                depth = depth.reshape(h, w).cpu().numpy()
                depths.append(depth)

                if savedir is not None:
                    self.save_imgs(rgb, depth, savedir, i)
        
            rgbs = np.stack(rgbs, 0)
            depths = np.stack(depths, 0) 

        else:
            """
            Non equirectangular datasets
            """
            psnrs = []
            for i in trange(len(poses)):
                c2w = poses[i]

                rgb, depth, _, _ = \
                    self.render(*self.prepare_rays(c2w=c2w[:3,:4]), **render_kwargs)
                if i == 0:
                    print(rgb.shape, depth.shape)

                rgb = rgb.cpu().numpy()
                rgbs.append(rgb)
                depth = ((depth - self.near) / (self.far - self.near)).cpu().numpy()
                depths.append(depth)

                if gt_imgs is not None and render_factor == 1:
                    try:
                        gt_img = gt_imgs[i].cpu().numpy()
                    except:
                        gt_img = gt_imgs[i]
                    
                    p = psnr(rgb, gt_img)
                    psnrs.append(p)

                if savedir is not None:
                    self.save_imgs(rgb, depth, savedir, i)         

            rgbs = np.stack(rgbs, 0)
            depths = np.stack(depths, 0)

            if gt_imgs is not None and render_factor == 1:
                avg_psnr = sum(psnrs)/len(psnrs)
                print("Avg PSNR over Test set: ", avg_psnr)
                with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
                    pickle.dump(psnrs, fp)

        return rgbs, depths

    def prepare_rays(self, rays = None, c2w = None,
            ndc: bool = True, c2w_staticcam = None):
        """
        Prepare rays for rendering in batches
        """
        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = get_rays(self.h, self.w, c2w, 
                                      focal=self.focal, K=self.k)
        else:
            # use provided ray batch
            rays_o, rays_d = rays

        # use viewdirs (rays_d)
        if self.use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(self.h, self.w, c2w_staticcam, focal=self.focal, K=self.k)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = viewdirs.reshape(-1,3).float()

        # we take note of the rays' shape 
        # prior to flattening 
        # for reshaping later
        sh = rays_d.shape # [..., 3]
        
        # normalized device coordinates, for forward facing scenes
        if ndc:
            rays_o, rays_d = get_ndc_rays(self.h, self.w, focal=self.k[0][0], 
                                          near=1., rays_o=rays_o, rays_d=rays_d)

        # Create ray batch
        # (N, 3)
        rays_o = rays_o.reshape(-1,3).float()
        rays_d = rays_d.reshape(-1,3).float()

        near, far = self.near * torch.ones_like(rays_d[...,:1]), self.far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        return rays, list(sh[:-1])

    def render(self, rays, reshape_to, **kwargs):

        """
        Main ray rendering function

        Args:
        rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.
        
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
        """
        
        # Render and reshape
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        render_outputs = {}
        for i in range(0, rays.shape[0], self.proc_chunk):
            chunk_output = self.render_rays(rays[i:i + self.proc_chunk], **kwargs)

            for k in chunk_output:
                if k not in render_outputs:
                    render_outputs[k] = []
                render_outputs[k].append(chunk_output[k])
        render_outputs = {k : torch.cat(render_outputs[k], 0) for k in render_outputs}

        for k in render_outputs:
            k_sh = reshape_to + list(render_outputs[k].shape[1:])
            render_outputs[k] = torch.reshape(render_outputs[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        return_priority = [render_outputs[k] for k in k_extract]
        ret_dict = {k: render_outputs[k] for k in render_outputs if k not in k_extract}
        return return_priority + [ret_dict]

    def render_rays(self, rays,
                    network_fn,
                    network_query_fn,
                    N_samples,
                    embed_fn=None,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    network_fine=None,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    verbose=False,
                    seed: int = None):
        """Volumetric rendering.
        Args:
        rays: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
            in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """

        """
        RAY BATCH
        Shape: (chunk_size, 8) or (chunk_size, 11) (if use_viewdirs)

        0:3 - ray origin
        3:6 - ray direction
        6:8 - near and far bounds
        8:11 (if use_viewdirs) - viewing direction
        """
        if seed:
            torch.manual_seed(0)

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:,0:3], rays[:,3:6] # [N_rays, 3] each
        viewdirs = rays[:,-3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[...,6:8], [-1,1,2]) # (chunk_size, 1, 2)
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        # z_vals is now of shape (N_rays, N_samples)
        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = \
            raw2outputs(raw, z_vals, rays_d, 
                        raw_noise_std, white_bkgd, seed=seed)

        if N_importance > 0:

            rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = rgb_map, depth_map, acc_map, sparsity_loss

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, 
                                det=(perturb==0.), seed=seed)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            run_fn = network_fn if network_fine is None else network_fine
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = \
                raw2outputs(raw, z_vals, rays_d, 
                            raw_noise_std, white_bkgd, seed=seed)

        ret = {
            'rgb_map': rgb_map, 
            'depth_map': depth_map, 
            'acc_map': acc_map, 
            'sparsity_loss': sparsity_loss
        }

        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['sparsity_loss0'] = sparsity_loss_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, net_chunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    output = []
    for i in range(0, embedded.shape[0], net_chunk):
        output.append(fn(embedded[i:i + net_chunk]))

    outputs_flat = torch.cat(output, 0)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
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

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        # sigma
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    ones = torch.ones((alpha.shape[0], 1))
    weights = alpha * torch.cumprod(torch.cat([ones, 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    try:
        entropy = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
    except:
        pdb.set_trace()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

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