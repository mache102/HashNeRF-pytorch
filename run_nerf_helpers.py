import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import time
import imageio

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ray_util import get_rays, get_ndc_rays

from dataclasses import dataclass

@dataclass 
class RayPredictions:
    rgb: torch.Tensor
    depth: torch.Tensor
    disparity: torch.Tensor
    accumulation: torch.Tensor
    weights: torch.Tensor
    sparsity_loss: torch.Tensor

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

DEBUG = False
"""
render_path():
    ...
    for each pose 
        ...
        render():
            ...
            get_rays()
            get_ndc_rays()
            ...
            for each chunk of rays_flat
                render_rays():
                    network_query_fn() == run_network():
                        batchify()
                    ...
                    raw2outputs()
                    sample_pdf()
                    raw2outputs()
"""

def render_path(render_poses, hwf, render_kwargs, args, K=None, chunk=None, gt_imgs=None, savedir=None, render_factor=0):
    """
    Rendering for test set
    """

    """
    preproc (get hwf, render factor, etc)
    """
    dataset_type = args.dataset_type
    if len(hwf) == 3:
        H, W, focal = hwf
    else:
        H, W = hwf
        focal = None
    near, far = render_kwargs['near'], render_kwargs['far']

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        if focal is not None:
            focal = focal/render_factor

    rgbs = []
    depths = []

    """
    rendering pipeline for equirect and standard rectilinear options

    (most operations take place in render() and its called funcs)
    """
    if dataset_type == "equirect":
        rays_o, rays_d = render_poses
        t = time.time()
        batch = H*W
        
        for i in tqdm(range(rays_o.shape[0] // batch)):
            print(i, time.time() - t)
            t = time.time()
            rgb, dep, grad,  _ = render(H, W, rays=[rays_o[i*batch:(i+1)*batch], rays_d[i*batch:(i+1)*batch]], **render_kwargs, ndc=False)
            if i==0:
                print(rgb.shape, dep.shape)

            if savedir is not None:
                rgb8 = rgb.reshape(H, W, 3).cpu().numpy()
                rgbs.append(rgb8)
                
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, to8b(rgb8))
                
                dep8 = dep.reshape(H, W).cpu().numpy()
                depths.append(dep8)
                filename = os.path.join(savedir, 'd_{:03d}.png'.format(i))
                
                imageio.imwrite(filename, to8b(dep8))            
        
        
        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0) 
    else:
        psnrs = []

        t = time.time()
        for i, c2w in enumerate(tqdm(render_poses)):
            print(i, time.time() - t)
            t = time.time()
            rgb, depth, acc, _ = render(H, W, K=K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)

            rgbs.append(rgb.cpu().numpy())
            # normalize depth to [0,1]
            depth = (depth - near) / (far - near)
            depths.append(depth.cpu().numpy())
            if i==0:
                print(rgb.shape, depth.shape)

            if gt_imgs is not None and render_factor==0:
                try:
                    gt_img = gt_imgs[i].cpu().numpy()
                except:
                    gt_img = gt_imgs[i]
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
                print(p)
                psnrs.append(p)

            if savedir is not None:
                # save rgb and depth as a figure
                fig = plt.figure(figsize=(25,15))
                ax = fig.add_subplot(1, 2, 1)
                rgb8 = to8b(rgbs[-1])
                ax.imshow(rgb8)
                ax.axis('off')
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(depths[-1], cmap='plasma', vmin=0, vmax=1)
                ax.axis('off')
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                # save as png
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                # imageio.imwrite(filename, rgb8)


        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0)
        if gt_imgs is not None and render_factor==0:
            avg_psnr = sum(psnrs)/len(psnrs)
            print("Avg PSNR over Test set: ", avg_psnr)
            with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
                pickle.dump(psnrs, fp)

    return rgbs, depths


def render(H, W, focal=None, K=None, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """
    Main ray rendering function
    ====

    ====
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near, far: ray distances
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
       
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, c2w, focal=focal, K=K)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, c2w_staticcam, focal=focal, K=K)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = get_ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    # (N, 3)
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    # all_ret = batchify_rays(rays, chunk, **kwargs)
    """
    Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(rays[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    # end batchify_rays()


    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_rays(rays,
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
                pytest=False):
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
    # unpack batch of rays
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:,0:3], rays[:,3:6]
    viewdirs = rays[:,-3:] if rays.shape[-1] > 8 else None
    bounds = torch.reshape(rays[...,6:8], [-1,1,2]) # [N_rays, 1, 2]
    near, far = bounds[...,0], bounds[...,1] # [N_rays, 1]

    # prepare for sampling
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

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
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
    # raw = run_network(inputs={"pos": positional, "dir": viewdirs},
    #                         model_name="coarse")
    raw = network_query_fn(pts, viewdirs, network_fn)
    coarse_preds = raw2outputs(raw, z_vals, rays_d, 
                                    raw_noise_std=raw_noise_std)
    weights = coarse_preds.weights

    if N_importance > 0:
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1],  N_samples,
                            det=(perturb==0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # concatenate positional samples
        # [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] 
        # model_name = "fine" if models["fine"] is not None else "coarse"
        # raw = run_network(inputs={"pos": positional, "dir": viewdirs},
        #                         model_name=model_name)
        raw = network_query_fn(pts, viewdirs, network_fine)
        fine_preds = raw2outputs(raw, z_vals, rays_d, 
                                        raw_noise_std=raw_noise_std)

    if N_importance > 0:
        ret = {
            "rgb0": coarse_preds.rgb,
            "depth0": coarse_preds.depth,
            "acc0": coarse_preds.accumulation,
            "sparsity_loss0": coarse_preds.sparsity_loss,     

            "rgb_map": fine_preds.rgb,
            "depth_map": fine_preds.depth,
            "acc_map": fine_preds.accumulation,
            "sparsity_loss": fine_preds.sparsity_loss,
            "z_std": torch.std(z_samples, dim=-1, unbiased=False) # (N_rays,)
        }
    else: 
        ret = {
            "rgb_map": coarse_preds.rgb,
            "depth_map": coarse_preds.depth,
            "accumulation_map": coarse_preds.accumulation,
            "sparsity_loss": coarse_preds.sparsity_loss,
        }
    if retraw:
        ret['raw'] = raw

    # debug_dict(ret)
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

    outputs_flat = batchify(fn, net_chunk)(embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


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


    return RayPredictions(rgb=rgb_map, depth=depth_map, disparity=disp_map,
                        accumulation=acc_map, weights=weights,
                        sparsity_loss=sparsity_loss)


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



# a js script that constantly listens for elements with class combination = "style-scope ytd-popup-container"
# when such an element is found, remove the element and log the operation in console

# targetElement = "style-scope ytd-popup-container"

# # event listener here
# document.addEventListener('DOMContentLoaded', function() {
#     var element = document.getElementsByClassName(targetElement);
#     element.parentNode.removeChild(element);
# }, false);