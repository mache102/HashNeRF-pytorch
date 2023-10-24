import torch 

from einops import rearrange
from load.load_data import CameraConfig
from ray_util import get_rays, get_ndc_rays

def parse_rays(rays, samples, usage): 
    """
    Prepare rays for embedding
    """
    # unpack batch of rays
    rays_o = rays[:, 0:3] # (render_bsz, 3)
    rays_d = rays[:, 3:6] # (render_bsz, 3)

    xyz = rearrange(rays_o, 'r c -> r () c') \
        + rearrange(rays_d, 'r c -> r () c') \
        * rearrange(samples, 'r cs -> r cs ()')
    xyz_shape = xyz.shape
    xyz = rearrange(xyz, 'r cs c -> (r cs) c') # (render_bsz * N_coarse, 3)

    dir = None
    if usage["dirs"]:
        dir = rays[:, 8:11]
        dir = rearrange(dir, 'r c -> r () c').expand(xyz_shape)
        dir = rearrange(dir, 'r cs c -> (r cs) c') # (render_bsz * N_coarse, 3)

    inputs = {
        "xyz": xyz,
        "dir": dir,
        "appearance": rays["appearance"],
        "transient": rays["transient"]
    }

    return inputs

def prepare_rays(cc: CameraConfig, usage: dict,
                rays = None, c2w = None,
                c2w_staticcam = None, ndc: bool = False):
    """
    Prepare rays for batch rendering
    """
    if c2w is not None:
        # special case to render full image
        # both (h, w, 3)
        rays_o, rays_d = get_rays(cc.height, cc.width, c2w, 
                                  focal=cc.focal, K=cc.k)
    else:
        # usage provided ray batch
        rays_o, rays_d = rays["o"], rays["d"]

    # we take note of the rays' shape 
    # prior to flattening 
    # for reshaping later
    # (train_bsz,) or (h, w) 
    og_shape = list(rays_d.shape[:-1])

    # let us refer to h*w as train_bsz

    if usage["dirs"]:
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

    # (train_bsz, 3 + 3 + 1 + 1) 
    # +3 if dir 
    # +1 if appearance
    # +1 if transient
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if usage["dirs"]:
        rays = torch.cat([rays, viewdirs], -1)
    if usage["appearance"]:
        appearance = rays["appearance"]
        if len(appearance) == 3:
            appearance = rearrange(appearance, 'h w c -> (h w) c')  
        rays = torch.cat(rays[rays, appearance], -1)

    if usage["transient"]:
        transient = rays["transient"]
        if len(transient) == 3:
            transient = rearrange(transient, 'h w c -> (h w) c')  
        rays = torch.cat(rays[rays, transient], -1)

    # (train_bsz, ?) and ((train_bsz,) or (h, w))
    return rays, og_shape

def post_process(inputs, raw, bbox):
    max_xyz = torch.max(inputs["xyz"], dim=-1).values
    min_xyz = torch.min(inputs["xyz"], dim=-1).values
    keep_mask = (max_xyz >= bbox[0]) & (min_xyz <= bbox[1])
    keep_mask = keep_mask.all(dim=-1)
    # (net_bsz, ?)
    raw = torch.cat(raw, 0)
    raw[~keep_mask, 3] = 0 # set sigma to 0 for invalid points
    return raw