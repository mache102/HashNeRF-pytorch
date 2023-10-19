import torch

from ray_util import *
from load.load_data import CameraConfig

def prepare_rays(cc: CameraConfig, 
                rays = None, c2w = None,
                c2w_staticcam = None, ndc: bool = False, 
                use_viewdirs: bool = False):
    """
    Prepare rays for rendering in batches

    returns:
        rays: (N, 8) or (N, 11) if use_viewdirs
        reshape_to: shape of rays_d before flattening
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(cc.h, cc.w, c2w, 
                                  focal=cc.focal, K=cc.k)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    # use viewdirs (rays_d)
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(cc.h, cc.w, c2w_staticcam, focal=cc.focal, K=cc.k)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = viewdirs.reshape(-1,3).float()

    # we take note of the rays' shape 
    # prior to flattening 
    # for reshaping later
    sh = rays_d.shape # [..., 3]
    
    # normalized device coordinates, for forward facing scenes
    if ndc:
        rays_o, rays_d = get_ndc_rays(cc.h, cc.w, focal=cc.k[0][0], 
                                        near=1., rays_o=rays_o, rays_d=rays_d)

    # Create ray batch
    # (N, 3)
    rays_o = rays_o.reshape(-1,3).float()
    rays_d = rays_d.reshape(-1,3).float()

    near, far = cc.near * torch.ones_like(rays_d[...,:1]), cc.far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    reshape_to = list(sh[:-1])
    return rays, reshape_to
