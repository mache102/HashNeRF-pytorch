import numpy as np
import torch
from kornia import create_meshgrid

"""
Ray utils for bbox 
"""
def get_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    dir_bounds = directions.view(-1, 3)
    # print("Directions ", directions[0,0,:], directions[H-1,0,:], directions[0,W-1,:], directions[H-1, W-1, :])
    # print("Directions ", dir_bounds[0], dir_bounds[W-1], dir_bounds[H*W-W], dir_bounds[H*W-1])

    return directions

def ray_from_directions(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

"""
Other/ multi-use
"""
def get_rays(H, W, c2w, focal=None, K=None):
    """
    Rays from 
    height, width, camera intrinsics, camera to world matrix

    Return shapes:
        rays_o: (H, W, 3)
        rays_d: (H, W, 3)
    """
    if focal is None and K is None:
        raise ValueError("focal and K cannot be both None")
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()

    if K is None:
        dirs = torch.stack([(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, c2w, focal=None, K=None):
    """
    get_rays in numpy
    """
    if focal is None and K is None:
        raise ValueError("focal and K cannot be both None")
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    if K is None:
        dirs = np.stack([(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -np.ones_like(i)], -1)
    else:
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dirs becomes (H, W, 1, 3)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (render_bsz) or float, the depths of the near plane
        rays_o: (render_bsz, 3), the origin of the rays in world coordinate
        rays_d: (render_bsz, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (render_bsz, 3), the origin of the rays in NDC
        rays_d: (render_bsz, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    # (render_bsz) or (H, W)
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    # both are (render_bsz)
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    # ^^
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    # all are (N, 3)
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    # both are (N, 3)
    rays_o = torch.stack([o0, o1, o2], -1) 
    rays_d = torch.stack([d0, d1, d2], -1) 
    
    return rays_o, rays_d



if __name__ == '__main__':
    # test get_ray_directions
    H, W, focal = 400, 400, 1.0
    directions = get_directions(H, W, focal)
    print(directions.shape)

    # plot rays 3d

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dots = directions.view(-1, 3).numpy()
    ax.scatter(dots[:,0], dots[:,1], dots[:,2], c='r', marker='o')
    plt.show()
