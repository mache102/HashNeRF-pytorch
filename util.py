import numpy as np
import os
import imageio 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from embedding.embedder import Embedder
from embedding.hash_encoding import HashEmbedder 
from embedding.spherical_harmonic import SHEncoder

# TODO: reorganize utils

def get_transform_matrix(translation, rotation):
    """
    torch tensor transformation matrix 
    from translation and rotation
    
    translation: (3,) torch tensor
    rotation: (2,) torch tensor
    ^ theta and phi angles
    
    returns: (4, 4) torch tensor
    """
    transform_matrix = torch.zeros((4, 4))
    transform_matrix[3, 3] = 1.0
    transform_matrix[:3, 3] = translation
    
    # Calculate rotation matrix from theta and phi
    theta, phi = rotation
    
    # Construct the rotation matrix
    theta_matx = torch.eye(3)
    theta_matx[0, 0] = torch.cos(theta)
    theta_matx[0, 1] = -torch.sin(theta)
    theta_matx[1, 0] = torch.sin(theta)
    theta_matx[1, 1] = torch.cos(theta)
    theta_matx[2, 2] = 1.0
    
    # Apply the phi rotation (rotation in the XY plane)
    phi_matx = torch.eye(3)
    phi_matx[0, 0] = torch.cos(phi)
    phi_matx[0, 1] = -torch.sin(phi)
    phi_matx[1, 0] = torch.sin(phi)
    phi_matx[1, 1] = torch.cos(phi)
    
    # Combine the two rotations
    rot_matx = torch.mm(theta_matx, phi_matx)
    
    # Copy the 3x3 rotation matrix into the top-left of the 4x4 transform_matrix
    transform_matrix[:3, :3] = rot_matx
    
    return transform_matrix

# kornia's create_meshgrid in numpy 
def create_meshgrid_np(H, W, normalized_coordinates=True):
    if normalized_coordinates:
        xs = np.linspace(-1, 1, W)
        ys = np.linspace(-1, 1, H)
    else:
        xs = np.linspace(0, W-1, W)
        ys = np.linspace(0, H-1, H)
    
    grid = np.stack(np.meshgrid(xs, ys), -1) # H, W, 2
    # transpose is not needed as the resulting grid 
    # is already the same as the one from kornia
    # grid = np.transpose(grid, [1, 0, 2]) # W, H, 2
    return grid


def create_expname(args):
    args.expname += f"_{args.i_embed}XYZ"
    args.expname += f"_{args.i_embed_views}VIEW"
    
    args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
    args.expname += "_lr"+str(args.lr) + "_decay"+str(args.lr_decay)
    args.expname += "_RAdam"
    if args.sparse_loss_weight > 0:
        args.expname += "_sparse" + str(args.sparse_loss_weight)
    args.expname += "_TV" + str(args.tv_loss_weight)
    #args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')

    return args.expname

def all_to_tensor(rays, device):
    """
    Iterate over all rays and convert to torch tensor
    (dataclass)
    """
    for key, value in rays.__dict__.items():
        if value is not None:
            rays.__dict__[key] = torch.Tensor(value).to(device)
    return rays

def shuffle_rays(rays):
    perm_anchor = rays.rgb 
    rand_idx = torch.randperm(perm_anchor.shape[0])

    for key, value in rays.__dict__.items():
        if value is not None:
            rays.__dict__[key] = value[rand_idx]

    return rays


def to_8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def psnr(pred_img, gt_img):
    return -10. * np.log10(np.mean(np.square(pred_img - gt_img)))

def img2mse(x, y):
    return np.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def save_configs(args):
    with open(os.path.join(args.savepath, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        with open(os.path.join(args.savepath, 'config.txt'), 'w') as file:
            file.write(open(args.config, 'r').read())

def debug_dict(d, name="dict", depth=0):
    """
    Check whether any bottommost layer of a dict
    contains nan or inf.

    (depths may vary by dict key, 
    so recursively check all layers)
    """
    for k in d:
        if isinstance(d[k], dict):    
            debug_dict(d[k], name=f"{name}.{k}", depth=depth + 1)
        else:
            if torch.isnan(d[k]).any():
                print(f"! [Numerical Error] {name}.{k} contains nan")
            if torch.isinf(d[k]).any():
                print(f"! [Numerical Error] {name}.{k} contains inf")

# example of debug_dict:
# x = {"a": torch.randn(3,3), "b": {"c": torch.Tensor([np.inf]), "d": torch.randn(3,3), "e": {"f": torch.randn(3,3)}}, "nan_val": torch.Tensor([np.nan])}
# debug_dict(x)

def save_imgs(rgb, depth, idx, savepath,
            method="imageio"):
    """
    save rgb and depth as a figure
    """
    if savepath is None:
        return 
    
    if method == "imageio":
        fn = os.path.join(savepath, f'rgb_{idx:03d}.png')
        imageio.imwrite(fn, to_8b(rgb))
    
        fn = os.path.join(savepath, f'd_{idx:03d}.png')
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

        fn = os.path.join(savepath, f'{idx:03d}.png')
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        raise NotImplementedError
    
def get_embedder(name, args, multires=None):
    """
    none: no embedding
    pos: Standard positional encoding (Nerf, section 5.1)
    hash: Hashed pos encoding
    sh: Spherical harmonic encoding
    """
    if name == "none":
        return nn.Identity(), 3
    elif name == "positional":
        assert multires is not None
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires - 1,
                    'num_freqs' : multires, 
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif name == "hash":
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif name == "sh":
        embed = SHEncoder()
        out_dim = embed.out_dim

    return embed, out_dim
