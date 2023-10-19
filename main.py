import imageio
import numpy as np
import os
import pdb
import pickle
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Optional

import torch

from renderer import *
from create_nerf import create_nerf
from ray_util import *
from loss import sigma_sparsity_loss, total_variation_loss

from load.load_data import load_data

from util import create_expname, all_to_tensor, shuffle_rays, to_8b, save_configs
from parse_args import config_parser

from util import img2mse, mse2psnr

from dataclasses import dataclass 

# 20231010 15:25
np.random.seed(0)
DEBUG = False

   
def main():
    dataset = load_data(args)
    """
    Experiment saving
    """
    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = create_expname(args)
    expname = args.expname
    savepath = os.path.join(basedir, expname)
    args.savepath = savepath # for convenience
    os.makedirs(savepath, exist_ok=True)
    save_configs(args)

    """
    Create nerf model

    set render kwargs
    """
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "coarse": None,
        "fine": None
    }
    embedders = {
        "pos": None,
        "dir": None
    }

    """
    step 1: create embedding functions
    """
    
    # input ch as in model input ch
    embedders["pos"], input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        pos_embedder_params = list(embedders["pos"].parameters())

    input_ch_views = 0
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embedders["dir"], input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    """
    Step 2: Create coarse and fine models
    """
    @dataclass
class SigmaNetConfig:
    input_ch: int = 3
    layers: int = 3
    hdim: int = 64
    geo_feat_dim: int = 15
    skips: List[int] = field(default_factory=lambda: [4])

@dataclass
class ColorNetConfig:
    input_ch: int = 3
    layers: int = 4
    hdim: int = 64

    model_configs = {
        "coarse": {
            "sigma": SigmaNetConfig(input_ch=input_ch, 
                                    layers=2 if args.i_embed == 1 else 3,)
        }    
    }

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.i_embed==1:
        model_coarse = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        
    elif not args.use_gradient:
        model_coarse = NeRF(D=args.netdepth, W=args.netwidth,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    else:
        model_coarse = NeRFGradient(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        
    models["coarse"] = model_coarse
    grad_vars = list(models["coarse"].parameters())

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        elif not args.use_gradient:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRFGradient(D=args.netdepth_fine, W=args.netwidth_fine,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

        models["fine"] = model_fine
        grad_vars += list(models["fine"].parameters())
        
    

    """
    Step 3: create optimizer
    """
    # Create optimizer
    if args.i_embed == 1:
        optimizer = \
            RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': pos_embedder_params, 'eps': 1e-15}
            ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = \
            torch.optim.Adam(
                params=grad_vars, 
                lr=args.lrate, 
                betas=(0.9, 0.999)
            )

    start = 0
    savepath = args.savepath

    """
    Step 5: Load checkpoints if available
    """
    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(savepath, f) for f in sorted(os.listdir(os.path.join(savepath))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        models["coarse"].load_state_dict(ckpt['network_fn_state_dict'])
        if models["fine"] is not None:
            models["fine"].load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embedders["pos"].load_state_dict(ckpt['pos_embedder_state_dict'])

    if args.dataset_type == 'equirect':
        # all to tensor
        rays = shuffle_rays(all_to_tensor(rays, device))
        rays_test = all_to_tensor(rays_test, device)

        renderer = VolRenderer(dataset_type=args.dataset_type, h=H, w=W, 
                               proc_chunk=args.chunk, near=near, far=far, 
                               use_viewdirs=args.use_viewdirs)

    else:
        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        renderer = VolRenderer(dataset_type=args.dataset_type, h=H, w=W,
                               focal=focal, k=K, 
                               proc_chunk=args.chunk, near=near, far=far, 
                               use_viewdirs=args.use_viewdirs)
    """
    Skip to render only
    """
    # Short circuit if only rendering out from trained model
    
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.dataset_type == 'equirect':
                if args.stage > 0:
                    testsavedir = os.path.join(savepath, 'renderonly_stage_{}_{:06d}'.format(args.stage, start))
                else:
                    testsavedir = os.path.join(savepath, 'renderonly_train_{}_{:06d}'.format('test' if args.render_test else 'path', start))

                eval_test_omninerf(renderer, savedir=testsavedir, rays_test=rays_test)
            else:
                if args.render_test:
                    # render_test switches to test poses
                    images = images[i_test]
                else:
                    # Default is smoother render_poses path
                    images = None

                testsavedir = os.path.join(savepath, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', render_poses.shape)

                rgbs, _ = renderer.render_path(render_poses gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to_8b(rgbs), duration=1000//30, quality=8)

            return

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    args.N_iters += 1
    if args.dataset_type != 'llff':
        args.ndc = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
