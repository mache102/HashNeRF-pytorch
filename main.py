import imageio
import numpy as np
import os
import json
import pdb
import pickle
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Optional

import torch

from networks.hash_nerf import HashNeRF
from networks.vanilla_nerf import VanillaNeRF

from renderer import *
from create_nerf import create_nerf
from ray_util import *
from loss import sigma_sparsity_loss, total_variation_loss
from create_nerf import get_embedder
from load.load_data import load_data

from util import create_expname, all_to_tensor, shuffle_rays, to_8b, save_configs
from parse_args import config_parser

from util import img2mse, mse2psnr

from dataclasses import dataclass 

# 20231010 15:25
np.random.seed(0)
DEBUG = False

   
def main():
    # load dataset
    dataset = load_data(args)
    """
    Experiment savepath, savename, etc.
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
    Load model config
    """
    fp = os.path.join("model_configs", f"{args.model_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Model configuration not found: {fp}")
    with open(fp, "r") as f:
        model_config = json.load(f)

    """
    Create embedding functions for position & viewdirs
    """
    embedders = {
        "pos": None,
        "dir": None
    }
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
    Create coarse and fine models
    """
    models = {
        "coarse": None,
        "fine": None
    }
    if args.i_embed == 1:
        model_coarse = HashNeRF(model_config["coarse"], input_ch=input_ch, 
                                input_ch_views=input_ch_views).to(device)
        
    elif not args.use_gradient:
        if args.importance > 0:
            model_config["coarse"]["output_ch"] += 1
            model_config["fine"]["output_ch"] += 1
        model_coarse = VanillaNeRF(model_config["coarse"], input_ch=input_ch, 
                                   input_ch_views=input_ch_views, 
                                   use_viewdirs=args.use_viewdirs,
                                   use_gradient=args.use_gradient).to(device)
    models["coarse"] = model_coarse
    grad_vars = list(models["coarse"].parameters())

    if args.N_importance > 0:
        if args.i_embed == 1:
            model_fine = HashNeRF(model_config["fine"], input_ch=input_ch, 
                                    input_ch_views=input_ch_views).to(device)
            
        elif not args.use_gradient:
            model_fine = VanillaNeRF(model_config["fine"], input_ch=input_ch, 
                                    input_ch_views=input_ch_views, 
                                    use_viewdirs=args.use_viewdirs,
                                    use_gradient=args.use_gradient).to(device)
        models["fine"] = model_fine
        grad_vars += list(models["fine"].parameters())

    """
    Create optimizer
    """
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

    """
    Load checkpoints if available
    """
    global_step = 0
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(savepath, f) for f in sorted(os.listdir(os.path.join(savepath))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        global_step = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        models["coarse"].load_state_dict(ckpt['network_fn_state_dict'])
        if models["fine"] is not None:
            models["fine"].load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embedders["pos"].load_state_dict(ckpt['pos_embedder_state_dict'])

    """
    Create trainer
    """
    if args.dataset_type == "equirect":
        trainer = EquirectTrainer(dataset=dataset, start=global_step,
                                  models=models, optimizer=optimizer,
                                  embedders=embedders, args=args)
    else:
        trainer = StandardTrainer(dataset=dataset, start=global_step,
                                  models=models, optimizer=optimizer,
                                  embedders=embedders, args=args)
    """
    Skip to render only
    """
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
