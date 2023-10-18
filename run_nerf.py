import configargparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pickle
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from run_nerf_helpers import *
from create_nerf import create_nerf
from ray_util import *
from loss import sigma_sparsity_loss, total_variation_loss

from load.load_llff import load_llff_data
from load.load_deepvoxels import load_dv_data
from load.load_blender import load_blender_data
from load.load_scannet import load_scannet_data
from load.load_LINEMOD import load_LINEMOD_data
from load.load_st3d import load_st3d_data

from util import create_expname, all_to_tensor, shuffle_rays

# 20231010 15:25
np.random.seed(0)
DEBUG = False

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--net_chunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / st3d / multi_mp3d')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## scannet flags
    parser.add_argument("--scannet_sceneID", type=str, default='scene0000_00',
                        help='sceneID to load from scannet')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    ## st3d flags
    # use_depth:
    # use_gradient:
    # stage: used only in load_st3d_data
    parser.add_argument("--use_depth", action='store_true', 
                        help='use depth to update')
    parser.add_argument("--use_gradient", action='store_true', 
                        help='use gradient to update')
    parser.add_argument("--stage", type=int, default=0,
                        help='use iterative training by defining stage, if 0: don\'t use')


    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    return parser

def eval_test_omninerf(H, W, savedir: str, rays_test, render_kwargs_test: Dict):
    os.makedirs(savedir, exist_ok=True)
    with torch.no_grad():
        rgbs, _ = render_path([rays_test.o, rays_test.d], [H, W], render_kwargs=render_kwargs_test, 
                              args=args, chunk=None, savedir=savedir, render_factor=args.render_factor)
    print('Done rendering', savedir)
    
    # calculate MSE and PSNR for last image(gt pose)
    gt_loss = img2mse(torch.tensor(rgbs[-1]), torch.tensor(rays_test.rgb[-1]))
    gt_psnr = mse2psnr(gt_loss)
    print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
    with open(os.path.join(savedir, 'statistics.txt'), 'w') as f:
        f.write('loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
    
    rgbs = np.concatenate([rgbs[:-1],rgbs[:-1][::-1]])
    imageio.mimwrite(os.path.join(savedir, 'video2.gif'), to8b(rgbs), duration=1000//10)
    print('Saved test set')   
   
def main():

    """
    Load data

    Supported types:
    - blender
    - llff
    - scannet
    - deepvoxels
    - LINEMOD

    images (samples, H, W, 4) # rgba
    poses (samples, 4, 4) # frame transformation matrix 
    render_poses (41, 4, 4) # full 360 poses 
    hwf (3, ) # height, width, focal = .5 * W / np.tan(.5 * camera_angle_x) 
    i_split (3, None) # imgs indices for train, val, test 
    bounding_box (2, 3) # pair of 3d coords

    """
    K = None
    # Load data
    if args.dataset_type == 'st3d':
        rays, rays_test, H, W = load_st3d_data(args.datadir, args.stage)
        # rays_o, rays_d, rays_g, rays_rgb, rays_depth, hw = load_st3d_data(args.datadir, args.stage)
        # rays_o, rays_o_test = rays_o
        # rays_d, rays_d_test = rays_d
        # rays_rgb, rays_rgb_test = rays_rgb
        # rays_depth, rays_depth_test = rays_depth
        near, far = 0.0, 2.0
        print(f"Near Far bounds are: {near}, {far}")
        args.bounding_box = (torch.tensor([-1.5, -1.5, -1.0]), torch.tensor([1.5, 1.5, 1.0]))
        
    elif args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
        args.bounding_box = bounding_box
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'scannet':
        images, poses, render_poses, hwf, i_split, bounding_box = load_scannet_data(args.datadir, args.scannet_sceneID, args.half_res)
        args.bounding_box = bounding_box
        print('Loaded scannet', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    """ 
    Process HWF
    (and convert poses into np array)
    that's about it

    st3d has rays batch (origin,direction,rgb,depth,gradient)
    directly 
    whereas other datasets have images and poses
    (rays to be obtained from poses)
    """
    if args.dataset_type != 'st3d':
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        if args.render_test:
            render_poses = np.array(poses[i_test])
    
    """
    Experiment saving
    """
    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = create_expname(args)
    expname = args.expname

    savepath = os.path.join(basedir, expname)
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(savepath, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        with open(os.path.join(savepath, 'config.txt'), 'w') as file:
            file.write(open(args.config, 'r').read())

    """
    Create nerf model

    set render kwargs
    """
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.dataset_type != 'st3d':
        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)
    else:
        # all to tensor
        rays = shuffle_rays(all_to_tensor(rays, device))
        rays_test = all_to_tensor(rays_test, device)
    """
    Skip to render only
    """
    # Short circuit if only rendering out from trained model
    
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.dataset_type == 'st3d':
                if args.stage > 0:
                    testsavedir = os.path.join(savepath, 'renderonly_stage_{}_{:06d}'.format(args.stage, start))
                else:
                    testsavedir = os.path.join(savepath, 'renderonly_train_{}_{:06d}'.format('test' if args.render_test else 'path', start))

                eval_test_omninerf(H, W, savedir=testsavedir, rays_test=rays_test, 
                                   render_kwargs_test=render_kwargs_test)
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

                rgbs, _ = render_path(render_poses, hwf, K=K, render_kwargs=render_kwargs_test, args=args, 
                                      chunk=args.chunk, gt_imgs=images, savedir=testsavedir, 
                                      render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), duration=1000//30, quality=8)

            return

    """
    Prepare raybatch tensor if batching random rays
    """
    if args.dataset_type == 'st3d':
        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        i_batch = 0
        N_iters = 200000 + 1
        start = start + 1
        print('Begin, iter: %d' % start)

        for i in trange(start, N_iters):
            time0 = time.time()
            
            batch_o = rays.o[i_batch:i_batch+N_rand]
            batch_d = rays.d[i_batch:i_batch+N_rand]
            
            target_rgb = rays.rgb[i_batch:i_batch+N_rand]      
            target_d = rays.d[i_batch:i_batch+N_rand]
            target_g = rays.g[i_batch:i_batch+N_rand]

            i_batch += N_rand
            if i_batch >= rays.rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rays = shuffle_rays(rays)
                i_batch = 0

            #####  Core optimization loop  #####

            rgb, dep, grad, extras = render(H, W, rays=[batch_o, batch_d], **render_kwargs_train, ndc=False)

            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            img_loss = img2mse(rgb, target_rgb)
            # depth_loss
            if args.use_depth:
                depth_loss = torch.abs(dep - target_d).mean()
            else:
                depth_loss = torch.tensor(0.0)
                
            if args.use_gradient:
                grad_loss = img2mse(grad, target_g)
            else:
                grad_loss = torch.tensor(0.0)


            loss = img_loss + depth_loss + grad_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_rgb)
                loss = loss + img_loss0
                if args.use_depth:
                    depth_loss0 = torch.abs(extras['depth0'] - target_d).mean()
                    loss = loss + depth_loss0
                if args.use_gradient:
                    grad_loss0 = img2mse(extras['grad0'], target_g)
                    loss = loss + grad_loss0
                    
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time()-time0

            # Rest is logging
            if i%args.i_weights==0:
                path = os.path.join(savepath, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            
            if i%args.i_testset==0 and i > 0:
                if args.stage > 0:
                    testsavedir = os.path.join(savepath, 'stage{}_test_{:06d}'.format(args.stage, i))
                else:
                    testsavedir = os.path.join(savepath, 'testset_{:06d}'.format(i))
                eval_test_omninerf(H, W, savedir=testsavedir, rays_test=rays_test,
                                      render_kwargs_test=render_kwargs_test)

                
            if i%args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            global_step += 1

    else:
        N_rand = args.N_rand
        use_batching = not args.no_batching
        if use_batching:
            # For random ray batching
            print("Using random ray batching")

            rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print("Concatenated direction and origin rays")
            rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)

            np.random.shuffle(rays_rgb)
            print("Shuffled rays")

            i_batch = 0

        # Move training data to GPU
        if use_batching:
            images = torch.Tensor(images).to(device)
            rays_rgb = torch.Tensor(rays_rgb).to(device)
        poses = torch.Tensor(poses).to(device)

        N_iters = 50000 + 1
        print('Start Training')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        loss_list = []
        psnr_list = []
        time_list = []
        start = start + 1
        time0 = time.time()

        for i in trange(start, N_iters):
            # Sample random ray batch
            # omninerf assumes use batching
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0


                batch_o = rays_o[i_batch:i_batch+N_rand]
                batch_d = rays_d[i_batch:i_batch+N_rand]
                
                target_rgb = rays_rgb[i_batch:i_batch+N_rand]      
                target_d = rays_d[i_batch:i_batch+N_rand]
                target_g = rays_g[i_batch:i_batch+N_rand]
                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_o = rays_o[rand_idx]
                    rays_d = rays_d[rand_idx]
                    rays_rgb = rays_rgb[rand_idx]
                    if args.use_gradient:
                        rays_g = rays_g[rand_idx]
                    
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(i_train)
                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3,:4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, depth, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
            loss = loss + sparsity_loss

            # add Total Variation loss
            if args.i_embed==1:
                n_levels = render_kwargs_train["embed_fn"].n_levels
                min_res = render_kwargs_train["embed_fn"].base_resolution
                max_res = render_kwargs_train["embed_fn"].finest_resolution
                log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
                TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                                min_res, max_res, \
                                                i, log2_hashmap_size, \
                                                n_levels=n_levels) for i in range(n_levels))
                loss = loss + args.tv_loss_weight * TV_loss
                if i>1000:
                    args.tv_loss_weight = 0.0

            loss.backward()
            # pdb.set_trace()
            optimizer.step()

            """
            UPDATE LEARNING RATE
            """
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            t = time.time() - time0
            """
            ################################################################################
            REST IS LOGGING
            ################################################################################
            """
            
            """
            SAVE CHECKPOINT
            """
            if i%args.i_weights==0:
                path = os.path.join(savepath, '{:06d}.tar'.format(i))
                if args.i_embed==1:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                print('Saved checkpoints at', path)

            """
            RENDER VIDEO
            """
            if i%args.i_video==0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    rgbs, disps = render_path(render_poses, hwf, K=K, 
                                              render_kwargs=render_kwargs_test, 
                                              args=args, chunk=args.chunk)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(savepath, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), duration=1000//30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), duration=1000//30, quality=8)

            """
            RENDER TEST SET
            """
            if i%args.i_testset==0 and i > 0:
                testsavedir = os.path.join(savepath, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K=K,
                                render_kwargs=render_kwargs_test, args=args,
                                chunk=args.chunk, gt_imgs=images[i_test], 
                                savedir=testsavedir)
                print('Saved test set')

            """
            PRINT TRAINING PROGRESS
            """
            if i%args.i_print==0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
                loss_list.append(loss.item())
                psnr_list.append(psnr.item())
                time_list.append(t)
                loss_psnr_time = {
                    "losses": loss_list,
                    "psnr": psnr_list,
                    "time": time_list
                }
                with open(os.path.join(savepath, "loss_vs_time.pkl"), "wb") as fp:
                    pickle.dump(loss_psnr_time, fp)

            global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
