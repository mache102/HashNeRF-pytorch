import imageio
import numpy as np
import os
import pdb
import pickle
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Optional

import torch

from run_nerf_helpers import *
from create_nerf import create_nerf
from ray_util import *
from loss import sigma_sparsity_loss, total_variation_loss

from load.load_llff import load_llff_data
from load.load_deepvoxels import load_dv_data
from load.load_blender import load_blender_data
from load.load_scannet import load_scannet_data
from load.load_LINEMOD import load_LINEMOD_data
from load.load_equirect import load_equirect_data

from util import create_expname, all_to_tensor, shuffle_rays, to_8b
from parse_args import config_parser

# 20231010 15:25
np.random.seed(0)
DEBUG = False

def eval_test_omninerf(renderer, savedir: str, rays_test, render_kwargs_test: Dict):
    os.makedirs(savedir, exist_ok=True)
    with torch.no_grad():
        rgbs, _ = renderer.render_path([rays_test.o, rays_test.d], 
                                       render_kwargs=render_kwargs_test, 
                            savedir=savedir, render_factor=args.render_factor)
    print('Done rendering', savedir)
    
    # calculate MSE and PSNR for last image(gt pose)
    gt_loss = img2mse(torch.tensor(rgbs[-1]), torch.tensor(rays_test.rgb[-1]))
    gt_psnr = mse2psnr(gt_loss)
    print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
    with open(os.path.join(savedir, 'statistics.txt'), 'w') as f:
        f.write('loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
    
    rgbs = np.concatenate([rgbs[:-1],rgbs[:-1][::-1]])
    imageio.mimwrite(os.path.join(savedir, 'video2.gif'), to_8b(rgbs), duration=1000//10)
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
    if args.dataset_type == 'equirect':
        rays, rays_test, H, W = load_equirect_data(args.datadir, args.stage)

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

    equirect has rays batch (origin,direction,rgb,depth,gradient)
    directly 
    whereas other datasets have images and poses
    (rays to be obtained from poses)
    """
    if args.dataset_type != 'equirect':
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

                eval_test_omninerf(renderer, savedir=testsavedir, rays_test=rays_test, 
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

                rgbs, _ = renderer.render_path(render_poses, render_kwargs=render_kwargs_test, 
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to_8b(rgbs), duration=1000//30, quality=8)

            return

    """
    Prepare raybatch tensor if batching random rays
    """
    if args.dataset_type == 'equirect':
        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        i_batch = 0
        start = start + 1
        print('Begin, iter: %d' % start)

        for i in trange(start, args.N_iters):
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
            rays, reshape_to = renderer.prepare_rays(rays=[batch_o, batch_d], ndc=False)
            rgb, dep, grad, extras = renderer.render(rays=rays, **render_kwargs_train)

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

            images = torch.Tensor(images).to(device)
            rays_rgb = torch.Tensor(rays_rgb).to(device)

            
        poses = torch.Tensor(poses).to(device)

        print('Start Training')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        loss_list = []
        psnr_list = []
        time_list = []
        start = start + 1
        time0 = time.time()

        for i in trange(start, args.N_iters):
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
            rays, reshape_to = renderer.prepare_rays(rays=batch_rays, ndc=False)
            rgb, depth, acc, extras = renderer.render(rays=rays, verbose=i < 10, retraw=True,
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
                    rgbs, disps = renderer.render_path(render_poses,
                                              render_kwargs=render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(savepath, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to_8b(rgbs), duration=1000//30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to_8b(disps / np.max(disps)), duration=1000//30, quality=8)

            """
            RENDER TEST SET
            """
            if i%args.i_testset==0 and i > 0:
                testsavedir = os.path.join(savepath, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    renderer.render_path(torch.Tensor(poses[i_test]).to(device), 
                                render_kwargs=render_kwargs_test, gt_imgs=images[i_test], 
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
    args.N_iters += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
