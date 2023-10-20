from dataclasses import dataclass 
import numpy as np
from typing import Optional


import numpy as np
import torch

from load.load_llff import load_llff_data
from load.load_deepvoxels import load_dv_data
from load.load_blender import load_blender_data
from load.load_scannet import load_scannet_data
from load.load_LINEMOD import load_LINEMOD_data
from load.load_equirect import load_equirect_data

from data_classes import  *

def load_data(args):
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
        
    if args.dataset_type == 'equirect':
        print("Data type: Equirectangular")
        rays, rays_test, h, w = load_equirect_data(args.datadir, args.stage)
        print(f"Loaded equirectangular; h: {h}, w: {w}")

        near, far = 0.0, 2.0
        bbox = (torch.tensor([-1.5, -1.5, -1.0]), torch.tensor([1.5, 1.5, 1.0]))

        cc = CameraConfig(height=h, width=w, near=near, far=far)
        dataset = EquirectDataset(cc=cc, rays_train=rays, 
                                  rays_test=rays_test, bbox=bbox)
        
    elif args.dataset_type == 'llff':
        print("Data type: LLFF")
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print(f'Loaded LLFF; Images: {images.shape}, Poses: {poses.shape}, HWF: {hwf}')
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        if not args.ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        
        cc = CameraConfig(height=hwf[1], width=hwf[0], focal=hwf[2],
                          near=near, far=far)
        dataset = StandardDataset(cc=cc, images=images, poses=poses, 
                                  render_poses=render_poses,
                                  bbox=bounding_box,
                                  train=i_train, val=i_val, test=i_test)
        
    elif args.dataset_type == 'blender':
        print("Data type: Blender")
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded Blender; Images: {images.shape}, Poses: {poses.shape}, HWF: {hwf}')
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        cc = CameraConfig(height=hwf[1], width=hwf[0], focal=hwf[2],
                            near=near, far=far)
        dataset = StandardDataset(cc=cc, images=images, poses=poses,
                                  render_poses=render_poses,
                                  bbox=bounding_box,
                                  train=i_train, val=i_val, test=i_test)

    elif args.dataset_type == 'scannet':
        print("Data type: Scannet")
        images, poses, render_poses, hwf, i_split, bounding_box = load_scannet_data(args.datadir, args.scannet_sceneID, args.half_res)
        print(f'Loaded Scannet; Images: {images.shape}, Poses: {poses.shape}, HWF: {hwf}')
        i_train, i_val, i_test = i_split

        near = 0.1
        far = 10.0

        cc = CameraConfig(height=hwf[1], width=hwf[0], focal=hwf[2],
                            near=near, far=far)
        dataset = StandardDataset(cc=cc, images=images, poses=poses,
                                  render_poses=render_poses,
                                  bbox=bounding_box,
                                  train=i_train, val=i_val, test=i_test)

    elif args.dataset_type == 'LINEMOD':
        print("Data type: LINEMOD")
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        cc = CameraConfig(height=hwf[1], width=hwf[0], focal=hwf[2], k=K,
                            near=near, far=far)
        dataset = StandardDataset(cc=cc, images=images, poses=poses,
                                  render_poses=render_poses,
                                  bbox=bounding_box,
                                  train=i_train, val=i_val, test=i_test)
        
    elif args.dataset_type == 'deepvoxels':
        print("Data type: DeepVoxels")
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)
        print(f'Loaded DeepVoxels; Images: {images.shape}, Poses: {poses.shape}, HWF: {hwf}')
        # print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

        cc = CameraConfig(height=hwf[1], width=hwf[0], focal=hwf[2],
                            near=near, far=far)
        dataset = StandardDataset(cc=cc, images=images, poses=poses,
                                  render_poses=render_poses,
                                  bbox=bounding_box,
                                  train=i_train, val=i_val, test=i_test)
    else:
        raise NotImplementedError(f'Unknown dataset type: {args.dataset_type}')
        
    print("Camera Config:")
    print(dataset.cc)
    return dataset