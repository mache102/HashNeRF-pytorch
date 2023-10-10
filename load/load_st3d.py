import numpy as np
from PIL import Image
import os
import math
import cv2
import glob

from dataclasses import dataclass 
from typing import Optional, List 

@dataclass 
class EquirectRays:
    o: List[any] = None
    d: List[any] = None
    rgb: List[any] = None
    depth: List[any] = None
    g: Optional[List[any]] = None # gradient (not present in test set)

def concat_all(batch):
    """
    Iterate over the batch and concatenate all the lists
    """
    for key, value in batch.__dict__.items():
        if value is not None:
            batch.__dict__[key] = np.concatenate(value, axis=0)

    return batch

def load_st3d_data(baseDir: str, stage=0):

    basename = baseDir.split('/')[-1]+'_'
    rgb = np.asarray(Image.open(os.path.join(baseDir, basename+'rgb.png'))) / 255.0
    
    # load depth.
    # if using matterport3d dataset, then depth is stored in .exr format
    # else (structured3d), depth is stored in .png format
    # TODO: what about google maps street view dataset?
    # it could be .png
    if baseDir.split('/')[-2] == 'mp3d':
        print(os.path.join(baseDir, basename+'depth.exr'))
        d = cv2.imread(os.path.join(baseDir, basename+'depth.exr'), cv2.IMREAD_ANYDEPTH)
        d = d.astype(np.float)

    else:
        d = np.asarray(Image.open(os.path.join(baseDir, basename+'d.png')))

    # gradient is obtained from rgb image's laplacian
    gradient = cv2.Laplacian(rgb, cv2.CV_64F)
    gradient = 2 * (gradient - np.min(gradient)) / np.ptp(gradient) -1
        
    # normalize to [0, 1]
    max_depth = np.max(d)
    d = d.reshape(rgb.shape[0], rgb.shape[1], 1) / max_depth
    
    # fixed dimensions?
    H, W = 512, 1024
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

    _theta = (1 - 2 * (_x) / H) * np.pi/2 # latitude
    _phi = 2*math.pi*(0.5 - (_y)/W ) # longtitude

    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1)
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1)
    original_coord = np.concatenate((axis0, axis1, axis2), axis=2)
    coord = original_coord * d
    
    # load training camera poses
    # sample item: [x, y, z]
    image_coords = []
    with open(os.path.join(baseDir, 'cam_pos.txt'),'r') as fp:
        all_poses = fp.readlines()
        for p in all_poses:
            image_coords.append(np.array(p.split()).astype(float))
    
    # load testing camera poses
    with open(os.path.join(baseDir, 'test', 'cam_pos.txt'),'r') as fp:
        all_poses = fp.readlines()
        for p in all_poses:
            image_coords.append(np.array(p.split()).astype(float))
    
    image_coords.append([0.0, 0.0, 0.0])
    image_coords = np.array(image_coords)
    # image_coords = np.concatenate([image_coords, np.array([0.0, 0.0, 0.0]).reshape(1,3)])
    
    batch = EquirectRays()
    batch.g = []
    batch_test = EquirectRays()
    # rays_o, rays_d, rays_g, rays_rgb, rays_depth = [], [], [], [], []
    # rays_o_test, rays_d_test, rays_rgb_test, rays_depth_test = [], [], [], []
    if stage > 0:
        if stage == 1:
            x, z = coord[..., 0], coord[..., 2]
            max_idx = np.unravel_index((np.power(x, 2)+np.power(z, 2)).argmax(), x.shape[:2])
            xmax = x[max_idx] 
            zmax = z[max_idx] 
            xz_interval = np.linspace(np.array([xmax-0.2, 0.0, zmax])*0.15, -np.array([xmax, 0.0, zmax+0.5]) * 0.1, 60)
            
            image_coords = xz_interval
            print(image_coords.tolist())
            for i, p in enumerate(image_coords):
                raise NotImplementedError
            
                # depth, ray_dir, and images don't seem to be defined
                # depth.append(np.sqrt(np.sum(np.square(coord - p), axis=2)))
                # ray_dir.append(coord)
                # images.append(rgb) 

    else:
        for idx, c in enumerate(image_coords):
            dep = np.linalg.norm(coord - c, axis=-1)

            if idx < 100:
                dir = coord - c # direction = end point - start point
                dir = dir / np.linalg.norm(dir, axis=-1)[..., None]
                
                # the augmented equirects may have occluded regions
                # so we mask them out
                mask = np.asarray(Image.open(os.path.join(baseDir, 'rm_occluded', 'mask_%d.png' % idx))).copy() / 255

                batch.o.append(np.repeat(c.reshape(1, -1), (mask>0).sum(), axis=0))
                batch.d.append(dir[mask>0])
                batch.rgb.append(rgb[mask>0])
                batch.depth.append(dep[mask>0])
                batch.g.append(gradient[mask>0])

                # rays_o.append(np.repeat(c.reshape(1, -1), (mask>0).sum(), axis=0))
                # rays_g.append(gradient[mask>0])
                # rays_d.append(dir[mask>0])
                # rays_rgb.append(rgb[mask>0])
                # rays_depth.append(dep[mask>0])

            elif idx < 110:
                batch_test.o.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                batch_test.d.append(original_coord.reshape(-1, 3))
                batch_test.rgb.append(np.asarray(Image.open(os.path.join(baseDir, 'test', 'rgb_{}.png'.format(idx - 100)))).reshape(-1, 3))
                batch_test.depth.append(dep.reshape(-1))

                # rays_o_test.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                # rays_d_test.append(original_coord.reshape(-1, 3))
                # rays_rgb_test.append(np.asarray(Image.open(os.path.join(baseDir, 'test', 'rgb_{}.png'.format(idx - 100)))).reshape(-1 ,3))
                # rays_depth_test.append(dep.reshape(-1))

            elif idx == 110:
                batch_test.o.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                batch_test.d.append(coord.reshape(-1, 3))
                batch_test.rgb.append(rgb.reshape(-1, 3))
                batch_test.depth.append(dep.reshape(-1))

                # rays_o_test.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                # rays_d_test.append(coord.reshape(-1, 3))
                # rays_rgb_test.append(rgb.reshape(-1, 3))
                # rays_depth_test.append(dep.reshape(-1))

    # use this functio to reduce verbose code
    batch = concat_all(batch)
    batch_test = concat_all(batch_test)

    # rays_o, rays_o_test = np.concatenate(rays_o, axis=0), np.concatenate(rays_o_test, axis=0)
    # rays_d, rays_d_test = np.concatenate(rays_d, axis=0), np.concatenate(rays_d_test, axis=0)
    # rays_g = np.concatenate(rays_g, axis=0)
    # rays_rgb, rays_rgb_test = np.concatenate(rays_rgb, axis=0), np.concatenate(rays_rgb_test, axis=0)
    # rays_depth, rays_depth_test = np.concatenate(rays_depth, axis=0), np.concatenate(rays_depth_test, axis=0)
    
    # rays_o, rays_d, rays_g, rays_rgb, rays_depth, [H, W]
    # all in flatten format : [N(~H*W*100), 3 or 1]
    
    return batch, batch_test, H, W
    # return [rays_o, rays_o_test], [rays_d, rays_d_test], rays_g, [rays_rgb, rays_rgb_test], [rays_depth, rays_depth_test], [int(H), int(W)]
