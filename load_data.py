import numpy as np
import torch
from PIL import Image
import os
import math
import cv2

from tqdm import trange 
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class RayBundle:
    origin: List[any] = field(default_factory=list)
    direction: List[any] = field(default_factory=list)
    color: List[any] = field(default_factory=list)
    depth: List[any] = field(default_factory=list)
    gradient: List[any] = field(default_factory=list) # gradient (not present in the test set)
    embed: List[any] = field(default_factory=list) # idx embed for appearance, transience, etc

@dataclass
class CameraConfig:
    height: int
    width: int
    near: float
    far: float
    focal: Optional[float] = None
    k: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.k is None:
            self.k = np.array([
                [self.focal, 0, 0.5 * self.width],
                [0, self.focal, 0.5 * self.height],
                [0, 0, 1]
            ])

@dataclass 
class EquirectDataset:
    cc: CameraConfig
    rays_train: RayBundle
    rays_test: RayBundle
    bbox: Tuple[torch.Tensor, torch.Tensor] 

def concat_all(batch):
    """
    Iterate over the batch and concatenate all the lists
    """
    for key, value in batch.__dict__.items():
        if value is not None and not isinstance(value, np.ndarray):
            # also skip if already concatenated
            batch.__dict__[key] = np.concatenate(value, axis=0)

    return batch

def make_direction_rays(h, w):
    """
    Ray directions for equirectangular images
    ~= unit sphere
    """
    y = np.repeat(np.array(range(w)).reshape(1,w), h, axis=0)
    x = np.repeat(np.array(range(h)).reshape(1,h), w, axis=0).T

    lat = (1 - 2 * x / h) * np.pi / 2 # theta
    lon = 2 * np.pi * (0.5 - (y) / w) # phi

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    directions = np.stack((x, y, z), axis=2) # (h, w, 3)
    return directions 

def load_equirect_data(path: str, shape: Tuple[int, int], 
                       n_test_imgs: int = 10, seed: int = 0):
    h, w = shape
    rays_train = RayBundle()
    rays_test = RayBundle()

    color_path = os.path.join(path, "color")

    # use calibrated
    img_fns = sorted([fn for fn in os.listdir(color_path) if fn.startswith("p_")])
    n_imgs = len(img_fns) 
    assert n_imgs > n_test_imgs, "Not enough images for testing"
    idxs = np.arange(n_imgs)
    np.random.seed(seed)
    np.random.shuffle(idxs)
    test_idxs = idxs[:n_test_imgs]
    train_idxs = idxs[n_test_imgs:]
    n_train_imgs = len(train_idxs)

    cams = np.loadtxt(os.path.join(path, "cams.txt"), delimiter=" ")
    cams = np.insert(cams, 1, 0, axis=1)

    # prep directions in advance as they are the same for all images
    directions = make_direction_rays(h, w).reshape(-1, 3) # (h*w, 3)
    # (h*w*n_train_imgs, 3) and (h*w*n_test_imgs, 3)
    rays_train.directions = np.repeat(directions, len(train_idxs), axis=0)
    rays_test.directions = np.repeat(directions, len(test_idxs), axis=0)

    # same goes for embeds 
    # see https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/blender.py
    # the embed is initialized by frame/camera idx
    # (rays_o) is the img, len is constant in our case 
    # for t, frame in enumerate(self.meta['frames']):
    #     ...
    #     rays_t = t * torch.ones(len(rays_o), 1)
    rays_train.embed = np.concatenate([i * np.ones((h*w, 1)) for i in range(n_train_imgs)], axis=0)
    rays_test.embed = np.concatenate([i * np.ones((h*w, 1)) for i in range(n_test_imgs)], axis=0)

    print("Loading training data")
    for idx in trange(len(train_idxs)):
        fn = img_fns[train_idxs[idx]]
        origin = cams[train_idxs[idx]] # ray origin / cam pos

        # rgb image
        img = cv2.imread(os.path.join(color_path, fn), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(-1, 3)

        # gradient from rgb image's laplacian
        gradient = cv2.Laplacian(img, cv2.CV_64F)
        gradient = 2 * (gradient - np.min(gradient)) / np.ptp(gradient) - 1
                
        rays_train.origin.append(np.repeat(origin.reshape(1, -1), h * w, axis=0))
        rays_train.color.append(img) 
        rays_train.gradient.append(gradient)

    print("Loading testing data")
    for idx in trange(len(test_idxs)):
        fn = img_fns[test_idxs[idx]]
        origin = cams[test_idxs[idx]] # ray origin / cam pos

        # rgb image
        img = cv2.imread(os.path.join(color_path, fn), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img.reshape(-1, 3)

        # gradient from rgb image's laplacian
        gradient = cv2.Laplacian(img, cv2.CV_64F)
        gradient = 2 * (gradient - np.min(gradient)) / np.ptp(gradient) - 1
                
        rays_test.origin.append(np.repeat(origin.reshape(1, -1), h * w, axis=0))
        rays_test.color.append(img) 
        rays_test.gradient.append(gradient)

    rays_train = concat_all(rays_train)
    rays_test = concat_all(rays_test)
    return rays_train, rays_test, h, w

def load_equirect_data_(save_path: str, stage=0):

    basename = save_path.split('/')[-1]+'_'
    rgb = np.asarray(Image.open(os.path.join(save_path, basename+'rgb.png'))) / 255.0
    
    # load depth.
    # if using matterport3d dataset, then depth is stored in .exr format
    # else (structured3d), depth is stored in .png format
    if save_path.split('/')[-2] == 'mp3d':
        print(os.path.join(save_path, basename+'depth.exr'))
        d = cv2.imread(os.path.join(save_path, basename+'depth.exr'), cv2.IMREAD_ANYDEPTH)
        d = d.astype(np.float)

    else:
        d = np.asarray(Image.open(os.path.join(save_path, basename+'d.png')))

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
    with open(os.path.join(save_path, 'cam_pos.txt'),'r') as fp:
        all_poses = fp.readlines()
        for p in all_poses:
            image_coords.append(np.array(p.split()).astype(float))
    
    # load testing camera poses
    with open(os.path.join(save_path, 'test', 'cam_pos.txt'),'r') as fp:
        all_poses = fp.readlines()
        for p in all_poses:
            image_coords.append(np.array(p.split()).astype(float))
    
    image_coords.append([0.0, 0.0, 0.0])
    image_coords = np.array(image_coords)
    # image_coords = np.concatenate([image_coords, np.array([0.0, 0.0, 0.0]).reshape(1,3)])
    
    rays_train = RayBundle()
    rays_train.gradient = []
    rays_test = RayBundle()
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
            padded_idx = str(idx).zfill(3)
            dep = np.linalg.norm(coord - c, axis=-1)

            if idx < 100:
                dir = coord - c # direction = end point - start point
                dir = dir / np.linalg.norm(dir, axis=-1)[..., None]
                
                # the augmented equirects may have occluded regions
                # so we mask them out
                mask = np.asarray(Image.open(os.path.join(save_path, 'rm_occluded', f'mask_{padded_idx}.png'))).copy() / 255

                rays_train.origin.append(np.repeat(c.reshape(1, -1), (mask>0).sum(), axis=0))
                rays_train.direction.append(dir[mask>0])
                rgb_ = rgb[mask>0]
                rays_train.color.append(rgb_) 
                rays_train.depth.append(dep[mask>0])
                rays_train.gradient.append(gradient[mask>0])

            elif idx < 110:
                rays_test.origin.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                rays_test.direction.append(original_coord.reshape(-1, 3))
                rgb_ = np.asarray(Image.open(os.path.join(save_path, 'test', f'rgb_{str(idx - 100).zfill(3)}.png'))).reshape(-1, 3)
                rays_test.color.append(rgb_)
                rays_test.depth.append(dep.reshape(-1))

            elif idx == 110:
                rays_test.origin.append(np.repeat(c.reshape(1, -1), H*W, axis=0))
                rays_test.direction.append(coord.reshape(-1, 3))
                rgb_ = rgb.reshape(-1, 3)
                rays_test.color.append(rgb_)
                rays_test.depth.append(dep.reshape(-1))

            ts = idx * np.ones((len(rgb_), 1))
            if idx < 100:
                rays_train.ts.append(ts)
            else: 
                rays_test.ts.append(ts)

    # use this function to reduce verbose code
    rays_train = concat_all(rays_train)
    rays_test = concat_all(rays_test)

    # rays_o, rays_d, rays_g, rays_rgb, rays_depth, [H, W]
    # all in flatten format : [N(~H*W*100), 3 or 1]
    
    return rays_train, rays_test, H, W


def load(args):
    """
    origin, direction, color, gradient, embed: (h * w * n_train_imgs, 3)
    """
    print("Data type: Equirectangular")
    rays_train, rays_test, h, w = load_equirect_data(data_path=args.data_path, shape=args.shape,
                                                     n_test_imgs=args.n_test_imgs, seed=args.seed)
    print(f"Loaded equirectangular; h: {h}, w: {w}")

    near, far = 0.0, 2.0
    bbox = (torch.tensor([-1.5, -1.5, -1.0]), torch.tensor([1.5, 1.5, 1.0]))

    cc = CameraConfig(height=h, width=w, near=near, far=far)
    dataset = EquirectDataset(cc=cc, rays_train=rays_train, 
                                rays_test=rays_test, bbox=bbox)
        
    print("Camera Config:")
    print(dataset.cc)
    return dataset