"""
Test to retrieve depthmap from a single panorama tile.
"""
import asyncio
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import time 
from PIL import Image

from tqdm import trange
from typing import List, Tuple, Union, Optional, Dict

from Panorama.pn_retriever import get_panorama
from Panorama.pn_crop import crop_pano
from Panorama.pn_depthmap import get_depth_map

def get_pano_id(url):
    """
    exactly 22 chars. 
    nested between "!1s" and "!2e"
    """

    pattern = re.compile(r"!1s(.{22})!2e")
    pano_id = pattern.search(url).group(1)
    return pano_id

# url = "https://www.google.com/maps/@29.6950438,-95.3274211,3a,75y,225.52h,96.09t/data=!3m6!1e1!3m4!1sGo1mMho-l5se_RBtFpfgZA!2e0!7i16384!8i8192?entry=ttu"
url = "https://www.google.com/maps/@37.4282039,-121.8883669,3a,53.9y,355.03h,83t/data=!3m7!1e1!3m5!1sAj12is2gFFmxjPLaIwpFcQ!2e0!5s20200301T000000!7i16384!8i8192?entry=ttu"

pano_id = get_pano_id(url)
print(pano_id)

# pano_id = "FRt0Bxijfyb9vDBvdzlbBQ"# input("Enter pano_id: ")
zoom = 1
pano_img, times = asyncio.run(get_panorama(pano_id=pano_id, zoom=zoom, debug=True))
print(times)

depth_map = get_depth_map(pano_id)

pano_img = np.array(pano_img)
pano_img = crop_pano(pano_img)
# resize im to same shape as pano_img (upscale)

# resize depth map to same shape as pano_img.shape[:2]

depth_map = cv2.resize(depth_map, pano_img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

# replace nans with double the max depth
depth_map = np.nan_to_num(depth_map, nan=np.nanmax(depth_map))

# normalize
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
# reshape to (depth_map.shape[0], depth_map.shape[1], 3)
# (3rd dim 1 -> 3)
depth_map = np.repeat(depth_map.reshape(depth_map.shape[0], depth_map.shape[1], 1), 3, axis=2)

rgb = pano_img.copy()

# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.imshow(rgb)
# ax = fig.add_subplot(122)
# ax.imshow(depth_map)
# plt.show()
# exit()

def translate(crd: np.ndarray, rgb: np.ndarray,
            d: np.ndarray, cam: Tuple[float]):
    """
    Given an equirectangular panorama, a depth map, 
    and a relative camera pose,
    
    Translate the panorama to the provided (new) camera pose.

    """
    t1 = time.time()
    H, W = rgb.shape[0], rgb.shape[1]

    # depth = 0 becomes -1
    d = np.where(d==0, -1, d)
    
    # move coordinates to relative pose and calculate new depth
    # apply op to each pixel 's xyz coordinate
    tmp_coord = crd - cam
    new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2))
    
    
    # normalize: /depth
    new_coord = tmp_coord / new_d.reshape(H,W,1)
    new_depth = np.zeros(new_d.shape)
    img = np.zeros(rgb.shape)

    
    # backward: 3d coordinate to pano image
    [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]
    

    idx = np.where(new_d>0)
    
    # theta: horizontal angle, phi: vertical angle
    theta = np.zeros(y.shape)
    phi = np.zeros(y.shape)
    x1 = np.zeros(z.shape)
    y1 = np.zeros(z.shape)

    theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
    phi[idx] = np.arctan2(-z[idx], x[idx])
    
    x1[idx] = (0.5 - theta[idx] / np.pi) * H #- 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
    y1[idx] = (0.5 - phi[idx]/(2*np.pi))*W #- 0.5
    x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')
    
    img = np.zeros(rgb.shape)
    # Mask out
    mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)
    x = x[mask]
    y = y[mask]
    new_d = new_d[mask]
    rgb = rgb[mask]
    # Give smaller depth pixel higher priority
    reorder = np.argsort(-new_d)
    x = x[reorder]
    y = y[reorder]
    new_d = new_d[reorder]
    rgb = rgb[reorder]
    # Assign
    new_depth[x, y] = new_d
    img[x, y] = rgb
                
    # print(f"prep: {time.time() - t1}")

    # remove occluded pixels
    # img, new_depth2 = filter_depth(img, new_depth)
    
    mask = (new_depth != 0).astype(int)
    # mask2 = (new_depth2 != 0).astype(int)

    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.imshow(mask)
    # ax = fig.add_subplot(122)
    # ax.imshow(mask2)
    # plt.show()
    # exit()

    return img, new_depth.reshape(H,W,1), tmp_coord, cam.reshape(1, 1, 3), mask

def filter_depth(img, depth, m=4):
    """
    depth filtering loop
    """
    times = []

    # margin for filtering
    m = 4
    for i in trange(m, H, 2):
        for j in range(m, W, 2):
            t1 = time.time()

            x1 = max(0, i - m)
            x2 = min(H, i + m)

            y1 = max(0, j - m)
            y2 = min(W, j + m)
            
            # within this square, find pixels with depth > 0
            index = np.where(depth[x1:x2, y1:y2] > 0)
            if len(index[0]) == 0: continue # depthless

            mean = np.median(depth[x1:x2, y1:y2][index]) # median
            target_index = np.where(depth[x1:x2, y1:y2] > mean * 1.3)
            
            if len(target_index[0]) > m ** 2 // 2:
                # reduce block size
                img[x1:x2, y1:y2][target_index] = 0#np.array([255.0, 0.0, 0.0])
                depth[x1:x2, y1:y2][target_index] = 0
    
            times.append(time.time() - t1)

    print(f"Number of pixels: {len(times)}, average time: {np.mean(times)}")

    return img, depth
# normalize to [0, 1]
# d = d.reshape(rgb.shape[0], rgb.shape[1], 1) / np.max(d)
# d = np.where(d==0, 1, d)


H, W = depth_map.shape[:2] 
# [0, 1, ..., W-1] for H times
_y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)

# [0, ..., 0], [1, ..., 1], ..., [H-1, ..., H-1], each of length W
_x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T

_theta = (1 - 2 * (_x) / H) * np.pi / 2 # latitude
_phi = 2 * np.pi * (0.5 - (_y) / W) # longtitude

# axii, each (H, W)
a0 = np.cos(_theta) * np.cos(_phi)
a1 = np.sin(_theta)
a2 = -np.cos(_theta)*np.sin(_phi)

# (H, W, 3)
coord = np.stack((a0, a1, a2), axis=2) * depth_map
# plt.imshow(coord)
# plt.show()
# exit()

scale = 0.6
cam_pos = []


counts = 50

"""
generate coordinates relative to the center of the panorama
# one set of coordinates from min(x) to max(x)
# the other set from min(z) to max(z)
# y is the vertical axis, thus same for both sets

imagine a plus sign constructed of dots, 
that is what the coordinates look like
"""

################ XZ ONLY ################
max_, min_ = coord[256, 512, 0]*scale, coord[256, 0, 0]*scale
pos_ = np.stack([np.linspace(max_, min_, counts), np.zeros(counts), np.zeros(counts)], axis=1)
cam_pos.append(pos_)
print(max_, min_)
# Z
max_, min_ = coord[256, 640, 2]*scale, coord[256, 128, 2]*scale
pos_ = np.stack([np.zeros(counts), np.zeros(counts), np.linspace(max_, min_, counts)], axis=1)
cam_pos.append(pos_)
print(max_, min_)
################ XY ONLY ################

cam_pos = np.concatenate(cam_pos, axis=0)
# save training, testnig camera poses
# with open(os.path.join(baseDir, 'cam_pos.txt'), 'w') as fp:
#     for p in cam_pos: fp.write('%f %f %f\n' % (p[0], p[1], p[2])) 

"""
some custom set test poses
though this may change later
"""
test_pos = np.array([[-0.05,      0.,       -0.05    ],
                    [-0.03,      0.,       -0.05    ],
                    [-0.01,      0.,       -0.05    ],
                    [ 0.01,      0.,       -0.05    ],
                    [ 0.03,      0.,       -0.05    ],
                    [ 0.05,      0.,       -0.05    ],
                    [ 0.05,      0.,       -0.03    ],
                    [ 0.05,      0.,       -0.01    ],
                    [ 0.05,      0.,        0.01    ],
                    [ 0.05,      0.,        0.03    ]])

# with open(os.path.join(baseDir, 'test', 'cam_pos.txt'), 'w') as fp:
#     for p in test_pos: fp.write('%f %f %f\n' % (p[0], p[1], p[2])) 

"""
finally concat to create a comprehensive list of camera poses
"""
cam_pos = np.concatenate([cam_pos, test_pos, np.array([0.0, 0.0, 0.0]).reshape(1,3)])

# poses = 90
# radius = 0.3

# cam_pos = []
# # generate poses (circle)
# degs = np.linspace(0, 360, poses + 1)[:-1]
# for i in trange(len(degs)):
#     deg = degs[i]
#     deg = np.deg2rad(deg)

#     cam_pos.append([radius * np.cos(deg), 0, radius * np.sin(deg)])

# cam_pos = np.array(cam_pos)

# plot all cam pos on 3d 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*cam_pos.T)
# plt.show()
# exit()
base_dir = '../data/'
dir_name = "gmap_rail1"
save_path = os.path.join(base_dir, dir_name)
os.makedirs(os.path.join(save_path, 'rm_occluded'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)

# images = []
# depth = []
for i in trange(len(cam_pos)):
    cam = cam_pos[i]
    # if n<100: continue
    # img = rgb.copy()
    # print("Num ", n)
    tmp_coord = np.stack((a0, a1, a2), axis=2)
    img1, d1, _, _, mask1 = translate(coord, rgb, depth_map, cam)
    img2, d2, _, _, mask2 = translate(tmp_coord*d1, img1, d1, -cam)

    # # plot mask1 and mask2 side by side
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.imshow(np.uint8(mask1*255))
    # ax = fig.add_subplot(122)
    # ax.imshow(np.uint8(mask2*255))
    # plt.show()
    # exit()
    
    
    # plt.imshow(np.uint8(img1))
    # plt.show()
    
    if i < 100:
        Image.fromarray(np.uint8(mask2*255)).save(os.path.join(save_path, 
                                                               'rm_occluded', 
                                                               f'mask_{str(i).zfill(3)}.png'))
    Image.fromarray(np.uint8(img1)).save(os.path.join(save_path, 'test', f'rgb_{str(i).zfill(3)}.png'))
    # images.append(img)
