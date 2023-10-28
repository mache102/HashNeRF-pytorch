"""
Test to retrieve depthmap from a single panorama tile.

python3 -m equirect.generate_data --url "https://www.google.com/maps/@37.4237922,-121.8896494" --file_path imgs/
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from einops import reduce 
from tqdm import trange 
from typing import Tuple

from .Panorama.pn_utils import coords_from_url

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default=None, 
                        help="url containing coordinates")
    parser.add_argument("--coord", '-c', type=float, nargs=2, 
                        default=None, help="latitude and longitude")
    parser.add_argument("--file_path", type=str, 
                        help="Path to save images to (and load from)")

    return parser.parse_args()

class Transform:
    def __init__(self, coord):
        self.coord = coord
        self.h = coord.shape[0]
        self.w = coord.shape[1]
    
    def translate(self, color: np.ndarray, 
                    depth: np.ndarray, pose: Tuple[float]):
        """
        Given an equirectangular panorama, a depth map, 
        and a relative camera pose,
        
        Translate the panorama to the provided (new) camera pose.
        """
        depth = np.where(depth == 0, -1, depth)
        
        # move coordinates to relative pose and calculate new depth
        # apply op to each pixel 's xyz coordinate
        temp = self.coord * depth - pose
        new_d = np.sqrt(np.sum(np.square(temp), axis=2))
        
        new_coord = temp / new_d.reshape(self.h, self.w, 1)
        new_depth = np.zeros(new_d.shape)
        tl_color = np.zeros(color.shape)

        
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
        
        x1[idx] = (0.5 - theta[idx] / np.pi) * self.h #- 0.5  # (1 - np.sin(theta[idx]))*H/2 - 0.5
        y1[idx] = (0.5 - phi[idx]/(2*np.pi)) * self.w #- 0.5
        x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')

        # Mask out
        mask = (new_d > 0) & (self.h > x) & (x > 0) & (self.w > y) & (y > 0)
        x = x[mask]
        y = y[mask]
        new_d = new_d[mask]
        color = color[mask]
        # Give smaller depth pixel higher priority
        reorder = np.argsort(-new_d)
        x = x[reorder]
        y = y[reorder]
        new_d = new_d[reorder]
        color = color[reorder]
        # Assign
        new_depth[x, y] = new_d
        tl_color[x, y] = color
                    
        mask = (new_depth != 0).astype(int)
        # mask2 = (new_depth2 != 0).astype(int)

        return tl_color, new_depth.reshape(self.h, self.w, 1), mask

def make_coord(h, w):
    _y = np.repeat(np.array(range(w)).reshape(1,w), h, axis=0)
    _x = np.repeat(np.array(range(h)).reshape(1,h), w, axis=0).T

    _theta = (1 - 2 * (_x) / h) * np.pi/2 # latitude
    _phi = 2*np.pi*(0.5 - (_y)/w ) # longtitude

    axis0 = (np.cos(_theta)*np.cos(_phi))
    axis1 = np.sin(_theta)
    axis2 = (-np.cos(_theta)*np.sin(_phi))
    coord = np.stack((axis0, axis1, axis2), axis=2)

    return coord

class TestPoseGen:
    """
    Generate test poses 
    """

    @staticmethod 
    def circle(samples: int, radius: float):
        """n Test poses in a circle of radius"""
        angles = np.linspace(0, 2*np.pi, samples + 1)[:-1]
        return np.array([[np.cos(a), 0, np.sin(a)] for a in angles]) * radius
    
    @staticmethod 
    def linear(samples: int, p1: Tuple[float, float], p2: Tuple[float, float]):
        """n Test poses in a line between p1 and p2"""
        pos = np.linspace(p1, p2, samples)
        # insert 0 as y axis 
        pos = np.insert(pos, 1, 0, axis=1)
        return pos

def make_train_pos(coord, samples):
    """
    array of shape (samples, 3)
    train camera positions, relative
    """
    counts = samples // 2

    zeros = np.zeros(counts)

    train_pos = []
    # x
    xmin, xmax = coord[256, 0, 0], coord[256, 512, 0]
    pos = np.linspace(xmin, xmax, counts)
    pos = np.stack([pos, zeros, zeros], axis=1)
    train_pos.append(pos)

    # z
    zmin, zmax = coord[256, 128, 2], coord[256, 640, 2]
    pos = np.linspace(zmin, zmax, counts)
    pos = np.stack([zeros, zeros, pos], axis=1)
    train_pos.append(pos)

    train_pos = np.concatenate(train_pos, axis=0)
    return train_pos 


def main():
    os.makedirs(os.path.join(args.file_path, 'rm_occluded'), exist_ok=True)
    os.makedirs(os.path.join(args.file_path, "test"), exist_ok=True)

    raw_path = os.path.join(args.file_path, "raw")
    depth_path = os.path.join(args.file_path, "depth")

    panos = sorted(os.listdir(raw_path))
    depths = sorted(os.listdir(depth_path))
    cams = np.loadtxt(os.path.join(args.file_path, "cams.txt"), delimiter=" ")

    zoom = 1
    dim = (512 * zoom, 1024 * zoom)
    coord = make_coord(*dim)
    samples = 10
    rel_train_pos = make_train_pos(coord, samples) # relative to each pano
    all_train_pos = [] # relative to origin

    ts = Transform(coord=coord)

    for idx in range(len(panos)): 
        print(f"[{idx + 1} / {len(panos)}]")
        num_str = panos[idx].split(".")[0].split("_")[1]
        num = int(num_str)
        print(f"pano {num}")

        img = cv2.imread(os.path.join(raw_path, panos[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
        depth = np.load(os.path.join(depth_path, depths[idx]))
        # replace nan with max 
        # depth = np.nan_to_num(depth, nan=np.nanmax(depth))
        # (h, w, 1) -> (h, w, 3) (repeat)
        depth = np.repeat(depth.reshape(depth.shape[0], depth.shape[1], 1), 3, axis=2)

        cam = cams[num]
        cam_pos = np.array([cam[0], cam[1]])

        train_pos = rel_train_pos.copy()
        train_pos[:, 0] += cam_pos[0]
        train_pos[:, 2] += cam_pos[1]
        all_train_pos.append(train_pos)

        for sample_idx in trange(samples):
            pose = np.array([0, 0, 10])# rel_train_pos[sample_idx]
            pose *= -1

            img1, depth1, _ = ts.translate(img, depth, pose)
            img2, _, mask = ts.translate(img1, depth1, -pose)

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(np.uint8(img1))
            plt.show() 
            exit() 

            fn = f'mask_{num_str}_{str(sample_idx).zfill(4)}.png'
            fp = os.path.join(args.file_path, 'rm_occluded', fn)
            Image.fromarray(np.uint8(mask * 255)).save(fp)

    # (panos, samples, 3) -> (pano * samples, 3)
    all_train_pos = reduce(all_train_pos, "p s c -> (p s) c")
    np.savetxt(os.path.join(args.file_path, 'train_pos.txt'), 
               all_train_pos, fmt='%f', delimiter=" ")
    
    print("Training samples done. Now generating testing samples")
    # testing samples
    test_samples = 11
    all_test_pos = TestPoseGen.linear(test_samples, (-10, -10), (10, -10))
    np.savetxt(os.path.join(args.file_path, 'test_pos.txt'), 
               all_test_pos, fmt='%f', delimiter=" ")
    
    # use closest to origin to gen test
    img = cv2.imread(os.path.join(raw_path, panos[0]), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
    depth = np.load(os.path.join(depth_path, depths[0]))
    # (h, w, 1) -> (h, w, 3) (repeat)
    depth = np.repeat(depth.reshape(depth.shape[0], depth.shape[1], 1), 3, axis=2)

    cam = cams[0]
    cam_pos = np.array([cam[0], cam[1]])

    for sample_idx in trange(test_samples):
        pose = all_test_pos[sample_idx]
        # this pose is relative to origin, 
        # so we convert it to relative to cam
        pose[0] -= cam_pos[0]
        pose[2] -= cam_pos[1]
        pose *= -1

        img1, depth1, _ = ts.translate(img, depth, pose)
        img2, _, mask = ts.translate(img1, depth1, -pose)
        
        fn = f'mask_{num_str}_{str(sample_idx).zfill(4)}.png'
        fp = os.path.join(args.file_path, 'rm_occluded', fn)
        Image.fromarray(np.uint8(mask * 255)).save(fp)

if __name__ == '__main__':
    args = parse_args()
    assert not (args.url is None and args.coord is None), \
        "Must provide either url or coordinates"
    if args.coord is None: 
        args.coord = coords_from_url(args.url)
    args.file_path = \
        os.path.join(args.file_path, "gsv_" + "_".join([str(x) for x in args.coord]))
    assert os.path.exists(args.file_path), \
        f"Path {args.file_path} does not exist"

    main()