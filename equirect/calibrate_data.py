"""
calibrate panoramas obtained from get_data.py with metadata (cam pitch and roll in particular)

python3 -m equirect.calibrate_data --file_path imgs/ --url "https://www.google.com/maps/@37.4237922,-121.8896494"
"""
import argparse
import numpy as np
import os 
import cv2

from .Panorama.pn_utils import coords_from_url
from .Panorama.pn_crop import crop_pano

from tqdm import trange 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default=None, 
                        help="url containing coordinates")
    parser.add_argument("--coord", '-c', type=float, nargs=2, 
                        default=None, help="latitude and longitude")
    parser.add_argument("--file_path", type=str, 
                        help="Path to save images to (and load from)")

    return parser.parse_args()


def eul2rot(roll, pitch, heading):
    """euler angles to rotation matrix"""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]])
    R = R_x @ R_y @ R_z
    return R

def rotate_pixel(coord, R, w, h):
    """
    coord: (2,)
    R: (3, 3)
    w, h: int
    """
    coord = (np.pi * coord[0] / h, 2 * np.pi * coord[1] / w)
    # convert to cartesian
    vec_cartesian = np.array([-np.sin(coord[0])*np.cos(coord[1]), 
                              np.sin(coord[0])*np.sin(coord[1]), 
                              np.cos(coord[0])], dtype=np.float32)
    
    vec_cartesian_rot = np.dot(R, vec_cartesian)
    
    # convert back to spherical
    coord_rot = np.array([np.arccos(vec_cartesian_rot[2]), 
                          np.arctan2(vec_cartesian_rot[1], -vec_cartesian_rot[0])])
    if coord_rot[1] < 0:
        coord_rot[1] += 2*np.pi
    # convert to pixel
    coord_rot = np.array([int(h*coord_rot[0]/np.pi), 
                          int(w*coord_rot[1]/(2*np.pi))])
    return coord_rot

def main():
    color_path = os.path.join(args.file_path, "color")
    depth_path = os.path.join(args.file_path, "depth")
    panos = sorted([fn for fn in os.listdir(color_path) if fn.startswith("p_")])
    depths = sorted([fn for fn in os.listdir(depth_path) if fn.startswith("d_")])

    cams = np.loadtxt(os.path.join(args.file_path, "cams.txt"), delimiter=" ")
    for idx in range(len(panos)): 
        print(f"[{idx + 1} / {len(panos)}]")
        num_str = panos[idx].split(".")[0].split("_")[1]
        num = int(num_str)
        print(f"pano {num}")

        img = cv2.imread(os.path.join(color_path, panos[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_pano(img) # some panoramas have repetitive vertical strips of panorama data on the right
        h, w, _ = img.shape

        depth = np.load(os.path.join(depth_path, depths[idx]))

        cam = cams[num]
        pitch, roll = cam[3], cam[4]
        pitch = np.deg2rad(90 - pitch)
        roll = np.deg2rad(0 - roll)
        yaw = 0 # keep as is 

        R = eul2rot(roll, pitch, yaw)

        # ~12 seconds for a 512x1024 image, 40~50 it/s
        # = ~5 min for 25 imgs
        # consider optimizing?
        warp_pixel_arr = np.zeros((h*w, ), dtype=np.int32)
        for i in trange(h):
            for j in range(w):
                # inverse warp a pixel
                vec_pixel = rotate_pixel((i, j), R, w, h)
                i_ = int(vec_pixel[0])
                j_ = int(vec_pixel[1])
                if((i_ >= 0) and (j_ >= 0) and (i_ < h) and (j_ < w)):
                    warp_pixel_arr[i*w + j] = i_*w + j_
                    # warped_img[i, j, :] = img[i_, j_, :]

        warped_img = img.reshape(-1, 3)[warp_pixel_arr].reshape(h, w, 3)
        warped_depth = depth.flatten()[warp_pixel_arr].reshape(h, w)
        
        # save to args.file_path, pcal_xxxx.png
        fn = f"pcal_{num_str}.png"
        cv2.imwrite(os.path.join(color_path, fn), 
                    cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))
        fn = f"dcal_{num_str}.npy"
        np.save(os.path.join(depth_path, fn), warped_depth)


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