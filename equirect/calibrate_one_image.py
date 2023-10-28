"""
calibrate a single panorama
python3 calibrate_one_image.py --file_path image.png --roll 30 --pitch 30 --yaw 30
"""
import argparse
import numpy as np
import cv2

from tqdm import trange 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_path", type=str, 
                        help="image path")
    parser.add_argument("--roll", type=float, default=0,
                        help="roll angle in degrees")
    parser.add_argument("--pitch", type=float, default=0,
                        help="pitch angle in degrees")
    parser.add_argument("--yaw", type=float, default=0,
                        help="yaw angle in degrees")

    return parser.parse_args()

def eul2rot(roll, pitch, yaw):
    """euler angles to rotation matrix"""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
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
    img = cv2.imread(args.file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    n_pixels = h * w

    R = eul2rot(np.deg2rad(args.roll),
                np.deg2rad(args.pitch),
                np.deg2rad(args.yaw))

    # ~12 seconds for a 512x1024=524288 pixel image, ~50k pixels/s
    # 100 imgs (one scene): 1200 seconds, 20 mins
    # consider optimizing?
    warp_pixel_arr = np.zeros((h*w,), dtype=np.int32)
    for n in trange(n_pixels):
        i = n // w
        j = n % w
        # inverse warp a pixel
        vec_pixel = rotate_pixel((i, j), R, w, h)
        i_ = int(vec_pixel[0])
        j_ = int(vec_pixel[1])
        if((i_ >= 0) and (j_ >= 0) and (i_ < h) and (j_ < w)):
            warp_pixel_arr[i*w + j] = i_*w + j_

    warped_img = img.reshape(-1, 3)[warp_pixel_arr].reshape(h, w, 3)
    # flip about x axis 
    # warped_img = warped_img[::-1, :, :]
    cv2.imwrite("warped_image.png", cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    args = parse_args()
    main()