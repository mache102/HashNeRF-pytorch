"""
Given a maps url or location, retrieve nearby panoramas and save them to a directory.
This program does not generate equirectangular data for nerfs.

example:

python3 -m equirect.get_data_linear --url1 "https://www.google.com/maps/@37.342413,-121.8952678,21z?entry=ttu" --url2 "https://www.google.com/maps/@37.3417396,-121.8947369,21z?entry=ttu" --save_path gsvdata/ --min_date 2011-01
"""
import argparse
import numpy as np
import os 
import haversine 
import asyncio
import cv2 

from tqdm import trange 

from .Panorama.pn_utils import coords_from_url
from .Panorama.pn_search import search_panoramas
from .Panorama.pn_retriever import get_panorama
from .Panorama.pn_depthmap import get_depth_map
from .Panorama.pn_crop import crop_pano

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url1", type=str, default=None, 
                        help="coord 1")
    parser.add_argument("--url2", type=str, default=None,
                        help="coord 2")
    parser.add_argument("--min_date", type=str, default="2011-01", 
                        help="Minimum date for panoramas")
    parser.add_argument("--ignore_nodate", action="store_true",
                        help="Ignore panoramas without dates")
    parser.add_argument("--save_path", type=str, 
                        help="Path to save images to")

    return parser.parse_args()

def get_relative_dists(coord0, c1):
    """
    obtain relative distances in meters
    coords are (lat, lon)
    """
    lat0, lon0 = coord0
    lat1, lon1 = c1
    y = haversine.haversine((lat1, lon0), coord0, unit=haversine.Unit.METERS)
    x = haversine.haversine((lat0, lon1), coord0, unit=haversine.Unit.METERS)
    # fix signs of x and y
    if lat1 < lat0:
        y *= -1
    if lon1 < lon0:
        x *= -1

    return (x, y)

def rotate_heading(img, heading):
    """
    Rotate image by heading
    """
    # rotate image by heading
    img = np.array(img)
    img = np.roll(img, int(heading * img.shape[1] / (2 * np.pi)), axis=1)
    return img

class ProcessImage:
    def __init__(self, headings):
        self.headings = headings

        # find median of headings 
        self.heading = np.median(headings)

    def pano(self, id, idx):
        heading = self.headings[idx]
        img = asyncio.run(get_panorama(id, zoom=1))
        img = crop_pano(np.array(img))
        img = rotate_heading(img, self.heading - heading)
        
        fn = f"p_{str(idx).zfill(4)}.png"
        # save img np array as png
        cv2.imwrite(os.path.join(args.save_path, "color", fn), 
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # depth_map = get_depth_map(pano.pano_id)
        # depth_map = cv2.resize(depth_map, np.array(img).shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        # # depth_map = np.nan_to_num(depth_map, nan=10 * np.nanmax(depth_map))
        # # depth_map = depth_map.astype(np.float32)
        # fn = f"d_{str(idx).zfill(4)}"
        # # save depth map as np array instead 
        # np.save(os.path.join(args.save_path, "depth", fn), depth_map)

def main():
    os.makedirs(os.path.join(args.save_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "depth"), exist_ok=True)

    sample_coords = np.linspace(args.c1, args.c2, num=10)
    midpoint = ((args.c1[0] + args.c2[0]) / 2, (args.c1[1] + args.c2[1]) / 2)

    all_panos = []
    for i in trange(len(sample_coords)):
        panos = search_panoramas(sample_coords[i])
        # convert panos to list and add to all_panos
        all_panos += list(panos)
    all_panos = [[p.pano_id, p.latitude, p.longitude, 
                  p.heading, p.pitch, p.roll, p.date] for p in all_panos]
    all_panos = {p[0]: p[1:] for p in all_panos} # dirty way to remove duplicates
    coords = np.array([p[:2] for p in all_panos.values()])
    dates = np.array([p[-1] for p in all_panos.values()])
    norm_coords = [get_relative_dists(midpoint, c) for c in coords]

    angles = np.deg2rad([p[2:5] for p in all_panos.values()])
    angles[:, 0] = (5*np.pi/2 - angles[:, 0]) % (2*np.pi)
 
    # (n, 2) and (n, 3)
    cams = np.concatenate([norm_coords, angles], axis=1)
    with open(os.path.join(args.save_path, "cams.txt"), "w") as f:
        np.savetxt(f, cams, fmt="%.4f", delimiter=" ")
    with open(os.path.join(args.save_path, "info.txt"), "w") as f:
        # first line is median heading 
        f.write(f"{np.median(angles[:, 0])}\n\n")
        # rest are coords + dates
        for i in range(len(norm_coords)):
            f.write(f"{norm_coords[i][0]} {norm_coords[i][1]} {dates[i]}\n")

    r = 30
    pano_ids = np.array(list(all_panos.keys()))
    close_idx = np.where(np.linalg.norm(norm_coords, axis=1) < r)[0]
    pano_ids = pano_ids[close_idx]

    proc = ProcessImage(headings=angles[:, 0])

    for i in trange(len(pano_ids)):
        id = pano_ids[i]
        pano = all_panos[id]
        if pano[-1] is None:
            if args.ignore_nodate:
                continue
            else:
                proc.pano(id, i)
        elif pano[-1] > args.min_date:
            proc.pano(id, i)

if __name__ == '__main__':
    args = parse_args()

    args.c1 = coords_from_url(args.url1)
    args.c2 = coords_from_url(args.url2)
    args.save_path = \
        os.path.join(args.save_path, "gsv_" + "_".join([str(x) for x in args.c1]))
        
    main()