"""
Given a maps url or location, retrieve nearby panoramas and save them to a directory.
This program does not generate equirectangular data for nerfs.

example:

python3 -m equirect.get_data --url "https://www.google.com/maps/@37.3420419,-121.8948427" --save_path imgs/


python3 -m equirect.get_data --url "https://www.google.com/maps/@37.342413,-121.8952678,21z?entry=ttu" --url2 "https://www.google.com/maps/@37.3417396,-121.8947369,21z?entry=ttu" --save_path imgs/
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default=None, 
                        help="url containing coordinates")
    parser.add_argument("--coord", '-c', type=float, nargs=2, 
                        default=None, help="latitude and longitude")
    parser.add_argument("--min_date", type=str, default="2011-01", 
                        help="Minimum date for panoramas")
    parser.add_argument("--ignore_nodate", action="store_true",
                        help="Ignore panoramas without dates")
    parser.add_argument("--save_path", type=str, 
                        help="Path to save images to")

    return parser.parse_args()

def get_relative_dists(coord0, coord1):
    """
    obtain relative distances in meters
    coords are (lat, lon)
    """
    lat0, lon0 = coord0
    lat1, lon1 = coord1
    y = haversine.haversine((lat1, lon0), coord0, unit=haversine.Unit.METERS)
    x = haversine.haversine((lat0, lon1), coord0, unit=haversine.Unit.METERS)
    # fix signs of x and y
    if lat1 < lat0:
        y *= -1
    if lon1 < lon0:
        x *= -1

    return (x, y)

def process_pano(pano, idx):
    pano_img = asyncio.run(get_panorama(pano.pano_id, zoom=1))
    fn = f"p_{str(idx).zfill(4)}.png"
    pano_img.save(os.path.join(args.save_path, "color", fn))

    depth_map = get_depth_map(pano.pano_id)
    depth_map = cv2.resize(depth_map, np.array(pano_img).shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
    # depth_map = np.nan_to_num(depth_map, nan=10 * np.nanmax(depth_map))
    # depth_map = depth_map.astype(np.float32)
    fn = f"d_{str(idx).zfill(4)}"
    # save depth map as np array instead 
    np.save(os.path.join(args.save_path, "depth", fn), depth_map)

def main():

    os.makedirs(os.path.join(args.save_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "depth"), exist_ok=True)
    panos = search_panoramas(args.coord)
    # sort by offset (distance from search point)
    panos.sort(key=lambda x: x.offset)
    
    # get average lat, lon of panos
    lat = sum([p.latitude for p in panos]) / len(panos)
    lon = sum([p.longitude for p in panos]) / len(panos)
    print(f"Average lat, lon: {lat:.6f}, {lon:.6f}")

    coords = []
    rotations = []
    for pano in panos:
        print(pano)
        coord0 = (lat, lon)
        coord1 = (pano.latitude, pano.longitude)
        x, y = get_relative_dists(coord0, coord1)

        # deg2rad heading, match w/ cartesian
        heading = (450 - pano.heading) % 360 # np.pi/2 - np.deg2rad(pano.heading)
        coords.append([x, y])
        rotations.append([heading, pano.pitch, pano.roll])

    # (n, 2) and (n, 3)
    cams = np.concatenate([coords, rotations], axis=1)
    with open(os.path.join(args.save_path, "cams.txt"), "w") as f:
        np.savetxt(f, cams, fmt="%.4f", delimiter=" ")

    for i in trange(len(panos)):
        pano = panos[i]
        if pano.date is None:
            if args.ignore_nodate:
                continue
            else:
                process_pano(pano, i)
        elif pano.date > args.min_date:
            process_pano(pano, i)

if __name__ == '__main__':
    args = parse_args()
    assert not (args.url is None and args.coord is None), \
        "Must provide either url or coordinates"
    if args.coord is None: 
        args.coord = coords_from_url(args.url)
    args.save_path = \
        os.path.join(args.save_path, "gsv_" + "_".join([str(x) for x in args.coord]))
        
    main()