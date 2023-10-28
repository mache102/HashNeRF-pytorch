"""
Given a maps url or location, retrieve nearby panoramas and save them to a directory.
This program does not generate equirectangular data for nerfs.

example:
python3 -m equirect.plot_coords --url "https://www.google.com/maps/@37.3420419,-121.8948427,19.31z?entry=ttu" --file_path imgs/

python3 -m equirect.plot_coords --url "https://www.google.com/maps/@37.342413,-121.8952678,21z?entry=ttu" --file_path gsvdata/
"""
import argparse
import numpy as np
import os 
import matplotlib.pyplot as plt

from .Panorama.pn_utils import coords_from_url

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default=None, 
                        help="url containing coordinates")
    parser.add_argument("--coord", '-c', type=float, nargs=2, 
                        default=None, help="latitude and longitude")
    parser.add_argument("--file_path", type=str, 
                        help="Path to save images to")

    return parser.parse_args()

def main():
    cams = np.loadtxt(os.path.join(args.file_path, "cams.txt"), delimiter=" ")
    coords, headings = cams[:, 0:2], cams[:, 2]
    coords = coords[np.linalg.norm(coords, axis=1) < 30]
    unit_vecs = np.stack([np.cos(headings), np.sin(headings)], axis=1)

    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(coords[:, 0], coords[:, 1], color="black")

    for coord, unit_vec in zip(coords, unit_vecs):
        plt.arrow(*coord, *unit_vec, width=0.1, color="black")

    # dot at origin 
    plt.scatter(0, 0, c="r", marker="*")
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    assert not (args.url is None and args.coord is None), \
        "Must provide either url or coordinates"
    if args.coord is None: 
        args.coord = coords_from_url(args.url)
    args.file_path = \
        os.path.join(args.file_path, "gsv_" + "_".join([str(x) for x in args.coord]))
        
    main()