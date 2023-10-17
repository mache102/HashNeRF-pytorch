import asyncio
import argparse 
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys 
sys.path.append('./')
import time 
import cProfile 
import pstats

from constants import *
from data_reader import read_xings
from manager_base import BaseManager
from Panorama.pn_main import PanoramaManager
from pn_retriever import get_panorama
from MapTile.mt_utils import get_mt_list

def main():
    # get any panorama 
    args = argparse.Namespace()
    args.tile_zoom = 16 
    args.tile_layers = "r"
    args.date_filter = "latest"

    posxing = [1]
    state = "CA"
    county = "ALAMEDA"
    city = "SAN LEANDRO"
    pano_num = -9
    pano_zoom = 4

    ct_path = os.path.join(DATA_PATH, f"col_types.json")
    data_path = os.path.join(DATA_PATH, f"{state}_StateCrossingData.csv")
    record_path = os.path.join(RECORD_PATH, f"{state}_record.csv")

    df_xings_raw = read_xings(data_path, ct_path)
    df_xings = df_xings_raw[df_xings_raw[POS_XING].isin(posxing)]
    if county is not None:
        df_xings = df_xings[df_xings[COUNTY_NAME] == county]
    if city is not None:
        df_xings = df_xings[df_xings[CITY_NAME] == city]

    mt_dirname = os.path.join(MAP_TILE_PATH, state, county, city)
    mt_list = get_mt_list(mt_path=mt_dirname, args=args)
    mt_basename = mt_list[pano_num]

    latitude = float(mt_basename.split("_")[1])
    longitude = float(mt_basename.split("_")[2])
    coords = (latitude, longitude)

    pnm = PanoramaManager(args=None, xings=df_xings, record_path=record_path)
    panos = pnm.filter_panoramas(coords=coords, max_pano_count=1)
    pano = panos[0]
    pano.zoom = pano_zoom

    print('retrieve')
    t1 = time.time()
    pano_img, times = asyncio.run(get_panorama(pano_id=pano.pano_id, zoom=pano.zoom, debug=True))
    print(time.time() - t1)

    print(times)
    print(np.sum(times["fetch"]), np.sum(times["paste"]))

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    
    # Create a pstats.Stats object from the profiler
    stats = pstats.Stats(profiler)
    
    # Print the top 100 cumulative lines
    stats.strip_dirs().sort_stats("cumulative").print_stats(100)
"""
z=1
[4.5299530029296875e-06, 7.152557373046875e-07] 2.6226043701171875e-06

"""

# using bash open run_nerf_helper.py and comment out line 8
# python -m cProfile -o profile.out run_nerf_helpers.py
