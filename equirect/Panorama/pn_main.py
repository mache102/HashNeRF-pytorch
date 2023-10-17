import asyncio
import json
import logging
import multiprocessing
import numpy as np 
import os
import pandas as pd
import time

from tqdm import tqdm
from typing import Tuple, Union

from constants import *
from manager_base import BaseManager
from MapTile.mt_crossing_detect import CrossingDetector
from MapTile.mt_main import MapTileManager
from .pn_search import search_panoramas
from .pn_retriever import get_panorama
from .pn_utils import pn_fn_extractor, PanoramaInfo
from utils import today

class PanoramaManager(BaseManager):
    def __init__(self, args, xings: pd.DataFrame, record_path: str) -> None:
        """
        The manager for panorama retrieval and processing.
        """
        self.args = args 
        self.xings = xings 

        # the records may need to be saved, so we store the path in the class.
        self.record_path = record_path 
        self.records = pd.read_csv(record_path)

        self.cdet = CrossingDetector(config_path=CDET_CONFIG_PATH)
        self.mtm = MapTileManager(args=args, xings=xings, record_path=record_path)
        self.no_match = "no_match"

        with open(PNF_CONFIG_PATH, "r") as f:
            self.pnf_config = json.load(f)

    def make_basename(self, coords: Tuple[float, float], pano: PanoramaInfo, date: str = "_today") -> str:
        """
        Create basename.

        date
            "_today": Today's date.
            "_wildcard": an asterisk (*) for searching basenames and comparing dates.
            otherwise, as is.
        """

        if date == "_today":
            date = today()
        elif date == "_wildcard":
            date = "*"

        basename = f"pn_{coords[0]}_{coords[1]}_{pano.offset:.4f}_{date}_{pano.zoom}__{pano.pano_id}.png"

        return basename  
    
    def filter_panoramas(self, coords: Tuple[float, float], max_pano_count: int = 1) -> Union[list, None]:
        """
        Filter panoramas (and prioritize) by distance from the crossing and the panorama's date.
        """

        panos = search_panoramas(coords)
        if panos == []:
            return None 
        
        panos = sorted(panos, key=lambda x: x.offset)

        ok_panos = []
        pano_indices = np.arange(len(panos))
        for idx, pano in enumerate(panos):
            if pano.date is not None and \
                pano.offset < self.pnf_config["distance_range"][1] \
                and pano.date > self.pnf_config["date_range"][0]: 

                # more recent date range is preferred
                ok_panos.append(pano)
                pano_indices = np.delete(pano_indices, np.where(pano_indices == idx))

            if len(ok_panos) >= max_pano_count:
                break 
        
        next_idx = 0
        while len(ok_panos) < max_pano_count:
            ok_panos.append(panos[pano_indices[next_idx]])
            next_idx += 1 

        return ok_panos
    
    def check_pn_download(self, pn_savepath: str) -> bool:
        """
        Check if a panorama has been downloaded before.
        """

        return os.path.exists(pn_savepath)
    
    def create_entry(self, pano: PanoramaInfo) -> list:
        return [pano.latitude, pano.longitude, round(pano.offset, 4),
                pano.date, pano.zoom, pano.pano_id]

    def update_history(self, matching_files: list, dl_history: list) -> list:
        """
        the panorama has been downloaded before
        if the history is incomplete, update it
        """

        for file in matching_files:

            # get filename metadata
            pn_info = pn_fn_extractor(file, is_basename=False)

            entry = self.create_entry(pn_info)
            if entry not in dl_history:
                dl_history.append(entry)

        return dl_history

    def download_single_pano(self, index: int, coords: Tuple[float, float], 
                             pano: PanoramaInfo, pn_dirname: str,
                             dl_history: list) -> None: 
        """
        Download a single panorama.
        """
        pn_basename = self.make_basename(coords=coords, pano=pano, date=pano.date)
        pn_savepath = os.path.join(pn_dirname, pn_basename)

        pn_downloaded = False 
        if not self.args.redownload:
            # not redownloading?
            # check if the pano's been downloaded b4
            pn_downloaded = self.check_pn_download(pn_savepath)

        if pn_downloaded:
            dl_history = self.update_history(matching_files=[pn_savepath], dl_history=dl_history)
            self.records.at[index, PANORAMA_RECORD] = json.dumps(dl_history)
            return None 

        entry = self.create_entry(pano)
        try:
            pano_img = asyncio.run(get_panorama(pano_id=pano.pano_id, zoom=pano.zoom))
            pano_img.save(pn_savepath)

            # check if entry isn't duplicated
            if entry not in dl_history:
                dl_history.append(entry)
            self.records.at[index, PANORAMA_RECORD] = json.dumps(dl_history)

        except Exception as e:
            logging.error(f"Error downloading panorama for {coords}: {e}")
            self.records.at[index, PANORAMA_RECORD] = json.dumps(dl_history)

        return None

    def process_record(self, index: int, xing_record: pd.Series) -> None:

        # get the xings row with the matching crossing id
        matching_xings = self.xings[self.xings[CROSSING_ID] == xing_record[CROSSING_ID]]
        if len(matching_xings) == 0:
            return None
        
        xing = matching_xings.iloc[0]
        
        """
        invalid or unselected xing_record
        """
        if xing[POS_XING] not in self.args.posxing \
            or xing_record[LONGITUDE] == -999 \
            or xing_record[LATITUDE] == -999:
            return None


        coords = (xing[LATITUDE], xing[LONGITUDE])
        """
        obtain available panorama information
        """
        if self.args.detect_crossings:
            # requires the corresponding map tile
            mt_dirname = os.path.join(MAP_TILE_PATH, self.args.state, 
                                    xing_record[COUNTY_NAME], xing_record[CITY_NAME])
            mt_basename = self.mtm.make_basename(coords=coords)
            mt_path = os.path.join(mt_dirname, mt_basename)
            xing_positions = self.cdet.detect_crossings(image_path=mt_path)
        else:
            xing_positions = [coords]

        if xing_positions == []:
            return None


        dl_history = self.get_history(xing_record, PANORAMA_RECORD)
        pn_dirname = os.path.join(PANORAMA_PATH, self.args.state, xing[COUNTY_NAME], xing[CITY_NAME])
        if not os.path.exists(pn_dirname):
            os.makedirs(pn_dirname)

        """
        all the positions found in the maptile 
        (or simply the central pixel if crossing detection 
        isn't enabled)
        """
        for pos in xing_positions:
            panos = self.filter_panoramas(coords=pos, max_pano_count=self.args.pano_per_xing)
            if panos is None:
                continue 

            for pano in panos:
                """
                download a single panorama.
                """
                pano.zoom = self.args.pano_zoom
                self.download_single_pano(index=index, coords=coords, 
                                            pano=pano, pn_dirname=pn_dirname,
                                            dl_history=dl_history)


    def download_panoramas(self) -> None:
        """
        Download panoramas.
        """

        records_subset = self.prepare_records()

        t1 = time.time()
        # with multiprocessing.Pool() as pool:
        #     pool.starmap(self.process_record, records_subset.iterrows())

        tasks = []
        for num, (index, xing_record) in tqdm(enumerate(records_subset.iterrows()), 
                                              total=len(records_subset)):
            self.process_record(index, xing_record)

        print(time.time() - t1)

        #     # if (num + 1) % self.args.pano_record_save_every == 0:
        #     #     logging.info(f"{num + 1} crossings, saving record sheet at {self.record_path}")
        #     #     self.records.to_csv(self.record_path, index=False)
        # end of loop, save record sheet (separate from the loop to be safe)
        logging.info(f"{len(records_subset)} crossings, saving record sheet at {self.record_path}")
        self.records.to_csv(self.record_path, index=False)

# class Class:
#     # ...

#     def func_b(self):
#         # ...
#         var3 = request_func() # the requests in this function take a few seconds


#     def func_a(self):
#         # ...

#         for i in iterable_a1: # probably only one iteration most of the time
#             for j in iterable_a2: # probably only one iteration most of the time
#                 # ...
#                 self.func_b()

#     def entry_point(self):
#         # ...

#         for var1, var2 in iterable_1:
#             self.func_a(var1, var2)


# def main():
#     # ...

#     c = Class()
#     c.entry_point()
