import asyncio
import aiohttp
import itertools
import numpy as np
import time 


from dataclasses import dataclass
from io import BytesIO
from typing import Generator, Tuple, List

from PIL import Image

"""
Retrievers for Google Maps panoramas and map tiles.

Panorama retrievers taken from https://github.com/robolyst/streetview/ with extensive modifications.
"""

TILE_WIDTH = 512
TILE_WIDTH = 512

@dataclass
class PanoramaTile:
    x: int
    y: int
    fileurl: str

def get_width_and_height_from_zoom(zoom: int) -> Tuple[int, int]:
    """
    Returns the width and height of a panorama at a given zoom level.
    """
    return 2 ** zoom, 2 ** (zoom - 1)

def make_download_url(pano_id: str, zoom: int, x: int, y: int) -> str:
    """
    Returns the URL to download a tile.
    """
    return (
        "https://cbk0.google.com/cbk"
        f"?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    )

async def fetch_panorama_tile_async(tile_info: PanoramaTile, session: aiohttp.ClientSession) -> BytesIO:
    while True:
        try:
            async with session.get(tile_info.fileurl) as response:
                response_content = await response.read()
                return BytesIO(response_content)
        except aiohttp.ClientConnectionError:
            print("Connection error. Trying again in 2 seconds.")
            await asyncio.sleep(2)

async def get_panorama(pano_id: str, zoom: int = 1, debug=False) -> np.ndarray:
    total_width, total_height = get_width_and_height_from_zoom(zoom)
    panorama = Image.new("RGB", (total_width * TILE_WIDTH, total_height * TILE_WIDTH))

    async with aiohttp.ClientSession() as session:
        tasks = []
        times = {
            "fetch": [],
            "paste": [],
        }  # time measures

        for x, y in itertools.product(range(total_width), range(total_height)):
            tile_info = PanoramaTile(
                x=x,
                y=y,
                fileurl=make_download_url(pano_id=pano_id, zoom=zoom, x=x, y=y),
            )

            if debug:
                start_time = time.time()

            tasks.append(fetch_panorama_tile_async(tile_info, session))

            if debug:
                end_time = time.time()
                times["fetch"].append(end_time - start_time)

        tiles = await asyncio.gather(*tasks)

    # paste tiles
    for (x, y), tile in zip(itertools.product(range(total_width), range(total_height)), tiles):
        if debug:
            start_time = time.time()
        panorama.paste(im=Image.open(tile), box=(x * TILE_WIDTH, y * TILE_WIDTH))
        if debug:
            end_time = time.time()
            times["paste"].append(end_time - start_time)

    if debug:
        return panorama, times
    else:
        return panorama
