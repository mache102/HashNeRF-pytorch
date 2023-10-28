import os 
import re
import requests 
import json

from dataclasses import dataclass 
from typing import Optional, Tuple

@dataclass 
class PanoramaInfo:
    pano_id: str
    latitude: float
    longitude: float
    heading: float
    pitch: Optional[float]
    roll: Optional[float]
    date: Optional[str]
    zoom: Optional[int]
    offset: Optional[float]


# sample url: https://www.google.com/maps/@29.6950438,-95.3274211,3a,75y,225.52h,96.09t/data=!3m6!1e1!3m4!1sGo1mMho-l5se_RBtFpfgZA!2e0!7i16384!8i8192?entry=ttu

def pano_id_from_url(url):
    """
    Extract pano_id from url.
    The id is the 22 character string between !1 and !2
    """
    pattern = re.compile(r"!1s(.{22})!2")
    # find id (only 1 match)
    match = pattern.findall(url)[0]
    return match

def coords_from_url(url) -> Tuple[float, float]:
    """
    Extract lat, lon pair from url.

    ... @ <lat>,<lon>,...
    """
    coords = url.split("@")[1].split(",")[0:2]
    return (float(coords[0]), float(coords[1]))

def pn_fn_extractor(fn: str, is_basename: bool = False) -> PanoramaInfo:
    """
    extract metadata from the filename of a panorama.
    basename for reference:

    basename = f"pn_{coords[0]}_{coords[1]}_{date}_{self.args.pano_zoom}__{pano_id}.png"
    """

    if is_basename == False:
        fn = os.path.basename(fn)

    fn_id_split = fn.split("__")
    pano_id = fn_id_split[1].split(".")[0]

    fn_split = fn_id_split[0].split("_")
    latitude = float(fn_split[1])
    longitude = float(fn_split[2])
    offset = float(fn_split[3])
    date = fn_split[4]
    zoom = int(fn_split[5])

    return PanoramaInfo(pano_id=pano_id, 
                        latitude=latitude, 
                        longitude=longitude,
                        heading=None,
                        pitch=None,
                        roll=None,
                        date=date, 
                        zoom=zoom,
                        offset=offset)


def get_metadata(pano_id):
    """
    Retrieve the metadata associated with a given panorama id.
    WARNING: returned results may be lengthy
    """
    endpoint = 'https://www.google.com/maps/photometa/v1'

    # Set API parameters
    params = {
        'authuser': '0',
        'hl': 'en',
        'gl': 'us',
        'pb': f'!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s{pano_id}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
    }

    # Send GET request to API endpoint and retrieve response
    response = requests.get(endpoint, params=params, proxies=None)
    # response = requests.get(endpoint)

    # Extract image and depth map from response
    # print(response.links)
    response = response.content
    # print(len(response))
    
    response = json.loads(response[4:])
    return response