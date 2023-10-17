import json
import re
import requests

from haversine import haversine, Unit
from requests.models import Response
from typing import List, Tuple

from .pn_utils import PanoramaInfo

def make_search_url(coords: Tuple[float, float]) -> str:
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = (
        "https://maps.googleapis.com/maps/api/js/"
        "GeoPhotoService.SingleImageSearch"
        "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(*coords)


def search_request(coords: Tuple[float, float]) -> Response:
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    """
    url = make_search_url(coords)
    return requests.get(url)


def extract_panoramas(text: str) -> List[PanoramaInfo]:
    """
    Given a valid response from the panoids endpoint, return a list of all the
    panoids.
    """

    # The response is actually javascript code. It's a function with a single
    # input which is a huge deeply nested array of items.
    blob = re.findall(r"callbackfunc\( (.*) \)$", text)[0]
    data = json.loads(blob)

    if data == [[5, "generic", "Search returned no images."]]:
        return []

    if data == [[3, 'generic', 'Invalid location']]:
        return []

    subset = data[1][5][0]

    raw_panos = subset[3][0]

    if len(subset) < 9 or subset[8] is None:
        raw_dates = []
    else:
        raw_dates = subset[8]

    # For some reason, dates do not include a date for each panorama.
    # the n dates match the last n panos. Here we flip the arrays
    # so that the 0th pano aligns with the 0th date.
    raw_panos = raw_panos[::-1]
    raw_dates = raw_dates[::-1]

    dates = [f"{d[1][0]}-{d[1][1]:02d}" for d in raw_dates]

    return [
        PanoramaInfo(
            pano_id=pano[0][1],
            latitude=pano[2][0][2],
            longitude=pano[2][0][3],
            heading=pano[2][2][0],
            pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None,
            roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None,
            date=dates[i] if i < len(dates) else None,
            zoom=None,
            offset=None,
        )
        for i, pano in enumerate(raw_panos)
    ]


def search_panoramas(coords: Tuple[float, float]) -> List[PanoramaInfo]:
    """
    Gets the closest panoramas (ids) to the GPS coordinates.
    """

    resp = search_request(coords)
    try:
        panos = extract_panoramas(resp.text)
        for i in range(len(panos)):
            panos[i].offset = haversine((panos[i].latitude, panos[i].longitude), coords, unit=Unit.METERS)

    except Exception as e:
        print(resp.text)
        raise Exception(f"Error parsing response: {e}")
        
    
    return panos

if __name__ == '__main__':
    from .pn_utils import coords_from_url

    url = "https://www.google.com/maps/@29.2030191,-95.2176061,3a,75y,256.4h,90t/data=!3m7!1e1!3m5!1syGLFyQYAnt_VvLidyT1-SA!2e0!6shttps:%2F%2Fstreetviewpixels-pa.googleapis.com%2Fv1%2Fthumbnail%3Fpanoid%3DyGLFyQYAnt_VvLidyT1-SA%26cb_client%3Dmaps_sv.tactile.gps%26w%3D203%26h%3D100%26yaw%3D254.5042%26pitch%3D0%26thumbfov%3D100!7i16384!8i8192?entry=ttu"

    coords = coords_from_url(url)

    panos = search_panoramas(coords)
    # sort by offset (distance from search point)
    panos.sort(key=lambda x: x.offset)

    for pano in panos:
        print(pano)




