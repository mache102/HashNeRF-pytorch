from .pn_utils import coords_from_url
from haversine import haversine, Unit
"""
given a list of urls of length n,
calc the distance (in meters)
between the i-th and i+1-th urls
for i = 0, 1, ..., n-2
"""

urls = [
    "https://www.google.com/maps/@29.2018739,-95.218848,3a,75y,63.61h,83.05t/data=!3m6!1e1!3m4!1s7JbROvAPeGrBtiIeoEkwZQ!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.2019382,-95.2187787,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1svLOI4uIY2p8r41h0h0nyvw!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.2020027,-95.2187096,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1sO93Tohg4GX3bMGyy3K8GpA!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.2020668,-95.2186404,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1sggi3u3H5Y4RFUi9ZazRwDA!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.202131,-95.2185713,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1siDQ2nMTPCdyjpF9GbBl6-A!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.2021948,-95.2185021,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1soBxp_qNrk_fwQiZjZuyY1Q!2e0!7i16384!8i8192?entry=ttu",
    "https://www.google.com/maps/@29.2022587,-95.2184328,3a,75y,40.86h,64.48t/data=!3m6!1e1!3m4!1scGH5swgvyfdAGQCgxR5XTw!2e0!7i16384!8i8192?entry=ttu"
]

coords = [coords_from_url(url) for url in urls]

for i in range(len(coords) - 1):
    print(f"distance between {i} and {i+1}: {haversine(coords[i], coords[i+1], unit=Unit.METERS)}")

