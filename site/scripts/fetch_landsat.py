"""Fetch the 90 Landsat scenes bench/landsat.py uses, at 1/4 resolution (120 m) for the site.

Same STAC query, window and QA_PIXEL mask bits as bench/landsat.py; reads COG overviews.
Data: Landsat imagery courtesy of the U.S. Geological Survey (public domain).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / "data" / "landsat_046027_site512.npz"
S, D = 2048, 512
CENTER = (-122.33, 47.45)
BANDS = ("red", "green", "blue")
MASK_BITS = 0b11111

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
items = sorted(cat.search(collections=["landsat-c2-l2"], datetime="2023-01-01/2024-12-31",
                          query={"platform": {"in": ["landsat-8", "landsat-9"]},
                                 "landsat:wrs_path": {"eq": "046"}, "landsat:wrs_row": {"eq": "027"}}).items(),
               key=lambda i: i.datetime)
x, y = Transformer.from_crs(4326, 32610, always_xy=True).transform(*CENTER)
x0, y0 = x - S * 15, y + S * 15


def read(item):
    out = {}
    for band in BANDS + ("qa_pixel",):
        for attempt in range(5):
            try:
                with rasterio.open(item.assets[band].href) as src:
                    t = src.transform
                    col, row = round((x0 - t.c) / t.a), round((y0 - t.f) / t.e)
                    rs = Resampling.nearest if band == "qa_pixel" else Resampling.average
                    out[band] = src.read(1, window=Window(col, row, S, S), out_shape=(D, D), boundless=True,
                                         fill_value=1, resampling=rs)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return np.stack([out[b] for b in BANDS]).astype(np.uint16), (out["qa_pixel"] & MASK_BITS) != 0


t = time.perf_counter()
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(read, items))
print(f"read {len(items)} scenes in {time.perf_counter() - t:.0f} s")
np.savez(OUT, rgb=np.stack([r[0] for r in res]), bad=np.stack([r[1] for r in res]),
         dates=np.array([str(i.datetime.date()) for i in items]))
