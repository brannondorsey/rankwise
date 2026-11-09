"""Satellite composite on real data: the cloud-free median of 90 Landsat 8/9 scenes.

    taskset -c 1 python bench/landsat.py [--figures-only | --fetch-only]

Landsat Collection 2 Level-2 surface reflectance for WRS-2 path 46, row 27 (Seattle and
Tacoma), every Landsat 8 and 9 acquisition of 2023 and 2024: 90 scenes. A 2048×2048 window
(61 km) of the red, green and blue bands is read from Microsoft Planetary Computer, which
mirrors the USGS archive (the AWS copy is requester-pays). Pixels that QA_PIXEL flags as fill,
dilated cloud, cirrus, cloud or cloud shadow are set to the fill value 0.

rankwise reduces the uint16 red band directly with nodata=0; the baselines need float32 with
NaN, converted once and not timed. The median must be identical to np.nanmedian on those
floats. The figure composites all three bands.

Data: Landsat imagery courtesy of the U.S. Geological Survey (public domain).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common  # noqa: F401  (one-thread env first)
import numpy as np

import rankwise as rw
from common import best_of, machine, save, versions

FIGURES_ONLY = "--figures-only" in sys.argv
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "bench" / "data"
CACHE.mkdir(parents=True, exist_ok=True)
S = 2048
PATH, ROW = "046", "027"
CENTER = (-122.33, 47.45)  # lon, lat: between Seattle and Tacoma
BANDS = ("red", "green", "blue")
MASK_BITS = 0b11111  # QA_PIXEL bits 0-4: fill, dilated cloud, cirrus, cloud, cloud shadow


def fetch():
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import Window

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    items = sorted(cat.search(collections=["landsat-c2-l2"], datetime="2023-01-01/2024-12-31",
                              query={"platform": {"in": ["landsat-8", "landsat-9"]},
                                     "landsat:wrs_path": {"eq": PATH}, "landsat:wrs_row": {"eq": ROW}}).items(),
                   key=lambda i: i.datetime)
    x, y = Transformer.from_crs(4326, 32610, always_xy=True).transform(*CENTER)
    x0, y0 = x - S * 15, y + S * 15  # upper-left corner of the window, snapped per scene below

    def read(item):
        out = {}
        for band in BANDS + ("qa_pixel",):
            for attempt in range(4):
                try:
                    with rasterio.open(item.assets[band].href) as src:
                        t = src.transform  # 30 m grid, the same for every scene of a path/row
                        col, row = round((x0 - t.c) / t.a), round((y0 - t.f) / t.e)
                        out[band] = src.read(1, window=Window(col, row, S, S), boundless=True, fill_value=1)
                    break
                except Exception:  # transient HTTP errors
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        bad = (out["qa_pixel"] & MASK_BITS) != 0
        return [out[b].astype(np.uint16) for b in BANDS], bad

    t = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(read, items))
    print(f"read {len(items)} scenes in {time.perf_counter() - t:.0f} s")
    rgb = np.stack([np.stack([r[0][b] for r in res]) for b in range(3)])  # (3, N, S, S), as delivered
    bad = np.stack([r[1] for r in res])  # (N, S, S): fill, cloud, cirrus or shadow
    return dict(rgb=rgb, bad=np.packbits(bad, axis=-1), ids=np.array([i.id for i in items]),
                dates=np.array([str(i.datetime.date()) for i in items]))


f = CACHE / f"landsat_{PATH}{ROW}_{S}.npz"
if not f.exists():
    np.savez(f, **fetch())
if "--fetch-only" in sys.argv:
    sys.exit(0)
z = np.load(f)
raw, dates = z["rgb"], z["dates"]
bad = np.unpackbits(z["bad"], axis=-1, count=S).astype(bool)
masked = bad.mean(axis=(1, 2))
rgb = np.where(bad[None], np.uint16(0), raw)  # clouds and shadows set to the fill value 0
scenes = np.ascontiguousarray(rgb[0])  # red band, (N, S, S) uint16
N = scenes.shape[0]
px = S * S
print(f"{N} scenes {dates[0]}..{dates[-1]}, {(scenes == 0).mean() * 100:.0f}% of values masked")

if not FIGURES_ONLY:
    fl = np.where(scenes == 0, np.float32(np.nan), scenes.astype(np.float32))  # baselines' input (not timed)
    ref = np.nanmedian(fl, axis=0)
    rows = []

    def record(tool, op, fn, reps=3, family="competitor", check=None):
        t = best_of(fn, reps)
        ok = None if check is None else bool(check(np.asarray(fn())))
        rows.append(dict(tool=tool, op=op, family=family, seconds=t, ns_per_px=t / px * 1e9, identical=ok))
        print(f"{tool:38s} {op:10s} {t:7.3f} s  {t / px * 1e9:7.1f} ns/px  identical={ok}", flush=True)

    same = lambda r: np.array_equal(r.astype(np.float64), ref.astype(np.float64), equal_nan=True)  # noqa: E731
    out = np.empty((S, S), np.float64)
    for isa in rw.available_isas():
        label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "portable", "neon": "NEON"}[isa]
        record(f"rankwise · {label} (uint16, nodata=0)", "median",
               lambda: rw.median(scenes, axis=0, nodata=0, out=out, isa=isa), reps=7, family="rankwise", check=same)
    import bottleneck as bn  # noqa: E402
    import numbagg  # noqa: E402

    record("NumPy 2.5 nanmedian (float32)", "median", lambda: np.nanmedian(fl, axis=0), reps=1, check=same)
    record("Bottleneck 1.6 nanmedian (float32)", "median", lambda: bn.nanmedian(fl, axis=0), reps=2, check=same)
    record("numbagg 0.9 nanmedian (float32)", "median", lambda: numbagg.nanmedian(fl, axis=0), reps=2, check=same)
    save(f"landsat_{N}x{S}", dict(machine=machine(), scenes=N, side=S, nodata=0, threads=1,
                                  source=f"Landsat C2 L2 SR, WRS-2 {PATH}/{ROW}, {dates[0]} to {dates[-1]}, "
                                         "via Microsoft Planetary Computer (USGS, public domain)",
                                  cloud_fraction=float((scenes == 0).mean()),
                                  versions=versions("rankwise", "numpy", "bottleneck", "numbagg"), rows=rows))
    del fl

# ---------------------------------------------------------------- figure
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def display(bands):
    """Surface reflectance (scale 2.75e-5, offset -0.2) to sRGB-ish 8-bit, a fixed stretch."""
    r = np.stack(bands, -1).astype(np.float32) * 2.75e-5 - 0.2
    v = np.clip(r / 0.28, 0, 1) ** (1 / 1.9)
    return (v * 255 + 0.5).astype(np.uint8)


composite = [rw.median(rgb[b], axis=0, nodata=0, out=np.empty((S, S), np.float64)) for b in range(3)]
with_clouds = [np.median(raw[b], axis=0) for b in range(3)]
pick = int(np.argmin(np.abs(masked - 0.55)))  # a typical half-cloudy acquisition
single = [raw[b][pick] for b in range(3)]
valid = (rgb[0] != 0).sum(0)
import datetime as _dt  # noqa: E402

day = _dt.date.fromisoformat(str(dates[pick])).strftime("%-d %b %Y")
panels = [
    ("One acquisition", f"{day} · {masked[pick] * 100:.0f}% cloud or shadow", display(single)),
    ("Median, clouds kept", f"np.median of {N} scenes", display(with_clouds)),
    ("Median, clouds as nodata", f"rankwise.median(nodata=0) of {N}", display(composite)),
]
OUT = ROOT / "docs" / "img"
P = 1024  # panel size (downsampled 2×)
pad, head = 16, 136
W = 3 * P + 4 * pad
H = P + head + 2 * pad
canvas = Image.new("RGB", (W, H), (14, 14, 16))
dr = ImageDraw.Draw(canvas)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
fr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
for j, (title, sub, img) in enumerate(panels):
    x = pad + j * (P + pad)
    dr.rectangle([x, pad, x + P - 1, pad + head - 1], fill=(0, 0, 0))
    dr.text((x + 24, pad + 16), title, font=fb, fill=(255, 255, 255))
    dr.text((x + 24, pad + 82), sub, font=fr, fill=(200, 200, 205))
    canvas.paste(Image.fromarray(img).resize((P, P), Image.LANCZOS), (x, pad + head))
    if j == 2:
        dr.rectangle([x - 3, pad - 3, x + P + 2, pad + head + P + 2], outline=(74, 144, 226), width=5)
canvas.save(OUT / "landsat.jpg", quality=88, optimize=True, progressive=True)
print("wrote docs/img/landsat.jpg")
