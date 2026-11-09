"""Weather case study on real data: NOAA GEFS 2 m temperature from dynamical.org.

    taskset -c 1 python bench/gefs.py [--figures-only] [cache_dir]

Loads one run (init 2026-11-01T00) over the contiguous US for every member and every lead
time to 240 h (31 × 81 × 121 × 257, about 310 MB), then
* times p10/p50/p90 across members, one thread, against NumPy, numbagg and fastnanquantile,
  checking that rankwise is bit-identical to np.quantile;
* draws a 10-day fan chart for Denver and a map of the ensemble's p10–p90 spread.

Data: NOAA GEFS via dynamical.org, CC BY 4.0.
"""

import json
import sys
import time
from pathlib import Path

import common  # noqa: F401  (one-thread env first)
import numpy as np

import rankwise as rw
from common import best_of, machine, save, versions

FIGURES_ONLY = "--figures-only" in sys.argv
if FIGURES_ONLY:
    sys.argv.remove("--figures-only")
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "data"
CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"
INIT = "2026-11-01T00"
LAT = slice(52.0, 22.0)
LON = slice(-128.0, -64.0)


def load():
    f = CACHE / "gefs_t2m_conus.npz"
    if f.exists():
        z = np.load(f)
        return z["t2m"], z["lat"], z["lon"], z["lead_h"]
    import xarray as xr

    t = time.perf_counter()
    ds = xr.open_zarr(URL, chunks=None)
    run = ds["temperature_2m"].sel(init_time=INIT, latitude=LAT, longitude=LON).isel(lead_time=slice(0, 81))
    run = run.transpose("ensemble_member", "lead_time", "latitude", "longitude")
    t2m = np.ascontiguousarray(run.values.astype(np.float32))
    lat, lon = run.latitude.values, run.longitude.values
    lead_h = (run.lead_time.values / np.timedelta64(1, "h")).astype(np.float32)
    print(f"downloaded {t2m.shape} in {time.perf_counter() - t:.0f} s")
    np.savez(f, t2m=t2m, lat=lat, lon=lon, lead_h=lead_h)
    return t2m, lat, lon, lead_h


t2m, lat, lon, lead_h = load()
n, nl, nlat, nlon = t2m.shape
cells = nl * nlat * nlon
print(f"GEFS {INIT}: {n} members × {nl} leads × {nlat} × {nlon}, NaN: {int(np.isnan(t2m).sum())}")
QS = [0.1, 0.5, 0.9]
ref = None if FIGURES_ONLY else np.quantile(t2m, QS, axis=0)
rows = []


def record(tool, fn, reps=3, family="competitor"):
    t = best_of(fn, reps)
    ident = bool(np.array_equal(np.asarray(fn()).astype(ref.dtype), ref, equal_nan=True))
    rows.append(dict(tool=tool, family=family, seconds=t, ns_per_cell=t / cells * 1e9, identical_to_numpy=ident))
    print(f"{tool:30s} {t:7.3f} s  {t / cells * 1e9:7.1f} ns/cell  identical={ident}", flush=True)


out = np.empty((3,) + t2m.shape[1:], np.float64)
for isa in [] if FIGURES_ONLY else rw.available_isas():
    label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "portable", "neon": "NEON"}[isa]
    record(f"rankwise · {label}", lambda: rw.quantile(t2m, QS, axis=0, out=out, isa=isa), family="rankwise")
import fastnanquantile as fnq  # noqa: E402
import numbagg  # noqa: E402

if not FIGURES_ONLY:
    record("NumPy 2.5 quantile", lambda: np.quantile(t2m, QS, axis=0), reps=2)
    record("numbagg 0.9 nanquantile", lambda: numbagg.nanquantile(t2m, QS, axis=0), reps=2)
    record("fastnanquantile 0.0.2", lambda: np.stack([fnq.nanquantile(t2m, q, axis=0) for q in QS]), reps=2)
if not FIGURES_ONLY:
    save("gefs_conus", dict(machine=machine(), source=f"{URL} init {INIT}, temperature_2m, CONUS, leads 0-240 h (CC BY 4.0)",
                            shape=[n, nl, nlat, nlon], quantiles=QS, threads=1,
                            versions=versions("rankwise", "numpy", "numbagg", "fastnanquantile"), rows=rows))

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from figures import FONT, THEMES, headline, style  # noqa: E402

plt.rcParams.update({"font.family": FONT, "font.size": 10.5})
IMG = ROOT / "docs" / "img"
city, (clat, clon) = "Denver", (39.74, -104.99)
iy, ix = int(np.abs(lat - clat).argmin()), int(np.abs(lon - clon).argmin())
fan = rw.quantile(t2m[:, :, iy, ix], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)  # (5, leads)
spread = rw.quantile(t2m[:, 40], [0.1, 0.9], axis=0)  # lead index 40 = 120 h
spread = spread[1] - spread[0]
days = lead_h / 24

for theme, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(9.2, 4.2), facecolor=t["surface"])
    style(ax, t)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.xaxis.grid(False)
    blue = t["rw"][0]
    ax.fill_between(days, fan[0], fan[4], color=blue, alpha=0.18, linewidth=0)
    ax.fill_between(days, fan[1], fan[3], color=blue, alpha=0.30, linewidth=0)
    for m in range(n):
        ax.plot(days, t2m[m, :, iy, ix], color=t["muted"], linewidth=0.5, alpha=0.35)
    ax.plot(days, fan[2], color=blue, linewidth=2)
    ax.text(days[-1] + 0.12, fan[2][-1], "median", va="center", color=t["ink"], fontsize=9.5)
    ax.text(days[-1] + 0.12, fan[4][-1], "p90", va="center", color=t["ink2"], fontsize=9)
    ax.text(days[-1] + 0.12, fan[0][-1], "p10", va="center", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, days[-1] + 0.9)
    ax.set_xticks(range(0, 11))
    ax.set_xlabel("Days after 2026-11-01 00 UTC", color=t["ink2"])
    ax.set_ylabel("2 m temperature (°C)", color=t["ink2"])
    headline(fig, f"{city}: the 31-member GEFS forecast fans out after day 4",
             "Median with p25–p75 and p10–p90 bands from rankwise.quantile; thin lines are the members · "
             "NOAA GEFS via dynamical.org (CC BY 4.0)", t, top=0.975)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.8, bottom=0.13)
    fig.savefig(IMG / f"gefs_fan_{theme}.png", dpi=200, facecolor=t["surface"])
    plt.close(fig)

    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    if theme == "dark":
        ramp = ramp[::-1][:6][::-1]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    fig, ax = plt.subplots(figsize=(9.2, 5.9), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    im = ax.imshow(spread, extent=[lon[0], lon[-1], lat[-1], lat[0]], cmap=cmap, aspect=1.25, interpolation="nearest")
    ax.set_anchor("W")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(colors=t["ink2"], length=0)
    ax.plot([clon], [clat], marker="o", markersize=6, color=t["s2"], markeredgecolor=t["surface"])
    ax.text(clon + 0.8, clat, city, color=t["ink"], fontsize=9.5, va="center")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015, anchor=(0.0, 0.5))
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["ink2"], length=0)
    cb.set_label("p90 − p10 (°C)", color=t["ink2"])
    headline(fig, "Where the forecast is uncertain at day 5",
             "Spread between the 10th and 90th percentile of 31 members, 2 m temperature at +120 h · "
             "NOAA GEFS via dynamical.org", t, top=0.975)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.86, bottom=0.06)
    fig.savefig(IMG / f"gefs_spread_{theme}.png", dpi=200, facecolor=t["surface"])
    plt.close(fig)
print(json.dumps({"city": city, "grid_point": [float(lat[iy]), float(lon[ix])]}))
