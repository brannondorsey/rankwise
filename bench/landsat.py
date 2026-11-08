"""Satellite-composite workload: cloud-free median of uint16 scenes with a nodata value.

    taskset -c 1 python bench/landsat.py [scenes] [side]

Synthetic stand-in for one Landsat path/row over two years (the real archive is
requester-pays on AWS): 90 uint16 surface-reflectance scenes of 2048×2048, each with 20-70%
of its pixels under cloud, set to the Collection 2 fill value 0. rankwise works on the uint16
data directly with nodata=0; the baselines need float32 with NaN, so that conversion is done
once up front and not timed. The median must be identical to np.nanmedian on those floats.
"""

import sys

import common  # noqa: F401
import numpy as np
from scipy.ndimage import gaussian_filter

import rankwise as rw
from common import best_of, machine, save, versions

N = int(sys.argv[1]) if len(sys.argv) > 1 else 90
S = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
rng = np.random.default_rng(11)
land = gaussian_filter(rng.standard_normal((S, S)), 24)
land = (8000 + 6000 * (land - land.min()) / np.ptp(land)).astype(np.float32)
scenes = np.empty((N, S, S), np.uint16)
for k in range(N):
    season = 1 + 0.15 * np.sin(2 * np.pi * k / 45)
    x = land * season + rng.normal(0, 150, (S, S)).astype(np.float32)
    cloud = gaussian_filter(rng.standard_normal((S // 8, S // 8)), 3)
    cloud = np.kron(cloud > np.quantile(cloud, 1 - rng.uniform(0.2, 0.7)), np.ones((8, 8), bool))
    x[cloud] = 0
    scenes[k] = np.clip(x, 0, 65535).astype(np.uint16)
f = np.where(scenes == 0, np.float32(np.nan), scenes.astype(np.float32))  # baselines' input (not timed)
px = S * S
ref = np.nanmedian(f, axis=0)
rows = []


def record(tool, op, fn, reps=3, family="competitor", check=None):
    t = best_of(fn, reps)
    ok = None if check is None else bool(check(np.asarray(fn())))
    rows.append(dict(tool=tool, op=op, family=family, seconds=t, ns_per_px=t / px * 1e9, identical=ok))
    print(f"{tool:34s} {op:10s} {t:7.3f} s  {t / px * 1e9:7.1f} ns/px  identical={ok}", flush=True)


same = lambda r: np.array_equal(r.astype(np.float64), ref.astype(np.float64), equal_nan=True)  # noqa: E731
out = np.empty((S, S), np.float64)
for isa in rw.available_isas():
    label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "portable", "neon": "NEON"}[isa]
    record(f"rankwise · {label} (uint16, nodata=0)", "median", lambda: rw.median(scenes, axis=0, nodata=0, out=out, isa=isa),
           family="rankwise", check=same)
import bottleneck as bn  # noqa: E402
import numbagg  # noqa: E402

record("NumPy 2.5 nanmedian (float32)", "median", lambda: np.nanmedian(f, axis=0), reps=1, check=same)
record("Bottleneck 1.6 nanmedian (float32)", "median", lambda: bn.nanmedian(f, axis=0), reps=2, check=same)
record("numbagg 0.9 nanmedian (float32)", "median", lambda: numbagg.nanmedian(f, axis=0), reps=2, check=same)
save(f"landsat_{N}x{S}", dict(machine=machine(), scenes=N, side=S, nodata=0, threads=1,
                              cloud_fraction=float((scenes == 0).mean()),
                              versions=versions("rankwise", "numpy", "bottleneck", "numbagg"), rows=rows))
