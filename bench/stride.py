"""Power-of-two versus padded frame strides (handoff §4.6), one thread.

    taskset -c 1 python bench/stride.py [--depth-only] [--out NAME]

NumPy puts large arrays on 2 MiB pages, so a contiguous stack of 2^k-byte frames starts every
frame in the same L2 set. Two tables: three frame sizes at 31 frames, and a depth sweep at
1024×1024 with the read path rankwise chose (direct, skewed or staged).
"""

import sys

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, machine, save

rng = np.random.default_rng(0)
DEPTH_ONLY = "--depth-only" in sys.argv
NAME = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "stride"


def stack(n, npix, pad):
    buf = rng.normal(1000, 30, (n, npix + pad)).astype(np.float32)
    return buf, buf[:, :npix]


def path(st):
    return rw.describe(st, axis=0).split("path=")[1].split()[0]


ROUNDS = 4  # configurations alternate within each round, so VM noise hits them alike


def measure(cases, reps):
    """cases: {key: fn}. Best time per key over ROUNDS interleaved rounds of `reps` runs."""
    best = {k: float("inf") for k in cases}
    for _ in range(ROUNDS):
        for k, fn in cases.items():
            best[k] = min(best[k], best_of(fn, reps))
    return best


rows = []
for (h, w) in [] if DEPTH_ONLY else [(1024, 1024), (2048, 2048), (721, 1440)]:
    npix = h * w
    out = np.empty((h, w), np.float32)
    stacks = {pad: stack(31, npix, pad) for pad in (0, 16, 1024)}  # none, one line, one 4 KiB page
    views = {pad: f.reshape(31, h, w) for pad, (_, f) in stacks.items()}
    med = measure({pad: (lambda st=st: rw.median(st, axis=0, out=out)) for pad, st in views.items()}, 3)
    clip = measure({pad: (lambda st=st: rw.sigma_clip_mean(st, axis=0, out=out)) for pad, st in views.items()}, 2)
    for pad, st in views.items():
        r = dict(frame=f"{h}x{w}", stride_bytes=(npix + pad) * 4, pad_bytes=pad * 4, path=path(st),
                 median_ns_px=med[pad] / npix * 1e9, sigclip_ns_px=clip[pad] / npix * 1e9)
        rows.append(r)
        print(r, flush=True)
    del stacks, views

depth = []
npix = 1024 * 1024
out = np.empty((1024, 1024), np.float32)
for n in (8, 16, 24, 31, 40, 47, 63):
    stacks = {"pow2": stack(n, npix, 0), "padded": stack(n, npix, 16)}
    views = {k: f.reshape(n, 1024, 1024) for k, (_, f) in stacks.items()}
    med = measure({k: (lambda st=st: rw.median(st, axis=0, out=out)) for k, st in views.items()}, 3)
    clip = measure({k: (lambda st=st: rw.sigma_clip_mean(st, axis=0, out=out)) for k, st in views.items()}, 2)
    r = dict(n=n)
    for k, st in views.items():
        r[f"{k}_path"] = path(st)
        r[f"{k}_median"] = med[k] / npix * 1e9
        r[f"{k}_sigclip"] = clip[k] / npix * 1e9
    depth.append(r)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    del stacks, views
save(NAME, dict(machine=machine(), unit="ns per output pixel", n=31, threads=1, rankwise=rw.__version__ if hasattr(rw, "__version__") else None,
               rows=rows, depth=depth))
