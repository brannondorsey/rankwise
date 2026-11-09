"""End to end: read 31 FITS files, then stack them. One thread.

    taskset -c 1 python bench/e2e.py

The in-RAM benchmarks leave out reading the data, which can hide a faster kernel. Here every
Python pipeline reads the same 31 float32 FITS files of 2048×2048 (the astro benchmark's
stack, from the page cache after a warm-up) with astropy into one native-endian array, then
reduces it. Siril reads the same files itself (bench/siril/time_siril.py); its row is copied
from results/siril.json.
"""

import json

import common  # noqa: F401  (one-thread env first)
import numpy as np
from astropy.io import fits

import rankwise as rw
from common import RESULTS, astro_stack, best_of, machine, save, versions

ROOT = RESULTS.parents[1]
DIR = ROOT / "bench" / "data" / "fits_31x2048"
N, S = 31, 2048
paths = [DIR / f"light_{k + 1:05d}.fit" for k in range(N)]
if not all(p.exists() for p in paths):
    DIR.mkdir(parents=True, exist_ok=True)
    st = astro_stack(N, S, S, seed=1)
    for p, f in zip(paths, st):
        fits.writeto(p, f, overwrite=True)
    del st

stack = np.empty((N, S, S), np.float32)


def read():
    """FITS is big-endian: astropy memory-maps each file and the assignment byte-swaps it."""
    for k, p in enumerate(paths):
        with fits.open(p, memmap=True) as h:
            stack[k] = h[0].data
    return stack


import reducers as rd  # noqa: E402

out = np.empty((S, S), np.float32)
pipelines = {
    "median": {
        "rankwise · AVX-512": lambda st: rw.median(st, axis=0, out=out),
        "reducers 0.4": lambda st: rd.median(st, axis=0),
    },
    "sigclip": {
        "rankwise · AVX-512": lambda st: rw.sigma_clip_mean(st, axis=0, sigma=(3, 3), maxiters=5, out=out),
        "reducers 0.4": lambda st: rd.sigclip_combine(st, combine="mean", sigma=(3, 3), maxiters=5),
    },
}
read()  # warm the page cache
t_read = best_of(read, 5)
print(f"read 31 FITS files: {t_read:.3f} s")
rows = []
for op, tools in pipelines.items():
    for tool, fn in tools.items():
        t_total = best_of(lambda: fn(read()), 3 if tool.startswith("rankwise") else 2)
        rows.append(dict(tool=tool, op=op, read_seconds=t_read, total_seconds=t_total))
        print(rows[-1], flush=True)
siril = RESULTS / "siril.json"
if siril.exists():
    for r in json.loads(siril.read_text())["rows"]:
        rows.append(dict(tool="Siril 1.2.1", op=r["op"], read_seconds=None, total_seconds=r["seconds"]))
save("e2e_astro", dict(machine=machine(), files=N, side=S, threads=1, reader="astropy.io.fits (memmap, byte-swapped)",
                       versions=versions("rankwise", "numpy", "astropy", "reducers"), rows=rows))
