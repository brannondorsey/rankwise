"""Depth sweep: rankwise against the two fastest competitors, 1024×1024 frames, one thread.

    taskset -c 1 python bench/sweep.py [--rankwise-only]

--rankwise-only re-measures the rankwise columns and keeps the saved competitor columns.
"""

import json
import sys

import common  # noqa: F401
import numpy as np

import imcombiners as ic
import reducers as rd
import rankwise as rw
from common import RESULTS, astro_stack, best_of, machine, save, versions

ONLY_RW = "--rankwise-only" in sys.argv
old = {r["n"]: r for r in json.loads((RESULTS / "sweep_1024.json").read_text())["rows"]} if ONLY_RW else {}
rows = []
for n in (8, 15, 31, 63, 127, 255, 511):
    st = astro_stack(n, 1024, 1024, seed=n)
    px = st[0].size
    out = np.empty(st.shape[1:], np.float32)
    reps = 7 if n <= 63 else 3
    r = dict(old.get(n, {}), n=n)
    r["rankwise_path"] = rw.describe(st, axis=0).split("path=")[1].split()[0]
    r["rankwise_median"] = best_of(lambda: rw.median(st, axis=0, out=out), reps) / px * 1e9
    r["rankwise_sigclip"] = best_of(lambda: rw.sigma_clip_mean(st, axis=0, out=out), reps) / px * 1e9
    if "avx2" in rw.available_isas():
        r["rankwise_avx2_median"] = best_of(lambda: rw.median(st, axis=0, out=out, isa="avx2"), reps) / px * 1e9
        r["rankwise_avx2_sigclip"] = best_of(lambda: rw.sigma_clip_mean(st, axis=0, out=out, isa="avx2"), reps) / px * 1e9
    if not ONLY_RW:
        r["reducers_median"] = best_of(lambda: rd.median(st, axis=0), 2) / px * 1e9
        r["reducers_sigclip"] = best_of(
            lambda: rd.sigclip_combine(st, combine="mean", sigma=(3, 3), maxiters=5), 2) / px * 1e9
        r["imcombiners_median"] = best_of(lambda: ic.ndcombine(st, combine="median"), 2) / px * 1e9
        r["imcombiners_sigclip"] = best_of(
            lambda: ic.ndcombine(st, combine="mean", reject="sigclip", sigma=(3, 3), maxiters=5), 2) / px * 1e9
        r["numpy_median"] = best_of(lambda: np.median(st, axis=0), 1) / px * 1e9
    assert np.array_equal(rw.median(st, axis=0), np.median(st, axis=0))
    rows.append(r)
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    del st
save("sweep_1024", dict(machine=machine(), unit="ns per output pixel", threads=1,
                        versions=versions("rankwise", "numpy", "reducers", "imcombiners"), rows=rows))
