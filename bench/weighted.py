"""Weighted quantiles (p10/p50/p90, method="inverted_cdf") against NumPy's ``weights=``, one thread.

    taskset -c 1 python bench/weighted.py

Two cases: one weight per frame (31 frames of 512×512, e.g. per-frame quality weights), and a
weight map with one weight per value (31 × 256 × 256, e.g. inverse-variance maps). NumPy's
weighted path sorts with argsort and then calls searchsorted pixel by pixel from Python.
"""

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, machine, save, versions

QS = [0.1, 0.5, 0.9]
rng = np.random.default_rng(3)
cases = {
    "frames": (31, 512),
    "map": (31, 256),
}
rows = []
for case, (n, side) in cases.items():
    x = (rng.normal(1000, 30, (n, side, side))).astype(np.float32)
    w = rng.random(n) if case == "frames" else rng.random((n, side, side))
    px = side * side
    ref = np.quantile(x, QS, axis=0, method="inverted_cdf", weights=w)

    def record(tool, fn, reps, family="competitor"):
        t = best_of(fn, reps, warmup=reps > 1)
        out = fn() if reps > 1 else ref
        ident = bool(np.array_equal(np.asarray(out), ref))
        rows.append(dict(case=case, n=n, side=side, tool=tool, family=family, seconds=t, ns_per_px=t / px * 1e9,
                         identical_to_numpy=ident))
        print(f"{case:6s} {tool:28s} {t:8.3f} s  {t / px * 1e9:9.1f} ns/px  identical={ident}", flush=True)

    for isa in rw.available_isas():
        label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
        record(f"rankwise · {label}", lambda: rw.quantile(x, QS, axis=0, method="inverted_cdf", weights=w, isa=isa), 5,
               family="rankwise")
    record("NumPy 2.5 (weights=)", lambda: np.quantile(x, QS, axis=0, method="inverted_cdf", weights=w), 1)
    t = best_of(lambda: rw.quantile(x, QS, axis=0, method="inverted_cdf"), 5)
    rows.append(dict(case=case, n=n, side=side, tool="rankwise unweighted", family="reference", seconds=t,
                     ns_per_px=t / px * 1e9, identical_to_numpy=None))
    print(f"{case:6s} {'rankwise unweighted':28s} {t:8.3f} s  {t / px * 1e9:9.1f} ns/px", flush=True)

save("weighted", dict(machine=machine(), versions=versions("numpy", "rankwise"), qs=QS, rows=rows))
