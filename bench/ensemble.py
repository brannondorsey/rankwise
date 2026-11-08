"""Weather ensemble: p10/p50/p90 over 31 members on a 721×1440 grid, one thread.

    taskset -c 1 python bench/ensemble.py

Two comparisons, so each is like for like:
* plain quantile (GEFS fields have no NaN) against np.quantile;
* NaN-aware nanquantile against the NaN-aware tools (numbagg, fastnanquantile, NumPy).
"""

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, ensemble, machine, save, versions

QS = [0.1, 0.5, 0.9]
st = ensemble()
stn = ensemble(nan_frac=0.02, seed=8)
H, W = st.shape[1:]
cells = H * W
ref = np.quantile(st, QS, axis=0)
refn = np.nanquantile(stn, QS, axis=0)
rows = []


def record(tool, kind, fn, reps=3, family="competitor"):
    t = best_of(fn, reps)
    out = np.asarray(fn())
    want = ref if kind == "quantile" else refn
    ident = bool(np.array_equal(out.astype(want.dtype), want, equal_nan=True))
    rows.append(dict(tool=tool, kind=kind, family=family, seconds=t, ns_per_cell=t / cells * 1e9,
                     identical_to_numpy=ident))
    print(f"{tool:34s} {kind:12s} {t:7.3f} s  {t / cells * 1e9:7.1f} ns/cell  identical={ident}", flush=True)


out = np.empty((3, H, W), np.float64)
for isa in rw.available_isas():
    label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
    record(f"rankwise · {label}", "quantile", lambda: rw.quantile(st, QS, axis=0, out=out, isa=isa), family="rankwise")
    record(f"rankwise · {label}", "nanquantile", lambda: rw.nanquantile(stn, QS, axis=0, out=out, isa=isa),
           family="rankwise")

import fastnanquantile as fnq  # noqa: E402
import numbagg  # noqa: E402

record("NumPy 2.5 quantile", "quantile", lambda: np.quantile(st, QS, axis=0), reps=2)
record("NumPy 2.5 nanquantile", "nanquantile", lambda: np.nanquantile(stn, QS, axis=0), reps=1)
record("numbagg 0.9 nanquantile", "nanquantile", lambda: numbagg.nanquantile(stn, QS, axis=0), reps=2)
record("fastnanquantile 0.0.2", "nanquantile",
       lambda: np.stack([fnq.nanquantile(stn, q, axis=0) for q in QS]), reps=2)
save("ensemble_31x721x1440", dict(machine=machine(), members=31, grid=[H, W], quantiles=QS, threads=1,
                                  versions=versions("rankwise", "numpy", "numbagg", "fastnanquantile"),
                                  comparators={"full_sort": rw.network_size(31), "p10_p50_p90": rw.network_size(31, [3, 15, 27])},
                                  rows=rows))
