"""Moving-window median along time, and grouped (climatology) medians, one thread.

    taskset -c 1 python bench/rolling.py

Rolling: 1000 float32 frames of 256×256, trailing windows of 7, 31 and 101 frames
(``min_periods=1``), against bottleneck's ``move_median`` (what xarray's ``rolling`` uses when
bottleneck is installed), pandas and NumPy's ``sliding_window_view`` + ``nanmedian``.
Grouped: 30 "years" of 12 monthly frames of 256×256, the median of each calendar month,
against xarray's ``groupby`` (NumPy path) and a NumPy loop over the groups.
"""

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, machine, save, versions

rng = np.random.default_rng(5)
T, side = 1000, 256
x = rng.normal(100, 5, (T, side, side)).astype(np.float32)
x[rng.random(x.shape) < 0.01] = np.nan
px = side * side
rows = []


def record(case, tool, fn, ref, reps, family="competitor", per=T * px):
    t = best_of(fn, reps, warmup=reps > 1)
    out = np.asarray(fn()) if reps > 1 else ref
    ident = bool(np.array_equal(out, ref, equal_nan=True))
    rows.append(dict(case=case, tool=tool, family=family, seconds=t, ns_per_output=t / per * 1e9, identical=ident))
    print(f"{case:12s} {tool:30s} {t:8.3f} s  {t / per * 1e9:8.2f} ns/output  identical={ident}", flush=True)


def window_ref(w):
    xp = np.concatenate([np.full((w - 1, side, side), np.nan, np.float32), x])
    from numpy.lib.stride_tricks import sliding_window_view
    v = sliding_window_view(xp, w, axis=0)
    out = np.empty(x.shape, np.float32)
    for t0 in range(0, T, 50):  # in pieces: the window view is large
        out[t0:t0 + 50] = np.nanmedian(v[t0:t0 + 50], axis=-1)
    return out


for w in (7, 31, 101):
    case = f"rolling w={w}"
    ref = rw.rolling(x, w, func="median", min_count=1)
    for isa in rw.available_isas():
        label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
        record(case, f"rankwise · {label}", lambda: rw.rolling(x, w, func="median", min_count=1, isa=isa), ref, 3,
               family="rankwise")
    try:
        import bottleneck as bn
        record(case, f"bottleneck {bn.__version__} move_median",
               lambda: bn.move_median(x, w, min_count=1, axis=0), ref, 2)
    except ImportError:
        pass
    import pandas as pd
    flat = x.reshape(T, px)
    sub = 4096  # pandas is slow: time a slice of the pixels, scale up
    t = best_of(lambda: pd.DataFrame(flat[:, :sub]).rolling(w, min_periods=1).median().to_numpy(), 1, warmup=False)
    pv = pd.DataFrame(flat[:, :sub]).rolling(w, min_periods=1).median().to_numpy()
    rows.append(dict(case=case, tool=f"pandas {pd.__version__}", family="competitor", seconds=t * px / sub,
                     ns_per_output=t / (T * sub) * 1e9,
                     identical=bool(np.array_equal(pv, ref.reshape(T, px)[:, :sub], equal_nan=True))))
    print(f"{case:12s} {'pandas':30s} {t * px / sub:8.3f} s (scaled)  identical={rows[-1]['identical']}", flush=True)
    if w <= 31:
        record(case, "NumPy windows + nanmedian", lambda: window_ref(w), ref, 1)

# grouped: monthly climatology
years, months = 30, 12
g = rng.normal(280, 8, (years * months, side, side)).astype(np.float32)
labels = np.tile(np.arange(months), years)
gref, _ = rw.groupby_reduce(g, labels, func="median")
gper = months * px
for isa in rw.available_isas():
    label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
    record("groupby", f"rankwise · {label}", lambda: rw.groupby_reduce(g, labels, func="median", isa=isa)[0], gref, 3,
           family="rankwise", per=gper)
record("groupby", "NumPy (loop over months)",
       lambda: np.stack([np.median(g[labels == m], axis=0) for m in range(months)]), gref, 2, per=gper)
try:
    import xarray as xr
    da = xr.DataArray(g, dims=("time", "y", "x"), coords={"month": ("time", labels)})
    with xr.set_options(use_flox=False):
        record("groupby", f"xarray {xr.__version__} groupby",
               lambda: da.groupby("month").median("time").values, gref, 1, per=gper)
except ImportError:
    pass

save("rolling", dict(machine=machine(), versions=versions("numpy", "bottleneck", "pandas", "xarray", "rankwise"),
                     T=T, side=side, rows=rows))
