"""Head-to-head on an astro stack: median and sigma-clipped mean, one thread per tool.

    taskset -c 1 python bench/astro.py [n] [side]

Every tool gets the same float32 (n, side, side) array already in RAM, runs on one core,
and is timed as the best of 3 runs after a warm-up. Each result is checked: medians against
np.median, clipped means against astropy's sigma_clip mask (kept counts per pixel).
Siril is timed separately (bench/siril/time_siril.py) because it reads its own files.
"""

import sys

import common  # noqa: F401  (sets thread env vars first)
import numpy as np

import rankwise as rw
from common import astro_stack, best_of, machine, save, versions

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ONLY_RW = "--rankwise-only" in sys.argv  # re-measure rankwise rows, keep the saved competitor rows
n = int(args[0]) if args else 31
side = int(args[1]) if len(args) > 1 else 2048
st = astro_stack(n, side, side, seed=1)
px = side * side
print(f"stack {st.shape} float32, {st.nbytes / 2**30:.2f} GiB")

ref_med = np.median(st, axis=0)
from astropy.stats import sigma_clip  # noqa: E402

sc = sigma_clip(st, sigma=3, maxiters=5, cenfunc="median", stdfunc="std", axis=0)
ref_kept = (~sc.mask).sum(0)
ref_mean = sc.mean(axis=0).filled(np.nan).astype(np.float64)
del sc

rows = []


def record(tool, op, fn, reps=3, kept=None, family="competitor"):
    t = best_of(fn, reps)
    out = np.asarray(fn() if kept is None else fn()[0], np.float64)
    if op == "median":
        check = {"identical_to_np_median": bool(np.array_equal(out, ref_med))}
    else:
        rel = np.nanmax(np.abs(out - ref_mean) / np.abs(ref_mean))
        check = {"max_rel_diff_vs_astropy_mean": float(rel)}
        if kept is not None:
            check["kept_counts_identical_to_astropy"] = float(np.mean(kept() == ref_kept))
    rows.append(dict(tool=tool, op=op, family=family, seconds=t, ns_per_px=t / px * 1e9, **check))
    print(f"{tool:34s} {op:8s} {t:8.3f} s  {t / px * 1e9:7.1f} ns/px  {check}", flush=True)


out = np.empty((side, side), np.float32)
lo = np.empty((side, side), np.uint16)
hi = np.empty((side, side), np.uint16)
for isa in rw.available_isas():
    label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
    record(f"rankwise · {label}", "median", lambda: rw.median(st, axis=0, out=out, isa=isa), reps=7, family="rankwise")
    record(f"rankwise · {label}", "sigclip",
           lambda: (rw.sigma_clip_mean(st, axis=0, sigma=3, maxiters=5, out=out, low=lo, high=hi, isa=isa),),
           reps=7, kept=lambda: n - lo.astype(np.int64) - hi, family="rankwise")
    if isa == "avx512":
        record(f"rankwise · {label} · f32 bounds", "sigclip",
               lambda: (rw.sigma_clip_mean(st, axis=0, out=out, low=lo, high=hi, precision="f32", isa=isa),),
               reps=7, kept=lambda: n - lo.astype(np.int64) - hi, family="rankwise")

if ONLY_RW:
    import json  # noqa: E402

    from common import RESULTS  # noqa: E402

    old = json.loads((RESULTS / f"astro_{n}x{side}.json").read_text())
    old["rows"] = rows + [r for r in old["rows"] if r["family"] != "rankwise"]
    old["machine"] = machine()
    save(f"astro_{n}x{side}", old)
    sys.exit()

import bottleneck as bn  # noqa: E402
import imcombiners as ic  # noqa: E402
import numbagg  # noqa: E402
import reducers as rd  # noqa: E402

record("NumPy 2.5 median", "median", lambda: np.median(st, axis=0), reps=2)
record("Bottleneck 1.6", "median", lambda: bn.median(st, axis=0), reps=2)
record("numbagg 0.9 (nanmedian)", "median", lambda: numbagg.nanmedian(st, axis=0), reps=2)
record("reducers 0.4", "median", lambda: rd.median(st, axis=0), reps=2)
record("imcombiners 0.1", "median", lambda: ic.ndcombine(st, combine="median"), reps=2)
record("reducers 0.4", "sigclip", lambda: rd.sigclip_combine(st, combine="mean", sigma=(3, 3), maxiters=5), reps=2)
record("imcombiners 0.1", "sigclip",
       lambda: ic.ndcombine(st, combine="mean", reject="sigclip", sigma=(3, 3), maxiters=5), reps=2)
record("astropy 8.0 sigma_clip", "sigclip",
       lambda: sigma_clip(st, sigma=3, maxiters=5, cenfunc="median", stdfunc="std", axis=0).mean(axis=0), reps=1)
if px <= 2048 * 2048:
    from ccdproc import CCDData, Combiner

    def ccd(op):
        c = Combiner([CCDData(f, unit="adu") for f in st])
        if op == "median":
            return c.median_combine().data
        c.sigma_clipping(low_thresh=3, high_thresh=3, func=np.ma.median)  # one pass
        return c.average_combine().data

    record("ccdproc 2.5 Combiner", "median", lambda: ccd("median"), reps=1)
    record("ccdproc 2.5 Combiner (1 pass)", "sigclip", lambda: ccd("sigclip"), reps=1)

save(f"astro_{n}x{side}", dict(
    machine=machine(), n=n, side=side, dtype="float32", threads=1,
    method="best of 7 for rankwise (sub-second runs), 3 or 2 for slower tools, after a warm-up; taskset to one core",
    versions=versions("rankwise", "numpy", "bottleneck", "numbagg", "reducers", "imcombiners", "astropy", "ccdproc"),
    rows=rows))
