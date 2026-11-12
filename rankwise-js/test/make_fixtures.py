"""Reference results for the JS conformance tests, computed by NumPy, SciPy and astropy.

    pip install -r conformance/requirements.txt
    python rankwise-js/test/make_fixtures.py        # writes rankwise-js/test/fixtures/

Each case stores its input and the reference outputs as raw little-endian arrays plus a
manifest entry naming the rankwise call that must reproduce them (`test/conformance.test.mjs`).
The stacks reuse the native conformance suite's value families (conformance/helpers.py).
"""

from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.stats
from astropy.stats import sigma_clip

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "conformance"))
from helpers import make  # noqa: E402

OUT = HERE / "fixtures"
METHODS = [
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
    "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest",
]
DT = {np.dtype(np.float32): "float32", np.dtype(np.float64): "float64", np.dtype(np.uint16): "uint16",
      np.dtype(np.uint8): "uint8", np.dtype(np.int64): "int32", np.dtype(np.int32): "int32",
      np.dtype(np.bool_): "uint8"}

cases = []
_files = 0


def save(a) -> dict:
    global _files
    a = np.asarray(a)
    if a.dtype == np.int64:
        a = a.astype(np.int32)  # rankwise's JS indices are Int32Array
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    _files += 1
    name = f"{_files:04d}.bin"
    np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tofile(OUT / name)
    return {"file": name, "dtype": DT[a.dtype], "shape": list(a.shape)}


def case(name, fn, x, expect, *, args=(), opts=None, frames=False, axis=0, compare="exact", qform=None):
    entry = {"name": name, "fn": fn, "args": list(args), "opts": dict(opts or {}), "frames": frames,
             "input": save(x), "expect": [save(e) for e in expect], "compare": compare}
    if not frames:
        entry["opts"]["axis"] = axis
    if qform:
        entry["qform"] = qform
    cases.append(entry)


def quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def depths():
    # unrolled (2..32 on simd128 up to the unroll limit), hybrid, interpreted, staged
    return [1, 2, 3, 5, 8, 15, 16, 17, 31, 33, 64, 100, 129, 512]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # ---- medians over every network shape, NaN families, three dtypes
    for n in depths():
        for kind in ["normal", "ties", "nan_one", "nan30", "inf", "zeros", "extreme"]:
            x = make(n, 203, np.float32, kind, seed=n)
            case(f"median f32 n={n} {kind}", "median", x, [np.median(x, axis=0)])
            if kind.startswith("nan") or kind == "inf":
                case(f"nanmedian f32 n={n} {kind}", "nanmedian", x, [quiet(np.nanmedian, x, axis=0)])
        x = make(n, 77, np.float64, "nan30", seed=n)
        case(f"nanmedian f64 n={n}", "nanmedian", x, [quiet(np.nanmedian, x, axis=0)])
        u = make(n, 151, np.uint16, "normal", seed=n)
        case(f"median u16 n={n}", "median", u, [np.median(u, axis=0)])

    x = make(9, 200, np.float32, "nan_all", seed=3)
    case("nanmedian all-NaN slices", "nanmedian", x, [quiet(np.nanmedian, x, axis=0)])

    # ---- layouts: frame lists, inner axes (strided path), 1-D, uint8, chunked streaming
    x = make(31, 40 * 37, np.float32, "normal", seed=11).reshape(31, 40, 37)
    case("median frames list", "median", x, [np.median(x, axis=0).ravel()], frames=True)
    case("median axis 1", "median", x, [np.median(x, axis=1)], axis=1)
    case("median axis -1", "median", x, [np.median(x, axis=2)], axis=-1)
    case("quantile axis 1", "quantile", x, list(np.quantile(x, [0.25, 0.75], axis=1)), args=[[0.25, 0.75]], axis=1)
    v = make(1, 301, np.float64, "normal", seed=12)[0]
    case("median 1-D", "median", v, [np.median(v)])
    b = np.random.default_rng(13).integers(0, 256, (21, 999)).astype(np.uint8)
    case("median uint8", "median", b, [np.median(b, axis=0)])
    case("max uint8", "max", b, [np.max(b, axis=0)])
    big = make(5, 300_007, np.float32, "nan30", seed=14)  # > 1 chunk, ragged last chunk
    case("nanmedian chunked", "nanmedian", big, [quiet(np.nanmedian, big, axis=0)])
    case("nanmedian chunked frames", "nanmedian", big, [quiet(np.nanmedian, big, axis=0)], frames=True)

    # ---- quantiles: every method, every way of passing q
    for n in [1, 2, 7, 31, 64]:
        x = make(n, 101, np.float32, "normal", seed=20 + n)
        xn = make(n, 101, np.float32, "nan30", seed=20 + n)
        for m in METHODS:
            qs = [0.0, 0.1, 0.5, 0.9, 1.0]
            case(f"quantile {m} n={n} list", "quantile", x, list(np.quantile(x, qs, axis=0, method=m)),
                 args=[qs], opts={"method": m})
            case(f"quantile {m} n={n} float", "quantile", x, [np.quantile(x, 0.3, axis=0, method=m)],
                 args=[0.3], opts={"method": m}, qform="number")
            case(f"quantile {m} n={n} float32", "quantile", x,
                 list(np.quantile(x, np.array([0.3, 0.7], np.float32), axis=0, method=m)),
                 args=[[0.3, 0.7]], opts={"method": m}, qform="f32")
            case(f"nanquantile {m} n={n}", "nanquantile", xn,
                 list(quiet(np.nanquantile, xn, [0.25, 0.75], axis=0, method=m)), args=[[0.25, 0.75]],
                 opts={"method": m})
        d = make(n, 101, np.float64, "normal", seed=40 + n)
        case(f"percentile f64 n={n}", "percentile", d, list(np.percentile(d, [5, 50, 95], axis=0)), args=[[5, 50, 95]])
        u = make(n, 101, np.uint16, "normal", seed=50 + n)
        case(f"quantile u16 n={n}", "quantile", u, list(np.quantile(u, [0.1, 0.9], axis=0)), args=[[0.1, 0.9]])
        case(f"nanpercentile f32 n={n}", "nanpercentile", xn,
             [quiet(np.nanpercentile, xn, 42.0, axis=0)], args=[42.0], qform="number")

    # ---- min / max / ptp / arg-variants / select
    for n in [1, 3, 17, 31, 65]:
        x = make(n, 150, np.float32, "nan_one", seed=60 + n)
        c = make(n, 150, np.float32, "ties", seed=60 + n)
        for fn in ["min", "max", "ptp", "argmin", "argmax"]:
            case(f"{fn} n={n}", fn, c, [getattr(np, fn)(c, axis=0)])
            case(f"{fn} NaN n={n}", fn, x, [getattr(np, fn)(x, axis=0)])
        for fn in ["nanmin", "nanmax", "nanargmin", "nanargmax"]:
            if n == 1 and fn.startswith("nanarg"):
                continue  # every slice is all-NaN: NumPy raises, rankwise gives -1
            case(f"{fn} n={n}", fn, x, [quiet(getattr(np, fn), x, axis=0)])
        srt = np.argsort(c, axis=0, kind="stable")
        case(f"argmedian n={n}", "argmedian", c, [srt[(n - 1) // 2]])
        ranks = sorted({0, n // 3, n - 1})
        case(f"selectRanks n={n}", "selectRanks", x, list(np.sort(x, axis=0)[ranks]), args=[ranks])
        case(f"argselect n={n}", "argselect", c, list(srt[ranks]), args=[ranks])

    # ---- SciPy
    for n in [4, 10, 31, 64]:
        x = make(n, 120, np.float32, "normal", seed=70 + n)
        d = x.astype(np.float64)
        case(f"trimmedMean n={n}", "trimmedMean", d, [scipy.stats.trim_mean(d, 0.1, axis=0)], args=[0.1])
        case(f"iqr n={n}", "iqr", x, [scipy.stats.iqr(x, axis=0)])
        case(f"iqr normal n={n}", "iqr", d, [scipy.stats.iqr(d, axis=0, rng=(10, 90), scale="normal")],
             opts={"rng": [10, 90], "scale": "normal"})
        case(f"mad n={n}", "mad", d, [scipy.stats.median_abs_deviation(d, axis=0)])
        case(f"mad normal n={n}", "mad", x, [scipy.stats.median_abs_deviation(x, axis=0, scale="normal")],
             opts={"scale": "normal"})

    # ---- uint16 nodata = NaN-omitting statistics on the float equivalent
    for n in [3, 31, 90]:
        u = make(n, 211, np.uint16, "normal", seed=80 + n)
        u[np.random.default_rng(n).random(u.shape) < 0.5] = 0
        f = np.where(u == 0, np.nan, u.astype(np.float64))
        case(f"median nodata n={n}", "median", u, [quiet(np.nanmedian, f, axis=0)], opts={"nodata": 0})

    # ---- sigma clipping against astropy: identical masks and counts, mean within 1 ulp
    rng = np.random.default_rng(90)
    for n in [1, 2, 5, 16, 31, 33, 64, 129]:
        for variant, kw in [("default", {}), ("2/2.5 inf iters", {"sigma_lower": 2.0, "sigma_upper": 2.5, "maxiters": None}),
                            ("mean center", {"cenfunc": "mean"}), ("mad_std", {"stdfunc": "mad_std"})]:
            x = rng.normal(1000, 30, (n, 257))
            hits = rng.random(x.shape) < 0.04
            x[hits] += rng.uniform(300, 30000, hits.sum())
            x = x.astype(np.float32)
            x[rng.random(x.shape) < 0.05] = np.nan
            akw = {"sigma_lower": 3.0, "sigma_upper": 3.0, "maxiters": 5, "cenfunc": "median", "stdfunc": "std", **kw}
            sc = quiet(sigma_clip, x, axis=0, **akw)
            mask = np.ma.getmaskarray(sc)
            kept = np.where(mask, 0.0, x.astype(np.float64))
            cnt = (~mask).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                ref = (kept.sum(0) / cnt).astype(np.float32)
            rejected = (np.isfinite(x).sum(0) - cnt).astype(np.uint16)
            opts = {"sigmaLower": akw["sigma_lower"], "sigmaUpper": akw["sigma_upper"], "maxiters": akw["maxiters"],
                    "cenfunc": akw["cenfunc"], "stdfunc": akw["stdfunc"], "mask": True, "returnCounts": True}
            case(f"sigmaClipMean n={n} {variant}", "sigmaClipMean", x, [ref, mask, rejected], opts=opts,
                 compare="clip")

    (OUT / "manifest.json").write_text(json.dumps({"numpy": np.__version__, "scipy": scipy.__version__,
                                                   "cases": cases}, indent=1))
    print(f"{len(cases)} cases in {OUT}")


if __name__ == "__main__":
    main()
