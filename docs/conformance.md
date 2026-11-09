# Conformance

"Identical" means the same dtype, the same shape and equal values under IEEE equality, with
NaN equal to NaN and ±0 equal. The suite in [`conformance/`](../conformance) checks this on
every ISA the CPU supports (forced one at a time), against pinned references:
NumPy 2.5.3, SciPy 1.18.1 and astropy 8.0.2 (`conformance/requirements.txt`).

```bash
pip install -r conformance/requirements.txt
maturin build --release -m rankwise-py/Cargo.toml -o dist && pip install --find-links dist rankwise
python -m pytest conformance -q          # 1,039 tests, about a minute
```

## What is checked

| rankwise | Reference | Bar | Status |
|---|---|---|---|
| `median`, `nanmedian` | `np.median`, `np.nanmedian` | identical | ✅ |
| `quantile`, `nanquantile`, `percentile`, `nanpercentile`, all 13 methods | the same NumPy call, same form of `q`, same `out` | identical | ✅ |
| `min`, `max`, `nanmin`, `nanmax`, `ptp` | NumPy | identical | ✅ |
| `argmin`, `argmax`, `nanargmin`, `nanargmax` | NumPy (first occurrence; first NaN wins) | identical; all-NaN gives −1 + warning (`strict=True` raises) | ✅ |
| `argmedian`, `argselect` | `np.argsort(a, axis, kind="stable")` at the rank(s) | identical | ✅ |
| `select_ranks` | `np.sort(a, axis)[ranks]` | identical | ✅ |
| `iqr`, `mad` | `scipy.stats.iqr`, `scipy.stats.median_abs_deviation` | identical (propagate and omit) | ✅ |
| `trimmed_mean` | `scipy.stats.trim_mean` | within 1 ulp of an exact float64 reference | ✅ |
| `sigma_clip_mean` (std and mad_std, median and mean center) | `astropy.stats.sigma_clip(..., axis=...)` | identical masks and kept counts; mean within 1 ulp of float64 | ✅ |
| Siril presets | `siril-cli` 1.2.1 | identical reject maps where definitions allow | see [presets](presets.md) |
| C ABI | the Python package on the same data | identical | ✅ |

The matrix: every stack depth 1–17 plus 31–33, 63–65, 127–129, 255, 256, 511 and 512; pixel
counts that leave partial vectors (1, 7, 15, 16, 17, 33, 1001); float32, float64 and uint16
(uint8 is widened); frame lists, memory-mapped files, every axis, negative, odd and
power-of-two strides, Fortran order; NaN densities from none to whole all-NaN slices, ±inf,
±0, heavy ties, subnormals and ±FLT_MAX; q at 0, 1, 0.5, k/(n−1), 1/3 and random, as a Python
float, a list, float64 and float32 arrays, `np.float64` and `np.float32`, with and without
`out=`; one and several threads; xarray `apply_ufunc` with dask. Hypothesis adds random
shapes, values and q. The Rust side adds exhaustive 0-1 principle tests for n ≤ 20 and oracle
tests on every ISA, including NEON (CI runs on aarch64; locally under qemu).

## NumPy behaviours rankwise reproduces

Matching NumPy bit for bit meant reproducing some behaviours that are easy to miss:

* **How `q` is passed changes the arithmetic.** `type(q) in (int, float)` is a "weak" scalar:
  float32 data interpolates in float32, with `t` rounded to float32 and `(1 − t)` formed in
  float64 then rounded. A list or float64 array interpolates in float64. A float32 array
  computes even the virtual index in float32. A Python `int` with `method="linear"` takes an
  order statistic and keeps the input dtype (so `np.quantile(u16, 1)` is uint16).
* **`out=` changes the arithmetic too.** NumPy's `_lerp` runs the `t >= 0.5` branch as
  `subtract(..., dtype=out.dtype)`, so with a float32 `out` that subtraction happens in
  float32 even when the rest ran in float64. (The handoff note said "float64 math, rounded
  once"; that holds only for the `t < 0.5` branch.)
* **`np.nanquantile` does not do that**: it builds its result with `apply_along_axis`, so
  `out=` is a single cast at the end, and the result dtype is the *first slice's*: if that
  slice is all-NaN, an array-`q` float32 call returns float32, not float64.
* **`np.nanmedian` on 2-D and larger input averages the middle value with itself** for odd
  counts (it goes through `np.ma.median`): `(x + x) / 2`, which is `inf` for |x| > FLT_MAX/2.
  On 1-D input it does not. rankwise does the same in both cases.
* **Clamped quantile indices keep NumPy's weight.** For `q = 1` NumPy's weight is `n`, not 0;
  with an infinite maximum `inf − inf` makes the result NaN, as in NumPy.
* **`scipy.stats.iqr` uses `scipy.stats.quantile`, not NumPy's**: `(1 − g)·y[j] + g·y[j+1]`
  in the data's float dtype, and its `scale="normal"` constant is 1.3489795003921636.

## astropy behaviours rankwise reproduces

* Center, spread and bounds in float64 (data converted), population std, `0.5 * (upper +
  lower)` medians, keep `bound_lo <= x <= bound_hi`.
* Passes are cumulative, and stop when nothing changes, when nothing would be kept, or after
  `maxiters` (`maxiters=0` still runs one pass, as astropy's C loop does).
* The final mask applies the *last* bounds to every finite value, so a value rejected early
  can come back. NaN and ±inf are masked before clipping.
* The one difference: astropy sums in the order its partial sort leaves the buffer; rankwise
  sums in sorted order. Sums of float32 data in float64 are exact over the dynamic range of
  real images, so the means agree; in principle a value within an ulp of a bound could be
  decided differently. The suite has not found one.

## Known differences

* **Signed zeros**: equal under `==`; the sign bit can differ from NumPy and between ISAs.
* **`median(..., out=float64)` on float32 data**: NumPy averages in float64; rankwise averages
  in float32 and casts. Pass a float32 `out` (or none) for identical results.
* **`trimmed_mean`** is more accurate than SciPy's float32 summation (it sums in float64), so
  it agrees with SciPy to the float32 rounding of SciPy's own error, not bit for bit.
* **`nanargmin`/`nanargmax` on all-NaN slices** return −1 and warn; NumPy raises
  (`strict=True` restores that).
