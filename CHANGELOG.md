# Changelog

## Unreleased

* **Moving windows.** `rolling(a, window, func=...)` with xarray's `rolling` semantics (trailing
  or centered windows, `min_count` as `min_periods`, `step`): frames past either end and NaN are
  skipped, so results equal `np.nanmedian` (and friends) over the NaN-padded
  `sliding_window_view`, value and dtype. Median, quantiles (13 methods), min, max, ptp, ranks,
  IQR, MAD and every rejection method. Each lane keeps its window sorted and moves it with one
  branch-free delete-and-insert pass instead of a sort. Rust: `Rolling`, `reduce_rolling_raw`,
  `rolling_frames`; C: `rw_rolling_median_f32`, `rw_rolling_quantile_f32`.
* **Grouped reductions.** `groupby_reduce(a, labels, func=...)` runs any reduction on each
  group of frames (views, no copies) and returns `(result, groups)`; arg-variants report
  indices along the original axis. Rust: `reduce_groups_raw` and `group_frames` run all groups
  on one set of workers; C: `rw_group_median_f32`.
* **Fix:** `mad` of a slice whose median is infinite (so a deviation is `inf − inf`, NaN) could
  give inf; SciPy gives NaN.

## 0.1.0 (unreleased)

First release of the production library, rebuilt from the `lanestat-proto` prototype.

* **Core.** One generic kernel compiled for AVX-512, AVX2, NEON and a portable fallback,
  chosen at runtime (`RANKWISE_ISA` overrides). Rank-pruned Batcher networks,
  register-resident up to 32 frames, hybrid beyond; up to 512 frames.
* **Statistics.** `select_ranks`; median and the 13 NumPy quantile methods, with NumPy's
  arithmetic for every form of `q` and for `out=`; min/max/ptp; argmin/argmax/argmedian and
  `argselect`; trimmed mean, IQR and MAD with SciPy's definitions; NaN-aware variants for
  floats and `nodata` for uint16.
* **Rejection.** Sigma, MAD, median-sigma, winsorized and percentile clipping with low/high
  reject counts and an optional rejection map; astropy-identical decisions by default;
  Siril and PixInsight presets; per-frame normalization and weights in the same pass.
* **Inputs.** float32, float64 and uint16 (uint8 widened), any axis of a strided array or a
  list of frames, no whole-array copies; deep stacks and non-unit strides read through an
  in-cache tile; 8–47 frames that start in the same cache set (power-of-two frame strides)
  read skewed, a few cache lines apart per group of 8; full blocks start vector-aligned
  however the frames are aligned.
* **Bindings.** Rust crate (`ndarray` and `rayon` features), C ABI with a cbindgen header,
  and an abi3 Python wheel that releases the GIL.
* **Verification.** 1,092-test conformance suite against NumPy 2.5.3, SciPy 1.18.1 and
  astropy 8.0.2 on every ISA; exhaustive 0-1 principle tests; black-box checks against
  Siril 1.2.1; real-data checks on CCD calibration frames against astropy, ccdproc and Siril,
  real Landsat scenes and real NOAA GEFS forecasts.

Differences from the prototype: NaN-safe everywhere (no panics, no `unwrap` on NaN),
quantiles exact at fractional ranks, sigma-clip bounds in float64, no thread pool built per
call (single-threaded calls start none; `threads=` runs on rayon's global pool in the Python
wheel, capped at its size, scoped threads in Rust without `rayon` and in C), separate low/high
reject counts, `rw_*` symbols.
