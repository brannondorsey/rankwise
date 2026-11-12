# Changelog

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
* **JavaScript.** The `rankwise` npm package for Node, Bun, Deno and browsers: the Python
  package's API on TypedArrays, with two backends behind it. Natively, a Node-API addon
  (napi-rs) works on the caller's arrays in place, with AVX-512/AVX2/NEON and `threads`. It
  ships as one optional dependency per platform with `os`/`cpu`/`libc`, so package managers
  install only the matching one, and every region a call touches is bounds-checked.
  Everywhere else, WebAssembly: a simd128 backend in the core (x86's `minps` semantics on every
  host) and a portable build. The Node entry loads the addon and falls back to WebAssembly,
  saying why; browsers resolve a WebAssembly-only entry. `rw.backend`, `RANKWISE_BACKEND` and
  `init({ backend })` expose and override the choice. Identical to NumPy, SciPy and astropy on
  every backend in Node, Bun, Deno and Chromium; the install flow is tested against a local npm
  registry. Examples for Node, Bun, worker threads and the browser.
* **Verification.** 1,057-test conformance suite against NumPy 2.5.3, SciPy 1.18.1 and
  astropy 8.0.2 on every ISA; exhaustive 0-1 principle tests; black-box checks against
  Siril 1.2.1; real-data checks on CCD calibration frames against astropy, ccdproc and Siril,
  real Landsat scenes and real NOAA GEFS forecasts.

Differences from the prototype: NaN-safe everywhere (no panics, no `unwrap` on NaN),
quantiles exact at fractional ranks, sigma-clip bounds in float64, no OS threads spawned per
call, separate low/high reject counts, `rw_*` symbols.
