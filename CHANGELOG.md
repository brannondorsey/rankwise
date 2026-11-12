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
* **WebAssembly.** A simd128 backend (4 f32 lanes, x86's `minps` semantics on every host) and
  `rankwise-wasm`: simd128 and portable modules plus an ES module for Node, Bun, Deno and
  browsers with the Python package's API on TypedArrays, chunked streaming, zero-copy arrays,
  and TypeScript types. 10–14× faster than the fastest JavaScript on one thread in Node;
  identical to NumPy, SciPy and astropy on 5,239 checks in Node, Bun, Deno and Chromium. On
  wasm the per-block kernels are separate functions so engines optimize them quickly.
  Examples for Node, Bun, worker threads and the browser.
* **Verification.** 1,057-test conformance suite against NumPy 2.5.3, SciPy 1.18.1 and
  astropy 8.0.2 on every ISA; exhaustive 0-1 principle tests; black-box checks against
  Siril 1.2.1; real-data checks on CCD calibration frames against astropy, ccdproc and Siril,
  real Landsat scenes and real NOAA GEFS forecasts.

Differences from the prototype: NaN-safe everywhere (no panics, no `unwrap` on NaN),
quantiles exact at fractional ranks, sigma-clip bounds in float64, no OS threads spawned per
call, separate low/high reject counts, `rw_*` symbols.
