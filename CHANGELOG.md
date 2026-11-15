# Changelog

## Unreleased

* **Deep stacks: up to 65,535 frames** (was 512), on every ISA and for every operation. On
  AVX-512, stacks past 768 frames (where the buffer outgrows L1) sort with a register-blocked
  bitonic network: chunks of 32 sort in registers, then each merge stage's layers are split
  evenly into passes of at most four, and a pass of `r` layers loads groups of 2^r wires (8 or
  16), runs its layers on them in registers and stores them back, so the buffer is read once
  per three or four layers instead of twice per comparator. Rank pruning works per group;
  selecting only the minimum (or the maximum) keeps the hybrid network, whose pruned form is
  a tournament.
* Memory at depth: the network cache is capped at 64 MiB (a 65,535-frame network is 27 MB;
  rank sets varied in a loop grew it to gigabytes), quantile tables per count of valid
  values are built only up to 8 MiB and computed per lane beyond, and `Staging::Skew` uses a
  tile from 96 frames (its read-ahead ring grows as n²).
* PixInsight's Sn counts the distance between equal infinities as +inf (it could return
  −inf, a negative scale, when the stack held repeated infinities and `mask_infinite=False`).
* Staging tiles are no larger than the row they stage (inputs that collapse into many short
  rows each allocated and cleared a full tile).
* PixInsight's winsorized preset computes Sn in O(n log n) (was O(n²)).

## 0.1.0 (unreleased)

First release of the production library, rebuilt from the `lanestat-proto` prototype.

* **Core.** One generic kernel compiled for AVX-512, AVX2, NEON and a portable fallback,
  chosen at runtime (`RANKWISE_ISA` overrides). Rank-pruned Batcher networks,
  register-resident up to 32 frames, hybrid beyond; up to 512 frames.
* **Statistics.** `select_ranks`; median and the 13 NumPy quantile methods, with NumPy's
  arithmetic for every form of `q` and for `out=`; min/max/ptp; argmin/argmax/argmedian and
  `argselect`; trimmed mean, IQR and MAD with SciPy's definitions (`nan_policy` and
  `keepdims` included); NaN-aware variants for floats and `nodata` for uint16.
* **Rejection.** Sigma, MAD, median-sigma, winsorized and percentile clipping with low/high
  reject counts and an optional rejection map; astropy-identical decisions by default;
  Siril and PixInsight presets; per-frame normalization and weights in the same pass.
* **Inputs.** float32, float64 and uint16 (uint8 widened), any axis of a strided array or a
  list of frames, no whole-array copies; deep stacks and non-unit strides read through an
  in-cache tile; 8–47 frames that start in the same cache set (power-of-two frame strides)
  read skewed, a few cache lines apart per group of 8; full blocks start vector-aligned
  however the frames are aligned.
* **Bindings.** Rust crate (`ndarray` and `rayon` features), C ABI with a cbindgen header
  (enum-valued arguments are validated: out-of-range values return `RW_ERR_PARAM`), and an
  abi3 Python wheel that releases the GIL and runs on NumPy 1.24 and later (bit-identity is
  checked against the pinned NumPy 2.5.3).
* **Verification.** 1,175-test conformance suite against NumPy 2.5.3, SciPy 1.18.1 and
  astropy 8.0.2 on every ISA; exhaustive 0-1 principle tests; black-box checks against
  Siril 1.2.1; real-data checks on CCD calibration frames against astropy, ccdproc and Siril,
  real Landsat scenes and real NOAA GEFS forecasts.

Differences from the prototype: NaN-safe everywhere (no panics, no `unwrap` on NaN),
quantiles exact at fractional ranks, sigma-clip bounds in float64, no thread pool built per
call (single-threaded calls start none; `threads=` runs on rayon's global pool in the Python
wheel, capped at its size, scoped threads in Rust without `rayon` and in C), separate low/high
reject counts, `rw_*` symbols.
