# Benchmarks

Every number here comes from the scripts in [`bench/`](../bench) and the JSON they wrote to [`bench/results/`](../bench/results). Regenerate this page with `python bench/report.py` and the charts with `python bench/figures.py`.

## Method

* **Machine:** Intel(R) Xeon(R) Processor @ 2.10GHz, 4 vCPUs (cloud VM, KVM), ISAs available: avx512, avx2, scalar.
* **One core for every tool.** Scripts run under `taskset -c 1`, and thread pools are pinned with `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS` and friends set to 1 before any import. rankwise runs with `threads=1`.
* **Same input for everyone.** Each tool gets the same array, already in RAM. Siril reads its own FITS files (from the page cache) and is marked as such.
* **Timing:** best of N runs after a warm-up: 7 for rankwise's sub-second runs, which this VM's noise affects most, 3 for most tools, 2 or 1 for tools that take tens of seconds.
* **Every result is checked**, not just timed: medians against `np.median`, quantiles against `np.quantile`/`np.nanquantile`, sigma clipping against astropy's mask.
* **Like for like.** NaN-aware tools are compared with rankwise's NaN-aware functions; ccdproc's `Combiner.sigma_clipping` does a single pass and is labeled so.
* The machine is a shared cloud VM; repeated runs vary by about ±15%.

## Astro stack: 31 × 2048 × 2048 float32

<picture><source media="(prefers-color-scheme: dark)" srcset="img/bench_astro_dark.png"><img alt="Astro stacking benchmark" src="img/bench_astro_light.png"></picture>

| Tool | Operation | Seconds | ns / pixel | vs rankwise AVX-512 | Check |
|---|---|---|---|---|---|
| rankwise · AVX-512 · f32 bounds | sigma-clipped mean | 0.233 | 55.5 | 1.0× | astropy kept counts identical on 100% of pixels |
| rankwise · AVX-512 | sigma-clipped mean | 0.252 | 60.2 | 1.1× | astropy kept counts identical on 100% of pixels |
| rankwise · AVX2 | sigma-clipped mean | 0.364 | 86.9 | 1.6× | astropy kept counts identical on 100% of pixels |
| rankwise · scalar | sigma-clipped mean | 0.763 | 182.0 | 3.3× | astropy kept counts identical on 100% of pixels |
| reducers 0.4 | sigma-clipped mean | 3.5 | 833.4 | 15.0× | mean differs from astropy's by up to 1.9% |
| imcombiners 0.1 | sigma-clipped mean | 4.01 | 955.5 | 17.2× | mean differs from astropy's by up to 1.9% |
| astropy 8.0 sigma_clip | sigma-clipped mean | 4.82 | 1148.4 | 20.7× | mean differs from astropy's by up to 0% |
| ccdproc 2.5 Combiner (1 pass) | sigma-clipped mean | 9.54 | 2274.4 | 41.0× | mean differs from astropy's by up to 49% |
| rankwise · AVX-512 | median | 0.0852 | 20.3 | 1.0× | identical to np.median |
| rankwise · AVX2 | median | 0.143 | 34.1 | 1.7× | identical to np.median |
| rankwise · scalar | median | 0.349 | 83.2 | 4.1× | identical to np.median |
| reducers 0.4 | median | 1.55 | 369.5 | 18.2× | identical to np.median |
| imcombiners 0.1 | median | 1.63 | 389.2 | 19.2× | identical to np.median |
| Bottleneck 1.6 | median | 2.21 | 527.9 | 26.0× | identical to np.median |
| NumPy 2.5 median | median | 2.98 | 710.5 | 35.0× | identical to np.median |
| ccdproc 2.5 Combiner | median | 8.2 | 1954.8 | 96.3× | identical to np.median |
| numbagg 0.9 (nanmedian) | median | 12.6 | 3010.2 | 148.2× | identical to np.median |
| Siril 1.2.1 (reads FITS) | median | 2.72 | 647.4 | — | own I/O |
| Siril 1.2.1 (reads FITS) | sigma-clipped mean | 3.44 | 821.0 | — | own I/O |

reducers and imcombiners clip differently from astropy (they do not re-apply the final bounds to values rejected earlier), so their means differ from astropy's on some pixels. ccdproc runs one pass. The f32-bounds row is rankwise's opt-in float32 center/spread/bounds mode.

## Stack depth sweep: 1024 × 1024 float32

<picture><source media="(prefers-color-scheme: dark)" srcset="img/bench_sweep_dark.png"><img alt="Depth sweep" src="img/bench_sweep_light.png"></picture>

| Frames | rankwise median | rankwise σ-clip | reducers median | imcombiners median | reducers σ-clip | imcombiners σ-clip | NumPy median | median speedup | σ-clip speedup |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 3.1 | 19.2 | 92 | 83 | 187 | 187 | 139 | 26.5× | 9.7× |
| 15 | 5.2 | 27.3 | 205 | 198 | 438 | 471 | 287 | 38.2× | 16.1× |
| 31 | 18.3 | 57.0 | 380 | 357 | 839 | 945 | 753 | 19.5× | 14.7× |
| 63 | 44.8 | 130.4 | 638 | 674 | 1660 | 1905 | 1724 | 14.2× | 12.7× |
| 127 | 105.9 | 292.6 | 1195 | 1198 | 3714 | 3768 | 3392 | 11.3× | 12.7× |
| 255 | 277.5 | 687.7 | 2273 | 2314 | 8055 | 8626 | 6682 | 8.2× | 11.7× |
| 511 | 839.8 | 1772.5 | 4454 | 5001 | 17453 | 18738 | 13882 | 5.3× | 9.8× |

ns per output pixel, one core, AVX-512. Speedup is against the faster of reducers and imcombiners.

## Weather ensemble: p10 / p50 / p90 of 31 members, 721 × 1440

<picture><source media="(prefers-color-scheme: dark)" srcset="img/bench_ensemble_dark.png"><img alt="Ensemble quantile benchmark" src="img/bench_ensemble_light.png"></picture>

| Tool | Function | Seconds per lead time | ns / cell | Identical to NumPy |
|---|---|---|---|---|
| rankwise · AVX-512 | quantile | 0.016 | 15.4 | yes |
| rankwise · AVX2 | quantile | 0.025 | 24.1 | yes |
| rankwise · scalar | quantile | 0.102 | 97.9 | yes |
| NumPy 2.5 quantile | quantile | 1.33 | 1283.3 | yes |
| rankwise · AVX-512 | nanquantile | 0.0287 | 27.7 | yes |
| rankwise · AVX2 | nanquantile | 0.0622 | 59.9 | yes |
| rankwise · scalar | nanquantile | 0.116 | 111.6 | yes |
| fastnanquantile 0.0.2 | nanquantile | 0.882 | 850.0 | no |
| numbagg 0.9 nanquantile | nanquantile | 4.76 | 4582.0 | no |
| NumPy 2.5 nanquantile | nanquantile | 44.1 | 42481.5 | yes |

Synthetic GEFS-shaped data (heavy-tailed members in kelvin); the nanquantile rows add 2% NaN. The p10/p50/p90 network keeps 162 of 186 comparators. numbagg and fastnanquantile interpolate differently from NumPy, so their results differ in the last bits.

## Real data: NOAA GEFS 2 m temperature

https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr init 2026-11-01T00, temperature_2m, CONUS, leads 0-240 h (CC BY 4.0); array 31 × 81 × 121 × 257 (members × leads × lat × lon).

| Tool | Seconds | ns / cell | Identical to np.quantile |
|---|---|---|---|
| rankwise · AVX-512 | 0.032 | 12.7 | yes |
| rankwise · AVX2 | 0.0615 | 24.4 | yes |
| rankwise · portable | 0.192 | 76.1 | yes |
| fastnanquantile 0.0.2 | 2.3 | 913.2 | yes |
| NumPy 2.5 quantile | 2.62 | 1041.2 | yes |
| numbagg 0.9 nanquantile | 12 | 4769.5 | yes |

## Satellite composite: 90 uint16 scenes, 2048 × 2048, clouds as nodata

46% of values are cloud (fill value 0). rankwise reads the uint16 data with `nodata=0`; the baselines get float32 with NaN, converted once and not timed.

| Tool | Seconds | ns / pixel | Identical to np.nanmedian |
|---|---|---|---|
| rankwise · AVX-512 (uint16, nodata=0) | 0.237 | 56.4 | yes |
| rankwise · AVX2 (uint16, nodata=0) | 0.341 | 81.3 | yes |
| rankwise · portable (uint16, nodata=0) | 1.02 | 242.2 | yes |
| Bottleneck 1.6 nanmedian (float32) | 3.58 | 852.7 | yes |
| NumPy 2.5 nanmedian (float32) | 18.5 | 4419.0 | yes |
| numbagg 0.9 nanmedian (float32) | 19.3 | 4604.5 | yes |

## Power-of-two strides

A contiguous stack of 2^k-byte frames puts every frame's current cache line in the same set. 31 frames, rankwise AVX-512, ns per pixel:

| Frame | Stride | Median | Sigma clip |
|---|---|---|---|
| 1024x1024 | exactly 2^k | 18.1 | 56.8 |
| 1024x1024 | +64 B | 12.4 | 49.4 |
| 1024x1024 | +4096 B | 12.7 | 47.4 |
| 2048x2048 | exactly 2^k | 20.8 | 57.0 |
| 2048x2048 | +64 B | 14.9 | 53.4 |
| 2048x2048 | +4096 B | 14.2 | 49.1 |
| 721x1440 | not 2^k | 11.9 | 49.3 |
| 721x1440 | +64 B | 12.1 | 48.4 |
| 721x1440 | +4096 B | 11.3 | 49.1 |

