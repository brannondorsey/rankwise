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
| rankwise · AVX-512 · f32 bounds | sigma-clipped mean | 0.227 | 54.1 | 1.0× | astropy kept counts identical on 100% of pixels |
| rankwise · AVX-512 | sigma-clipped mean | 0.243 | 57.9 | 1.1× | astropy kept counts identical on 100% of pixels |
| rankwise · AVX2 | sigma-clipped mean | 0.333 | 79.3 | 1.5× | astropy kept counts identical on 100% of pixels |
| rankwise · scalar | sigma-clipped mean | 0.72 | 171.6 | 3.2× | astropy kept counts identical on 100% of pixels |
| reducers 0.4 | sigma-clipped mean | 3.5 | 833.4 | 15.4× | mean differs from astropy's by up to 1.9% |
| imcombiners 0.1 | sigma-clipped mean | 4.01 | 955.5 | 17.7× | mean differs from astropy's by up to 1.9% |
| astropy 8.0 sigma_clip | sigma-clipped mean | 4.82 | 1148.4 | 21.2× | mean differs from astropy's by up to 0% |
| ccdproc 2.5 Combiner (1 pass) | sigma-clipped mean | 9.54 | 2274.4 | 42.1× | mean differs from astropy's by up to 49% |
| rankwise · AVX-512 | median | 0.0751 | 17.9 | 1.0× | identical to np.median |
| rankwise · AVX2 | median | 0.109 | 26.0 | 1.5× | identical to np.median |
| rankwise · scalar | median | 0.295 | 70.4 | 3.9× | identical to np.median |
| reducers 0.4 | median | 1.55 | 369.5 | 20.6× | identical to np.median |
| imcombiners 0.1 | median | 1.63 | 389.2 | 21.7× | identical to np.median |
| Bottleneck 1.6 | median | 2.21 | 527.9 | 29.5× | identical to np.median |
| NumPy 2.5 median | median | 2.98 | 710.5 | 39.7× | identical to np.median |
| ccdproc 2.5 Combiner | median | 8.2 | 1954.8 | 109.2× | identical to np.median |
| numbagg 0.9 (nanmedian) | median | 12.6 | 3010.2 | 168.2× | identical to np.median |
| Siril 1.2.1 (reads FITS) | median | 2.72 | 647.4 | — | own I/O |
| Siril 1.2.1 (reads FITS) | sigma-clipped mean | 3.44 | 821.0 | — | own I/O |

reducers and imcombiners clip differently from astropy (they do not re-apply the final bounds to values rejected earlier), so their means differ from astropy's on some pixels. ccdproc runs one pass. The f32-bounds row is rankwise's opt-in float32 center/spread/bounds mode.

## Stack depth sweep: 1024 × 1024 float32

<picture><source media="(prefers-color-scheme: dark)" srcset="img/bench_sweep_dark.png"><img alt="Depth sweep" src="img/bench_sweep_light.png"></picture>

| Frames | rankwise median | rankwise σ-clip | reducers median | imcombiners median | reducers σ-clip | imcombiners σ-clip | NumPy median | median speedup | σ-clip speedup |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 2.8 | 19.8 | 92 | 83 | 187 | 187 | 139 | 29.9× | 9.4× |
| 15 | 5.0 | 26.7 | 205 | 198 | 438 | 471 | 287 | 39.6× | 16.4× |
| 31 | 14.7 | 54.8 | 380 | 357 | 839 | 945 | 753 | 24.2× | 15.3× |
| 63 | 45.4 | 134.7 | 638 | 674 | 1660 | 1905 | 1724 | 14.0× | 12.3× |
| 127 | 110.8 | 320.2 | 1195 | 1198 | 3714 | 3768 | 3392 | 10.8× | 11.6× |
| 255 | 361.1 | 685.6 | 2273 | 2314 | 8055 | 8626 | 6682 | 6.3× | 11.7× |
| 511 | 670.8 | 1983.2 | 4454 | 5001 | 17453 | 18738 | 13882 | 6.6× | 8.8× |

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

A contiguous stack of 2^k-byte frames puts every frame's current cache line in the same set; NumPy's 2 MiB pages keep that conflict in L2. rankwise reads 8–47 such frames skewed and 48 or more through its tile ([design](design.md#memory-layout-deep-stacks-and-the-power-of-two-stride)). rankwise AVX-512, ns per pixel, best of four interleaved rounds.

31 frames:

| Frame | Stride | Read path | Median | Sigma clip |
|---|---|---|---|---|
| 1024x1024 | exactly 2^k | skewed | 14.9 | 59.8 |
| 1024x1024 | +64 B | direct | 13.8 | 53.0 |
| 1024x1024 | +4096 B | direct | 13.5 | 51.9 |
| 2048x2048 | exactly 2^k | skewed | 18.2 | 57.7 |
| 2048x2048 | +64 B | direct | 15.5 | 50.1 |
| 2048x2048 | +4096 B | direct | 14.8 | 49.1 |
| 721x1440 | not 2^k | direct | 13.2 | 49.8 |
| 721x1440 | +64 B | direct | 12.2 | 49.5 |
| 721x1440 | +4096 B | direct | 13.2 | 48.6 |

1024 × 1024 frames by depth, exactly 2^k against +64 B:

| Frames | 2^k path | 2^k median | padded median | ratio | 2^k σ-clip | padded σ-clip | ratio |
|---|---|---|---|---|---|---|---|
| 8 | skewed | 3.1 | 3.4 | 0.91× | 19.6 | 20.4 | 0.96× |
| 16 | skewed | 6.0 | 5.6 | 1.06× | 28.2 | 28.3 | 1.00× |
| 24 | skewed | 11.4 | 10.0 | 1.14× | 44.1 | 40.3 | 1.09× |
| 31 | skewed | 15.2 | 13.5 | 1.12× | 57.6 | 51.4 | 1.12× |
| 40 | skewed | 25.8 | 21.3 | 1.21× | 75.5 | 71.8 | 1.05× |
| 47 | skewed | 31.3 | 25.0 | 1.25× | 94.6 | 89.5 | 1.06× |
| 63 | staged | 45.7 | 39.2 | 1.17× | 128.8 | 124.5 | 1.04× |

