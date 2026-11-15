# Rejection presets

Stacking tools agree on the names of their rejection algorithms but not on the details:
population or sample standard deviation, `x < c − kσ` or `c − x > kσ`, whether rejected
values come back, when to stop. rankwise implements each definition separately and says which
one you get.

| Call | Definition | Verified against |
|---|---|---|
| `rw.sigma_clip_mean(...)`, `clip_combine(method="sigma")` | astropy `sigma_clip`: median (or mean) center, population std (or `mad_std`), keep `c − kσ ≤ x ≤ c + kσ`, cumulative passes, last bounds re-applied to every value, NaN and ±inf masked | astropy 8.0.2: identical masks and kept counts on the whole conformance matrix |
| `clip_combine(method="mad")` | the same with `stdfunc="mad_std"` (1.482602218505602 × MAD) | astropy 8.0.2: identical; ccdproc 2.5.1 `Combiner.sigma_clipping` with `maxiters=1`: identical masks and means on real frames |
| `clip_combine(method="sigma", preset="siril")` | Siril "Sigma Clipping" (`rej s`): median center, **sample** std (n − 1), reject `c − x > kσ` or `x − c > kσ`, cumulative passes until nothing changes | Siril 1.2.1: identical reject maps and stacked values (exact ties: see [real frames](#real-frames)) |
| `clip_combine(method="median_sigma")` | Siril "Median Sigma Clipping" (`rej m`): every pass replaces values beyond the bounds by the median (the stack keeps its size), recomputes median and sample std, until a pass rejects nothing; the result is the mean of the replaced stack (other combines and weights raise) | Siril 1.2.1: identical reject maps and stacked values (exact ties and 1-ulp values on real frames: see [below](#real-frames)) |
| `clip_combine(method="percentile", sigma=(lo, hi))` | Siril "Percentile Clipping" (`rej p`): one pass, reject `(c − x)/c > lo` or `(x − c)/c > hi` | Siril 1.2.1: identical reject maps and stacked values |
| `clip_combine(method="winsorized")` | Siril "Winsorized Sigma Clipping" (`rej w`): median center, sample-std start; Huber loop clamps to c ± 1.5σ and sets σ = 1.134 × std(clamped) until \|Δσ\| ≤ 5·10⁻⁴ σ; reject `c − x > kσ` / `x − c > kσ`; repeat while more than three values remain | Siril 1.2.1: identical on 100% of pixels at 31 frames, 99.9% at 15, 98% at 8, 95% at 5 (see below) |
| `clip_combine(method="winsorized", preset="pixinsight")` | PixInsight PCL `WinsorizedSigmaClippingRejection` as documented: 1.1926 × Sn start, 1.5σ Huber clamp with the 1.134 correction, winsorization cutoff 5 on the first pass (beyond it a value becomes the center), center updated to the winsorized mean, ratio tests `(c − x)/σ > k` | not validated: PixInsight is not available to test against |

The Siril numbers come from `bench/siril/compare_presets.py`, which writes synthetic float32
FITS frames, stacks them with `siril-cli -s` using `-rejmaps`, and compares per-pixel low and
high reject counts and the stacked image with rankwise. Results are in
[`bench/siril/presets.json`](../bench/siril/presets.json); the conformance suite reruns the
exact cases when `siril-cli` is installed.

| Frames | sigma | median sigma | percentile | winsorized |
|---|---|---|---|---|
| 5 | 3072 / 3072 | 3072 / 3072 | 3072 / 3072 | 2920 / 3072 |
| 8 | 3072 / 3072 | 3072 / 3072 | 3072 / 3072 | 3021 / 3072 |
| 15 | 3072 / 3072 | 3072 / 3072 | 3072 / 3072 | 3069 / 3072 |
| 31 | 3072 / 3072 | 3072 / 3072 | 3072 / 3072 | 3072 / 3072 |

Pixels with identical low and high reject counts. Where counts agree, the stacked values are
bit-identical to Siril's float32 output.

### Real frames

`bench/ccd_real.py` repeats the comparison on real calibration frames: 10 bias, 10 dark and
10 flat frames of 4109 × 4096 from an Apogee Aspen CG16M (the raw data of Matt Craig's CCD
reduction guide, Zenodo 3245296, CC BY 4.0), 16.8 million pixels per stack. Results are in
[`bench/results/ccd_real.json`](../bench/results/ccd_real.json).

| Siril method | `precision="f64"` (default) | `precision="f32"` |
|---|---|---|
| sigma | 1 pixel per stack differs; values bit-identical | identical reject maps; values within 2 ulp |
| median sigma | 1 pixel per stack differs; values within 1 ulp | identical reject maps and values |
| percentile | identical reject maps and values | — |
| winsorized | 99.81% of pixels | 99.81–99.84% |

The one pixel is an exact tie. Integer ADU values can put a value exactly k standard
deviations from the median (at pixel (1567, 782) of the bias stack: median 1051, sample
std exactly 22/3, a value of 1029, so c − x = 3σ = 22). Siril computes in float32, where
rounding puts c − x just above 3σ and the value is rejected; in float64 it is not.
`precision="f32"` runs rankwise's statistics in float32 too and then decides every tie the
way Siril does; its sigma path also sums the survivors in float32, which moves the stacked
value by up to 2 ulp. Winsorized clipping at 10 frames shows the small-stack gap described
below.

## Open differences

* **Siril winsorized, small stacks.** Where rankwise and Siril disagree, Siril stops
  rejecting earlier, always leaving more values than rankwise. The black-box tests ruled out
  a cap on Huber iterations, float32 arithmetic, strict versus non-strict convergence, and a
  fixed minimum count of 3–6. Siril itself recommends percentile clipping below about six
  frames, where the gap is largest.
* **Siril k-MAD clipping (`rej a`).** Not reproduced: Siril rejects more than astropy's
  `mad_std` clipping and fewer than clipping at raw MAD, and none of the tested variants
  (scale 1 or 1.4826, one pass or iterated, MAD recomputed or fixed) matched. `preset="siril"`
  with `method="mad"` raises rather than return something that only looks like Siril's.
* **Linear-fit clipping and generalized ESD** are v0.2 (they need per-lane regressions and
  running statistics that the value-based rescans in v0.1 do not cover).

## Normalization and weights

`scale=` and `offset=` apply `x * scale[k] + offset[k]` to frame `k` as it loads (two
roundings, as NumPy would compute them), before rejection. `weights=` turns the final mean
into `Σ w_k x_k / Σ w_k` over the kept values. The tools' own normalization estimators (IKSS,
noise-weighted, star-count weights) are inputs here: compute them however your pipeline does
and pass the per-frame numbers.
