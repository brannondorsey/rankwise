//! What to compute: operations, NaN policy and rejection settings.

use crate::{Error, Result};

/// Element types rankwise reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    U16,
    I32,
    I64,
    /// One byte per element, 0 or 1 (NumPy `bool`).
    Bool,
    U8,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::U16 => 2,
            DType::Bool | DType::U8 => 1,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::U16 => "uint16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::Bool => "bool",
            DType::U8 => "uint8",
        }
    }
    pub fn from_name(s: &str) -> Option<DType> {
        Some(match s {
            "float32" | "f4" | "<f4" => DType::F32,
            "float64" | "f8" | "<f8" => DType::F64,
            "uint16" | "u2" | "<u2" => DType::U16,
            "int32" | "i4" | "<i4" => DType::I32,
            "int64" | "i8" | "<i8" => DType::I64,
            "bool" | "b1" | "|b1" => DType::Bool,
            "uint8" | "u1" | "|u1" => DType::U8,
            _ => return None,
        })
    }
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
}

/// What to do with NaN (floats) or `nodata` (integers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NanPolicy {
    /// A slice containing NaN produces NaN, like `np.median`.
    #[default]
    Propagate,
    /// NaN is ignored, like `np.nanmedian`. An all-NaN slice produces NaN.
    Omit,
}

/// NumPy's 13 quantile methods (Hyndman & Fan plus NumPy's legacy discontinuous ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QuantileMethod {
    InvertedCdf,
    AveragedInvertedCdf,
    ClosestObservation,
    InterpolatedInvertedCdf,
    Hazen,
    Weibull,
    #[default]
    Linear,
    MedianUnbiased,
    NormalUnbiased,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}

impl QuantileMethod {
    pub const ALL: [QuantileMethod; 13] = [
        QuantileMethod::InvertedCdf,
        QuantileMethod::AveragedInvertedCdf,
        QuantileMethod::ClosestObservation,
        QuantileMethod::InterpolatedInvertedCdf,
        QuantileMethod::Hazen,
        QuantileMethod::Weibull,
        QuantileMethod::Linear,
        QuantileMethod::MedianUnbiased,
        QuantileMethod::NormalUnbiased,
        QuantileMethod::Lower,
        QuantileMethod::Higher,
        QuantileMethod::Midpoint,
        QuantileMethod::Nearest,
    ];
    pub fn name(self) -> &'static str {
        match self {
            QuantileMethod::InvertedCdf => "inverted_cdf",
            QuantileMethod::AveragedInvertedCdf => "averaged_inverted_cdf",
            QuantileMethod::ClosestObservation => "closest_observation",
            QuantileMethod::InterpolatedInvertedCdf => "interpolated_inverted_cdf",
            QuantileMethod::Hazen => "hazen",
            QuantileMethod::Weibull => "weibull",
            QuantileMethod::Linear => "linear",
            QuantileMethod::MedianUnbiased => "median_unbiased",
            QuantileMethod::NormalUnbiased => "normal_unbiased",
            QuantileMethod::Lower => "lower",
            QuantileMethod::Higher => "higher",
            QuantileMethod::Midpoint => "midpoint",
            QuantileMethod::Nearest => "nearest",
        }
    }
    pub fn from_name(s: &str) -> Result<QuantileMethod> {
        QuantileMethod::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| Error::Param(format!("{s:?} is not a valid quantile method")))
    }
    /// Methods that pick an order statistic instead of interpolating.
    pub fn is_discrete(self) -> bool {
        matches!(
            self,
            QuantileMethod::InvertedCdf
                | QuantileMethod::ClosestObservation
                | QuantileMethod::Lower
                | QuantileMethod::Higher
                | QuantileMethod::Nearest
        )
    }
}

/// How `q` reached NumPy. NumPy's arithmetic, and therefore its last bit, depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QForm {
    /// A Python `float`: "weak" promotion, so f32 data interpolates in f32.
    Weak,
    /// A Python `int` (0 or 1).
    WeakInt,
    /// A float64 array, a list of floats, or `np.float64`: interpolation in f64.
    #[default]
    F64,
    /// A float32 array or `np.float32`: index and interpolation in f32.
    F32,
}

/// Which convention a `nan*` quantile follows when writing into `out`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OutRounding {
    /// `np.quantile(..., out=o)`: the `t >= 0.5` branch subtracts in `o`'s dtype.
    #[default]
    NumpyQuantile,
    /// Compute in the natural dtype and round once (`np.nanquantile`'s `apply_along_axis`).
    CastOnce,
}

/// Quantile settings.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileSpec {
    pub q: Vec<f64>,
    pub method: QuantileMethod,
    pub form: QForm,
    /// dtype of a caller-supplied `out` (it changes NumPy's arithmetic), else `None`.
    pub out_dtype: Option<DType>,
    pub out_rounding: OutRounding,
}

impl QuantileSpec {
    pub fn new(q: Vec<f64>) -> QuantileSpec {
        QuantileSpec {
            q,
            method: QuantileMethod::Linear,
            form: QForm::F64,
            out_dtype: None,
            out_rounding: OutRounding::NumpyQuantile,
        }
    }
}

/// Center estimator for sigma clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Center {
    #[default]
    Median,
    Mean,
}

/// Spread estimator for sigma clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Spread {
    /// Population standard deviation (`ddof = 0`), as `np.std`.
    #[default]
    Std,
    /// 1.482602218505602 × median absolute deviation, as `astropy.stats.mad_std`.
    MadStd,
}

/// Whose definition of an algorithm to follow, where tools disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Convention {
    /// `astropy.stats.sigma_clip`: population std, keep `c - low·s <= x <= c + high·s`,
    /// cumulative passes, then the last bounds re-applied to every value.
    #[default]
    Astropy,
    /// Siril 1.2: sample std (n - 1), reject `c - x > low·s` / `x - c > high·s`, cumulative
    /// passes until nothing changes. Verified black-box against Siril 1.2.1.
    Siril,
    /// PixInsight PCL as documented (winsorized only; not yet validated against PixInsight).
    PixInsight,
}

/// The rejection algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rejection {
    /// Keep everything.
    None,
    /// Iterative k-sigma clipping: reject values more than `low`/`high` spreads from the
    /// center, recompute on the survivors, stop when nothing changes or after `max_iters`
    /// (`None` = until convergence). `convention` picks astropy's or Siril's details.
    Sigma { low: f64, high: f64, max_iters: Option<u32>, center: Center, spread: Spread, convention: Convention },
    /// Siril's median sigma clipping: rejected values are replaced by the median (the stack
    /// keeps its size), median and sample std are recomputed, until nothing new is rejected.
    MedianSigma { low: f64, high: f64 },
    /// Winsorized sigma clipping (Huber): the spread comes from the stack with values beyond
    /// ±1.5 s clamped, iterated to convergence (×1.134); values beyond `low`/`high` robust
    /// sigmas from the center are rejected, and everything repeats on the survivors.
    Winsorized {
        low: f64,
        high: f64,
        max_iters: Option<u32>,
        /// First-pass winsorization cutoff in sigmas (PixInsight); `0` disables it.
        cutoff: f64,
        init: WinsorInit,
        /// `Siril`: center stays the median, sample std. `PixInsight`: center becomes the
        /// winsorized mean, Sn start, ratio test `(c - x)/s > k`.
        convention: Convention,
    },
    /// One-pass percentile clipping (Siril): reject `x` when `(c - x)/c > low` or
    /// `(x - c)/c > high`, with `c` the median. Verified against Siril 1.2.1.
    Percentile { low: f64, high: f64 },
}

/// Initial spread estimate for winsorized clipping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WinsorInit {
    /// Population standard deviation.
    #[default]
    Std,
    /// 1.1926 × Rousseeuw–Croux Sn, as documented for PixInsight's PCL.
    Sn,
}

/// How surviving values are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Combine {
    /// Mean of the kept values (weighted when weights are given).
    #[default]
    Mean,
    /// Median of the kept values.
    Median,
    /// Sum of the kept values.
    Sum,
}

/// Arithmetic precision for center, spread and bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// float64, as astropy's C implementation. Identical decisions to astropy.
    #[default]
    F64,
    /// float32 (f32 input only): faster, decisions may differ at exact ties with a bound.
    F32,
}

/// A complete rejection setup. Build with [`Clip::sigma`] and friends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    pub rejection: Rejection,
    pub combine: Combine,
    pub precision: Precision,
    /// Treat ±inf as missing as well as NaN (astropy does).
    pub mask_infinite: bool,
}

impl Clip {
    /// astropy-compatible sigma clipping: median center, population std, both sides.
    pub fn sigma(low: f64, high: f64) -> Clip {
        Clip {
            rejection: Rejection::Sigma {
                low,
                high,
                max_iters: Some(5),
                center: Center::Median,
                spread: Spread::Std,
                convention: Convention::Astropy,
            },
            combine: Combine::Mean,
            precision: Precision::F64,
            mask_infinite: true,
        }
    }
    /// MAD clipping: sigma clipping with `mad_std` as the spread.
    pub fn mad(low: f64, high: f64) -> Clip {
        let mut c = Clip::sigma(low, high);
        if let Rejection::Sigma { spread, .. } = &mut c.rejection {
            *spread = Spread::MadStd;
        }
        c
    }
    /// Siril's median sigma clipping: rejected values are replaced by the median.
    pub fn median_sigma(low: f64, high: f64) -> Clip {
        Clip { rejection: Rejection::MedianSigma { low, high }, ..Clip::sigma(low, high) }
    }
    /// Winsorized sigma clipping (Siril's definition: median center, sample std start).
    pub fn winsorized(low: f64, high: f64) -> Clip {
        Clip {
            rejection: Rejection::Winsorized {
                low,
                high,
                max_iters: None,
                cutoff: 0.0,
                init: WinsorInit::Std,
                convention: Convention::Siril,
            },
            combine: Combine::Mean,
            precision: Precision::F64,
            mask_infinite: true,
        }
    }
    /// Percentile clipping.
    pub fn percentile(low: f64, high: f64) -> Clip {
        Clip {
            rejection: Rejection::Percentile { low, high },
            combine: Combine::Mean,
            precision: Precision::F64,
            mask_infinite: true,
        }
    }
    /// No rejection: a plain (weighted, normalized) combine of the finite values.
    pub fn none() -> Clip {
        Clip { rejection: Rejection::None, combine: Combine::Mean, precision: Precision::F64, mask_infinite: true }
    }
    /// Cap the number of rejection passes (`None` = until nothing changes).
    pub fn max_iters(mut self, iters: impl Into<Option<u32>>) -> Clip {
        let it = iters.into();
        match &mut self.rejection {
            Rejection::Sigma { max_iters, .. } | Rejection::Winsorized { max_iters, .. } => *max_iters = it,
            _ => {}
        }
        self
    }
    pub fn center(mut self, c: Center) -> Clip {
        if let Rejection::Sigma { center, .. } = &mut self.rejection {
            *center = c;
        }
        self
    }
    pub fn spread(mut self, s: Spread) -> Clip {
        if let Rejection::Sigma { spread, .. } = &mut self.rejection {
            *spread = s;
        }
        self
    }
    pub fn precision(mut self, p: Precision) -> Clip {
        self.precision = p;
        self
    }
    /// Combine survivors with their mean.
    pub fn mean(mut self) -> Clip {
        self.combine = Combine::Mean;
        self
    }
    /// Combine survivors with their median.
    pub fn median(mut self) -> Clip {
        self.combine = Combine::Median;
        self
    }
    /// Combine survivors with their sum.
    pub fn sum(mut self) -> Clip {
        self.combine = Combine::Sum;
        self
    }
    pub fn combine(mut self, c: Combine) -> Clip {
        self.combine = c;
        self
    }

    /// Follow another tool's definition (sigma and winsorized clipping).
    pub fn convention(mut self, c: Convention) -> Clip {
        match &mut self.rejection {
            Rejection::Sigma { convention, .. } | Rejection::Winsorized { convention, .. } => *convention = c,
            _ => {}
        }
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let bad = |s: &str| Err(Error::Param(s.to_string()));
        match self.rejection {
            Rejection::Sigma { low, high, .. }
            | Rejection::Winsorized { low, high, .. }
            | Rejection::MedianSigma { low, high } => {
                if !(low >= 0.0 && high >= 0.0) {
                    return bad("sigma bounds must be non-negative");
                }
            }
            Rejection::Percentile { low, high } => {
                if !(low >= 0.0 && high >= 0.0) {
                    return bad("percentile bounds must be non-negative");
                }
            }
            Rejection::None => {}
        }
        if let Rejection::Winsorized { cutoff, .. } = self.rejection {
            if !(cutoff == 0.0 || cutoff >= 1.0) {
                return bad("winsorization cutoff must be 0 (off) or >= 1");
            }
        }
        if let Rejection::Sigma { convention: Convention::Siril | Convention::PixInsight, center, spread, .. } =
            self.rejection
        {
            if center != Center::Median || spread != Spread::Std {
                return bad("the Siril convention is defined for median center and std spread");
            }
        }
        Ok(())
    }
}

/// Named presets that follow a specific tool's published definition.
pub mod presets {
    use super::*;

    /// `astropy.stats.sigma_clip(..., cenfunc="median", stdfunc="std", maxiters=5)` followed
    /// by a mean of the survivors.
    pub fn astropy(sigma_lower: f64, sigma_upper: f64, maxiters: Option<u32>) -> Clip {
        Clip::sigma(sigma_lower, sigma_upper).max_iters(maxiters)
    }

    /// Siril's "Sigma Clipping" (rejection type `s`): median center, sample std, until
    /// nothing changes. Identical reject maps to Siril 1.2.1 in black-box tests.
    pub fn siril_sigma(low: f64, high: f64) -> Clip {
        Clip::sigma(low, high).max_iters(None).convention(Convention::Siril)
    }
    /// Siril's "Median Sigma Clipping" (`m`). Identical reject maps to Siril 1.2.1.
    pub fn siril_median_sigma(low: f64, high: f64) -> Clip {
        Clip::median_sigma(low, high)
    }
    /// Siril's "Winsorized Sigma Clipping" (`w`). Matches Siril 1.2.1 on 99.9% of pixels in
    /// black-box tests; see docs/presets.md for the known difference.
    pub fn siril_winsorized(low: f64, high: f64) -> Clip {
        Clip::winsorized(low, high)
    }
    /// Siril's "Percentile Clipping" (`p`). Identical to Siril 1.2.1 in black-box tests.
    pub fn siril_percentile(low: f64, high: f64) -> Clip {
        Clip::percentile(low, high)
    }
    /// PixInsight's winsorized sigma clipping as documented in PCL: Sn-based start, 1.5σ
    /// Huber clamp with the 1.134 correction, winsorization cutoff 5 on the first pass.
    /// Not yet validated against PixInsight itself.
    pub fn pixinsight_winsorized(low: f64, high: f64) -> Clip {
        Clip {
            rejection: Rejection::Winsorized {
                low,
                high,
                max_iters: None,
                cutoff: 5.0,
                init: WinsorInit::Sn,
                convention: Convention::PixInsight,
            },
            combine: Combine::Mean,
            precision: Precision::F64,
            mask_infinite: true,
        }
    }
}

/// Which order statistic an arg-variant reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgRank {
    /// Lower middle: rank `(n - 1) / 2` of the valid values.
    Median,
    /// A fixed rank of `np.argsort(a, kind="stable")` (NaN last).
    Fixed(usize),
}

/// One reduction.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// `np.sort(a, axis)[ranks]`: one output per rank, input dtype, NaN sorted last.
    SelectRanks(Vec<usize>),
    /// `np.median` / `np.nanmedian`.
    Median,
    /// `np.quantile` / `np.nanquantile`, one output per q.
    Quantile(QuantileSpec),
    /// `np.min` / `np.nanmin`.
    Min,
    /// `np.max` / `np.nanmax`.
    Max,
    /// `np.ptp` (max − min) in the input dtype.
    Range,
    /// `np.argmin` / `np.nanargmin` (first occurrence; all-NaN under `Omit` gives −1).
    ArgMin,
    /// `np.argmax` / `np.nanargmax`.
    ArgMax,
    /// Which frame supplied an order statistic (`np.argsort(kind="stable")` at that rank).
    ArgRank(ArgRank),
    /// `scipy.stats.trim_mean(a, proportion)`.
    TrimMean(f64),
    /// `scipy.stats.iqr(a, rng=(lo, hi), scale=scale)`; `lo`/`hi` in percent.
    Iqr { lo: f64, hi: f64, scale: f64 },
    /// `scipy.stats.median_abs_deviation(a, scale=scale)`.
    Mad { scale: f64 },
    /// Outlier rejection then combine. Outputs: value, low count, high count.
    Clip(Clip),
}

impl Op {
    /// Outputs the engine writes for this op.
    pub fn n_outputs(&self) -> usize {
        match self {
            Op::SelectRanks(r) => r.len(),
            Op::Quantile(q) => q.q.len(),
            Op::Clip(_) => 3,
            _ => 1,
        }
    }
}
