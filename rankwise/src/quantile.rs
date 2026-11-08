//! NumPy's quantile arithmetic, reproduced bit for bit.
//!
//! NumPy 2.x computes the virtual index in the dtype of `q` (float64 for Python floats, lists
//! and float64 arrays; float32 for float32 arrays), and interpolates in the promoted dtype of
//! the data and `q`, where a Python float is a "weak" scalar that adopts the data's dtype. The
//! tables here are computed once per call for every possible valid count `m`, so a NaN-aware
//! kernel can look up each lane's ranks and weight.

use crate::spec::{DType, OutRounding, QForm, QuantileMethod};

/// Where a quantile of `m` sorted values comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QIdx {
    /// An order statistic, returned unchanged (discrete methods).
    Take(usize),
    /// Interpolate between ranks `i0` and `i1` with weight `t` (held exactly; it is an f32
    /// value when the index dtype is f32).
    Lerp { i0: usize, i1: usize, t: f64 },
    /// No valid values.
    Empty,
}

trait Real:
    Copy + PartialOrd + core::ops::Add<Output = Self> + core::ops::Sub<Output = Self> + core::ops::Mul<Output = Self>
{
    fn c(x: f64) -> Self;
    fn f(self) -> f64;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn rint(self) -> Self;
}
impl Real for f64 {
    fn c(x: f64) -> Self {
        x
    }
    fn f(self) -> f64 {
        self
    }
    fn floor(self) -> Self {
        f64::floor(self)
    }
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    fn rint(self) -> Self {
        f64::round_ties_even(self)
    }
}
impl Real for f32 {
    fn c(x: f64) -> Self {
        x as f32
    }
    fn f(self) -> f64 {
        self as f64
    }
    fn floor(self) -> Self {
        f32::floor(self)
    }
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    fn rint(self) -> Self {
        f32::round_ties_even(self)
    }
}

fn alpha_beta(m: QuantileMethod) -> Option<(f64, f64)> {
    Some(match m {
        QuantileMethod::InterpolatedInvertedCdf => (0.0, 1.0),
        QuantileMethod::Hazen => (0.5, 0.5),
        QuantileMethod::Weibull => (0.0, 0.0),
        QuantileMethod::MedianUnbiased => (1.0 / 3.0, 1.0 / 3.0),
        QuantileMethod::NormalUnbiased => (3.0 / 8.0, 3.0 / 8.0),
        _ => return None,
    })
}

fn qidx_g<G: Real>(method: QuantileMethod, n: usize, q: G) -> QIdx {
    use QuantileMethod as M;
    if n == 0 {
        return QIdx::Empty;
    }
    let gn = G::c(n as f64);
    let gn1 = G::c((n - 1) as f64);
    let one = G::c(1.0);
    let clamp_take = |i: f64| -> QIdx {
        let i = if i < 0.0 { 0.0 } else { i };
        QIdx::Take((i as usize).min(n - 1))
    };
    match method {
        M::Lower => clamp_take((gn1 * q).floor().f()),
        M::Higher => clamp_take((gn1 * q).ceil().f()),
        M::Nearest => clamp_take((gn1 * q).rint().f()),
        M::InvertedCdf => {
            let idx = gn * q - one;
            let prev = idx.floor();
            let gamma = idx - prev;
            let r = if gamma == G::c(0.0) { prev } else { prev + one };
            clamp_take(r.f())
        }
        M::ClosestObservation => {
            let idx = gn * q - one - G::c(0.5);
            let prev = idx.floor();
            let gamma = idx - prev;
            let odd = prev.f().rem_euclid(2.0) == 1.0;
            let r = if gamma == G::c(0.0) && odd { prev } else { prev + one };
            clamp_take(r.f())
        }
        _ => {
            let v: G = match method {
                M::Linear => gn1 * q,
                M::AveragedInvertedCdf => gn * q - one,
                M::Midpoint => G::c(0.5) * ((gn1 * q).floor() + (gn1 * q).ceil()),
                _ => {
                    let (a, b) = alpha_beta(method).unwrap();
                    // n*q + (alpha + q*(1 - alpha - beta)) - 1, with (1 - alpha - beta) in Python f64
                    gn * q + (G::c(a) + q * G::c(1.0 - a - b)) - one
                }
            };
            // _get_indexes
            let (mut i0, mut i1): (i64, i64) = {
                let p = v.floor().f();
                (p as i64, p as i64 + 1)
            };
            if v.f() >= (n as f64) - 1.0 {
                i0 = -1;
                i1 = -1;
            }
            if v.f() < 0.0 {
                i0 = 0;
                i1 = 0;
            }
            // _get_gamma: (v - previous_indexes) promoted to f64, cast back to the index dtype.
            let mut gamma = G::c(v.f() - i0 as f64);
            match method {
                M::AveragedInvertedCdf => {
                    gamma = if gamma == G::c(0.0) { G::c(0.5) } else { G::c(1.0) };
                }
                M::Midpoint => {
                    gamma = if v.f().rem_euclid(1.0) == 0.0 { G::c(0.0) } else { G::c(0.5) };
                }
                _ => {}
            }
            let fix = |i: i64| -> usize {
                if i < 0 {
                    n - 1
                } else {
                    (i as usize).min(n - 1)
                }
            };
            QIdx::Lerp { i0: fix(i0), i1: fix(i1), t: gamma.f() }
        }
    }
}

/// Index and weight NumPy uses for quantile `q` of `n` values.
pub fn quantile_index(method: QuantileMethod, form: QForm, n: usize, q: f64) -> QIdx {
    match form {
        QForm::WeakInt if method == QuantileMethod::Linear => {
            if n == 0 {
                QIdx::Empty
            } else {
                QIdx::Take(((n - 1) as f64 * q) as usize)
            }
        }
        QForm::F32 => qidx_g::<f32>(method, n, q as f32),
        _ => qidx_g::<f64>(method, n, q),
    }
}

/// Interpolation arithmetic for one call: which dtype each step runs in.
#[derive(Clone, Copy, Debug)]
pub struct Lerp {
    /// Data dtype (`b - a` runs in it).
    pub data: DType,
    /// Promoted dtype of data and `q`.
    pub compute: DType,
    pub form: QForm,
    /// dtype the `t >= 0.5` subtraction runs in (a caller's `out`), if different.
    pub sub_dtype: DType,
}

impl Lerp {
    pub fn new(data: DType, form: QForm, out: Option<DType>, rounding: OutRounding) -> Lerp {
        let compute = match (data, form) {
            (DType::F64, _) => DType::F64,
            (DType::F32, QForm::Weak | QForm::WeakInt | QForm::F32) => DType::F32,
            (DType::F32, QForm::F64) => DType::F64,
            (_, QForm::F32) => DType::F32,
            _ => DType::F64,
        };
        let sub_dtype = match (out, rounding) {
            (Some(o), OutRounding::NumpyQuantile) => o,
            _ => compute,
        };
        Lerp { data, compute, form, sub_dtype }
    }

    /// The dtype NumPy returns when no `out` is given.
    pub fn natural_dtype(&self, discrete: bool) -> DType {
        if discrete {
            self.data
        } else {
            self.compute
        }
    }

    /// `_lerp(a, b, t)`. `a` and `b` are data values widened exactly to f64; the result is
    /// exactly representable in the dtype of the final step.
    #[inline]
    pub fn eval(&self, a: f64, b: f64, t: f64) -> f64 {
        let d = match self.data {
            DType::F32 => ((b as f32) - (a as f32)) as f64,
            _ => b - a,
        };
        let weak = matches!(self.form, QForm::Weak | QForm::WeakInt);
        if !(t >= 0.5) {
            match self.compute {
                DType::F32 => ((a as f32) + (d as f32) * (t as f32)) as f64,
                _ => a + d * t,
            }
        } else {
            let x = match self.compute {
                DType::F32 => {
                    let omt = if weak { (1.0 - t) as f32 } else { 1.0f32 - t as f32 };
                    ((d as f32) * omt) as f64
                }
                _ => d * (1.0 - t),
            };
            match self.sub_dtype {
                DType::F32 => ((b as f32) - (x as f32)) as f64,
                _ => b - x,
            }
        }
    }
}

/// `scipy.stats.quantile(..., method="linear")` as `scipy.stats.iqr` calls it: everything in
/// the data's floating dtype `F`, interpolating `(1 - g)·y[j] + g·y[j+1]`.
#[derive(Clone, Copy, Debug)]
pub struct ScipyLinear {
    pub f32: bool,
}

impl ScipyLinear {
    /// Returns `(j, jp1, g)` for probability `p` (already divided by 100) and count `n`.
    pub fn index(&self, p: f64, n: usize) -> (usize, usize, f64) {
        if self.f32 {
            let p = p as f32;
            let nf = n as f32;
            let m = 1.0f32 - p;
            let jg = p * nf + m;
            let jp1 = jg.floor();
            let j = jp1 - 1.0;
            let mut g = jg - jg.floor();
            if j < 0.0 {
                g = 0.0;
            }
            let clip = |x: f32| x.max(0.0).min(nf - 1.0) as usize;
            (clip(j), clip(jp1), g as f64)
        } else {
            let nf = n as f64;
            let m = 1.0 - p;
            let jg = p * nf + m;
            let jp1 = jg.floor();
            let j = jp1 - 1.0;
            let mut g = jg - jg.floor();
            if j < 0.0 {
                g = 0.0;
            }
            let clip = |x: f64| x.max(0.0).min(nf - 1.0) as usize;
            (clip(j), clip(jp1), g)
        }
    }

    #[inline]
    pub fn eval(&self, a: f64, b: f64, g: f64) -> f64 {
        if self.f32 {
            let g = g as f32;
            ((1.0f32 - g) * (a as f32) + g * (b as f32)) as f64
        } else {
            (1.0 - g) * a + g * b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_indices() {
        assert_eq!(quantile_index(QuantileMethod::Linear, QForm::F64, 5, 0.5), QIdx::Lerp { i0: 2, i1: 3, t: 0.0 });
        assert_eq!(quantile_index(QuantileMethod::Linear, QForm::F64, 5, 1.0), QIdx::Lerp { i0: 4, i1: 4, t: 5.0 });
        assert_eq!(quantile_index(QuantileMethod::Lower, QForm::F64, 5, 0.6), QIdx::Take(2));
        assert_eq!(quantile_index(QuantileMethod::Nearest, QForm::F64, 5, 0.625), QIdx::Take(2));
    }
}
