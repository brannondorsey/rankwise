//! C ABI for rankwise.
//!
//! Every function takes `n` frame pointers of `npix` contiguous values each (one per frame,
//! no copies), writes into caller-owned outputs and returns an [`rw_status`]. Inputs are
//! validated before any kernel runs; no Rust panic ever crosses the boundary.

#![allow(non_camel_case_types, clippy::missing_safety_doc, clippy::too_many_arguments)]

use rankwise::{
    presets, reduce_raw, ArgRank, Center, Clip, Combine, DType, Error, NanPolicy, Normalize, Op, Options, Precision,
    QForm, QuantileMethod, QuantileSpec, RawMask, RawOut, RawStack, Spread,
};
use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Result of every call.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rw_status {
    RW_OK = 0,
    /// `n` is 0 or above `RW_MAX_FRAMES`.
    RW_ERR_DEPTH = 1,
    /// Sizes disagree.
    RW_ERR_SHAPE = 2,
    /// A parameter is out of range.
    RW_ERR_PARAM = 3,
    /// Unsupported dtype for this operation.
    RW_ERR_DTYPE = 4,
    /// The requested ISA cannot run on this CPU.
    RW_ERR_ISA = 5,
    /// A required pointer is NULL.
    RW_ERR_NULL = 6,
    /// Internal error (a bug): the call was aborted safely.
    RW_ERR_INTERNAL = 7,
}

/// Deepest stack v0.1 reduces.
// A literal so cbindgen can emit it (it cannot evaluate `rankwise::MAX_FRAMES`).
pub const RW_MAX_FRAMES: usize = 512;
const _: () = assert!(RW_MAX_FRAMES == rankwise::MAX_FRAMES);

/// Rejection algorithms.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rw_rejection {
    RW_REJECT_NONE = 0,
    /// astropy `sigma_clip` semantics.
    RW_REJECT_SIGMA = 1,
    /// Sigma clipping with 1.4826 × MAD as the spread.
    RW_REJECT_MAD = 2,
    /// Sigma clipping; rejected values replaced by the median (Siril).
    RW_REJECT_MEDIAN_SIGMA = 3,
    /// Huber-winsorized sigma clipping (Siril's definition).
    RW_REJECT_WINSORIZED = 4,
    /// One pass around the median; `low`/`high` are fractions of the median (Siril).
    RW_REJECT_PERCENTILE = 5,
    /// PixInsight's winsorized sigma clipping as documented in PCL.
    RW_REJECT_WINSORIZED_PIXINSIGHT = 6,
    /// Siril's sigma clipping (sample std, until nothing changes; `maxiters` ignored).
    RW_REJECT_SIGMA_SIRIL = 7,
}

/// How kept values are combined.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum rw_combine {
    RW_COMBINE_MEAN = 0,
    RW_COMBINE_MEDIAN = 1,
    RW_COMBINE_SUM = 2,
}

/// A rejection setup. Build one with `rw_clip_sigma` and friends, then adjust fields.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct rw_clip {
    /// An `rw_rejection` value. (Integer fields, not enums, so that an out-of-range value from
    /// C is reported as `RW_ERR_PARAM` instead of being undefined behaviour in Rust.)
    pub method: i32,
    pub low: f32,
    pub high: f32,
    /// Maximum passes; negative = until nothing changes.
    pub maxiters: i32,
    /// 0 = median center, 1 = mean center (sigma / MAD methods).
    pub center_mean: i32,
    /// 0 = float64 center/spread/bounds (astropy-identical), 1 = float32.
    pub fast_f32: i32,
    /// An `rw_combine` value (ignored by `RW_REJECT_MEDIAN_SIGMA`, which averages).
    pub combine: i32,
}

/// Optional per-frame normalization (`x * scale[k] + offset[k]`) and weights. Any pointer may
/// be NULL; non-NULL arrays hold `n` values.
#[repr(C)]
pub struct rw_frame_params {
    pub scale: *const f64,
    pub offset: *const f64,
    pub weights: *const f64,
}

/// NumPy quantile methods.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub enum rw_quantile_method {
    RW_Q_INVERTED_CDF = 0,
    RW_Q_AVERAGED_INVERTED_CDF = 1,
    RW_Q_CLOSEST_OBSERVATION = 2,
    RW_Q_INTERPOLATED_INVERTED_CDF = 3,
    RW_Q_HAZEN = 4,
    RW_Q_WEIBULL = 5,
    RW_Q_LINEAR = 6,
    RW_Q_MEDIAN_UNBIASED = 7,
    RW_Q_NORMAL_UNBIASED = 8,
    RW_Q_LOWER = 9,
    RW_Q_HIGHER = 10,
    RW_Q_MIDPOINT = 11,
    RW_Q_NEAREST = 12,
}

static THREADS: AtomicUsize = AtomicUsize::new(1);

fn status(e: Error) -> rw_status {
    match e {
        Error::BadDepth(_) => rw_status::RW_ERR_DEPTH,
        Error::Shape(_) => rw_status::RW_ERR_SHAPE,
        Error::Param(_) => rw_status::RW_ERR_PARAM,
        Error::DType(_) => rw_status::RW_ERR_DTYPE,
        Error::UnsupportedIsa(_) | Error::UnknownIsa(_) => rw_status::RW_ERR_ISA,
    }
}

fn guarded(f: impl FnOnce() -> rw_status) -> rw_status {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(rw_status::RW_ERR_INTERNAL)
}

fn opts(nan: NanPolicy, nodata: Option<f64>) -> Options {
    Options { nan, nodata, threads: THREADS.load(Ordering::Relaxed), ..Options::default() }
}

/// Run `op` over frame pointers. `outs` are (pointer, dtype) pairs over `npix` contiguous values.
unsafe fn run(
    dtype: DType,
    frames: *const *const u8,
    n: usize,
    npix: usize,
    op: &Op,
    outs: &[(*mut u8, DType)],
    mask: Option<*const *mut u8>,
    o: &Options,
) -> rw_status {
    if frames.is_null() {
        return rw_status::RW_ERR_NULL;
    }
    if n == 0 || n > RW_MAX_FRAMES {
        return rw_status::RW_ERR_DEPTH;
    }
    let fr = std::slice::from_raw_parts(frames, n);
    if fr.iter().any(|p| p.is_null()) {
        return rw_status::RW_ERR_NULL;
    }
    let shape = [npix];
    let strides = [dtype.size() as isize];
    let stack = RawStack { dtype, frames: fr, shape: &shape, strides: &strides };
    let ostr: Vec<[isize; 1]> = outs.iter().map(|(_, d)| [d.size() as isize]).collect();
    let raw: Vec<RawOut> =
        outs.iter().zip(ostr.iter()).map(|(&(p, d), s)| RawOut { ptr: p, dtype: d, strides: s }).collect();
    let mptrs: Vec<*mut u8>;
    let mstr = [1isize];
    let rm;
    let mref = match mask {
        Some(m) if !m.is_null() => {
            mptrs = std::slice::from_raw_parts(m, n).to_vec();
            if mptrs.iter().any(|p| p.is_null()) {
                return rw_status::RW_ERR_NULL;
            }
            rm = RawMask { frames: &mptrs, dtype: DType::U8, strides: &mstr };
            Some(&rm)
        }
        _ => None,
    };
    match reduce_raw(&stack, op, &raw, mref, o) {
        Ok(()) => rw_status::RW_OK,
        Err(e) => status(e),
    }
}

fn nn<T>(p: *mut T) -> bool {
    !p.is_null()
}

// ------------------------------------------------------------------ info

/// Library version, e.g. "0.1.0" (static string).
#[no_mangle]
pub extern "C" fn rw_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Name of the ISA calls currently use ("avx512", "avx2", "neon" or "scalar").
#[no_mangle]
pub extern "C" fn rw_isa() -> *const c_char {
    match rankwise::resolve_isa(None) {
        Ok(rankwise::Isa::Avx512) => c"avx512".as_ptr(),
        Ok(rankwise::Isa::Avx2) => c"avx2".as_ptr(),
        Ok(rankwise::Isa::Neon) => c"neon".as_ptr(),
        _ => c"scalar".as_ptr(),
    }
}

/// Force an ISA by name ("auto" or NULL restores detection).
#[no_mangle]
pub unsafe extern "C" fn rw_set_isa(name: *const c_char) -> rw_status {
    guarded(|| {
        let isa = if name.is_null() {
            None
        } else {
            let s = match CStr::from_ptr(name).to_str() {
                Ok(s) => s,
                Err(_) => return rw_status::RW_ERR_PARAM,
            };
            if s == "auto" {
                None
            } else {
                match rankwise::Isa::from_name(s) {
                    Some(i) => Some(i),
                    None => return rw_status::RW_ERR_ISA,
                }
            }
        };
        match rankwise::set_isa_override(isa) {
            Ok(()) => rw_status::RW_OK,
            Err(e) => status(e),
        }
    })
}

/// Worker threads for subsequent calls (1 = the calling thread only).
#[no_mangle]
pub extern "C" fn rw_set_threads(threads: u32) {
    THREADS.store(threads.max(1) as usize, Ordering::Relaxed);
}

/// Human-readable description of a status code (static string).
#[no_mangle]
pub extern "C" fn rw_status_string(s: rw_status) -> *const c_char {
    (match s {
        rw_status::RW_OK => c"ok",
        rw_status::RW_ERR_DEPTH => c"stack depth out of range (1..=512 frames)",
        rw_status::RW_ERR_SHAPE => c"sizes disagree",
        rw_status::RW_ERR_PARAM => c"parameter out of range",
        rw_status::RW_ERR_DTYPE => c"unsupported dtype",
        rw_status::RW_ERR_ISA => c"requested ISA not available on this CPU",
        rw_status::RW_ERR_NULL => c"a required pointer is NULL",
        rw_status::RW_ERR_INTERNAL => c"internal error",
    })
    .as_ptr()
}

// ------------------------------------------------------------------ clip constructors

/// astropy-compatible sigma clipping (median center, population std, float64 bounds).
#[no_mangle]
pub extern "C" fn rw_clip_sigma(low: f32, high: f32, maxiters: i32) -> rw_clip {
    rw_clip {
        method: rw_rejection::RW_REJECT_SIGMA as i32,
        low,
        high,
        maxiters,
        center_mean: 0,
        fast_f32: 0,
        combine: rw_combine::RW_COMBINE_MEAN as i32,
    }
}

/// MAD clipping (sigma = 1.482602218505602 × MAD).
#[no_mangle]
pub extern "C" fn rw_clip_mad(low: f32, high: f32, maxiters: i32) -> rw_clip {
    rw_clip { method: rw_rejection::RW_REJECT_MAD as i32, ..rw_clip_sigma(low, high, maxiters) }
}

/// Winsorized sigma clipping.
#[no_mangle]
pub extern "C" fn rw_clip_winsorized(low: f32, high: f32, maxiters: i32) -> rw_clip {
    rw_clip { method: rw_rejection::RW_REJECT_WINSORIZED as i32, ..rw_clip_sigma(low, high, maxiters) }
}

/// Percentile clipping (`low`/`high` are fractions of the median).
#[no_mangle]
pub extern "C" fn rw_clip_percentile(low: f32, high: f32) -> rw_clip {
    rw_clip { method: rw_rejection::RW_REJECT_PERCENTILE as i32, ..rw_clip_sigma(low, high, -1) }
}

/// `None` when `method` or `combine` is not one of the enum's values.
fn to_clip(c: &rw_clip) -> Option<Clip> {
    const R: [rw_rejection; 8] = [
        rw_rejection::RW_REJECT_NONE,
        rw_rejection::RW_REJECT_SIGMA,
        rw_rejection::RW_REJECT_MAD,
        rw_rejection::RW_REJECT_MEDIAN_SIGMA,
        rw_rejection::RW_REJECT_WINSORIZED,
        rw_rejection::RW_REJECT_PERCENTILE,
        rw_rejection::RW_REJECT_WINSORIZED_PIXINSIGHT,
        rw_rejection::RW_REJECT_SIGMA_SIRIL,
    ];
    let method = *R.iter().find(|&&r| r as i32 == c.method)?;
    let combine = match c.combine {
        0 => Combine::Mean,
        1 => Combine::Median,
        2 => Combine::Sum,
        _ => return None,
    };
    let (lo, hi) = (c.low as f64, c.high as f64);
    let it = if c.maxiters < 0 { None } else { Some(c.maxiters as u32) };
    let mut k = match method {
        rw_rejection::RW_REJECT_NONE => Clip::none(),
        rw_rejection::RW_REJECT_SIGMA => Clip::sigma(lo, hi).max_iters(it),
        rw_rejection::RW_REJECT_MAD => Clip::mad(lo, hi).max_iters(it),
        rw_rejection::RW_REJECT_MEDIAN_SIGMA => Clip::median_sigma(lo, hi).max_iters(it),
        rw_rejection::RW_REJECT_WINSORIZED => Clip::winsorized(lo, hi).max_iters(it),
        rw_rejection::RW_REJECT_PERCENTILE => Clip::percentile(lo, hi),
        rw_rejection::RW_REJECT_WINSORIZED_PIXINSIGHT => presets::pixinsight_winsorized(lo, hi).max_iters(it),
        rw_rejection::RW_REJECT_SIGMA_SIRIL => presets::siril_sigma(lo, hi),
    };
    if c.center_mean != 0 {
        k = k.center(Center::Mean);
    }
    if c.fast_f32 != 0 {
        k = k.precision(Precision::F32);
    }
    if method != rw_rejection::RW_REJECT_MEDIAN_SIGMA {
        k = k.combine(combine);
    }
    let _ = Spread::Std;
    Some(k)
}

// ------------------------------------------------------------------ medians & quantiles

/// `np.median` over `n` frames of `npix` floats each.
#[no_mangle]
pub unsafe extern "C" fn rw_median_f32(frames: *const *const f32, n: usize, npix: usize, out: *mut f32) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        run(
            DType::F32,
            frames as _,
            n,
            npix,
            &Op::Median,
            &[(out as _, DType::F32)],
            None,
            &opts(NanPolicy::Propagate, None),
        )
    })
}

/// `np.nanmedian` over `n` frames (NaN ignored; all-NaN pixels give NaN).
#[no_mangle]
pub unsafe extern "C" fn rw_nanmedian_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    out: *mut f32,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        run(
            DType::F32,
            frames as _,
            n,
            npix,
            &Op::Median,
            &[(out as _, DType::F32)],
            None,
            &opts(NanPolicy::Omit, None),
        )
    })
}

/// Median of uint16 frames, as float64 (NumPy's dtype). `nodata` >= 0 marks missing values,
/// which are skipped; pass -1 for none.
#[no_mangle]
pub unsafe extern "C" fn rw_median_u16(
    frames: *const *const u16,
    n: usize,
    npix: usize,
    nodata: i32,
    out: *mut f64,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        let nd = if nodata >= 0 { Some(nodata as f64) } else { None };
        run(
            DType::U16,
            frames as _,
            n,
            npix,
            &Op::Median,
            &[(out as _, DType::F64)],
            None,
            &opts(NanPolicy::Propagate, nd),
        )
    })
}

/// `None` when `m` is not an `rw_quantile_method` value.
fn method(m: i32) -> Option<QuantileMethod> {
    usize::try_from(m).ok().and_then(|i| QuantileMethod::ALL.get(i).copied())
}

/// Quantiles of float32 frames into float32 outputs, with NumPy's arithmetic for a Python
/// float `q` (interpolation in float32): `np.quantile(stack, q, axis=0, method=...)`.
/// `outs` holds `nq` pointers of `npix` floats; `m` is an `rw_quantile_method` value.
#[no_mangle]
pub unsafe extern "C" fn rw_quantile_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    q: *const f64,
    nq: usize,
    m: i32,
    outs: *const *mut f32,
) -> rw_status {
    guarded(|| {
        if q.is_null() || outs.is_null() {
            return rw_status::RW_ERR_NULL;
        }
        let qs = std::slice::from_raw_parts(q, nq).to_vec();
        let os = std::slice::from_raw_parts(outs, nq);
        if os.iter().any(|p| p.is_null()) {
            return rw_status::RW_ERR_NULL;
        }
        let Some(m) = method(m) else { return rw_status::RW_ERR_PARAM };
        let op = Op::Quantile(QuantileSpec { method: m, form: QForm::Weak, ..QuantileSpec::new(qs) });
        let o: Vec<(*mut u8, DType)> = os.iter().map(|&p| (p as *mut u8, DType::F32)).collect();
        run(DType::F32, frames as _, n, npix, &op, &o, None, &opts(NanPolicy::Propagate, None))
    })
}

/// Quantiles of float32 frames into float64 outputs, as `np.quantile(stack, [q...], axis=0)`
/// with a list of `q` (interpolation in float64). `nan_omit` != 0 gives `np.nanquantile`;
/// `m` is an `rw_quantile_method` value.
#[no_mangle]
pub unsafe extern "C" fn rw_quantile_f32_f64(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    q: *const f64,
    nq: usize,
    m: i32,
    nan_omit: i32,
    outs: *const *mut f64,
) -> rw_status {
    guarded(|| {
        if q.is_null() || outs.is_null() {
            return rw_status::RW_ERR_NULL;
        }
        let qs = std::slice::from_raw_parts(q, nq).to_vec();
        let os = std::slice::from_raw_parts(outs, nq);
        if os.iter().any(|p| p.is_null()) {
            return rw_status::RW_ERR_NULL;
        }
        let Some(m) = method(m) else { return rw_status::RW_ERR_PARAM };
        let op = Op::Quantile(QuantileSpec { method: m, form: QForm::F64, ..QuantileSpec::new(qs) });
        let o: Vec<(*mut u8, DType)> = os.iter().map(|&p| (p as *mut u8, DType::F64)).collect();
        let nan = if nan_omit != 0 { NanPolicy::Omit } else { NanPolicy::Propagate };
        run(DType::F32, frames as _, n, npix, &op, &o, None, &opts(nan, None))
    })
}

/// `np.sort(stack, axis=0)[ranks]` for float32 frames, one output per rank (NaN sorts last).
#[no_mangle]
pub unsafe extern "C" fn rw_select_ranks_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    ranks: *const usize,
    nranks: usize,
    outs: *const *mut f32,
) -> rw_status {
    guarded(|| {
        if ranks.is_null() || outs.is_null() {
            return rw_status::RW_ERR_NULL;
        }
        let r = std::slice::from_raw_parts(ranks, nranks).to_vec();
        let os = std::slice::from_raw_parts(outs, nranks);
        if os.iter().any(|p| p.is_null()) {
            return rw_status::RW_ERR_NULL;
        }
        let o: Vec<(*mut u8, DType)> = os.iter().map(|&p| (p as *mut u8, DType::F32)).collect();
        run(DType::F32, frames as _, n, npix, &Op::SelectRanks(r), &o, None, &opts(NanPolicy::Propagate, None))
    })
}

/// Per-pixel minimum (`which` = 0), maximum (1) or range (2) of float32 frames.
/// `nan_omit` != 0 ignores NaN.
#[no_mangle]
pub unsafe extern "C" fn rw_minmax_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    which: i32,
    nan_omit: i32,
    out: *mut f32,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        let op = match which {
            0 => Op::Min,
            1 => Op::Max,
            2 => Op::Range,
            _ => return rw_status::RW_ERR_PARAM,
        };
        let nan = if nan_omit != 0 { NanPolicy::Omit } else { NanPolicy::Propagate };
        run(DType::F32, frames as _, n, npix, &op, &[(out as _, DType::F32)], None, &opts(nan, None))
    })
}

/// Which frame holds the minimum (`which` = 0), maximum (1) or lower median (2) of each pixel.
/// First occurrence wins; with `nan_omit` != 0, NaN is skipped and all-NaN pixels give -1.
#[no_mangle]
pub unsafe extern "C" fn rw_arg_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    which: i32,
    nan_omit: i32,
    out: *mut i64,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        let op = match which {
            0 => Op::ArgMin,
            1 => Op::ArgMax,
            2 => Op::ArgRank(ArgRank::Median),
            _ => return rw_status::RW_ERR_PARAM,
        };
        let nan = if nan_omit != 0 { NanPolicy::Omit } else { NanPolicy::Propagate };
        run(DType::F32, frames as _, n, npix, &op, &[(out as _, DType::I64)], None, &opts(nan, None))
    })
}

/// `scipy.stats.trim_mean(stack, proportion, axis=0)` for float32 frames.
#[no_mangle]
pub unsafe extern "C" fn rw_trimmed_mean_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    proportion: f64,
    out: *mut f32,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        run(
            DType::F32,
            frames as _,
            n,
            npix,
            &Op::TrimMean(proportion),
            &[(out as _, DType::F32)],
            None,
            &opts(NanPolicy::Propagate, None),
        )
    })
}

/// `scipy.stats.median_abs_deviation(stack, axis=0, scale=scale)` for float32 frames.
#[no_mangle]
pub unsafe extern "C" fn rw_mad_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    scale: f64,
    out: *mut f32,
) -> rw_status {
    guarded(|| {
        if !nn(out) {
            return rw_status::RW_ERR_NULL;
        }
        run(
            DType::F32,
            frames as _,
            n,
            npix,
            &Op::Mad { scale },
            &[(out as _, DType::F32)],
            None,
            &opts(NanPolicy::Propagate, None),
        )
    })
}

// ------------------------------------------------------------------ rejection

/// Reject outliers and average the survivors: `out[p]` is the clipped mean of pixel `p`.
/// `rejected`, if not NULL, receives the number of rejected frames per pixel.
#[no_mangle]
pub unsafe extern "C" fn rw_clip_mean_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    clip: *const rw_clip,
    out: *mut f32,
    rejected: *mut u16,
) -> rw_status {
    if rejected.is_null() {
        return rw_clip_combine_f32(
            frames,
            n,
            npix,
            clip,
            core::ptr::null(),
            out,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null(),
        );
    }
    let mut hi = vec![0u16; npix];
    let s = rw_clip_combine_f32(
        frames,
        n,
        npix,
        clip,
        core::ptr::null(),
        out,
        rejected,
        hi.as_mut_ptr(),
        core::ptr::null(),
    );
    if s == rw_status::RW_OK {
        let r = std::slice::from_raw_parts_mut(rejected, npix);
        for (a, b) in r.iter_mut().zip(hi.iter()) {
            *a += *b;
        }
    }
    s
}

/// The full rejection call: optional normalization and weights (`params`, may be NULL),
/// low/high reject counts (may be NULL) and a rejection map (`mask`: `n` pointers to `npix`
/// bytes, 1 = rejected or missing; may be NULL).
#[no_mangle]
pub unsafe extern "C" fn rw_clip_combine_f32(
    frames: *const *const f32,
    n: usize,
    npix: usize,
    clip: *const rw_clip,
    params: *const rw_frame_params,
    out: *mut f32,
    low: *mut u16,
    high: *mut u16,
    mask: *const *mut u8,
) -> rw_status {
    guarded(|| {
        if clip.is_null() || out.is_null() {
            return rw_status::RW_ERR_NULL;
        }
        if n == 0 || n > RW_MAX_FRAMES {
            return rw_status::RW_ERR_DEPTH;
        }
        let Some(c) = to_clip(&*clip) else { return rw_status::RW_ERR_PARAM };
        let mut o = opts(NanPolicy::Propagate, None);
        if !params.is_null() {
            let p = &*params;
            let arr = |q: *const f64| if q.is_null() { None } else { Some(std::slice::from_raw_parts(q, n).to_vec()) };
            let (s, off) = (arr(p.scale), arr(p.offset));
            if s.is_some() || off.is_some() {
                o.normalize = Some(Normalize { scale: s, offset: off });
            }
            o.weights = arr(p.weights);
        }
        let outs = [(out as *mut u8, DType::F32), (low as *mut u8, DType::U16), (high as *mut u8, DType::U16)];
        let m = if mask.is_null() { None } else { Some(mask) };
        run(DType::F32, frames as _, n, npix, &Op::Clip(c), &outs, m, &o)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clip_mean_roundtrip() {
        let frames: Vec<Vec<f32>> = (0..9).map(|k| vec![if k == 4 { 1000.0 } else { k as f32 }; 33]).collect();
        let ptrs: Vec<*const f32> = frames.iter().map(|f| f.as_ptr()).collect();
        let mut out = vec![0f32; 33];
        let mut rej = vec![0u16; 33];
        let c = rw_clip_sigma(3.0, 3.0, 5);
        let s = unsafe { rw_clip_mean_f32(ptrs.as_ptr(), 9, 33, &c, out.as_mut_ptr(), rej.as_mut_ptr()) };
        assert_eq!(s, rw_status::RW_OK);
        assert_eq!(rej[0], 1);
        let s = unsafe { rw_median_f32(ptrs.as_ptr(), 0, 33, out.as_mut_ptr()) };
        assert_eq!(s, rw_status::RW_ERR_DEPTH);
    }

    /// Values C can put in the enum-valued fields and parameters are checked, not trusted
    /// (an out-of-range `rw_clip.method` used to crash).
    #[test]
    fn out_of_range_enum_values_are_param_errors() {
        let frames: Vec<Vec<f32>> = (0..5).map(|k| vec![k as f32; 33]).collect();
        let ptrs: Vec<*const f32> = frames.iter().map(|f| f.as_ptr()).collect();
        let mut out = vec![0f32; 33];
        let null = core::ptr::null_mut();
        for (method, combine) in [(42, 0), (-1, 0), (8, 0), (1, 3), (1, -1)] {
            let c = rw_clip { method, combine, ..rw_clip_sigma(3.0, 3.0, 5) };
            let s = unsafe {
                rw_clip_combine_f32(
                    ptrs.as_ptr(),
                    5,
                    33,
                    &c,
                    core::ptr::null(),
                    out.as_mut_ptr(),
                    null,
                    null,
                    core::ptr::null(),
                )
            };
            assert_eq!(s, rw_status::RW_ERR_PARAM, "method {method} combine {combine}");
        }
        let q = [0.5f64];
        let mut o64 = vec![0f64; 33];
        let o32p = [out.as_mut_ptr()];
        let o64p = [o64.as_mut_ptr()];
        for m in [13, -1, 99] {
            let s = unsafe { rw_quantile_f32(ptrs.as_ptr(), 5, 33, q.as_ptr(), 1, m, o32p.as_ptr()) };
            assert_eq!(s, rw_status::RW_ERR_PARAM);
            let s = unsafe { rw_quantile_f32_f64(ptrs.as_ptr(), 5, 33, q.as_ptr(), 1, m, 0, o64p.as_ptr()) };
            assert_eq!(s, rw_status::RW_ERR_PARAM);
        }
        let s = unsafe {
            rw_quantile_f32(ptrs.as_ptr(), 5, 33, q.as_ptr(), 1, rw_quantile_method::RW_Q_NEAREST as i32, o32p.as_ptr())
        };
        assert_eq!(s, rw_status::RW_OK);
        assert_eq!(out[0], 2.0);
    }

    /// Median sigma clipping averages the replaced stack; weights are refused, not misapplied.
    #[test]
    fn median_sigma_refuses_weights() {
        let frames: Vec<Vec<f32>> = (0..9).map(|k| vec![if k == 4 { 1000.0 } else { k as f32 }; 33]).collect();
        let ptrs: Vec<*const f32> = frames.iter().map(|f| f.as_ptr()).collect();
        let mut out = vec![0f32; 33];
        let w = [1.0f64; 9];
        let params = rw_frame_params { scale: core::ptr::null(), offset: core::ptr::null(), weights: w.as_ptr() };
        let c = rw_clip { method: rw_rejection::RW_REJECT_MEDIAN_SIGMA as i32, ..rw_clip_sigma(3.0, 3.0, 5) };
        let null = core::ptr::null_mut();
        let s = unsafe {
            rw_clip_combine_f32(ptrs.as_ptr(), 9, 33, &c, &params, out.as_mut_ptr(), null, null, core::ptr::null())
        };
        assert_eq!(s, rw_status::RW_ERR_PARAM);
    }
}
