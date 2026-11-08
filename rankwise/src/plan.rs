//! Validation and planning: turn an [`Op`] over a [`RawStack`] into a [`Job`], then run it.

use crate::arch;
use crate::clip::ClipPlan;
use crate::engine::{Job, Kern, Missing, Unit};
use crate::layout::{collapse, RawMask, RawOut, RawStack};
use crate::net;
use crate::netgen;
use crate::quantile::{quantile_index, Lerp, QIdx, ScipyLinear};
use crate::simd::{resolve_isa, Isa};
use crate::spec::{ArgRank, Combine, DType, NanPolicy, Op, Precision};
use crate::{Error, Result, MAX_FRAMES};

/// When to read frames through an in-cache tile instead of straight from memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Staging {
    /// Stage only when the layout needs it (non-unit pixel stride, dtype conversion).
    /// Power-of-two frame strides cost 1.1-1.25x on a 2 MiB-L2 Sapphire Rapids core, less
    /// than the tile copy, so they read directly.
    #[default]
    Auto,
    /// Same as `Auto` (kept for explicitness in benchmarks).
    Never,
    /// Always.
    Always,
}

/// Per-frame normalization `x * scale[k] + offset[k]`, applied on load (two roundings).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Normalize {
    pub scale: Option<Vec<f64>>,
    pub offset: Option<Vec<f64>>,
}

/// Execution and missing-data options shared by every call.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Options {
    /// NaN handling (floats) or `nodata` handling (integers).
    pub nan: NanPolicy,
    /// Integer value that marks missing data (u16 input). Missing values are always
    /// omitted, as NaN is in the `nan*` functions.
    pub nodata: Option<f64>,
    pub normalize: Option<Normalize>,
    /// Per-frame weights for clipped means.
    pub weights: Option<Vec<f64>>,
    /// Force an ISA (default: `RANKWISE_ISA`, else the fastest available).
    pub isa: Option<Isa>,
    /// Worker threads; 0 or 1 runs on the calling thread.
    pub threads: usize,
    pub staging: Staging,
}

/// The dtype rankwise writes for `op` on `input` when the caller does not choose one.
pub fn natural_dtype(op: &Op, input: DType) -> DType {
    match op {
        Op::SelectRanks(_) | Op::Min | Op::Max | Op::Range => input,
        Op::Median => match input {
            DType::F32 => DType::F32,
            _ => DType::F64,
        },
        Op::Quantile(q) => {
            Lerp::new(input, q.form, None, q.out_rounding).natural_dtype(q.method.is_discrete()
                || (q.form == crate::spec::QForm::WeakInt && q.method == crate::spec::QuantileMethod::Linear))
        }
        Op::ArgMin | Op::ArgMax | Op::ArgRank(_) => DType::I64,
        Op::TrimMean(_) | Op::Iqr { .. } | Op::Mad { .. } => match input {
            DType::F32 => DType::F32,
            _ => DType::F64,
        },
        Op::Clip(_) => match input {
            DType::F64 => DType::F64,
            _ => DType::F32,
        },
    }
}

fn needed_union(tab: &[QIdx], n: usize, out: &mut Vec<usize>) {
    match tab[n] {
        QIdx::Take(i) => out.push(i),
        QIdx::Lerp { i0, i1, .. } => {
            out.push(i0);
            out.push(i1);
        }
        QIdx::Empty => {}
    }
}

/// Validate and plan. `outs` must have [`Op::n_outputs`] entries.
pub(crate) fn plan(
    stack: &RawStack,
    op: &Op,
    outs: &[RawOut],
    mask: Option<&RawMask>,
    opts: &Options,
) -> Result<Job> {
    let n = stack.frames.len();
    if n == 0 || n > MAX_FRAMES {
        return Err(Error::BadDepth(n));
    }
    let in_dtype = stack.dtype;
    if !matches!(in_dtype, DType::F32 | DType::F64 | DType::U16) {
        return Err(Error::DType(format!("input must be float32, float64 or uint16, got {}", in_dtype.name())));
    }
    if outs.len() != op.n_outputs() {
        return Err(Error::Shape(format!("{:?} writes {} outputs, got {}", op, op.n_outputs(), outs.len())));
    }
    if stack.shape.len() != stack.strides.len() {
        return Err(Error::Shape("input shape and strides differ in length".into()));
    }
    let isa = resolve_isa(opts.isa)?;
    let omit = opts.nan == NanPolicy::Omit || (in_dtype == DType::U16 && opts.nodata.is_some());
    let has_norm = opts
        .normalize
        .as_ref()
        .map(|z| z.scale.is_some() || z.offset.is_some())
        .unwrap_or(false);
    let sdt = if in_dtype == DType::U16 && has_norm { DType::F32 } else { in_dtype };
    if let Some(v) = opts.nodata {
        if in_dtype != DType::U16 {
            return Err(Error::Param("nodata applies to integer input; floats use NaN".into()));
        }
        if !(v >= 0.0 && v <= 65535.0 && v.fract() == 0.0) {
            return Err(Error::Param(format!("nodata {v} is not a uint16 value")));
        }
    }
    let missing = match (sdt, op) {
        (DType::U16, _) => match opts.nodata {
            Some(v) => Missing::NoData(v as u16),
            None => Missing::None,
        },
        (_, Op::Clip(c)) if c.mask_infinite => Missing::NonFinite,
        _ => Missing::Nan,
    };
    let norm = if has_norm {
        let z = opts.normalize.as_ref().unwrap();
        let s = z.scale.clone().unwrap_or_else(|| vec![1.0; n]);
        let o = z.offset.clone().unwrap_or_else(|| vec![0.0; n]);
        if s.len() != n || o.len() != n {
            return Err(Error::Shape(format!("normalization needs {n} scales and offsets")));
        }
        Some((s, o))
    } else {
        None
    };
    if let Some(w) = &opts.weights {
        if w.len() != n {
            return Err(Error::Shape(format!("{} weights for {n} frames", w.len())));
        }
        if !matches!(op, Op::Clip(_)) {
            return Err(Error::Param("weights apply to clipped means only".into()));
        }
        if w.iter().any(|x| !(x.is_finite() && *x >= 0.0)) {
            return Err(Error::Param("weights must be finite and non-negative".into()));
        }
    }
    let unroll = arch::unroll_for(isa, sdt);
    let full_sort = || net::plan(n, None, unroll);
    let mut fast_needed: Option<Vec<usize>> = None;

    let kern = match op {
        Op::SelectRanks(r) => {
            if let Some(&bad) = r.iter().find(|&&k| k >= n) {
                return Err(Error::Param(format!("rank {bad} out of range for {n} frames")));
            }
            fast_needed = Some(r.clone());
            Kern::Select { ranks: r.clone() }
        }
        Op::Median => {
            fast_needed = Some(netgen::median_ranks(n));
            Kern::Median { omit }
        }
        Op::Quantile(q) => {
            if q.q.is_empty() {
                return Err(Error::Param("no quantiles requested".into()));
            }
            if q.q.iter().any(|x| !(*x >= 0.0 && *x <= 1.0)) {
                return Err(Error::Param("Quantiles must be in the range [0, 1]".into()));
            }
            let tabs: Vec<Vec<QIdx>> = q
                .q
                .iter()
                .map(|&qq| (0..=n).map(|m| quantile_index(q.method, q.form, m, qq)).collect())
                .collect();
            let mut need = Vec::new();
            for t in &tabs {
                needed_union(t, n, &mut need);
            }
            fast_needed = Some(need);
            let lerp = Lerp::new(sdt_for_lerp(in_dtype, sdt), q.form, q.out_dtype, q.out_rounding);
            Kern::Quantile { tabs, lerp, omit }
        }
        Op::Iqr { lo, hi, scale } => {
            if !(lo.is_finite() && hi.is_finite()) {
                return Err(Error::Param("`rng` must not contain NaNs.".into()));
            }
            let (a, b) = if lo < hi { (lo / 100.0, hi / 100.0) } else { (hi / 100.0, lo / 100.0) };
            if a < 0.0 || b > 1.0 {
                return Err(Error::Param("Elements of `rng` must be in the range [0, 100].".into()));
            }
            let sp = ScipyLinear { f32: in_dtype == DType::F32 && sdt == DType::F32 };
            let tlo: Vec<_> = (0..=n).map(|m| if m == 0 { (0, 0, 0.0) } else { sp.index(a, m) }).collect();
            let thi: Vec<_> = (0..=n).map(|m| if m == 0 { (0, 0, 0.0) } else { sp.index(b, m) }).collect();
            fast_needed = Some(vec![tlo[n].0, tlo[n].1, thi[n].0, thi[n].1]);
            Kern::Iqr { tlo, thi, sp, scale: *scale, omit }
        }
        Op::Mad { scale } => {
            if !(*scale > 0.0) {
                return Err(Error::Param("scale must be positive".into()));
            }
            Kern::Mad { scale: *scale, omit }
        }
        Op::TrimMean(p) => {
            if !(p.is_finite() && *p >= 0.0) {
                return Err(Error::Param("proportiontocut must be a non-negative number".into()));
            }
            let lo = (p * n as f64) as usize;
            if lo > n.saturating_sub(lo) {
                return Err(Error::Param("Proportion too big.".into()));
            }
            let hi = n - lo;
            fast_needed = Some((lo..hi).collect());
            Kern::TrimMean { lo, hi }
        }
        Op::Min => Kern::MinMax { which: 0, omit },
        Op::Max => Kern::MinMax { which: 1, omit },
        Op::Range => Kern::MinMax { which: 2, omit },
        Op::ArgMin => Kern::Arg { max: false, omit },
        Op::ArgMax => Kern::Arg { max: true, omit },
        Op::ArgRank(r) => {
            match r {
                ArgRank::Fixed(k) => {
                    if *k >= n {
                        return Err(Error::Param(format!("rank {k} out of range for {n} frames")));
                    }
                    fast_needed = Some(vec![*k]);
                }
                ArgRank::Median => fast_needed = Some(vec![(n - 1) / 2]),
            }
            Kern::ArgRank { rank: *r, omit }
        }
        Op::Clip(c) => {
            c.validate()?;
            if c.precision == Precision::F32 && sdt != DType::F32 {
                return Err(Error::Param("f32 precision needs float32 input".into()));
            }
            if opts.weights.is_some() && !matches!(c.combine, Combine::Mean | Combine::Sum) {
                return Err(Error::Param("weights apply to mean or sum combines".into()));
            }
            Kern::Clip(ClipPlan {
                rej: c.rejection,
                combine: c.combine,
                f32: c.precision == Precision::F32,
                weights: opts.weights.clone(),
                has_mask: mask.is_some(),
            })
        }
    };

    if let Some(m) = mask {
        if !matches!(op, Op::Clip(_)) {
            return Err(Error::Param("a rejection map applies to clipping only".into()));
        }
        if m.frames.len() != n {
            return Err(Error::Shape("rejection map needs one frame per input frame".into()));
        }
        if !matches!(m.dtype, DType::Bool | DType::U8) {
            return Err(Error::DType("rejection map must be bool or uint8".into()));
        }
    }

    let net_full = full_sort();
    let net_fast = match &fast_needed {
        Some(r) => net::plan(n, Some(r), unroll),
        None => net_full.clone(),
    };
    let net_bitonic = net::bitonic_plan(n);

    let out_strides: Vec<&[isize]> = outs.iter().map(|o| if o.ptr.is_null() { stack.strides_zero() } else { o.strides }).collect();
    let space = collapse(stack.shape, stack.strides, &out_strides, mask.map(|m| m.strides))?;
    let esz = in_dtype.size() as isize;
    let need_stage = space.npix > 0 && (space.in_ps != esz || sdt != in_dtype);
    let staged = match opts.staging {
        Staging::Always => true,
        Staging::Never => need_stage,
        Staging::Auto => need_stage,
    };
    Ok(Job {
        n,
        in_dtype,
        sdt,
        frames: stack.frames.iter().map(|&p| p as usize).collect(),
        space,
        outs: outs.iter().map(|o| (o.ptr as usize, o.dtype)).collect(),
        mask: mask.map(|m| (m.frames.iter().map(|&p| p as usize).collect(), m.dtype)),
        missing,
        norm,
        kern,
        net_fast,
        net_full,
        net_bitonic,
        staged,
        isa,
        threads: opts.threads.max(1),
    })
}

fn sdt_for_lerp(input: DType, sdt: DType) -> DType {
    if input == DType::U16 && sdt == DType::F32 {
        // normalized integers interpolate like float32 data
        DType::F32
    } else {
        input
    }
}

impl RawStack<'_> {
    fn strides_zero(&self) -> &'static [isize] {
        const Z: [isize; 32] = [0; 32];
        &Z[..self.shape.len().min(32)]
    }
}

/// Split the work into units for `threads` workers.
fn units(job: &Job) -> Vec<Unit> {
    let rows = job.space.rows();
    let npix = job.space.npix;
    if npix == 0 || rows == 0 {
        return vec![];
    }
    let t = job.threads;
    let total = rows * npix;
    let target = if t <= 1 { usize::MAX } else { (total / (t * 8)).max(4096) };
    let mut v = Vec::new();
    for r in 0..rows {
        if npix <= target {
            v.push(Unit { row: r, p0: 0, p1: npix });
        } else {
            let chunk = target.div_ceil(64) * 64;
            let mut p = 0;
            while p < npix {
                let e = (p + chunk).min(npix);
                v.push(Unit { row: r, p0: p, p1: e });
                p = e;
            }
        }
    }
    v
}

pub(crate) fn execute(job: &Job) {
    let us = units(job);
    if us.is_empty() {
        return;
    }
    if job.threads <= 1 || us.len() == 1 {
        for u in us {
            unsafe { arch::run(job, u) };
        }
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(job.threads).build();
        match pool {
            Ok(p) => p.install(|| us.par_iter().for_each(|&u| unsafe { arch::run(job, u) })),
            Err(_) => us.iter().for_each(|&u| unsafe { arch::run(job, u) }),
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let workers = job.threads.min(us.len());
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= us.len() {
                        break;
                    }
                    unsafe { arch::run(job, us[i]) };
                });
            }
        });
    }
}

/// Compute `op` over a raw strided stack.
///
/// # Safety
/// Every frame pointer must be valid for reads over `shape`/`strides`, every non-null output
/// (and the mask) valid for writes over its strides, and outputs must not overlap inputs or
/// each other.
pub unsafe fn reduce_raw(
    stack: &RawStack,
    op: &Op,
    outs: &[RawOut],
    mask: Option<&RawMask>,
    opts: &Options,
) -> Result<()> {
    let job = plan(stack, op, outs, mask, opts)?;
    execute(&job);
    Ok(())
}

/// Describe how a call would run: ISA, lanes, network size and path.
pub fn describe(stack: &RawStack, op: &Op, opts: &Options) -> Result<String> {
    let outs: Vec<RawOut> = (0..op.n_outputs()).map(|_| RawOut::none()).collect();
    let job = plan(stack, op, &outs, None, opts)?;
    Ok(format!(
        "isa={} lanes={} sort_dtype={} n={} comparators(fast)={} comparators(full)={} path={} rows={} pixels/row={}",
        job.isa,
        arch::lanes_for(job.isa, job.sdt),
        job.sdt.name(),
        job.n,
        job.net_fast.comparators(),
        job.net_full.comparators(),
        if job.staged { "staged" } else { "direct" },
        job.space.rows(),
        job.space.npix,
    ))
}

/// Comparators in the network rankwise runs for ranks `needed` of `n` values.
pub fn network_size(n: usize, needed: Option<&[usize]>) -> usize {
    match needed {
        None => netgen::batcher(n).len(),
        Some(r) => netgen::prune(&netgen::batcher(n), n, r).len(),
    }
}
