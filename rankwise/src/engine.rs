//! The block engine: load L pixels from n frames into n registers, handle missing values,
//! run the network, then hand the sorted block to the operation's epilogue.

use crate::layout::Space;
use crate::net::NetPlan;
use crate::quantile::{Lerp, QIdx, ScipyLinear};
use crate::simd::{Element, FVector, Isa, Mask, Vector};
use crate::spec::{ArgRank, DType};
use core::any::TypeId;

/// "No index" marker that every sort dtype represents exactly.
pub(crate) const SENTINEL: usize = 65535;

/// Most lanes any vector type has (u16 on AVX-512).
pub(crate) const MAXL: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Missing {
    None,
    Nan,
    NonFinite,
    NoData(u16),
}

/// Per-operation data the epilogues need.
#[derive(Debug, Clone)]
pub(crate) enum Kern {
    Select { ranks: Vec<usize> },
    Median { omit: bool },
    Quantile { tabs: Vec<Vec<QIdx>>, lerp: Lerp, omit: bool },
    Iqr { tlo: Vec<(usize, usize, f64)>, thi: Vec<(usize, usize, f64)>, sp: ScipyLinear, scale: f64, omit: bool },
    Mad { scale: f64, omit: bool },
    TrimMean { lo: usize, hi: usize },
    MinMax { which: u8, omit: bool },
    Arg { max: bool, omit: bool },
    ArgRank { rank: ArgRank, omit: bool },
    Clip(crate::clip::ClipPlan),
}

impl Kern {
    pub fn streaming(&self) -> bool {
        matches!(self, Kern::MinMax { .. } | Kern::Arg { .. })
    }
    pub fn needs_raw(&self) -> bool {
        match self {
            Kern::ArgRank { .. } => true,
            Kern::Clip(c) => c.needs_raw(),
            _ => false,
        }
    }
    /// Whether a block with missing values needs the full network (per-lane ranks) rather
    /// than the fixed-rank one.
    pub fn per_lane_ranks(&self) -> bool {
        match self {
            Kern::Select { .. } | Kern::TrimMean { .. } => false,
            Kern::Median { omit } | Kern::Quantile { omit, .. } | Kern::Iqr { omit, .. } => *omit,
            Kern::Mad { .. } => true,
            Kern::ArgRank { rank, omit } => *omit || matches!(rank, ArgRank::Median),
            Kern::Clip(_) => true,
            Kern::MinMax { .. } | Kern::Arg { .. } => false,
        }
    }
}

pub(crate) struct Job {
    pub n: usize,
    /// dtype in memory.
    pub in_dtype: DType,
    /// dtype the network sorts (u16 data with normalization is widened to f32).
    pub sdt: DType,
    pub frames: Vec<usize>,
    pub space: Space,
    pub outs: Vec<(usize, DType)>,
    pub mask: Option<(Vec<usize>, DType)>,
    pub missing: Missing,
    pub norm: Option<(Vec<f64>, Vec<f64>)>,
    pub kern: Kern,
    pub net_fast: NetPlan,
    pub net_full: NetPlan,
    pub net_bitonic: NetPlan,
    /// Read through an in-cache tile (non-unit pixel stride, dtype conversion, or the
    /// power-of-two frame-stride fix).
    pub staged: bool,
    pub isa: Isa,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Unit {
    pub row: usize,
    pub p0: usize,
    pub p1: usize,
}

/// An output row: base pointer (null = skip), dtype, pixel stride in bytes.
#[derive(Clone, Copy)]
pub(crate) struct OutRow {
    pub ptr: *mut u8,
    pub dtype: DType,
    pub ps: isize,
}

/// Write `lanes` values (exactly representable in the destination, or rounded once) at
/// pixel `p`.
#[inline(always)]
pub(crate) unsafe fn put(o: &OutRow, p: usize, vals: &[f64; MAXL], lanes: usize) {
    if o.ptr.is_null() {
        return;
    }
    let base = o.ptr.offset(p as isize * o.ps);
    macro_rules! w {
        ($t:ty, $conv:expr) => {{
            let sz = core::mem::size_of::<$t>() as isize;
            if o.ps == sz {
                let d = base as *mut $t;
                for i in 0..lanes {
                    d.add(i).write_unaligned($conv(vals[i]));
                }
            } else {
                for i in 0..lanes {
                    (base.offset(i as isize * o.ps) as *mut $t).write_unaligned($conv(vals[i]));
                }
            }
        }};
    }
    match o.dtype {
        DType::F32 => w!(f32, |x: f64| x as f32),
        DType::F64 => w!(f64, |x: f64| x),
        DType::U16 => w!(u16, |x: f64| x as u16),
        DType::I32 => w!(i32, |x: f64| x as i32),
        DType::I64 => w!(i64, |x: f64| x as i64),
        DType::Bool | DType::U8 => w!(u8, |x: f64| (x != 0.0) as u8),
    }
}

/// Store a vector's lanes as f64.
#[inline(always)]
pub(crate) unsafe fn lanes_f64<S: Vector>(v: S, out: &mut [f64; MAXL]) {
    let mut a = [S::E::ZERO; MAXL];
    v.store(a.as_mut_ptr());
    for i in 0..S::L {
        out[i] = a[i].to_f64();
    }
}

#[inline(always)]
pub(crate) unsafe fn lanes_usize<S: Vector>(v: S) -> [usize; MAXL] {
    let mut a = [S::E::ZERO; MAXL];
    v.store(a.as_mut_ptr());
    let mut r = [0usize; MAXL];
    for i in 0..S::L {
        r[i] = a[i].to_usize();
    }
    r
}

/// Element `lane` of vector `k` of a buffer of vectors.
#[inline(always)]
pub(crate) unsafe fn at<S: Vector>(buf: *const S, k: usize, lane: usize) -> S::E {
    *(buf.add(k) as *const S::E).add(lane)
}

#[inline(always)]
pub(crate) unsafe fn miss_mask<S: Vector>(x: S, mode: Missing) -> S::M {
    match mode {
        Missing::None => S::mask_none(),
        Missing::Nan => x.nan_mask(),
        Missing::NonFinite => x.nonfinite_mask(),
        Missing::NoData(v) => x.eq(S::splat(S::E::from_f64(v as f64))),
    }
}

/// NumPy's median of the pair of middle values, as f64 (exact in the result dtype).
#[inline(always)]
fn np_mid<E: Element>(a: E, b: E) -> f64 {
    if TypeId::of::<E>() == TypeId::of::<f32>() {
        let (a, b) = (a.to_f64() as f32, b.to_f64() as f32);
        ((a + b) * 0.5) as f64
    } else {
        (a.to_f64() + b.to_f64()) * 0.5
    }
}

/// The value written for "no valid data" in an output of the sort dtype.
#[inline(always)]
fn fill_value(job: &Job) -> f64 {
    match job.missing {
        Missing::NoData(v) if !job.sdt.is_float() => v as f64,
        _ => f64::NAN,
    }
}

/// Block context handed to epilogues.
pub(crate) struct Blk<'a, S: Vector> {
    pub buf: *mut S,
    pub raw: *const S,
    pub m: S,
    pub any_missing: bool,
    pub lanes: usize,
    pub p: usize,
    pub outs: &'a [OutRow],
    pub mask: &'a [*mut u8],
    pub mask_ps: isize,
    /// 64-byte aligned scratch for `3 * n` vectors of the widest type.
    pub scratch: *mut u8,
}

/// Process one unit of work: one row, pixels `p0..p1`.
#[inline(always)]
pub(crate) unsafe fn run_unit<A: crate::arch::Arch, S: crate::arch::SortVec>(job: &Job, u: Unit) {
    let n = job.n;
    let l = S::L;
    let mut out_off = vec![0isize; job.outs.len()];
    let (in_off, mask_off) = job.space.row_offsets(u.row, &mut out_off);
    let outs: Vec<OutRow> = job
        .outs
        .iter()
        .enumerate()
        .map(|(j, &(ptr, dt))| OutRow {
            ptr: if ptr == 0 { core::ptr::null_mut() } else { (ptr as isize + out_off[j]) as *mut u8 },
            dtype: dt,
            ps: job.space.out_ps[j],
        })
        .collect();
    let mask: Vec<*mut u8> = match &job.mask {
        Some((m, _)) => m.iter().map(|&a| (a as isize + mask_off) as *mut u8).collect(),
        None => vec![],
    };
    let src: Vec<*const u8> = job.frames.iter().map(|&f| (f as isize + in_off) as *const u8).collect();
    let in_ps = job.space.in_ps;
    let esz = core::mem::size_of::<S::E>();

    let mut buf: Vec<S> = vec![S::splat(S::E::ZERO); n.max(1)];
    let mut raw: Vec<S> = if job.kern.needs_raw() { vec![S::splat(S::E::ZERO); n] } else { vec![] };
    // Staging tile: n rows of `ts` elements.
    let tile_px = if job.staged {
        // Large enough that each frame's copy streams a long run from memory (the hardware
        // prefetcher ramps up once per run), small enough that the tile stays in L2.
        let budget = tile_budget();
        let target = budget / (n * esz).max(1);
        ((target / l).max(1) * l).min(16384).max(l)
    } else {
        u.p1 - u.p0
    };
    let ts = tile_px.div_ceil(l) * l + if job.staged { 16 } else { 0 };
    let mut tile: Vec<S::E> = if job.staged { vec![S::E::ZERO; n * ts] } else { vec![] };
    let mut ptrs: Vec<*const S::E> = vec![core::ptr::null(); n];
    let mut scratch_v: Vec<u8> = vec![0u8; 3 * n * 64 + 64];
    let scratch = {
        let p = scratch_v.as_mut_ptr();
        p.add(p.align_offset(64))
    };
    let mut pad = [S::E::ZERO; MAXL];

    let scale_off: Option<(Vec<S::E>, Vec<S::E>)> = job
        .norm
        .as_ref()
        .map(|(s, o)| (s.iter().map(|&x| S::E::from_f64(x)).collect(), o.iter().map(|&x| S::E::from_f64(x)).collect()));

    let mut t0 = u.p0;
    while t0 < u.p1 {
        let t1 = (t0 + tile_px).min(u.p1);
        let cnt = t1 - t0;
        if job.staged {
            stage::<S>(job, &src, in_ps, t0, cnt, &mut tile, ts);
            for k in 0..n {
                ptrs[k] = tile.as_ptr().add(k * ts);
            }
        } else {
            for k in 0..n {
                ptrs[k] = src[k].offset(t0 as isize * in_ps) as *const S::E;
            }
        }
        let mut b = 0;
        while b < cnt {
            let lanes = (cnt - b).min(l);
            let full = lanes == l || job.staged;
            let blk_p = t0 + b;
            if job.kern.streaming() {
                streaming::<S>(job, &ptrs, b, lanes, full, &mut pad, &scale_off, &outs, blk_p);
                b += l;
                continue;
            }
            // ---- load
            let mut anym = S::mask_none();
            for k in 0..n {
                let mut x = if full {
                    S::load(ptrs[k].add(b))
                } else {
                    for i in 0..lanes {
                        pad[i] = *ptrs[k].add(b + i);
                    }
                    S::load(pad.as_ptr())
                };
                if let Some((s, o)) = &scale_off {
                    x = norm::<S>(x, s[k], o[k]);
                }
                anym = anym.or(miss_mask(x, job.missing));
                *buf.as_mut_ptr().add(k) = x;
            }
            let any_missing = anym.any();
            if !raw.is_empty() {
                raw.copy_from_slice(&buf);
            }
            let mut m = S::splat(S::E::from_usize(n));
            if any_missing {
                let one = S::splat(S::E::from_usize(1));
                let big = S::splat(S::E::MAX);
                for k in 0..n {
                    let x = *buf.as_ptr().add(k);
                    let mm = miss_mask(x, job.missing);
                    *buf.as_mut_ptr().add(k) = S::select(mm, big, x);
                    m = S::select(mm, m.sub(one), m);
                }
            }
            let plan = if any_missing && job.kern.per_lane_ranks() { &job.net_full } else { &job.net_fast };
            S::apply_net(buf.as_mut_ptr(), plan);
            let blk = Blk {
                buf: buf.as_mut_ptr(),
                raw: raw.as_ptr(),
                m,
                any_missing,
                lanes,
                p: blk_p,
                outs: &outs,
                mask: &mask,
                mask_ps: job.space.mask_ps,
                scratch,
            };
            epilogue::<A, S>(job, &blk);
            b += l;
        }
        t0 = t1;
    }
}

/// Staging tile size in bytes (all frames together). `RANKWISE_TILE_BYTES` overrides it for
/// experiments.
fn tile_budget() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("RANKWISE_TILE_BYTES").ok().and_then(|s| s.parse().ok()).unwrap_or(768 * 1024))
}

#[inline(always)]
unsafe fn norm<S: Vector>(x: S, s: S::E, o: S::E) -> S {
    // x * s + o, two roundings. Only float sort dtypes carry normalization.
    if TypeId::of::<S::E>() == TypeId::of::<f32>() || TypeId::of::<S::E>() == TypeId::of::<f64>() {
        x.mul(S::splat(s)).add(S::splat(o))
    } else {
        x
    }
}

/// Copy (and convert) `cnt` pixels of every frame into the tile, `ts` elements per frame.
#[inline(always)]
unsafe fn stage<S: Vector>(
    job: &Job,
    src: &[*const u8],
    ps: isize,
    t0: usize,
    cnt: usize,
    tile: &mut [S::E],
    ts: usize,
) {
    let n = job.n;
    let tp = tile.as_mut_ptr();
    macro_rules! go {
        ($t:ty) => {{
            let fstep = if n > 1 { (src[1] as isize) - (src[0] as isize) } else { 0 };
            let uniform = n > 1 && (0..n).all(|k| src[k] as isize == src[0] as isize + k as isize * fstep);
            if uniform && fstep.unsigned_abs() < ps.unsigned_abs() {
                // Frames interleaved (reduced axis innermost): walk pixels, frames inside.
                for p in 0..cnt {
                    let base = src[0].offset((t0 + p) as isize * ps);
                    for k in 0..n {
                        let v = (base.offset(k as isize * fstep) as *const $t).read_unaligned();
                        *tp.add(k * ts + p) = S::E::from_f64(v as f64);
                    }
                }
            } else {
                for k in 0..n {
                    let base = src[k].offset(t0 as isize * ps);
                    let dst = tp.add(k * ts);
                    if ps == core::mem::size_of::<$t>() as isize && TypeId::of::<$t>() == TypeId::of::<S::E>() {
                        core::ptr::copy_nonoverlapping(base as *const S::E, dst, cnt);
                    } else {
                        for p in 0..cnt {
                            let v = (base.offset(p as isize * ps) as *const $t).read_unaligned();
                            *dst.add(p) = S::E::from_f64(v as f64);
                        }
                    }
                }
            }
        }};
    }
    match job.in_dtype {
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        DType::U16 => go!(u16),
        _ => unreachable!(),
    }
}

/// min / max / range / argmin / argmax: one pass over the frames, no network.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn streaming<S: Vector>(
    job: &Job,
    ptrs: &[*const S::E],
    b: usize,
    lanes: usize,
    full: bool,
    pad: &mut [S::E; MAXL],
    scale_off: &Option<(Vec<S::E>, Vec<S::E>)>,
    outs: &[OutRow],
    p: usize,
) {
    let n = job.n;
    let load = |k: usize, pad: &mut [S::E; MAXL]| -> S {
        let mut x = if full {
            S::load(ptrs[k].add(b))
        } else {
            for i in 0..lanes {
                pad[i] = *ptrs[k].add(b + i);
            }
            S::load(pad.as_ptr())
        };
        if let Some((s, o)) = scale_off {
            x = norm::<S>(x, s[k], o[k]);
        }
        x
    };
    let mut vals = [0f64; MAXL];
    match job.kern {
        Kern::MinMax { which, omit } => {
            let hi_fill = S::splat(S::E::MAX);
            let lo_fill = S::splat(lowest::<S::E>());
            let mut mn = hi_fill;
            let mut mx = lo_fill;
            let mut seen_missing = S::mask_none();
            let mut seen_valid = S::mask_none();
            for k in 0..n {
                let x = load(k, pad);
                let mm = miss_mask(x, job.missing);
                seen_missing = seen_missing.or(mm);
                if omit {
                    seen_valid = seen_valid.or(miss_mask(x, job.missing).not());
                    mn = mn.min(S::select(mm, hi_fill, x));
                    mx = mx.max(S::select(mm, lo_fill, x));
                } else {
                    mn = if k == 0 { x } else { mn.min(x) };
                    mx = if k == 0 { x } else { mx.max(x) };
                }
            }
            let r = match which {
                0 => mn,
                1 => mx,
                _ => mx.sub(mn),
            };
            lanes_f64(r, &mut vals);
            let bad = if omit { seen_valid.bits() ^ lane_bits(S::L) } else { seen_missing.bits() };
            if bad != 0 {
                let f = fill_value(job);
                for (i, v) in vals.iter_mut().enumerate().take(lanes) {
                    if (bad >> i) & 1 == 1 {
                        *v = f;
                    }
                }
            }
            put(&outs[0], p, &vals, lanes);
        }
        Kern::Arg { max, omit } => {
            let mut best = S::splat(S::E::ZERO);
            let mut idx = S::splat(S::E::from_usize(SENTINEL));
            let mut frozen = S::mask_none(); // propagate: a NaN was seen
            let mut found = S::mask_none(); // omit: a valid value was seen
            for k in 0..n {
                let x = load(k, pad);
                let kv = S::splat(S::E::from_usize(k));
                let mm = miss_mask(x, job.missing);
                if k == 0 && !omit {
                    best = x;
                    idx = kv;
                    frozen = mm;
                    continue;
                }
                let better = if max { x.gt(best) } else { x.lt(best) };
                let upd = if omit {
                    let valid = mm.not();
                    valid.and(better.or(found.not()))
                } else {
                    better.or(mm).andnot(frozen)
                };
                best = S::select(upd, x, best);
                idx = S::select(upd, kv, idx);
                if omit {
                    found = found.or(mm.not());
                } else {
                    frozen = frozen.or(mm);
                }
            }
            let ix = lanes_usize(idx);
            for i in 0..lanes {
                vals[i] = if ix[i] >= n { -1.0 } else { ix[i] as f64 };
            }
            put(&outs[0], p, &vals, lanes);
        }
        _ => unreachable!(),
    }
}

#[inline(always)]
fn lane_bits(l: usize) -> u64 {
    if l >= 64 {
        u64::MAX
    } else {
        (1u64 << l) - 1
    }
}

#[inline(always)]
fn lowest<E: Element>() -> E {
    if TypeId::of::<E>() == TypeId::of::<f32>() || TypeId::of::<E>() == TypeId::of::<f64>() {
        E::from_f64(f64::NEG_INFINITY)
    } else {
        E::ZERO
    }
}

/// Dispatch the block to the operation's epilogue.
#[inline(always)]
unsafe fn epilogue<A: crate::arch::Arch, S: crate::arch::SortVec>(job: &Job, blk: &Blk<S>) {
    let n = job.n;
    let lanes = blk.lanes;
    let mut vals = [0f64; MAXL];
    let ms = lanes_usize(blk.m);
    match &job.kern {
        Kern::Select { ranks } => {
            let fill = fill_value(job);
            for (j, &r) in ranks.iter().enumerate() {
                lanes_f64(*blk.buf.add(r), &mut vals);
                if blk.any_missing {
                    for i in 0..lanes {
                        if r >= ms[i] {
                            vals[i] = fill;
                        }
                    }
                }
                put(&blk.outs[j], blk.p, &vals, lanes);
            }
        }
        Kern::Median { omit } => {
            // np.nanmedian on 2-D+ input goes through np.ma.median, which averages the
            // middle value with itself for odd counts: (x + x) / 2 (inf near FLT_MAX).
            if !blk.any_missing || !omit {
                let mut lo = [S::E::ZERO; MAXL];
                let mut hi = [S::E::ZERO; MAXL];
                (*blk.buf.add((n - 1) / 2)).store(lo.as_mut_ptr());
                (*blk.buf.add(n / 2)).store(hi.as_mut_ptr());
                if n % 2 == 1 && !omit {
                    for i in 0..S::L {
                        vals[i] = hi[i].to_f64();
                    }
                } else {
                    for i in 0..S::L {
                        vals[i] = np_mid(lo[i], hi[i]);
                    }
                }
                if blk.any_missing {
                    for i in 0..lanes {
                        if ms[i] < n {
                            vals[i] = f64::NAN;
                        }
                    }
                }
            } else {
                for i in 0..lanes {
                    let m = ms[i];
                    // (m - 1) / 2 == m / 2 for odd m: np.ma.median's (x + x) / 2
                    vals[i] =
                        if m == 0 { f64::NAN } else { np_mid(at(blk.buf, (m - 1) / 2, i), at(blk.buf, m / 2, i)) };
                }
            }
            put(&blk.outs[0], blk.p, &vals, lanes);
        }
        Kern::Quantile { tabs, lerp, omit } => {
            for (j, tab) in tabs.iter().enumerate() {
                if !blk.any_missing || !omit {
                    match tab[n] {
                        QIdx::Take(i0) => lanes_f64(*blk.buf.add(i0), &mut vals),
                        QIdx::Lerp { i0, i1, t } => {
                            let mut a = [0f64; MAXL];
                            let mut b = [0f64; MAXL];
                            lanes_f64(*blk.buf.add(i0), &mut a);
                            lanes_f64(*blk.buf.add(i1), &mut b);
                            lerp_lanes(lerp, &a, &b, t, &mut vals, S::L);
                        }
                        QIdx::Empty => vals = [f64::NAN; MAXL],
                    }
                    if blk.any_missing {
                        for i in 0..lanes {
                            if ms[i] < n {
                                vals[i] = f64::NAN;
                            }
                        }
                    }
                } else {
                    // Per-lane ranks and weights: gather lane by lane, interpolate as a block.
                    let mut a = [0f64; MAXL];
                    let mut b = [0f64; MAXL];
                    let mut t = [0f64; MAXL];
                    let mut special = 0u64; // lanes that take a value as is (or are empty)
                    for i in 0..lanes {
                        match tab[ms[i]] {
                            QIdx::Take(i0) => {
                                a[i] = at(blk.buf, i0, i).to_f64();
                                special |= 1 << i;
                            }
                            QIdx::Lerp { i0, i1, t: w } => {
                                a[i] = at(blk.buf, i0, i).to_f64();
                                b[i] = at(blk.buf, i1, i).to_f64();
                                t[i] = w;
                            }
                            QIdx::Empty => {
                                a[i] = f64::NAN;
                                special |= 1 << i;
                            }
                        }
                    }
                    lerp_lanes_var(lerp, &a, &b, &t, &mut vals, S::L);
                    if special != 0 {
                        for i in 0..lanes {
                            if (special >> i) & 1 == 1 {
                                vals[i] = a[i];
                            }
                        }
                    }
                }
                put(&blk.outs[j], blk.p, &vals, lanes);
            }
        }
        Kern::Iqr { tlo, thi, sp, scale, omit } => {
            let f32m = sp.f32;
            for i in 0..lanes {
                let m = if *omit { ms[i] } else { n };
                if m == 0 || (!*omit && ms[i] < n) {
                    vals[i] = f64::NAN;
                    continue;
                }
                let (j0, j1, g0) = tlo[m];
                let (k0, k1, g1) = thi[m];
                let q0 = sp.eval(at(blk.buf, j0, i).to_f64(), at(blk.buf, j1, i).to_f64(), g0);
                let q1 = sp.eval(at(blk.buf, k0, i).to_f64(), at(blk.buf, k1, i).to_f64(), g1);
                vals[i] = if f32m {
                    let mut d = (q1 as f32) - (q0 as f32);
                    if *scale != 1.0 {
                        d /= *scale as f32;
                    }
                    d as f64
                } else {
                    let mut d = q1 - q0;
                    if *scale != 1.0 {
                        d /= *scale;
                    }
                    d
                };
            }
            put(&blk.outs[0], blk.p, &vals, lanes);
        }
        Kern::Mad { scale, omit } => {
            if job.sdt == DType::F32 {
                mad::<S, A::F32>(job, blk, *scale, *omit, &ms, &mut vals);
            } else {
                mad::<S, A::F64>(job, blk, *scale, *omit, &ms, &mut vals);
            }
            put(&blk.outs[0], blk.p, &vals, lanes);
        }
        Kern::TrimMean { lo, hi } => {
            let mut acc = [0f64; MAXL];
            let mut tmp = [0f64; MAXL];
            for k in *lo..*hi {
                lanes_f64(*blk.buf.add(k), &mut tmp);
                for i in 0..S::L {
                    acc[i] += tmp[i];
                }
            }
            let c = (*hi - *lo) as f64;
            for i in 0..S::L {
                vals[i] = acc[i] / c;
            }
            if blk.any_missing || hi == lo {
                for i in 0..lanes {
                    if *hi > ms[i] || hi == lo {
                        vals[i] = f64::NAN;
                    }
                }
            }
            put(&blk.outs[0], blk.p, &vals, lanes);
        }
        Kern::ArgRank { rank, omit } => {
            argrank::<S>(job, blk, *rank, *omit, &ms, &mut vals);
            put(&blk.outs[0], blk.p, &vals, lanes);
        }
        Kern::Clip(plan) => {
            if plan.f32 {
                crate::clip::clip_block::<S, A::F32>(job, plan, blk, &ms);
            } else {
                crate::clip::clip_block::<S, A::F64>(job, plan, blk, &ms);
            }
        }
        Kern::MinMax { .. } | Kern::Arg { .. } => unreachable!(),
    }
}

/// NumPy's `_lerp` for a whole block with a uniform weight `t`. Each arm is a straight loop
/// so the compiler vectorizes it.
#[inline(always)]
fn lerp_lanes(r: &Lerp, a: &[f64; MAXL], b: &[f64; MAXL], t: f64, out: &mut [f64; MAXL], l: usize) {
    let weak = matches!(r.form, crate::spec::QForm::Weak | crate::spec::QForm::WeakInt);
    let f32d = r.data == DType::F32;
    let f32c = r.compute == DType::F32;
    let f32s = r.sub_dtype == DType::F32;
    if !(t >= 0.5) {
        if f32c {
            let tc = t as f32;
            for i in 0..l {
                let d = if f32d { (b[i] as f32) - (a[i] as f32) } else { (b[i] - a[i]) as f32 };
                out[i] = ((a[i] as f32) + d * tc) as f64;
            }
        } else if f32d {
            for i in 0..l {
                let d = ((b[i] as f32) - (a[i] as f32)) as f64;
                out[i] = a[i] + d * t;
            }
        } else {
            for i in 0..l {
                out[i] = a[i] + (b[i] - a[i]) * t;
            }
        }
    } else if f32c {
        let omt = if weak { (1.0 - t) as f32 } else { 1.0f32 - t as f32 };
        for i in 0..l {
            let d = if f32d { (b[i] as f32) - (a[i] as f32) } else { (b[i] - a[i]) as f32 };
            let x = d * omt;
            out[i] = if f32s { ((b[i] as f32) - x) as f64 } else { b[i] - x as f64 };
        }
    } else {
        let omt = 1.0 - t;
        for i in 0..l {
            let d = if f32d { ((b[i] as f32) - (a[i] as f32)) as f64 } else { b[i] - a[i] };
            let x = d * omt;
            out[i] = if f32s { ((b[i] as f32) - (x as f32)) as f64 } else { b[i] - x };
        }
    }
}

/// `_lerp` with a weight per lane: both branches computed, then selected, so the loop has
/// no branches and vectorizes. Bit-identical to [`Lerp::eval`] lane by lane.
#[inline(always)]
fn lerp_lanes_var(r: &Lerp, a: &[f64; MAXL], b: &[f64; MAXL], t: &[f64; MAXL], out: &mut [f64; MAXL], l: usize) {
    let weak = matches!(r.form, crate::spec::QForm::Weak | crate::spec::QForm::WeakInt);
    let f32d = r.data == DType::F32;
    let f32s = r.sub_dtype == DType::F32;
    if r.compute == DType::F32 {
        for i in 0..l {
            let d = if f32d { (b[i] as f32) - (a[i] as f32) } else { (b[i] - a[i]) as f32 };
            let lo = (a[i] as f32) + d * (t[i] as f32);
            let omt = if weak { (1.0 - t[i]) as f32 } else { 1.0f32 - t[i] as f32 };
            let x = d * omt;
            let hi = if f32s { ((b[i] as f32) - x) as f64 } else { b[i] - x as f64 };
            out[i] = if t[i] >= 0.5 { hi } else { lo as f64 };
        }
    } else {
        for i in 0..l {
            let d = if f32d { ((b[i] as f32) - (a[i] as f32)) as f64 } else { b[i] - a[i] };
            let lo = a[i] + d * t[i];
            let x = d * (1.0 - t[i]);
            let hi = if f32s { ((b[i] as f32) - (x as f32)) as f64 } else { b[i] - x };
            out[i] = if t[i] >= 0.5 { hi } else { lo };
        }
    }
}

/// Load lanes `g*C::L ..` of vector `k` of a sort-dtype buffer as compute-dtype lanes.
#[inline(always)]
pub(crate) unsafe fn load_group<S: Vector, C: FVector>(src: *const S, k: usize, g: usize) -> C {
    let p = (src.add(k) as *const S::E).add(g * C::L);
    if TypeId::of::<S::E>() == TypeId::of::<C::E>() {
        C::load(p as *const C::E)
    } else {
        let mut t = [C::E::ZERO; MAXL];
        for i in 0..C::L {
            t[i] = C::E::from_f64((*p.add(i)).to_f64());
        }
        C::load(t.as_ptr())
    }
}

/// scipy.stats.median_abs_deviation: median(|x - median(x)|) / scale, in the data's float
/// dtype. The deviations of a sorted stack fall then rise, so a bitonic merger sorts them.
#[inline(always)]
unsafe fn mad<S: crate::arch::SortVec, C: crate::arch::SortVec + FVector>(
    job: &Job,
    blk: &Blk<S>,
    scale: f64,
    omit: bool,
    ms: &[usize; MAXL],
    vals: &mut [f64; MAXL],
) {
    let n = job.n;
    let groups = S::L / C::L;
    let d = blk.scratch as *mut C;
    for g in 0..groups {
        // per-lane median in the compute dtype
        let mut med = [C::E::ZERO; MAXL];
        let mut mcnt = [C::E::ZERO; MAXL];
        for i in 0..C::L {
            let lane = g * C::L + i;
            let m = if omit { ms[lane] } else { n };
            mcnt[i] = C::E::from_usize(m);
            med[i] = if m == 0 {
                C::E::ZERO
            } else if m % 2 == 1 {
                C::E::from_f64(at(blk.buf, m / 2, lane).to_f64())
            } else {
                C::E::from_f64(np_mid(at(blk.buf, (m - 1) / 2, lane), at(blk.buf, m / 2, lane)))
            };
        }
        let medv = C::load(med.as_ptr());
        let mv = C::load(mcnt.as_ptr());
        let inf = C::splat(C::E::MAX);
        for k in 0..n {
            let x: C = load_group::<S, C>(blk.buf, k, g);
            let dev = x.sub(medv).abs();
            let valid = C::splat(C::E::from_usize(k)).lt(mv);
            *d.add(k) = C::select(valid, dev, inf);
        }
        C::apply_net(d, &job.net_bitonic);
        for i in 0..C::L {
            let lane = g * C::L + i;
            if lane >= blk.lanes {
                break;
            }
            let m = if omit { ms[lane] } else { n };
            let bad = if omit { m == 0 } else { ms[lane] < n };
            if bad {
                vals[lane] = f64::NAN;
                continue;
            }
            let r = if m % 2 == 1 {
                at(d as *const C, m / 2, i).to_f64()
            } else {
                np_mid(at(d as *const C, (m - 1) / 2, i), at(d as *const C, m / 2, i))
            };
            vals[lane] = if TypeId::of::<C::E>() == TypeId::of::<f32>() {
                ((r as f32) / (scale as f32)) as f64
            } else {
                r / scale
            };
        }
    }
}

/// Which frame holds rank r in a stable argsort (NaN last): count values below the target,
/// then find the right occurrence of the target among equal values, in frame order.
#[inline(always)]
unsafe fn argrank<S: Vector>(
    job: &Job,
    blk: &Blk<S>,
    rank: ArgRank,
    omit: bool,
    ms: &[usize; MAXL],
    vals: &mut [f64; MAXL],
) {
    let n = job.n;
    let l = S::L;
    // Per-lane rank and target value.
    let mut rk = [S::E::ZERO; MAXL];
    let mut tv = [S::E::ZERO; MAXL];
    let mut nanreg = 0u64; // lanes whose rank lands in the missing tail
    let mut none = 0u64; // lanes with no answer (-1)
    for i in 0..l {
        let m = ms[i];
        let r = match rank {
            ArgRank::Fixed(r) => r,
            ArgRank::Median => {
                let c = if omit { m } else { n };
                if c == 0 {
                    none |= 1 << i;
                    0
                } else {
                    (c - 1) / 2
                }
            }
        };
        if r >= m {
            nanreg |= 1 << i;
            rk[i] = S::E::from_usize(r - m);
        } else {
            rk[i] = S::E::from_usize(r);
            tv[i] = at(blk.buf, r, i);
        }
    }
    let target = S::load(tv.as_ptr());
    // occurrences of target strictly below it: count valid raw < target
    let one = S::splat(S::E::from_usize(1));
    let zero = S::splat(S::E::ZERO);
    let mut below = zero;
    for k in 0..n {
        let x = *blk.raw.add(k);
        let mm = miss_mask(x, job.missing);
        below = S::select(x.lt(target).andnot(mm), below.add(one), below);
    }
    let mut want = [S::E::ZERO; MAXL];
    let mut bl = [S::E::ZERO; MAXL];
    below.store(bl.as_mut_ptr());
    for i in 0..l {
        want[i] = if (nanreg >> i) & 1 == 1 { rk[i] } else { S::E::from_usize(rk[i].to_usize() - bl[i].to_usize()) };
    }
    let wantv = S::load(want.as_ptr());
    let mut occ = zero;
    let mut idx = S::splat(S::E::from_usize(SENTINEL));
    // lanes in the NaN region look for missing frames; others for frames equal to target
    let mut nr = [S::E::ZERO; MAXL];
    for i in 0..l {
        nr[i] = S::E::from_usize(((nanreg >> i) & 1) as usize);
    }
    let in_nan = S::load(nr.as_ptr()).eq(one);
    for k in 0..n {
        let x = *blk.raw.add(k);
        let mm = miss_mask(x, job.missing);
        let hit_val = x.eq(target).andnot(mm);
        let hit = mm.and(in_nan).or(hit_val.andnot(in_nan));
        let take = hit.and(occ.eq(wantv));
        idx = S::select(take, S::splat(S::E::from_usize(k)), idx);
        occ = S::select(hit, occ.add(one), occ);
    }
    let ix = lanes_usize(idx);
    for i in 0..blk.lanes {
        vals[i] = if (none >> i) & 1 == 1 || ix[i] >= n { -1.0 } else { ix[i] as f64 };
    }
}
