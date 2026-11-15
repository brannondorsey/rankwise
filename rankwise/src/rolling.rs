//! Moving windows along the frame axis: every lane keeps its window sorted.
//!
//! A window of `w` frames is a sorted stack per lane, kept in an L1/L2-resident buffer. Moving
//! it by one frame deletes the value that leaves and inserts the one that enters in a single
//! branch-free pass over the window (a compare, a blend, a max and a min per rank), about `4w`
//! vector operations instead of a sort. Because the window stays fully sorted, every epilogue
//! of the static engine (median, quantiles, ranks, IQR, MAD, rejection) runs on it unchanged.
//!
//! Frames outside the stack and missing values (NaN, `nodata`) count as missing: they sit at
//! the top of the window as +inf and the per-lane count `m` skips them, as the `nan*`
//! reductions do; outputs with fewer than `min_count` valid values get the fill value.
//!
//! Pixels are processed in tiles. For each frame that enters, the tile's run of pixels is
//! copied into a small ring (one contiguous run per frame, so the frame axis streams
//! sequentially whatever the input layout), and every block of the tile advances its window.

use crate::engine::{
    epilogue, fill_value, lanes_f64, lanes_usize, miss_mask, norm, put, stage_frames, Blk, Job, Kern, OutRow, MAXL,
};
use crate::simd::{Element, Mask, Vector};

/// Moving windows along the frame axis (xarray / pandas `rolling` semantics).
///
/// Output `t` (for `t` in `0..count`) reduces frames `start + t·step .. start + t·step +
/// window`; frames outside the stack are missing. Missing values (outside the stack, NaN, or
/// `nodata`) are skipped, and an output with fewer than `min_count` valid values is NaN (or
/// `nodata` for integer outputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rolling {
    /// Frames per window (1..=[`crate::MAX_FRAMES`]).
    pub window: usize,
    /// First frame of output 0's window; negative starts before frame 0.
    pub start: isize,
    /// Frames between consecutive outputs' windows.
    pub step: usize,
    /// Number of outputs.
    pub count: usize,
    /// Fewest valid values that give a result (1..=`window`).
    pub min_count: usize,
}

impl Rolling {
    /// Trailing windows, one output per frame: output `t` covers frames `t - window + 1 ..= t`
    /// (`xarray.DataArray.rolling(dim=window)`, `pandas.DataFrame.rolling(window)`).
    pub fn trailing(window: usize, frames: usize) -> Rolling {
        Rolling { window, start: 1 - window as isize, step: 1, count: frames, min_count: window }
    }

    /// Centered windows, one output per frame: output `t` covers frames from
    /// `t - window / 2` (`rolling(..., center=True)`).
    pub fn centered(window: usize, frames: usize) -> Rolling {
        Rolling { window, start: -((window / 2) as isize), step: 1, count: frames, min_count: window }
    }

    /// Only the windows that fit inside the stack (`numpy.lib.stride_tricks.sliding_window_view`).
    pub fn valid(window: usize, frames: usize) -> Rolling {
        Rolling { window, start: 0, step: 1, count: (frames + 1).saturating_sub(window), min_count: window }
    }

    /// Fewest valid values that give a result (`min_periods`).
    pub fn min_count(mut self, c: usize) -> Rolling {
        self.min_count = c;
        self
    }

    /// Keep every `step`-th output (`rolling(..., step=step)`): the number of outputs shrinks
    /// to `ceil(count / step)` and windows move `step` frames at a time.
    pub fn step(mut self, step: usize) -> Rolling {
        if step > 0 {
            self.count = self.count.div_ceil(step);
        }
        self.step = step;
        self
    }
}

/// Copy frame `f`'s pixels `t0..t0 + cnt` into its ring slot.
#[inline(always)]
unsafe fn stage_one<S: Vector>(
    job: &Job,
    src: &[*const u8],
    f: usize,
    t0: usize,
    cnt: usize,
    ring: *mut S::E,
    ts: usize,
) {
    let slot = f % (job.n + 1);
    let dst = core::slice::from_raw_parts_mut(ring.add(slot * ts), ts);
    stage_frames::<S>(job, &src[f..f + 1], job.space.in_ps, t0, cnt, dst, ts);
}

/// Process one unit of work (one row, pixels `p0..p1`) for every output window.
#[inline(always)]
pub(crate) unsafe fn run_unit<A: crate::arch::Arch, S: crate::arch::SortVec>(job: &Job, u: crate::engine::Unit) {
    let roll = job.rolling.as_ref().unwrap();
    let w = job.n;
    let nf = job.frames.len() as isize;
    let l = S::L;
    let esz = core::mem::size_of::<S::E>();
    let mut out_off = vec![0isize; job.outs.len()];
    let (in_off, _) = job.space.row_offsets(u.row, &mut out_off);
    let src: Vec<*const u8> = job.frames.iter().map(|&f| (f as isize + in_off) as *const u8).collect();
    let base_outs: Vec<OutRow> = job
        .outs
        .iter()
        .enumerate()
        .map(|(j, &(ptr, dt))| OutRow {
            ptr: if ptr == 0 { core::ptr::null_mut() } else { (ptr as isize + out_off[j]) as *mut u8 },
            dtype: dt,
            ps: job.space.out_ps[j],
        })
        .collect();
    let mut outs = base_outs.clone();

    // Tile: the windows (w vectors per block) and the ring (w + 1 frames) in about the
    // staging budget.
    let npx = u.p1 - u.p0;
    let budget = crate::engine::tile_budget();
    let tile_px = ((budget / ((2 * w + 1) * esz)).max(l) / l * l).min(npx.div_ceil(l) * l).max(l);
    let nb_max = tile_px / l;
    let ts = tile_px;
    let mut ring: Vec<S> = vec![S::splat(S::E::ZERO); (w + 1) * ts / l];
    let ring_e = ring.as_mut_ptr() as *mut S::E;
    let mut win: Vec<S> = vec![S::splat(S::E::MAX); nb_max * w];
    let mut cnt_v: Vec<S> = vec![S::splat(S::E::ZERO); nb_max];
    // Values that count toward min_count: the non-NaN ones, as xarray counts them. Only
    // rejection (which also treats ±inf as missing) needs a count of its own.
    let nonfinite = job.missing == crate::engine::Missing::NonFinite;
    let mut gcnt_v: Vec<S> = vec![S::splat(S::E::ZERO); if nonfinite { nb_max } else { 0 }];
    let mut scratch_v: Vec<u8> = vec![0u8; 3 * w * 64 + 64];
    let scratch = {
        let p = scratch_v.as_mut_ptr();
        p.add(p.align_offset(64))
    };
    let scale_off: Option<(Vec<S::E>, Vec<S::E>)> = job
        .norm
        .as_ref()
        .map(|(s, o)| (s.iter().map(|&x| S::E::from_f64(x)).collect(), o.iter().map(|&x| S::E::from_f64(x)).collect()));
    let big = S::splat(S::E::MAX);
    let one = S::splat(S::E::from_usize(1));
    let zero = S::splat(S::E::ZERO);
    let wv = S::splat(S::E::from_usize(w));
    let mcv = S::splat(S::E::from_usize(roll.min_count));
    let minmax = matches!(job.kern, Kern::MinMax { .. });
    // Re-sorting a window costs about 3 operations per comparator; moving it by one frame
    // about 6 per rank.
    let resort = roll.step * 6 * w > 3 * job.net_full.comparators() + 2 * w;
    // NaN in float outputs; nodata (or 0) in integer ones
    let fill = fill_value(job);
    // outputs that carry values (a clip's reject counts stay as computed)
    let value_outs = if matches!(job.kern, Kern::Clip(_)) { 1 } else { job.outs.len() };

    // Frame f's value for block b of the tile, missing values as +inf, and its validity.
    macro_rules! value {
        ($f:expr, $b:expr) => {{
            let f: isize = $f;
            if f < 0 || f >= nf {
                (big, S::mask_none(), S::mask_none())
            } else {
                let f = f as usize;
                let mut x = S::load(ring_e.add((f % (w + 1)) * ts + $b * l));
                if let Some((s, o)) = &scale_off {
                    x = norm::<S>(x, *s.get_unchecked(f), *o.get_unchecked(f));
                }
                let mm = miss_mask(x, job.missing);
                let counted = if nonfinite { x.nan_mask().not() } else { mm.not() };
                (S::select(mm, big, x), mm.not(), counted)
            }
        }};
    }

    let mut t0 = u.p0;
    while t0 < u.p1 {
        let t1 = (t0 + tile_px).min(u.p1);
        let cnt = t1 - t0;
        let nb = cnt.div_ceil(l);
        // Frames are needed in increasing order and each one stays in the ring until w + 1
        // later frames have been staged, so a window's frames are always there.
        let mut staged: isize = 0; // frames below this were staged (or skipped)
        macro_rules! need {
            ($f:expr) => {{
                let f: isize = $f;
                if f >= staged && f < nf {
                    stage_one::<S>(job, &src, f as usize, t0, cnt, ring_e, ts);
                    staged = f + 1;
                }
            }};
        }
        let mut lo = roll.start; // first frame of the current window
        for t in 0..roll.count {
            let first = roll.start + (t * roll.step) as isize;
            if t == 0 || resort {
                // sort the window from scratch
                for f in first..first + w as isize {
                    need!(f);
                }
                for b in 0..nb {
                    let wb = win.as_mut_ptr().add(b * w);
                    let (mut m, mut g) = (zero, zero);
                    for k in 0..w {
                        let (x, ok, counted) = value!(first + k as isize, b);
                        *wb.add(k) = x;
                        m = S::select(ok, m.add(one), m);
                        g = S::select(counted, g.add(one), g);
                    }
                    S::apply_net(wb, &job.net_full);
                    *cnt_v.as_mut_ptr().add(b) = m;
                    if nonfinite {
                        *gcnt_v.as_mut_ptr().add(b) = g;
                    }
                }
            } else {
                // slide: delete the frame that leaves, insert the one that enters
                for d in 0..roll.step as isize {
                    let fin = lo + w as isize + d;
                    need!(fin);
                    let fout = lo + d;
                    for b in 0..nb {
                        let (xi, oki, gi) = value!(fin, b);
                        let (xo, oko, go) = value!(fout, b);
                        let wb = win.as_mut_ptr().add(b * w);
                        // e[j] = min(max(d[j-1], xi), d[j]) with d the window without xo:
                        // d[j] = s[j] if s[j] < xo else s[j + 1]
                        let mut prev_d = S::splat(lowest::<S::E>());
                        let mut s_j = *wb;
                        for j in 0..w {
                            let s_next = if j + 1 < w { *wb.add(j + 1) } else { big };
                            let d_j = S::select(s_j.lt(xo), s_j, s_next);
                            *wb.add(j) = prev_d.max(xi).min(d_j);
                            prev_d = d_j;
                            s_j = s_next;
                        }
                        let c = cnt_v.as_mut_ptr().add(b);
                        let mut m = *c;
                        m = S::select(oki, m.add(one), m);
                        m = S::select(oko, m.sub(one), m);
                        *c = m;
                        if nonfinite {
                            let c = gcnt_v.as_mut_ptr().add(b);
                            let mut g = *c;
                            g = S::select(gi, g.add(one), g);
                            g = S::select(go, g.sub(one), g);
                            *c = g;
                        }
                    }
                }
            }
            lo = first;
            for (j, o) in outs.iter_mut().enumerate() {
                if !o.ptr.is_null() {
                    o.ptr = base_outs[j].ptr.offset(t as isize * job.rolling_tstrides[j]);
                }
            }
            for b in 0..nb {
                let lanes = (cnt - b * l).min(l);
                let p = t0 + b * l;
                let wb = win.as_mut_ptr().add(b * w);
                let m = *cnt_v.as_ptr().add(b);
                let blk = Blk {
                    buf: wb,
                    raw: core::ptr::null(),
                    m,
                    any_missing: m.lt(wv).any(),
                    lanes,
                    p,
                    outs: &outs,
                    mask: &[],
                    mask_ps: 0,
                    scratch,
                };
                if minmax {
                    window_minmax::<S>(job, &blk);
                } else {
                    epilogue::<A, S>(job, &blk);
                }
                let g = if nonfinite { *gcnt_v.as_ptr().add(b) } else { m };
                let gated = g.lt(mcv).bits() & if lanes >= 64 { u64::MAX } else { (1u64 << lanes) - 1 };
                if gated != 0 {
                    for o in outs.iter().take(value_outs) {
                        let v = if o.dtype.is_float() { f64::NAN } else { fill };
                        put_lanes(o, p, v, gated, lanes);
                    }
                }
            }
        }
        t0 = t1;
    }
}

/// Write `v` to the lanes of `bits` at pixel `p`.
#[inline(always)]
unsafe fn put_lanes(o: &OutRow, p: usize, v: f64, bits: u64, lanes: usize) {
    if o.ptr.is_null() {
        return;
    }
    let cur = [v; MAXL];
    for i in 0..lanes {
        if (bits >> i) & 1 == 1 {
            put(o, p + i, &cur, 1);
        }
    }
}

#[inline(always)]
fn lowest<E: Element>() -> E {
    if core::any::TypeId::of::<E>() == core::any::TypeId::of::<f32>()
        || core::any::TypeId::of::<E>() == core::any::TypeId::of::<f64>()
    {
        E::from_f64(f64::NEG_INFINITY)
    } else {
        E::ZERO
    }
}

/// min / max / range of a sorted window's valid values.
#[inline(always)]
unsafe fn window_minmax<S: Vector>(job: &Job, blk: &Blk<S>) {
    let Kern::MinMax { which, .. } = job.kern else { unreachable!() };
    let ms = lanes_usize(blk.m);
    let mut hi = [S::E::ZERO; MAXL];
    for i in 0..S::L {
        hi[i] = if ms[i] > 0 { crate::engine::at(blk.buf, ms[i] - 1, i) } else { S::E::ZERO };
    }
    let lo = *blk.buf;
    let hi = S::load(hi.as_ptr());
    let r = match which {
        0 => lo,
        1 => hi,
        _ => hi.sub(lo),
    };
    let mut vals = [0f64; MAXL];
    lanes_f64(r, &mut vals);
    let f = if blk.outs[0].dtype.is_float() { f64::NAN } else { fill_value(job) };
    for i in 0..blk.lanes {
        if ms[i] == 0 {
            vals[i] = f;
        }
    }
    put(&blk.outs[0], blk.p, &vals, blk.lanes);
}
