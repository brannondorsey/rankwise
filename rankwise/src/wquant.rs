//! Weighted quantiles: NumPy's `method="inverted_cdf"` with `weights=`.
//!
//! Weights belong to frames, so frame identity has to ride through the network: every value
//! carries its frame index, and compare-exchanges order `(value, index)` pairs, which is a
//! stable sort (ties in frame order) whatever network runs, so every ISA sums the weights in
//! the same order. NumPy then needs, per pixel, the cumulative weights in sorted order (summed
//! one by one in float64), divided by their total and cast to the dtype of `q`; the quantile
//! is the first value whose normalized cumulative weight reaches `q`. Missing values (NaN or
//! `nodata`, omitted) carry index `k + n`, so they sort after every valid value (a real +inf
//! included) and weigh nothing: their cumulative weights repeat the last valid one.

use crate::engine::{at, put, Job, Kern, OutRow, MAXL};
use crate::netgen::Comparator;
use crate::simd::{Element, FVector, Mask, Vector};
use std::sync::atomic::Ordering;

/// Compare-exchange `(value, index)` pairs: after it, wire `i` holds the smaller pair.
#[inline(always)]
pub(crate) unsafe fn pair_sort<S: Vector>(b: *mut S, ix: *mut S, net: &[Comparator]) {
    for &(i, j) in net {
        let (i, j) = (i as usize, j as usize);
        let (x, y) = (*b.add(i), *b.add(j));
        let (xi, yi) = (*ix.add(i), *ix.add(j));
        let swap = y.lt(x).or(y.eq(x).and(yi.lt(xi)));
        *b.add(i) = S::select(swap, y, x);
        *b.add(j) = S::select(swap, x, y);
        *ix.add(i) = S::select(swap, yi, xi);
        *ix.add(j) = S::select(swap, xi, yi);
    }
}

/// Where a block's weights come from.
pub(crate) enum Weights<'a> {
    /// One weight per frame.
    Frames(&'a [f64]),
    /// A weight map staged as float64: frame `k`'s weight for lane `i` of the block is at
    /// `tile[k * ts + off + i]`.
    Map { tile: *const f64, ts: usize, off: usize },
}

/// Scratch a weighted block needs: the index payload (`n` vectors) and every lane's
/// cumulative weights (`n * L`, rank-major).
pub(crate) struct Scratch<S: Vector> {
    pub ix: Vec<S>,
    pub cs: Vec<f64>,
    /// Per quantile: search thresholds, their counts, and the values written.
    pub thr: Vec<[f64; MAXL]>,
    pub cnt: Vec<[f64; MAXL]>,
    pub vals: Vec<[f64; MAXL]>,
}

impl<S: Vector> Scratch<S> {
    pub fn new(n: usize, nq: usize) -> Self {
        Scratch {
            ix: vec![unsafe { S::splat(S::E::ZERO) }; n],
            cs: vec![0.0; n * S::L],
            thr: vec![[0.0; MAXL]; nq],
            cnt: vec![[0.0; MAXL]; nq],
            vals: vec![[0.0; MAXL]; nq],
        }
    }
}

/// Sort one block (`b`, missing values already +inf; `raw`, the values as loaded) with frame
/// indices, then find every lane's weighted quantiles and write them.
#[inline(always)]
pub(crate) unsafe fn block<S: Vector, C: FVector<E = f64>>(
    job: &Job,
    b: *mut S,
    raw: *const S,
    ms: &[usize; MAXL],
    any_missing: bool,
    lanes: usize,
    p: usize,
    outs: &[OutRow],
    w: &Weights,
    sc: &mut Scratch<S>,
) {
    let Kern::WQuantile { qs, f32cmp, omit } = &job.kern else { unreachable!() };
    let (omit, f32cmp) = (*omit, *f32cmp);
    let n = job.n;
    let ix = sc.ix.as_mut_ptr();
    // Missing values of an omitted slice sort after everything (index k + n).
    for k in 0..n {
        let kv = S::splat(S::E::from_usize(k));
        *ix.add(k) = if omit && any_missing {
            let mm = crate::engine::miss_mask(*raw.add(k), job.missing);
            S::select(mm, S::splat(S::E::from_usize(k + n)), kv)
        } else {
            kv
        };
    }
    pair_sort(b, ix, job.net_pairs.as_slice());

    let fill = crate::engine::fill_value(job);
    let nq = qs.len();
    let l = S::L;
    let cs = sc.cs.as_mut_ptr();
    // Cumulative weights in sorted order, one by one in float64 (np.cumsum). Rank-major with
    // the lanes innermost, so the lanes' add chains run side by side.
    let mut run = [0f64; MAXL];
    let mut ids = [S::E::ZERO; MAXL];
    macro_rules! cumsum {
        ($weight:expr) => {
            for k in 0..n {
                (*ix.add(k)).store(ids.as_mut_ptr());
                let row = cs.add(k * l);
                for i in 0..l {
                    let id = ids[i].to_usize();
                    run[i] += if id >= n { 0.0 } else { $weight(id, i) };
                    *row.add(i) = run[i];
                }
            }
        };
    }
    match w {
        Weights::Frames(f) => cumsum!(|id: usize, _i: usize| *f.get_unchecked(id)),
        Weights::Map { tile, ts, off } => {
            let base = tile.add(*off);
            cumsum!(|id: usize, i: usize| *base.add(id * ts + i))
        }
    }
    // Per lane: what to write when there is no answer, else the search thresholds. No
    // cumulative weight below q·total·(1 - 2^-20) can reach q once divided by the total and
    // cast (each rounds by at most 2^-24 relative), so the first one at or above it is where
    // the exact search starts. A subnormal product may round up by more: start from 0 then.
    let thr_all = &mut sc.thr;
    thr_all.iter_mut().for_each(|t| *t = [0f64; MAXL]);
    let mut status = [0u8; MAXL]; // 0 search, 1 fill, 2 NaN
    for lane in 0..lanes {
        let m = ms[lane];
        let total = run[lane];
        status[lane] = if omit && m == 0 {
            // np.nanquantile: an all-NaN slice gives NaN before any weight is looked at.
            1
        } else if !(total.is_finite() && total > 0.0) {
            // NumPy raises "Weights included NaN, inf or were all zero." when
            // cdf[-1] / cdf[-1] is NaN, before it looks at NaN data.
            job.bad_weights.store(true, Ordering::Relaxed);
            2
        } else if !omit && m < n {
            2
        } else {
            0
        };
        if status[lane] == 0 {
            for (j, &q) in qs.iter().enumerate() {
                let t = q * total * (1.0 - 1.0 / 1048576.0);
                thr_all[j][lane] = if t < f64::MIN_POSITIVE { 0.0 } else { t };
            }
        }
    }
    // First rank at or above each threshold: cumulative weights never decrease, so it is the
    // count of those below, counted for C::L lanes at a time.
    let groups = l / C::L;
    let (one, zero) = (C::splat(1.0), C::splat(0.0));
    let cnt = &mut sc.cnt;
    for j in 0..nq {
        for g in 0..groups {
            let t = C::load(thr_all[j].as_ptr().add(g * C::L));
            let mut c = zero;
            for k in 0..n {
                let x = C::load(cs.add(k * l + g * C::L));
                c = c.add(C::select(x.lt(t), one, zero));
            }
            c.store(cnt[j].as_mut_ptr().add(g * C::L));
        }
    }
    let at_cs = |k: usize, lane: usize| *cs.add(k * l + lane);
    let vals = &mut sc.vals;
    for lane in 0..lanes {
        match status[lane] {
            1 => vals.iter_mut().for_each(|v| v[lane] = fill),
            2 => vals.iter_mut().for_each(|v| v[lane] = f64::NAN),
            _ => {
                let total = run[lane];
                let count = if omit { ms[lane] } else { n };
                // cdf[k] = cs[k] / total cast to q's dtype, zeros replaced by -1: it never
                // decreases, so np.searchsorted(cdf, q, side="left") is the first k reaching q.
                let reached = |x: f64, q: f64| -> bool {
                    let r = x / total;
                    let r = if f32cmp { (r as f32) as f64 } else { r };
                    let r = if r == 0.0 { -1.0 } else { r };
                    r >= q
                };
                for (j, &q) in qs.iter().enumerate() {
                    let mut k = cnt[j][lane] as usize;
                    while k < n {
                        let x = at_cs(k, lane);
                        if reached(x, q) {
                            break;
                        }
                        k += 1;
                        while k < n && at_cs(k, lane) == x {
                            k += 1; // equal sums (zero weights) reach q together
                        }
                    }
                    vals[j][lane] = at(b, k.min(count - 1), lane).to_f64();
                }
            }
        }
    }
    for (j, v) in sc.vals.iter().enumerate() {
        put(&outs[j], p, v, lanes);
    }
}
