//! Outlier rejection on sorted stacks.
//!
//! After the network, every lane holds one pixel's sorted stack. Rejection keeps a pair of
//! per-lane rank bounds `[lo, hi)` and moves them inward with masked passes over the
//! registers, so all lanes iterate together and nothing branches on pixel values. Center,
//! spread and bounds run in the compute dtype `C` (f64 by default, as astropy does).

use crate::arch::SortVec;
use crate::engine::{at, lanes_f64, load_group, miss_mask, put, Blk, Job, MAXL};
use crate::simd::{Element, FVector, Mask, Vector};
use crate::spec::{Center, Combine, Convention, Rejection, Spread, WinsorInit};

#[derive(Debug, Clone)]
pub(crate) struct ClipPlan {
    pub rej: Rejection,
    pub combine: Combine,
    /// Center/spread/bounds in f32 (fast mode) instead of f64.
    pub f32: bool,
    pub weights: Option<Vec<f64>>,
    pub has_mask: bool,
}

impl ClipPlan {
    pub fn needs_raw(&self) -> bool {
        self.weights.is_some() || self.has_mask
    }
}

#[inline(always)]
unsafe fn sp<C: Vector>(x: f64) -> C {
    C::splat(C::E::from_f64(x))
}


/// Median of sorted ranks `[lo, hi)` per lane, astropy's way: `0.5 * (upper + lower)`.
/// Gathers the two middle values lane by lane (cheaper than a masked pass over n).
#[inline(always)]
unsafe fn median_range<C: FVector>(cb: *const C, _n: usize, lo: C, hi: C) -> C {
    let mut a = [C::E::ZERO; MAXL];
    let mut b = [C::E::ZERO; MAXL];
    lo.store(a.as_mut_ptr());
    hi.store(b.as_mut_ptr());
    let mut lower = [C::E::ZERO; MAXL];
    let mut upper = [C::E::ZERO; MAXL];
    let mut odd = [C::E::ZERO; MAXL];
    for i in 0..C::L {
        let (l, h) = (a[i].to_usize(), b[i].to_usize());
        if h > l {
            let s = l + h;
            lower[i] = at(cb, (s - 1) / 2, i);
            upper[i] = at(cb, s / 2, i);
            odd[i] = C::E::from_usize((h - l) % 2);
        }
    }
    let (ma, mb) = (C::load(lower.as_ptr()), C::load(upper.as_ptr()));
    let odd = C::load(odd.as_ptr()).eq(sp::<C>(1.0));
    C::select(odd, mb, sp::<C>(0.5).mul(mb.add(ma)))
}

/// Per lane: how many of the sorted values in `[lo, hi)` satisfy `low(x)` scanning up from
/// `lo`, and how many satisfy `high(x)` scanning down from `hi`. Rejections are few, so the
/// scans stop after a step or two.
#[inline(always)]
unsafe fn end_scan<C: FVector>(
    cb: *const C,
    lo: C,
    hi: C,
    low: impl Fn(usize, C::E) -> bool,
    high: impl Fn(usize, C::E) -> bool,
) -> (C, C) {
    let mut a = [C::E::ZERO; MAXL];
    let mut b = [C::E::ZERO; MAXL];
    lo.store(a.as_mut_ptr());
    hi.store(b.as_mut_ptr());
    let mut nl = [C::E::ZERO; MAXL];
    let mut nh = [C::E::ZERO; MAXL];
    for i in 0..C::L {
        let (l, h) = (a[i].to_usize(), b[i].to_usize());
        let mut x = l;
        while x < h && low(i, at(cb, x, i)) {
            x += 1;
        }
        let mut y = h;
        while y > x && high(i, at(cb, y - 1, i)) {
            y -= 1;
        }
        nl[i] = C::E::from_usize(x - l);
        nh[i] = C::E::from_usize(h - y);
    }
    (C::load(nl.as_ptr()), C::load(nh.as_ptr()))
}

/// First and last+1 ranks of the values within `[lb, ub]` over the whole valid range
/// `[0, m)`, found by walking from the last iteration's `[lo, hi)` (usually zero steps).
#[inline(always)]
unsafe fn final_range<C: FVector>(cb: *const C, m: C, lo: C, hi: C, lb: C, ub: C) -> (C, C) {
    let mut ma = [C::E::ZERO; MAXL];
    let mut loa = [C::E::ZERO; MAXL];
    let mut hia = [C::E::ZERO; MAXL];
    let mut la = [C::E::ZERO; MAXL];
    let mut ua = [C::E::ZERO; MAXL];
    m.store(ma.as_mut_ptr());
    lo.store(loa.as_mut_ptr());
    hi.store(hia.as_mut_ptr());
    lb.store(la.as_mut_ptr());
    ub.store(ua.as_mut_ptr());
    let mut fl = [C::E::ZERO; MAXL];
    let mut fh = [C::E::ZERO; MAXL];
    for i in 0..C::L {
        let m = ma[i].to_usize();
        let (l, u) = (la[i], ua[i]);
        if l.is_nan() || u.is_nan() {
            continue; // no bounds: nothing kept
        }
        // lower_bound(l): first rank with x >= l
        let mut a = loa[i].to_usize().min(m);
        while a > 0 && !(at(cb, a - 1, i) < l) {
            a -= 1;
        }
        while a < m && at(cb, a, i) < l {
            a += 1;
        }
        // upper_bound(u): first rank with x > u
        let mut b = hia[i].to_usize().clamp(a, m);
        while b < m && !(at(cb, b, i) > u) {
            b += 1;
        }
        while b > a && at(cb, b - 1, i) > u {
            b -= 1;
        }
        fl[i] = C::E::from_usize(a);
        fh[i] = C::E::from_usize(b);
    }
    (C::load(fl.as_ptr()), C::load(fh.as_ptr()))
}

#[inline(always)]
unsafe fn in_range<C: Vector>(k: usize, lo: C, hi: C) -> C::M {
    let kk = sp::<C>(k as f64);
    kk.lt(hi).andnot(kk.lt(lo))
}

/// Whether every lane's range is the whole stack `[0, n)` (no masks needed).
#[inline(always)]
unsafe fn is_full<C: Vector>(n: usize, lo: C, hi: C) -> bool {
    !lo.eq(sp::<C>(0.0)).and(hi.eq(sp::<C>(n as f64))).not().any()
}

/// Sum over ranks `[lo, hi)`. Four accumulators keyed by `k % 4`, combined as
/// `(a0 + a1) + (a2 + a3)`: the order is fixed per lane, so every ISA gets the same bits,
/// and the add chains are short enough to keep the FP ports busy.
#[inline(always)]
pub(crate) unsafe fn sum_range<C: FVector>(cb: *const C, n: usize, lo: C, hi: C) -> C {
    let z = sp::<C>(0.0);
    if is_full(n, lo, hi) {
        acc4(n, |k| *cb.add(k))
    } else {
        acc4(n, |k| {
            let kk = sp::<C>(k as f64);
            C::select(kk.lt(hi).andnot(kk.lt(lo)), *cb.add(k), z)
        })
    }
}

/// `sum_k f(k)` with four accumulators keyed by `k % 4`, combined `(a0 + a1) + (a2 + a3)`.
#[inline(always)]
unsafe fn acc4<C: FVector>(n: usize, f: impl Fn(usize) -> C) -> C {
    let z = sp::<C>(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let mut k = 0;
    while k + 4 <= n {
        a0 = a0.add(f(k));
        a1 = a1.add(f(k + 1));
        a2 = a2.add(f(k + 2));
        a3 = a3.add(f(k + 3));
        k += 4;
    }
    if k < n {
        a0 = a0.add(f(k));
    }
    if k + 1 < n {
        a1 = a1.add(f(k + 1));
    }
    if k + 2 < n {
        a2 = a2.add(f(k + 2));
    }
    a0.add(a1).add(a2.add(a3))
}

/// Standard deviation over ranks `[lo, hi)` (two-pass), dividing by `cnt - ddof`.
#[inline(always)]
unsafe fn std_range<C: FVector>(cb: *const C, n: usize, lo: C, hi: C, mean: C, cnt: C, ddof: f64) -> C {
    let z = sp::<C>(0.0);
    let ss = if is_full(n, lo, hi) {
        acc4(n, |k| {
            let d = (*cb.add(k)).sub(mean);
            d.mul(d)
        })
    } else {
        acc4(n, |k| {
            let d = (*cb.add(k)).sub(mean);
            let kk = sp::<C>(k as f64);
            C::select(kk.lt(hi).andnot(kk.lt(lo)), d.mul(d), z)
        })
    };
    ss.div(cnt.sub(sp::<C>(ddof))).sqrt()
}

/// Median of |x - med| over ranks `[lo, hi)`: the deviations of a sorted range form a valley,
/// which the bitonic merger sorts.
#[inline(always)]
unsafe fn mad_range<C: FVector + SortVec>(
    cb: *const C,
    d: *mut C,
    n: usize,
    lo: C,
    hi: C,
    med: C,
    bitonic: &crate::net::NetPlan,
) -> C {
    let inf = C::splat(C::E::MAX);
    for k in 0..n {
        let dev = (*cb.add(k)).sub(med).abs();
        *d.add(k) = C::select(in_range(k, lo, hi), dev, inf);
    }
    C::apply_net(d, bitonic);
    let cnt = hi.sub(lo);
    median_range(d as *const C, n, sp::<C>(0.0), cnt)
}



/// astropy `sigma_clip` (fast path) on sorted lanes. Returns the final kept rank range.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn sigma<C: FVector + SortVec>(
    cb: *const C,
    d: *mut C,
    n: usize,
    m: C,
    low: f64,
    high: f64,
    max_iters: Option<u32>,
    center: Center,
    spread: Spread,
    conv: Convention,
    bitonic: &crate::net::NetPlan,
) -> (C, C) {
    let siril = conv != Convention::Astropy;
    let ddof = if siril { 1.0 } else { 0.0 };
    let zero = sp::<C>(0.0);
    let nan = sp::<C>(f64::NAN);
    let (vlow, vhigh) = (sp::<C>(low), sp::<C>(high));
    let mut lo = zero;
    let mut hi = m;
    let mut lb = nan;
    let mut ub = nan;
    let mut active = m.gt(zero);
    let need_med = center == Center::Median || spread == Spread::MadStd;
    let need_mean = center == Center::Mean || spread == Spread::Std;
    let mut it = 0u32;
    while active.any() {
        let cnt = hi.sub(lo);
        let med = if need_med { median_range(cb, n, lo, hi) } else { zero };
        let mean = if need_mean { sum_range(cb, n, lo, hi).div(cnt) } else { zero };
        let cen = if center == Center::Median { med } else { mean };
        let s = match spread {
            Spread::Std => std_range(cb, n, lo, hi, mean, cnt, ddof),
            Spread::MadStd => mad_range(cb, d, n, lo, hi, med, bitonic).mul(sp::<C>(1.482602218505602)),
        };
        let nlb = cen.sub(vlow.mul(s));
        let nub = cen.add(vhigh.mul(s));
        let (nl, nh) = if siril {
            // Siril: reject c - x > low*s, x - c > high*s
            let (mut ca, mut ka, mut kb) = ([C::E::ZERO; MAXL], [C::E::ZERO; MAXL], [C::E::ZERO; MAXL]);
            cen.store(ca.as_mut_ptr());
            vlow.mul(s).store(ka.as_mut_ptr());
            vhigh.mul(s).store(kb.as_mut_ptr());
            end_scan(cb, lo, hi, |i, x| sub_gt(ca[i], x, ka[i]), |i, x| sub_gt(x, ca[i], kb[i]))
        } else {
            let mut la = [C::E::ZERO; MAXL];
            let mut ua = [C::E::ZERO; MAXL];
            nlb.store(la.as_mut_ptr());
            nub.store(ua.as_mut_ptr());
            end_scan(cb, lo, hi, |i, x| x < la[i], |i, x| x > ua[i])
        };
        lb = C::select(active, nlb, lb);
        ub = C::select(active, nub, ub);
        let nlo = lo.add(nl);
        let nhi = hi.sub(nh);
        let changed = nl.gt(zero).or(nh.gt(zero));
        lo = C::select(active, nlo, lo);
        hi = C::select(active, nhi, hi);
        active = active.and(changed).and(hi.gt(lo));
        it += 1;
        if let Some(mx) = max_iters {
            if it >= mx {
                break;
            }
        }
    }
    if siril {
        return (lo, hi);
    }
    // astropy re-applies the last bounds to every finite value.
    final_range(cb, m, lo, hi, lb, ub)
}

/// `a - b > t`, evaluated in the compute dtype.
#[inline(always)]
fn sub_gt<E: Element>(a: E, b: E, t: E) -> bool {
    if core::any::TypeId::of::<E>() == core::any::TypeId::of::<f32>() {
        (a.to_f64() as f32) - (b.to_f64() as f32) > t.to_f64() as f32
    } else {
        a.to_f64() - b.to_f64() > t.to_f64()
    }
}

/// Rousseeuw–Croux Sn (without the consistency constant) of a sorted slice.
fn sn(x: &[f64]) -> f64 {
    let c = x.len();
    if c < 2 {
        return 0.0;
    }
    let mut meds = Vec::with_capacity(c);
    let k = c / 2; // high median of c distances (0-based)
    for i in 0..c {
        // merge distances to the left and right, ascending; self-distance 0 first
        let (mut l, mut r) = (i as isize - 1, i + 1);
        let mut v = 0.0;
        for _ in 0..k {
            let dl = if l >= 0 { x[i] - x[l as usize] } else { f64::INFINITY };
            let dr = if r < c { x[r] - x[i] } else { f64::INFINITY };
            if dl <= dr {
                v = dl;
                l -= 1;
            } else {
                v = dr;
                r += 1;
            }
        }
        meds.push(v);
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    meds[(c + 1) / 2 - 1]
}

/// Winsorized sigma clipping (Huber).
///
/// Siril convention (verified black-box): center = median of the range, spread starts as
/// the sample std; the Huber loop clamps to c ± 1.5 s and sets s = 1.134 × std(clamped)
/// until |Δs| <= 5e-4 s; reject `c - x > low·s` / `x - c > high·s`; repeat while more than
/// three values remain. PixInsight convention (PCL documentation): Sn-based start, first-pass
/// cutoff, center updated to the winsorized mean, ratio tests `(c - x)/s > low`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn winsorized<C: FVector>(
    cb: *const C,
    n: usize,
    m: C,
    low: f64,
    high: f64,
    max_iters: Option<u32>,
    cutoff: f64,
    init: WinsorInit,
    conv: Convention,
) -> (C, C) {
    let pi = conv == Convention::PixInsight;
    let zero = sp::<C>(0.0);
    let one = sp::<C>(1.0);
    let mut lo = zero;
    let mut hi = m;
    let mut active = m.gt(sp::<C>(3.0 - if pi { 1.0 } else { 0.0 }));
    let mut it = 0u32;
    while active.any() {
        let cnt = hi.sub(lo);
        let mut c = median_range(cb, n, lo, hi);
        let mut s = match init {
            WinsorInit::Std => {
                let mean = sum_range(cb, n, lo, hi).div(cnt);
                std_range(cb, n, lo, hi, mean, cnt, 1.0)
            }
            WinsorInit::Sn => {
                let mut los = [C::E::ZERO; MAXL];
                let mut his = [C::E::ZERO; MAXL];
                lo.store(los.as_mut_ptr());
                hi.store(his.as_mut_ptr());
                let mut out = [C::E::ZERO; MAXL];
                let mut tmp = Vec::with_capacity(n);
                for i in 0..C::L {
                    tmp.clear();
                    for k in los[i].to_usize()..his[i].to_usize() {
                        tmp.push(at(cb, k, i).to_f64());
                    }
                    out[i] = C::E::from_f64(1.1926 * sn(&tmp));
                }
                C::load(out.as_ptr())
            }
        };
        let tiny = one.add(s).eq(one);
        let mut hact = active.andnot(tiny);
        let mut hit = 0;
        while hact.any() && hit < 100 {
            let t0 = c.sub(sp::<C>(1.5).mul(s));
            let t1 = c.add(sp::<C>(1.5).mul(s));
            let (c0, c1) = (c.sub(sp::<C>(cutoff).mul(s)), c.add(sp::<C>(cutoff).mul(s)));
            let use_cut = pi && it == 0 && cutoff > 0.0;
            let mut ys = zero;
            for k in 0..n {
                let y = wins(*cb.add(k), t0, t1, c, c0, c1, use_cut);
                ys = ys.add(C::select(in_range(k, lo, hi), y, zero));
            }
            let wm = ys.div(cnt);
            let mut ss = zero;
            for k in 0..n {
                let y = wins(*cb.add(k), t0, t1, c, c0, c1, use_cut).sub(wm);
                ss = ss.add(C::select(in_range(k, lo, hi), y.mul(y), zero));
            }
            let snew = sp::<C>(1.134).mul(ss.div(cnt.sub(one)).sqrt());
            let delta = s.sub(snew).abs();
            let done = if pi {
                if hit >= 1 { delta.div(s).lt(sp::<C>(5e-4)) } else { C::mask_none() }
            } else {
                // |s_new - s| <= 5e-4 * s
                delta.gt(sp::<C>(5e-4).mul(s)).not()
            };
            if pi {
                c = C::select(hact, wm, c);
            }
            s = C::select(hact, snew, s);
            hact = hact.andnot(done).andnot(one.add(s).eq(one));
            hit += 1;
        }
        let ok = active.andnot(one.add(s).eq(one));
        let mut ca = [C::E::ZERO; MAXL];
        let mut sa = [C::E::ZERO; MAXL];
        c.store(ca.as_mut_ptr());
        s.store(sa.as_mut_ptr());
        let (lowe, highe) = (C::E::from_f64(low), C::E::from_f64(high));
        let (nl, nh) = if pi {
            end_scan(cb, lo, hi, |i, x| div_gt(ca[i], x, sa[i], lowe), |i, x| div_gt(x, ca[i], sa[i], highe))
        } else {
            let mut ka = [C::E::ZERO; MAXL];
            let mut kb = [C::E::ZERO; MAXL];
            sp::<C>(low).mul(s).store(ka.as_mut_ptr());
            sp::<C>(high).mul(s).store(kb.as_mut_ptr());
            end_scan(cb, lo, hi, |i, x| sub_gt(ca[i], x, ka[i]), |i, x| sub_gt(x, ca[i], kb[i]))
        };
        let nlo = lo.add(nl);
        let nhi = hi.sub(nh);
        let changed = nl.gt(zero).or(nh.gt(zero));
        lo = C::select(ok, nlo, lo);
        hi = C::select(ok, nhi, hi);
        let floor = if pi { 2.0 } else { 3.0 };
        active = ok.and(changed).and(hi.sub(lo).gt(sp::<C>(floor)));
        it += 1;
        if let Some(mx) = max_iters {
            if it >= mx {
                break;
            }
        }
    }
    (lo, hi)
}

/// Siril's median sigma clipping on one lane's sorted valid values: every pass computes the
/// median and sample std of the whole stack, and replaces each value beyond the bounds by
/// that median (the stack keeps its size), until a pass rejects nothing. Returns the kept rank
/// range of the original sorted stack and the mean of the final stack.
fn median_sigma_lane(v: &[f64], low: f64, high: f64) -> (usize, usize, f64) {
    let m = v.len();
    if m == 0 {
        return (0, 0, f64::NAN);
    }
    let mut w: Vec<f64> = v.to_vec();
    let mut flag = vec![false; m];
    let mut s = Vec::with_capacity(m);
    for _ in 0..4 * m + 8 {
        s.clear();
        s.extend_from_slice(&w);
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let c = if m % 2 == 1 { s[m / 2] } else { (s[m / 2 - 1] + s[m / 2]) * 0.5 };
        let mean = s.iter().sum::<f64>() / m as f64;
        let sd = if m > 1 {
            (s.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (m - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        let (kl, kh) = (low * sd, high * sd);
        let mut any = false;
        for k in 0..m {
            if c - w[k] > kl || w[k] - c > kh {
                w[k] = c;
                flag[k] = true;
                any = true;
            }
        }
        if !any {
            break;
        }
    }
    let lo = flag.iter().take_while(|&&f| f).count();
    let hi = m - flag.iter().rev().take_while(|&&f| f).count();
    let mean = w.iter().sum::<f64>() / m as f64;
    (lo, hi.max(lo), mean)
}

#[inline(always)]
unsafe fn wins<C: FVector>(x: C, t0: C, t1: C, c: C, c0: C, c1: C, cut: bool) -> C {
    if cut {
        let below = C::select(x.gt(c0), t0, c);
        let above = C::select(x.lt(c1), t1, c);
        C::select(x.lt(t0), below, C::select(x.gt(t1), above, x))
    } else {
        x.max(t0).min(t1)
    }
}

/// `(a - b) / s > t`, evaluated in the compute dtype.
#[inline(always)]
pub(crate) fn div_gt<E: Element>(a: E, b: E, s: E, t: E) -> bool {
    if core::any::TypeId::of::<E>() == core::any::TypeId::of::<f32>() {
        let (a, b, s, t) = (a.to_f64() as f32, b.to_f64() as f32, s.to_f64() as f32, t.to_f64() as f32);
        (a - b) / s > t
    } else {
        let (a, b, s, t) = (a.to_f64(), b.to_f64(), s.to_f64(), t.to_f64());
        (a - b) / s > t
    }
}

/// Percentile clipping: one pass around the median.
#[inline(always)]
unsafe fn percentile<C: FVector>(cb: *const C, n: usize, m: C, low: f64, high: f64) -> (C, C) {
    let zero = sp::<C>(0.0);
    let c = median_range(cb, n, zero, m);
    let mut ca = [C::E::ZERO; MAXL];
    c.store(ca.as_mut_ptr());
    let (lowe, highe) = (C::E::from_f64(low), C::E::from_f64(high));
    let (nl, nh) = end_scan(cb, zero, m, |i, x| div_gt(ca[i], x, ca[i], lowe), |i, x| div_gt(x, ca[i], ca[i], highe));
    (nl, m.sub(nh))
}

/// Lanes of an unsorted frame as compute lanes, with missing values turned into NaN.
#[inline(always)]
unsafe fn load_raw_group<S: Vector, C: FVector>(raw: *const S, k: usize, g: usize, job: &Job) -> C {
    let v = *raw.add(k);
    let miss = miss_mask(v, job.missing).bits() >> (g * C::L);
    let p = (raw.add(k) as *const S::E).add(g * C::L);
    let mut t = [C::E::ZERO; MAXL];
    for i in 0..C::L {
        t[i] = if (miss >> i) & 1 == 1 { C::E::from_f64(f64::NAN) } else { C::E::from_f64((*p.add(i)).to_f64()) };
    }
    C::load(t.as_ptr())
}

/// Reject, combine and write one block.
#[inline(always)]
pub(crate) unsafe fn clip_block<S: SortVec, C: FVector + SortVec>(job: &Job, plan: &ClipPlan, blk: &Blk<S>, ms: &[usize; MAXL]) {
    let n = job.n;
    let groups = S::L / C::L;
    let cb = blk.scratch as *mut C;
    let d = cb.add(n);
    let mut val = [0f64; MAXL];
    let mut nlow = [0f64; MAXL];
    let mut nhigh = [0f64; MAXL];
    let mut vlo = [0f64; MAXL];
    let mut vhi = [0f64; MAXL];
    for g in 0..groups {
        for k in 0..n {
            *cb.add(k) = load_group::<S, C>(blk.buf, k, g);
        }
        let mut mm = [C::E::ZERO; MAXL];
        for i in 0..C::L {
            mm[i] = C::E::from_usize(ms[g * C::L + i]);
        }
        let mut ms_vals: Option<[f64; MAXL]> = None;
        let m = C::load(mm.as_ptr());
        let (lo, hi) = match plan.rej {
            Rejection::None => (sp::<C>(0.0), m),
            Rejection::Sigma { low, high, max_iters, center, spread, convention } => {
                sigma(cb, d, n, m, low, high, max_iters, center, spread, convention, &job.net_bitonic)
            }
            Rejection::Winsorized { low, high, max_iters, cutoff, init, convention } => {
                winsorized(cb, n, m, low, high, max_iters, cutoff, init, convention)
            }
            Rejection::MedianSigma { low, high } => {
                let (mut la, mut ha, mut va) = ([C::E::ZERO; MAXL], [C::E::ZERO; MAXL], [0f64; MAXL]);
                let mut col = Vec::with_capacity(n);
                for i in 0..C::L {
                    let mi = ms[g * C::L + i];
                    col.clear();
                    for k in 0..mi {
                        col.push(at(cb, k, i).to_f64());
                    }
                    let (l, h, v) = median_sigma_lane(&col, low, high);
                    la[i] = C::E::from_usize(l);
                    ha[i] = C::E::from_usize(h);
                    va[i] = v;
                }
                ms_vals = Some(va);
                (C::load(la.as_ptr()), C::load(ha.as_ptr()))
            }
            Rejection::Percentile { low, high } => percentile(cb, n, m, low, high),
        };
        // ---- combine
        let zero = sp::<C>(0.0);
        let cnt = hi.sub(lo);
        let mut res = [0f64; MAXL];
        match plan.combine {
            Combine::Mean | Combine::Sum if plan.weights.is_none() => {
                let s = sum_range(cb, n, lo, hi);
                let r = if plan.combine == Combine::Mean { s.div(cnt) } else { s };
                lanes_f64(r, &mut res);
            }
            Combine::Median => {
                let r = median_range(cb, n, lo, hi);
                lanes_f64(r, &mut res);
                // np.median would use (a + b) * 0.5 too; empty ranges are NaN below.
            }
            _ => {}
        }
        if let Some(v) = ms_vals {
            // median sigma clipping averages the stack with rejected values replaced
            res[..C::L].copy_from_slice(&v[..C::L]);
        }
        // kept value thresholds, for weights and the rejection map
        let mut lo_a = [C::E::ZERO; MAXL];
        let mut hi_a = [C::E::ZERO; MAXL];
        lo.store(lo_a.as_mut_ptr());
        hi.store(hi_a.as_mut_ptr());
        for i in 0..C::L {
            let (l, h) = (lo_a[i].to_usize(), hi_a[i].to_usize());
            let lane = g * C::L + i;
            if h > l {
                vlo[lane] = at(cb, l, i).to_f64();
                vhi[lane] = at(cb, h - 1, i).to_f64();
            } else {
                vlo[lane] = f64::INFINITY;
                vhi[lane] = f64::NEG_INFINITY;
                if ms_vals.is_none() {
                    res[i] = f64::NAN;
                }
            }
            nlow[lane] = l as f64;
            nhigh[lane] = (ms[lane] - h.min(ms[lane])) as f64;
            val[lane] = res[i];
        }
        if let Some(w) = &plan.weights {
            // weighted mean of the kept values, rescanning the unsorted stack
            let mut tl = [C::E::ZERO; MAXL];
            let mut th = [C::E::ZERO; MAXL];
            for i in 0..C::L {
                tl[i] = C::E::from_f64(vlo[g * C::L + i]);
                th[i] = C::E::from_f64(vhi[g * C::L + i]);
            }
            let (tl, th) = (C::load(tl.as_ptr()), C::load(th.as_ptr()));
            let mut sw = zero;
            let mut swx = zero;
            for k in 0..n {
                let x: C = load_raw_group::<S, C>(blk.raw, k, g, job);
                let keep = x.lt(tl).or(x.gt(th)).or(x.nan_mask()).not();
                let wk = sp::<C>(w[k]);
                sw = sw.add(C::select(keep, wk, zero));
                swx = swx.add(C::select(keep, wk.mul(x), zero));
            }
            let r = if plan.combine == Combine::Sum { swx } else { swx.div(sw) };
            let mut wr = [0f64; MAXL];
            lanes_f64(r, &mut wr);
            for i in 0..C::L {
                let lane = g * C::L + i;
                val[lane] = if vlo[lane] <= vhi[lane] { wr[i] } else { f64::NAN };
            }
        }
    }
    let lanes = blk.lanes;
    put(&blk.outs[0], blk.p, &val, lanes);
    if blk.outs.len() > 1 {
        put(&blk.outs[1], blk.p, &nlow, lanes);
    }
    if blk.outs.len() > 2 {
        put(&blk.outs[2], blk.p, &nhigh, lanes);
    }
    if plan.has_mask {
        let mut xs = [0f64; MAXL];
        for k in 0..n {
            let x = *blk.raw.add(k);
            let miss = miss_mask(x, job.missing).bits();
            lanes_f64(x, &mut xs);
            let dst = blk.mask[k].offset(blk.p as isize * blk.mask_ps);
            for i in 0..lanes {
                let rej = (miss >> i) & 1 == 1 || !(xs[i] >= vlo[i] && xs[i] <= vhi[i]);
                *dst.offset(i as isize * blk.mask_ps) = rej as u8;
            }
        }
    }
}
