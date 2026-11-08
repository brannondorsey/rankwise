#![allow(clippy::needless_range_loop, clippy::unnecessary_unwrap)]
//! Every kernel on every ISA against a scalar sort-based oracle.

use rankwise::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn stack(n: usize, npix: usize, nan_frac: f64, ties: bool, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|_| {
            (0..npix)
                .map(|_| {
                    if r.f() < nan_frac {
                        f32::NAN
                    } else if ties {
                        (r.next() % 4) as f32
                    } else {
                        ((r.f() - 0.5) * 1000.0) as f32
                    }
                })
                .collect()
        })
        .collect()
}

fn column(st: &[Vec<f32>], p: usize) -> Vec<f32> {
    st.iter().map(|f| f[p]).collect()
}

fn sort_nan_last(v: &mut [f32]) {
    v.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        _ => a.partial_cmp(b).unwrap(),
    });
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

fn isas() -> Vec<Isa> {
    Isa::available()
}

fn run_f32(st: &[Vec<f32>], op: &Op, opts: &Options, nout: usize, dt: DType) -> Vec<Vec<f64>> {
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let npix = st[0].len();
    match dt {
        DType::F32 => {
            let mut bufs = vec![vec![0f32; npix]; nout];
            let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F32(b.as_mut_slice())).collect();
            reduce_frames(&views, op, &mut outs, None, opts).unwrap();
            bufs.into_iter().map(|b| b.into_iter().map(|x| x as f64).collect()).collect()
        }
        DType::F64 => {
            let mut bufs = vec![vec![0f64; npix]; nout];
            let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F64(b.as_mut_slice())).collect();
            reduce_frames(&views, op, &mut outs, None, opts).unwrap();
            bufs
        }
        _ => {
            let mut bufs = vec![vec![0i64; npix]; nout];
            let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::I64(b.as_mut_slice())).collect();
            reduce_frames(&views, op, &mut outs, None, opts).unwrap();
            bufs.into_iter().map(|b| b.into_iter().map(|x| x as f64).collect()).collect()
        }
    }
}

const NS: &[usize] =
    &[1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 63, 64, 65, 100, 127, 128, 129, 255, 256, 300, 511, 512];
const NPIX: usize = 77; // leaves a partial vector on every ISA

#[test]
fn median_and_nanmedian() {
    for &isa in &isas() {
        for &n in NS {
            for (nanf, ties) in [(0.0, false), (0.0, true), (0.2, false)] {
                let st = stack(n, NPIX, nanf, ties, n as u64 * 7 + 1);
                for omit in [false, true] {
                    let opts = Options {
                        isa: Some(isa),
                        nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                        ..Default::default()
                    };
                    let got = run_f32(&st, &Op::Median, &opts, 1, DType::F32);
                    for p in 0..NPIX {
                        let mut c = column(&st, p);
                        let has_nan = c.iter().any(|x| x.is_nan());
                        sort_nan_last(&mut c);
                        let m = if omit { c.iter().filter(|x| !x.is_nan()).count() } else { n };
                        let want = if (!omit && has_nan) || m == 0 {
                            f64::NAN
                        } else if m % 2 == 1 {
                            c[m / 2] as f64
                        } else {
                            ((c[m / 2 - 1] + c[m / 2]) * 0.5) as f64
                        };
                        assert!(same(got[0][p], want), "{isa} n={n} omit={omit} p={p}: {} vs {want}", got[0][p]);
                    }
                }
            }
        }
    }
}

#[test]
fn select_ranks_matches_sort() {
    for &isa in &isas() {
        for &n in NS {
            let st = stack(n, NPIX, 0.1, false, n as u64 * 13 + 5);
            let ranks: Vec<usize> = vec![0, n / 3, n / 2, n - 1];
            let opts = Options { isa: Some(isa), ..Default::default() };
            let got = run_f32(&st, &Op::SelectRanks(ranks.clone()), &opts, ranks.len(), DType::F32);
            for p in 0..NPIX {
                let mut c = column(&st, p);
                sort_nan_last(&mut c);
                for (j, &r) in ranks.iter().enumerate() {
                    assert!(same(got[j][p], c[r] as f64), "{isa} n={n} r={r}");
                }
            }
        }
    }
}

#[test]
fn quantiles_all_methods_and_forms() {
    let lerp_ref = |form: QForm, method: QuantileMethod, c: &[f32], m: usize, q: f64| -> f64 {
        let l = rankwise::quantile::Lerp::new(DType::F32, form, None, OutRounding::NumpyQuantile);
        match rankwise::quantile::quantile_index(method, form, m, q) {
            rankwise::quantile::QIdx::Take(i) => c[i] as f64,
            rankwise::quantile::QIdx::Lerp { i0, i1, t } => l.eval(c[i0] as f64, c[i1] as f64, t),
            rankwise::quantile::QIdx::Empty => f64::NAN,
        }
    };
    for &isa in &isas() {
        for &n in &[1usize, 2, 5, 16, 31, 40, 129] {
            let st = stack(n, NPIX, 0.15, false, n as u64 + 99);
            for method in QuantileMethod::ALL {
                for form in [QForm::Weak, QForm::F64, QForm::F32] {
                    for omit in [false, true] {
                        let qs = vec![0.0, 0.1, 1.0 / 3.0, 0.5, 0.77, 1.0];
                        let spec = QuantileSpec { form, method, ..QuantileSpec::new(qs.clone()) };
                        let opts = Options {
                            isa: Some(isa),
                            nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                            ..Default::default()
                        };
                        let got = run_f32(&st, &Op::Quantile(spec), &opts, qs.len(), DType::F64);
                        for p in 0..NPIX {
                            let mut c = column(&st, p);
                            let has_nan = c.iter().any(|x| x.is_nan());
                            sort_nan_last(&mut c);
                            let m = if omit { c.iter().filter(|x| !x.is_nan()).count() } else { n };
                            for (j, &q) in qs.iter().enumerate() {
                                let want = if !omit && has_nan { f64::NAN } else { lerp_ref(form, method, &c, m, q) };
                                assert!(
                                    same(got[j][p], want),
                                    "{isa} n={n} {method:?} {form:?} omit={omit} q={q}: {} vs {want}",
                                    got[j][p]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn min_max_arg() {
    for &isa in &isas() {
        for &n in NS {
            let st = stack(n, NPIX, 0.1, true, n as u64 * 3 + 7);
            for omit in [false, true] {
                let opts = Options {
                    isa: Some(isa),
                    nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                    ..Default::default()
                };
                let mn = run_f32(&st, &Op::Min, &opts, 1, DType::F32);
                let mx = run_f32(&st, &Op::Max, &opts, 1, DType::F32);
                let amn = run_f32(&st, &Op::ArgMin, &opts, 1, DType::I64);
                let amx = run_f32(&st, &Op::ArgMax, &opts, 1, DType::I64);
                let amed = run_f32(&st, &Op::ArgRank(ArgRank::Median), &opts, 1, DType::I64);
                for p in 0..NPIX {
                    let c = column(&st, p);
                    let first_nan = c.iter().position(|x| x.is_nan());
                    let valid: Vec<(usize, f32)> = c.iter().cloned().enumerate().filter(|(_, x)| !x.is_nan()).collect();
                    let (wmn, wmx, wamn, wamx) = if !omit && first_nan.is_some() {
                        (f64::NAN, f64::NAN, first_nan.unwrap() as f64, first_nan.unwrap() as f64)
                    } else if valid.is_empty() {
                        (f64::NAN, f64::NAN, -1.0, -1.0)
                    } else {
                        let mut bmn = valid[0];
                        let mut bmx = valid[0];
                        for &(i, x) in &valid {
                            if x < bmn.1 {
                                bmn = (i, x);
                            }
                            if x > bmx.1 {
                                bmx = (i, x);
                            }
                        }
                        (bmn.1 as f64, bmx.1 as f64, bmn.0 as f64, bmx.0 as f64)
                    };
                    assert!(same(mn[0][p], wmn), "{isa} min n={n} omit={omit}");
                    assert!(same(mx[0][p], wmx), "{isa} max n={n} omit={omit}");
                    assert_eq!(amn[0][p], wamn, "{isa} argmin n={n} omit={omit} p={p} col={c:?}");
                    assert_eq!(amx[0][p], wamx, "{isa} argmax n={n} omit={omit}");
                    // stable argsort, NaN last
                    let mut idx: Vec<usize> = (0..n).collect();
                    idx.sort_by(|&a, &b| {
                        let (x, y) = (c[a], c[b]);
                        match (x.is_nan(), y.is_nan()) {
                            (true, true) => core::cmp::Ordering::Equal,
                            (true, false) => core::cmp::Ordering::Greater,
                            (false, true) => core::cmp::Ordering::Less,
                            _ => x.partial_cmp(&y).unwrap(),
                        }
                    });
                    let m = valid.len();
                    let want = if omit {
                        if m == 0 {
                            -1.0
                        } else {
                            idx[(m - 1) / 2] as f64
                        }
                    } else {
                        idx[(n - 1) / 2] as f64
                    };
                    assert_eq!(amed[0][p], want, "{isa} argmedian n={n} omit={omit} p={p}");
                }
            }
        }
    }
}

/// astropy's compute_sigma_clipped_bounds on one slice (sums in sorted order).
fn astropy_clip(c: &[f64], lo_s: f64, hi_s: f64, maxiters: i32) -> (f64, f64) {
    let mut buf: Vec<f64> = c.iter().cloned().filter(|x| x.is_finite()).collect();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if buf.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let (mut lb, mut ub);
    let mut it = 0;
    loop {
        let k = buf.len();
        let med = if k.is_multiple_of(2) { 0.5 * (buf[k / 2] + buf[k / 2 - 1]) } else { buf[(k - 1) / 2] };
        let mean = buf.iter().sum::<f64>() / k as f64;
        let std = (buf.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / k as f64).sqrt();
        lb = med - lo_s * std;
        ub = med + hi_s * std;
        let nb: Vec<f64> = buf.iter().cloned().filter(|&x| x >= lb && x <= ub).collect();
        if nb.len() == k || nb.is_empty() {
            break;
        }
        buf = nb;
        it += 1;
        if maxiters != -1 && it >= maxiters {
            break;
        }
    }
    (lb, ub)
}

#[test]
fn sigma_clip_matches_astropy_algorithm() {
    for &isa in &isas() {
        for &n in &[1usize, 2, 3, 4, 8, 15, 16, 31, 40, 64, 127, 255] {
            let mut st = stack(n, NPIX, 0.03, false, n as u64 * 17 + 3);
            // outliers
            let mut r = Rng(n as u64 + 1);
            for f in st.iter_mut() {
                for x in f.iter_mut() {
                    if r.f() < 0.05 {
                        *x += 5000.0;
                    }
                }
            }
            let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
            let mut out = vec![0f32; NPIX];
            let mut lo = vec![0u16; NPIX];
            let mut hi = vec![0u16; NPIX];
            let mut map: Vec<Vec<bool>> = vec![vec![false; NPIX]; n];
            let mut mrefs: Vec<&mut [bool]> = map.iter_mut().map(|v| v.as_mut_slice()).collect();
            let opts = Options { isa: Some(isa), ..Default::default() };
            let mut outs = [Out::F32(&mut out), Out::U16(&mut lo), Out::U16(&mut hi)];
            reduce_frames(&views, &Op::Clip(Clip::sigma(3.0, 3.0)), &mut outs, Some(&mut mrefs), &opts).unwrap();
            for p in 0..NPIX {
                let c: Vec<f64> = column(&st, p).iter().map(|&x| x as f64).collect();
                let (lb, ub) = astropy_clip(&c, 3.0, 3.0, 5);
                let kept: Vec<f64> = c.iter().cloned().filter(|&x| x.is_finite() && x >= lb && x <= ub).collect();
                let finite = c.iter().filter(|x| x.is_finite()).count();
                let wl = c.iter().filter(|&&x| x.is_finite() && x < lb).count();
                let wh = c.iter().filter(|&&x| x.is_finite() && x > ub).count();
                assert_eq!((lo[p] as usize, hi[p] as usize), (wl, wh), "{isa} n={n} p={p}");
                assert_eq!(finite - wl - wh, kept.len());
                let mut ks = kept.clone();
                ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let want =
                    if ks.is_empty() { f64::NAN } else { (ks.iter().sum::<f64>() / ks.len() as f64) as f32 as f64 };
                assert!(same(out[p] as f64, want), "{isa} n={n} p={p} {} vs {want}", out[p]);
                for k in 0..n {
                    let rej = !(c[k].is_finite() && c[k] >= lb && c[k] <= ub);
                    assert_eq!(map[k][p], rej, "{isa} n={n} p={p} k={k}");
                }
            }
        }
    }
}

#[test]
fn isas_agree_bit_for_bit() {
    let all = isas();
    for &n in &[7usize, 31, 64, 200] {
        let st = stack(n, 1001, 0.05, false, 42 + n as u64);
        let ops = [
            Op::Median,
            Op::Quantile(QuantileSpec { form: QForm::Weak, ..QuantileSpec::new(vec![0.1, 0.9]) }),
            Op::Mad { scale: 1.0 },
            Op::Iqr { lo: 25.0, hi: 75.0, scale: 1.0 },
            Op::TrimMean(0.1),
            Op::Clip(Clip::sigma(2.5, 2.5)),
            Op::Clip(Clip::mad(3.0, 3.0)),
            Op::Clip(presets::pixinsight_winsorized(3.0, 3.0)),
        ];
        for op in &ops {
            let nout = op.n_outputs();
            let mut base: Option<Vec<Vec<f64>>> = None;
            for &isa in &all {
                let opts = Options { isa: Some(isa), nan: NanPolicy::Omit, ..Default::default() };
                let got = run_f32(&st, op, &opts, nout, DType::F64);
                if let Some(b) = &base {
                    for j in 0..nout {
                        for p in 0..1001 {
                            assert!(
                                same(b[j][p], got[j][p]),
                                "{op:?} n={n} {isa} out {j} p={p}: {} vs {}",
                                b[j][p],
                                got[j][p]
                            );
                        }
                    }
                } else {
                    base = Some(got);
                }
            }
        }
    }
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_any_axis_matches_frames() {
    use ndarray::{Array2, Array3, Axis};
    let st = stack(17, 30 * 40, 0.05, false, 77);
    let flat: Vec<f32> = st.iter().flatten().cloned().collect();
    let a = Array3::from_shape_vec((17, 30, 40), flat).unwrap();
    let mut out = Array2::<f32>::zeros((30, 40));
    a.reduce_into(Axis(0), Op::Median, out.view_mut()).unwrap();
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let mut want = vec![0f32; 1200];
    reduce_frames(&views, &Op::Median, &mut [Out::F32(&mut want)], None, &Options::default()).unwrap();
    for (g, w) in out.iter().zip(want.iter()) {
        assert!(same(*g as f64, *w as f64));
    }
    // reduce the middle axis of a permuted view
    let p = a.view().permuted_axes([1, 0, 2]);
    let mut o2 = Array2::<f32>::zeros((30, 40));
    p.reduce_into(Axis(1), Op::Median, o2.view_mut()).unwrap();
    assert!(o2.iter().zip(out.iter()).all(|(x, y)| same(*x as f64, *y as f64)));
    let mut o3 = Array2::<f32>::zeros((30, 40));
    a.reduce_into(Axis(0), Clip::sigma(3.0, 3.0).max_iters(5).mean(), o3.view_mut()).unwrap();
}
