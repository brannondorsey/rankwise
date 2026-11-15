#![allow(clippy::needless_range_loop)]
//! Weighted quantiles on every ISA against a scalar transcription of NumPy's algorithm
//! (`np.quantile(..., method="inverted_cdf", weights=w)` with a stable argsort).

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
                    let u = r.f();
                    if u < nan_frac {
                        f32::NAN
                    } else if u < nan_frac + 0.02 {
                        f32::INFINITY
                    } else if ties {
                        (r.next() % 4) as f32 - 1.0
                    } else {
                        ((r.f() - 0.5) * 1000.0) as f32
                    }
                })
                .collect()
        })
        .collect()
}

/// NumPy's weighted inverted_cdf for one slice. `cmp32`: q is a float32 array.
fn reference(vals: &[f64], w: &[f64], q: f64, omit: bool, cmp32: bool) -> Result<f64> {
    let (v, w): (Vec<f64>, Vec<f64>) = if omit {
        vals.iter().zip(w).filter(|(x, _)| !x.is_nan()).map(|(&x, &y)| (x, y)).unzip()
    } else {
        (vals.to_vec(), w.to_vec())
    };
    if v.is_empty() {
        return Ok(f64::NAN);
    }
    let mut order: Vec<usize> = (0..v.len()).collect();
    // stable argsort, NaN last
    order.sort_by(|&a, &b| match (v[a].is_nan(), v[b].is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Greater,
        (false, true) => core::cmp::Ordering::Less,
        _ => v[a].partial_cmp(&v[b]).unwrap(),
    });
    let mut cdf = Vec::with_capacity(v.len());
    let mut c = 0f64;
    for &i in &order {
        c += w[i];
        cdf.push(c);
    }
    let total = c;
    let cdf: Vec<f64> = cdf.iter().map(|x| x / total).collect();
    if cdf.last().unwrap().is_nan() {
        return Err(Error::Param("Weights included NaN, inf or were all zero.".into()));
    }
    if v[order[v.len() - 1]].is_nan() {
        return Ok(f64::NAN);
    }
    let cdf: Vec<f64> = cdf.iter().map(|&x| if cmp32 { (x as f32) as f64 } else { x }).collect();
    let cdf: Vec<f64> = cdf.iter().map(|&x| if x == 0.0 { -1.0 } else { x }).collect();
    let qq = if cmp32 { (q as f32) as f64 } else { q };
    let i = cdf.iter().position(|&x| x >= qq).unwrap_or(v.len()).min(v.len() - 1);
    Ok(v[order[i]])
}

fn weights(n: usize, seed: u64, kind: u8) -> Vec<f64> {
    let mut r = Rng(seed | 1);
    (0..n)
        .map(|k| match kind {
            0 => r.f(),
            // integers with zeros (exact sums, ties in the cdf)
            1 => (r.next() % 4) as f64,
            // leading zero weights, then tiny and huge ones
            _ => {
                if k < n / 3 {
                    0.0
                } else if k % 2 == 0 {
                    1e-30 * r.f()
                } else {
                    1e30 * r.f()
                }
            }
        })
        .collect()
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b && a.is_sign_negative() == b.is_sign_negative())
}

const NS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129, 255, 256, 300, 512];
const NPIX: usize = 45;
const QS: &[f64] = &[0.0, 1e-9, 0.1, 0.25, 1.0 / 3.0, 0.5, 0.7, 0.9, 1.0];

#[test]
fn weighted_quantiles_match_numpy_algorithm() {
    for &isa in &Isa::available() {
        for &n in NS {
            for (nanf, ties, wk) in [(0.0, false, 0u8), (0.0, true, 1), (0.15, false, 0), (0.15, true, 2)] {
                let st = stack(n, NPIX, nanf, ties, n as u64 * 31 + wk as u64);
                let w = weights(n, n as u64 + 5, wk);
                if w.iter().sum::<f64>() == 0.0 {
                    continue;
                }
                let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
                for omit in [false, true] {
                    for cmp in [DType::F64, DType::F32] {
                        let op = Op::WeightedQuantile(WeightedQuantileSpec { q: QS.to_vec(), cmp });
                        let opts = Options {
                            isa: Some(isa),
                            nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                            weights: Some(w.clone()),
                            ..Default::default()
                        };
                        let mut bufs = vec![vec![0f32; NPIX]; QS.len()];
                        let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F32(b)).collect();
                        let res = reduce_frames(&views, &op, &mut outs, None, &opts);
                        let mut bad = false;
                        for p in 0..NPIX {
                            let col: Vec<f64> = st.iter().map(|f| f[p] as f64).collect();
                            for (j, &q) in QS.iter().enumerate() {
                                match reference(&col, &w, q, omit, cmp == DType::F32) {
                                    Ok(want) => assert!(
                                        same(bufs[j][p] as f64, want),
                                        "{isa} n={n} omit={omit} {cmp:?} wk={wk} q={q} p={p}: {} vs {want}",
                                        bufs[j][p]
                                    ),
                                    Err(_) => bad = true,
                                }
                            }
                        }
                        assert_eq!(res.is_err(), bad, "{isa} n={n} omit={omit} wk={wk}: {res:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn weight_maps_match_per_pixel_reference() {
    for &isa in &Isa::available() {
        for &n in &[1usize, 3, 16, 33, 130] {
            let st = stack(n, NPIX, 0.1, true, n as u64 + 77);
            let mut r = Rng(n as u64 * 3 + 1);
            // f64 map, frame-major, padded rows (pixel stride 16 bytes) to exercise strides
            let wm: Vec<Vec<f64>> =
                (0..n).map(|_| (0..2 * NPIX).map(|_| (r.next() % 5) as f64 * 0.25).collect()).collect();
            let wm32: Vec<Vec<f32>> = wm.iter().map(|f| f.iter().map(|&x| x as f32).collect()).collect();
            for omit in [false, true] {
                for use32 in [false, true] {
                    let ptrs: Vec<*const u8> = st.iter().map(|f| f.as_ptr() as *const u8).collect();
                    let shape = [NPIX];
                    let strides = [4isize];
                    let stack = RawStack { dtype: DType::F32, frames: &ptrs, shape: &shape, strides: &strides };
                    let wptrs: Vec<*const u8> = if use32 {
                        wm32.iter().map(|f| f.as_ptr() as *const u8).collect()
                    } else {
                        wm.iter().map(|f| f.as_ptr() as *const u8).collect()
                    };
                    let wstr = [if use32 { 8isize } else { 16 }];
                    let wst = RawStack {
                        dtype: if use32 { DType::F32 } else { DType::F64 },
                        frames: &wptrs,
                        shape: &shape,
                        strides: &wstr,
                    };
                    let mut bufs = vec![vec![0f64; NPIX]; QS.len()];
                    let ostr = [8isize];
                    let outs: Vec<RawOut> = bufs
                        .iter_mut()
                        .map(|b| RawOut { ptr: b.as_mut_ptr() as *mut u8, dtype: DType::F64, strides: &ostr })
                        .collect();
                    let op = Op::WeightedQuantile(WeightedQuantileSpec::new(QS.to_vec()));
                    let opts = Options {
                        isa: Some(isa),
                        nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                        ..Default::default()
                    };
                    let res = unsafe { reduce_raw_weighted(&stack, &wst, &op, &outs, &opts) };
                    let mut bad = false;
                    for p in 0..NPIX {
                        let col: Vec<f64> = st.iter().map(|f| f[p] as f64).collect();
                        let wc: Vec<f64> = wm.iter().map(|f| f[2 * p]).collect();
                        for (j, &q) in QS.iter().enumerate() {
                            match reference(&col, &wc, q, omit, false) {
                                Ok(want) => assert!(same(bufs[j][p], want), "{isa} n={n} omit={omit} q={q} p={p}"),
                                Err(_) => bad = true,
                            }
                        }
                    }
                    assert_eq!(res.is_err(), bad, "{isa} n={n} omit={omit}");
                }
            }
        }
    }
}

#[test]
fn uint16_nodata_and_float64() {
    for &isa in &Isa::available() {
        for &n in &[2usize, 9, 40, 200] {
            let mut r = Rng(n as u64 + 9);
            let st: Vec<Vec<u16>> = (0..n).map(|_| (0..NPIX).map(|_| (r.next() % 6) as u16).collect()).collect();
            let w = weights(n, n as u64, 0);
            let views: Vec<&[u16]> = st.iter().map(|v| v.as_slice()).collect();
            let op = Op::WeightedQuantile(WeightedQuantileSpec::new(QS.to_vec()));
            let opts = Options { isa: Some(isa), nodata: Some(0.0), weights: Some(w.clone()), ..Default::default() };
            let mut bufs = vec![vec![0u16; NPIX]; QS.len()];
            let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::U16(b)).collect();
            reduce_frames(&views, &op, &mut outs, None, &opts).unwrap();
            for p in 0..NPIX {
                let col: Vec<f64> = st.iter().map(|f| if f[p] == 0 { f64::NAN } else { f[p] as f64 }).collect();
                for (j, &q) in QS.iter().enumerate() {
                    let want = reference(&col, &w, q, true, false).unwrap();
                    let want = if want.is_nan() { 0.0 } else { want };
                    assert_eq!(bufs[j][p] as f64, want, "{isa} n={n} q={q} p={p}");
                }
            }
            // float64 data
            let st64: Vec<Vec<f64>> = st.iter().map(|f| f.iter().map(|&x| x as f64 * 0.1 - 0.2).collect()).collect();
            let v64: Vec<&[f64]> = st64.iter().map(|v| v.as_slice()).collect();
            let mut out = [vec![0f64; NPIX]];
            let mut outs = [&mut out[0][..]];
            weighted_quantile_frames(&v64, &w, &[0.3], &mut outs).unwrap();
            for p in 0..NPIX {
                let col: Vec<f64> = st64.iter().map(|f| f[p]).collect();
                assert_eq!(out[0][p], reference(&col, &w, 0.3, false, false).unwrap());
            }
        }
    }
}

#[test]
fn validation() {
    let f = vec![vec![1f32; 4]; 3];
    let views: Vec<&[f32]> = f.iter().map(|v| v.as_slice()).collect();
    let mut o = vec![0f32; 4];
    let neg = weighted_quantile_frames(&views, &[1.0, -1.0, 1.0], &[0.5], &mut [&mut o]);
    assert_eq!(neg, Err(Error::Param("Weights must be non-negative.".into())));
    let zero = weighted_quantile_frames(&views, &[0.0, 0.0, 0.0], &[0.5], &mut [&mut o]);
    assert_eq!(zero, Err(Error::Param("Weights included NaN, inf or were all zero.".into())));
    let nan = weighted_quantile_frames(&views, &[1.0, f64::NAN, 1.0], &[0.5], &mut [&mut o]);
    assert!(nan.is_err());
    assert!(weighted_quantile_frames(&views, &[1.0, 1.0], &[0.5], &mut [&mut o]).is_err());
    assert!(weighted_quantile_frames(&views, &[1.0; 3], &[1.5], &mut [&mut o]).is_err());
    let op = Op::WeightedQuantile(WeightedQuantileSpec::new(vec![0.5]));
    let r = reduce_frames(&views, &op, &mut [Out::F32(&mut o)], None, &Options::default());
    assert!(matches!(r, Err(Error::Param(_))));
    // weighted median: frame weights decide
    let g: Vec<Vec<f32>> = (0..5).map(|k| vec![k as f32 * 10.0; 2]).collect();
    let gv: Vec<&[f32]> = g.iter().map(|v| v.as_slice()).collect();
    let mut m = vec![0f32; 2];
    weighted_median_frames(&gv, &[1.0, 1.0, 1.0, 1.0, 10.0], &mut m).unwrap();
    assert_eq!(m, vec![40.0, 40.0]);
    weighted_median_frames(&gv, &[1.0; 5], &mut m).unwrap();
    assert_eq!(m, vec![20.0, 20.0]);
}

#[test]
fn isas_and_threads_agree() {
    let n = 77;
    let npix = 5003;
    let st = stack(n, npix, 0.05, true, 5);
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let w = weights(n, 3, 0);
    let op = Op::WeightedQuantile(WeightedQuantileSpec::new(vec![0.1, 0.5, 0.9]));
    let mut base: Option<Vec<Vec<f32>>> = None;
    for &isa in &Isa::available() {
        for threads in [1, 4] {
            for staging in [Staging::Auto, Staging::Always, Staging::Skew] {
                let opts = Options {
                    isa: Some(isa),
                    threads,
                    staging,
                    nan: NanPolicy::Omit,
                    weights: Some(w.clone()),
                    ..Default::default()
                };
                let mut bufs = vec![vec![0f32; npix]; 3];
                let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F32(b)).collect();
                reduce_frames(&views, &op, &mut outs, None, &opts).unwrap();
                match &base {
                    None => base = Some(bufs),
                    Some(b) => {
                        for j in 0..3 {
                            for p in 0..npix {
                                assert!(same(b[j][p] as f64, bufs[j][p] as f64), "{isa} t={threads} {staging:?} p={p}");
                            }
                        }
                    }
                }
            }
        }
    }
}
