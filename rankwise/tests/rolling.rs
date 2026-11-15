#![allow(clippy::needless_range_loop)]
//! Moving windows and grouped reductions on every ISA, against the static engine run on each
//! window's (or group's) frames.

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
                    } else if u < nan_frac + 0.01 {
                        f32::INFINITY
                    } else if ties {
                        (r.next() % 5) as f32
                    } else {
                        ((r.f() - 0.5) * 1000.0) as f32
                    }
                })
                .collect()
        })
        .collect()
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

fn ops(w: usize) -> Vec<Op> {
    vec![
        Op::Median,
        Op::Quantile(QuantileSpec { form: QForm::Weak, ..QuantileSpec::new(vec![0.0, 0.3, 0.9, 1.0]) }),
        Op::Quantile(QuantileSpec {
            method: QuantileMethod::Hazen,
            out_rounding: OutRounding::CastOnce,
            ..QuantileSpec::new(vec![0.25, 0.5])
        }),
        Op::Quantile(QuantileSpec { method: QuantileMethod::Nearest, ..QuantileSpec::new(vec![0.6]) }),
        Op::SelectRanks(vec![0, w / 2, w - 1]),
        Op::Min,
        Op::Max,
        Op::Range,
        Op::Iqr { lo: 25.0, hi: 75.0, scale: 1.0 },
        Op::Mad { scale: 1.0 },
        Op::Clip(Clip::sigma(2.0, 2.0)),
        Op::Clip(Clip::median_sigma(2.5, 2.5)),
    ]
}

/// Run `op` over `frames` with outputs of dtype F64, `per` blocks of `npix` values each.
fn run_static(frames: &[&[f32]], op: &Op, opts: &Options) -> Vec<Vec<f64>> {
    let npix = frames[0].len();
    let nout = op.n_outputs();
    let mut bufs = vec![vec![0f64; npix]; nout];
    let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F64(b)).collect();
    reduce_frames(frames, op, &mut outs, None, opts).unwrap();
    bufs
}

/// The reference: each window's in-stack frames through the static engine (NaN omitted),
/// then windows with fewer than min_count valid values set to NaN.
fn reference(st: &[Vec<f32>], roll: &Rolling, op: &Op, isa: Isa) -> Vec<Vec<f64>> {
    let npix = st[0].len();
    let nout = op.n_outputs();
    let mut res = vec![vec![f64::NAN; roll.count * npix]; nout];
    let opts = Options { isa: Some(isa), nan: NanPolicy::Omit, ..Default::default() };
    let values = if matches!(op, Op::Clip(_)) { 1 } else { nout };
    for t in 0..roll.count {
        let a = roll.start + (t * roll.step) as isize;
        let idx: Vec<usize> =
            (a..a + roll.window as isize).filter(|&f| f >= 0 && (f as usize) < st.len()).map(|f| f as usize).collect();
        // min_count counts non-NaN values (±inf too, even for rejection), as xarray does
        let valid: Vec<usize> = (0..npix).map(|p| idx.iter().filter(|&&k| !st[k][p].is_nan()).count()).collect();
        if idx.is_empty() {
            for j in 0..nout {
                for p in 0..npix {
                    res[j][t * npix + p] = if j < values { f64::NAN } else { 0.0 };
                }
            }
            continue;
        }
        let views: Vec<&[f32]> = idx.iter().map(|&k| st[k].as_slice()).collect();
        let got = match op {
            // ranks of the whole window: positions past the in-stack frames are missing
            Op::SelectRanks(r) => {
                let rr: Vec<usize> = r.iter().map(|&k| k.min(idx.len() - 1)).collect();
                let mut g = run_static(&views, &Op::SelectRanks(rr), &opts);
                for (j, &k) in r.iter().enumerate() {
                    for p in 0..npix {
                        if k >= valid[p] {
                            g[j][p] = f64::NAN;
                        }
                    }
                }
                g
            }
            _ => run_static(&views, op, &opts),
        };
        for j in 0..nout {
            for p in 0..npix {
                let mut v = got[j][p];
                if j < values && valid[p] < roll.min_count {
                    v = f64::NAN;
                }
                res[j][t * npix + p] = v;
            }
        }
    }
    res
}

fn run_rolling(st: &[Vec<f32>], roll: &Rolling, op: &Op, opts: &Options) -> Vec<Vec<f64>> {
    let npix = st[0].len();
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let mut bufs = vec![vec![0f64; roll.count * npix]; op.n_outputs()];
    let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F64(b)).collect();
    rolling_frames(&views, roll, op, &mut outs, opts).unwrap();
    bufs
}

fn check(st: &[Vec<f32>], roll: &Rolling, op: &Op, isa: Isa, what: &str) {
    let got = run_rolling(st, roll, op, &Options { isa: Some(isa), ..Default::default() });
    let want = reference(st, roll, op, isa);
    let npix = st[0].len();
    for j in 0..got.len() {
        for i in 0..got[j].len() {
            assert!(
                same(got[j][i], want[j][i]),
                "{what} {isa} {op:?} {roll:?} out {j} window {} pixel {}: {} vs {}",
                i / npix,
                i % npix,
                got[j][i],
                want[j][i]
            );
        }
    }
}

const NPIX: usize = 37;

#[test]
fn rolling_matches_static_windows() {
    for &isa in &Isa::available() {
        for &(w, nf) in
            &[(1usize, 4usize), (2, 9), (3, 1), (3, 12), (5, 40), (8, 8), (16, 33), (31, 90), (33, 70), (64, 130)]
        {
            for (nanf, ties) in [(0.0, false), (0.1, false), (0.05, true)] {
                let st = stack(nf, NPIX, nanf, ties, (w * 1000 + nf) as u64);
                let rolls = [
                    Rolling::trailing(w, nf),
                    Rolling::centered(w, nf).min_count(1),
                    Rolling::valid(w, nf),
                    Rolling::trailing(w, nf).min_count(w.div_ceil(2)).step(2),
                    Rolling::centered(w, nf).step(3).min_count(1),
                    Rolling::trailing(w, nf).step(w + 2).min_count(1),
                ];
                for roll in rolls {
                    for op in ops(w) {
                        check(&st, &roll, &op, isa, "frames");
                    }
                }
            }
        }
    }
}

#[test]
fn rolling_layouts_tiles_threads_and_dtypes() {
    let isa = Isa::best();
    // a stack deep enough in pixels that each unit runs several tiles
    let (w, nf, npix) = (100usize, 150usize, 4100usize);
    let st = stack(nf, npix, 0.02, false, 9);
    let roll = Rolling::centered(w, nf).min_count(10);
    let base = run_rolling(&st, &roll, &Op::Median, &Options { isa: Some(isa), ..Default::default() });
    let want = reference(&st, &roll, &Op::Median, isa);
    for i in 0..base[0].len() {
        assert!(same(base[0][i], want[0][i]), "tiled i={i}");
    }
    for &isa2 in &Isa::available() {
        for threads in [1, 3] {
            let got = run_rolling(&st, &roll, &Op::Median, &Options { isa: Some(isa2), threads, ..Default::default() });
            for i in 0..got[0].len() {
                assert!(same(got[0][i], base[0][i]), "{isa2} t={threads} i={i}");
            }
        }
    }
    // frames interleaved (the frame axis innermost, as xarray.apply_ufunc lays it out)
    let (nf, h) = (50usize, 23usize);
    let st = stack(nf, h, 0.05, false, 4);
    let mut inter = vec![0f32; nf * h];
    for k in 0..nf {
        for p in 0..h {
            inter[p * nf + k] = st[k][p];
        }
    }
    let ptrs: Vec<*const u8> = (0..nf).map(|k| unsafe { (inter.as_ptr() as *const u8).add(4 * k) }).collect();
    let shape = [h];
    let strides = [(4 * nf) as isize];
    let rs = RawStack { dtype: DType::F32, frames: &ptrs, shape: &shape, strides: &strides };
    let roll = Rolling::trailing(7, nf).min_count(3);
    for op in ops(7) {
        let nout = op.n_outputs();
        let mut bufs = vec![vec![0f64; roll.count * h]; nout];
        let ostr = [8isize];
        let outs: Vec<RawOut> = bufs
            .iter_mut()
            .map(|b| RawOut { ptr: b.as_mut_ptr() as *mut u8, dtype: DType::F64, strides: &ostr })
            .collect();
        let tstr = vec![(8 * h) as isize; nout];
        unsafe { reduce_rolling_raw(&rs, &roll, &op, &outs, &tstr, &Options::default()).unwrap() };
        let want = run_rolling(&st, &roll, &op, &Options::default());
        for j in 0..nout {
            for i in 0..bufs[j].len() {
                assert!(same(bufs[j][i], want[j][i]), "interleaved {op:?} {j} {i}");
            }
        }
    }
    // uint16 with nodata, float32 outputs, normalization
    let nf = 40;
    let mut r = Rng(3);
    let u: Vec<Vec<u16>> = (0..nf).map(|_| (0..NPIX).map(|_| (r.next() % 7) as u16).collect()).collect();
    let uv: Vec<&[u16]> = u.iter().map(|v| v.as_slice()).collect();
    let as_f: Vec<Vec<f32>> =
        u.iter().map(|f| f.iter().map(|&x| if x == 0 { f32::NAN } else { x as f32 }).collect()).collect();
    let roll = Rolling::centered(9, nf).min_count(2);
    for op in [Op::Median, Op::Max, Op::Quantile(QuantileSpec::new(vec![0.2, 0.8]))] {
        let nout = op.n_outputs();
        let mut bufs = vec![vec![0f64; roll.count * NPIX]; nout];
        let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F64(b)).collect();
        rolling_frames(&uv, &roll, &op, &mut outs, &Options { nodata: Some(0.0), ..Default::default() }).unwrap();
        let want = run_rolling(&as_f, &roll, &op, &Options::default());
        for j in 0..nout {
            for i in 0..bufs[j].len() {
                assert!(same(bufs[j][i], want[j][i]), "u16 {op:?} {i}: {} vs {}", bufs[j][i], want[j][i]);
            }
        }
    }
    let scale: Vec<f64> = (0..nf).map(|k| 1.0 + k as f64 * 0.01).collect();
    let offset: Vec<f64> = (0..nf).map(|k| k as f64 * 0.5).collect();
    let norm = Options {
        normalize: Some(Normalize { scale: Some(scale.clone()), offset: Some(offset.clone()) }),
        ..Default::default()
    };
    let st = stack(nf, NPIX, 0.0, false, 77);
    let scaled: Vec<Vec<f32>> = st
        .iter()
        .enumerate()
        .map(|(k, f)| f.iter().map(|&x| x * scale[k] as f32 + offset[k] as f32).collect())
        .collect();
    let a = run_rolling(&st, &roll, &Op::Median, &norm);
    let b = run_rolling(&scaled, &roll, &Op::Median, &Options::default());
    for i in 0..a[0].len() {
        assert!(same(a[0][i], b[0][i]));
    }
}

#[test]
fn rolling_validation() {
    let st = stack(5, 3, 0.0, false, 1);
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let mut o = vec![0f32; 15];
    let run = |roll: Rolling, op: &Op, o: &mut Vec<f32>| {
        rolling_frames(&views, &roll, op, &mut [Out::F32(o)], &Options::default())
    };
    assert!(run(Rolling::trailing(0, 5), &Op::Median, &mut o).is_err());
    assert!(run(Rolling::trailing(3, 5).min_count(0), &Op::Median, &mut o).is_err());
    assert!(run(Rolling::trailing(3, 5).min_count(4), &Op::Median, &mut o).is_err());
    assert!(run(Rolling::trailing(3, 5), &Op::ArgMin, &mut o).is_err());
    assert!(run(Rolling::trailing(3, 5), &Op::TrimMean(0.1), &mut o).is_err());
    assert!(run(Rolling::trailing(3, 4), &Op::Median, &mut o).is_err()); // wrong output length
    assert!(run(Rolling::trailing(MAX_FRAMES + 1, 5), &Op::Median, &mut o).is_err());
    let mut z = vec![0f32; 0];
    assert!(run(Rolling { count: 0, ..Rolling::valid(9, 5) }, &Op::Median, &mut z).is_ok());
}

#[test]
fn groups_match_static_groups() {
    for &isa in &Isa::available() {
        for &nf in &[1usize, 7, 40, 200] {
            let st = stack(nf, NPIX, 0.05, false, nf as u64);
            let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
            let mut r = Rng(nf as u64 + 1);
            let ng = (nf / 3).max(1);
            // every group non-empty, frames shuffled between groups
            let labels: Vec<usize> =
                (0..nf).map(|k| if k < ng { k } else { (r.next() % ng as u64) as usize }).collect();
            for omit in [false, true] {
                let opts = Options {
                    isa: Some(isa),
                    nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
                    ..Default::default()
                };
                for op in [
                    Op::Median,
                    Op::Quantile(QuantileSpec::new(vec![0.1, 0.9])),
                    Op::ArgMax,
                    Op::Clip(Clip::sigma(2.0, 2.0)),
                    Op::TrimMean(0.1),
                ] {
                    let nout = op.n_outputs();
                    let mut bufs = vec![vec![0f64; ng * NPIX]; nout];
                    let mut outs: Vec<Out> = bufs.iter_mut().map(|b| Out::F64(b)).collect();
                    let ok = matches!(op, Op::ArgMax);
                    let mut ibuf = vec![0i64; ng * NPIX];
                    let g = if ok {
                        group_frames(&views, &labels, &op, &mut [Out::I64(&mut ibuf)], &opts).unwrap()
                    } else {
                        group_frames(&views, &labels, &op, &mut outs, &opts).unwrap()
                    };
                    assert_eq!(g, ng);
                    for gi in 0..ng {
                        let idx: Vec<usize> = (0..nf).filter(|&k| labels[k] == gi).collect();
                        let gv: Vec<&[f32]> = idx.iter().map(|&k| views[k]).collect();
                        if ok {
                            let mut want = vec![0i64; NPIX];
                            reduce_frames(&gv, &op, &mut [Out::I64(&mut want)], None, &opts).unwrap();
                            assert_eq!(&ibuf[gi * NPIX..(gi + 1) * NPIX], &want[..], "{isa} argmax group {gi}");
                            continue;
                        }
                        let want = run_static(&gv, &op, &opts);
                        for j in 0..nout {
                            for p in 0..NPIX {
                                assert!(
                                    same(bufs[j][gi * NPIX + p], want[j][p]),
                                    "{isa} {op:?} group {gi} out {j} p {p}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let st = stack(4, 3, 0.0, false, 1);
    let views: Vec<&[f32]> = st.iter().map(|v| v.as_slice()).collect();
    let mut o = vec![0f32; 9];
    assert!(group_frames(&views, &[0, 2, 0, 2], &Op::Median, &mut [Out::F32(&mut o)], &Options::default()).is_err());
    assert!(group_frames(&views, &[0, 1], &Op::Median, &mut [Out::F32(&mut o)], &Options::default()).is_err());
}
