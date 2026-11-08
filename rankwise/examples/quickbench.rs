//! cargo run --release --example quickbench -- [n] [side] [pad_floats] [staging]
use rankwise::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(31);
    let side: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let pad: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let staging = match a.get(4).map(|s| s.as_str()) {
        Some("never") => Staging::Never,
        Some("always") => Staging::Always,
        _ => Staging::Auto,
    };
    let npix = side * side;
    let stride = npix + pad;
    let mut buf = vec![0f32; n * stride + 16];
    let mut s = 12345u64;
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = 1000.0 + ((s >> 40) as f32 / 16777216.0 - 0.5) * 60.0;
        if (s & 1023) == 0 {
            *x += 5000.0;
        }
    }
    let frames: Vec<&[f32]> = (0..n).map(|k| &buf[k * stride..k * stride + npix]).collect();
    let mut out = vec![0f32; npix];
    let ops: Vec<(&str, Op)> = vec![
        ("median", Op::Median),
        ("q10/50/90", Op::Quantile(QuantileSpec { form: QForm::Weak, ..QuantileSpec::new(vec![0.1, 0.5, 0.9]) })),
        ("sigclip f64", Op::Clip(Clip::sigma(3.0, 3.0))),
        ("sigclip f32", Op::Clip(Clip::sigma(3.0, 3.0).precision(Precision::F32))),
        ("clip none f64", Op::Clip(Clip::none())),
        ("clip none f32", Op::Clip(Clip::none().precision(Precision::F32))),
        ("sig it1 f64", Op::Clip(Clip::sigma(3.0, 3.0).max_iters(1))),
        ("sig it2 f64", Op::Clip(Clip::sigma(3.0, 3.0).max_iters(2))),
    ];
    for isa in Isa::available() {
        for (name, op) in &ops {
            let nout = op.n_outputs();
            let mut extra: Vec<Vec<f32>> = vec![vec![0f32; npix]; nout.saturating_sub(1)];
            let opts = Options { isa: Some(isa), threads: 1, staging, ..Default::default() };
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut outs: Vec<Out> = vec![Out::F32(&mut out)];
                if let Op::Clip(_) = op {
                    outs.push(Out::None);
                    outs.push(Out::None);
                } else {
                    for e in extra.iter_mut() {
                        outs.push(Out::F32(e));
                    }
                }
                let t = Instant::now();
                reduce_frames(&frames, op, &mut outs, None, &opts).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!(
                "{:7} {:12} n={n} {side}x{side} pad={pad}: {:.3} s  {:6.1} ns/px",
                isa.name(),
                name,
                best,
                best / npix as f64 * 1e9
            );
        }
    }
}
