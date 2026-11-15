//! Median (or `sigclip`, `q`) of `n` frames of `npix` float32 values, best of three, one core:
//! `cargo run --release --example deepbench -- n npix [median|sigclip|q] [auto|never|always]`.
use rankwise::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let npix: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(65536);
    let which = a.get(3).cloned().unwrap_or("median".into());
    let mut buf = vec![0f32; n * npix];
    let mut s = 12345u64;
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = 1000.0 + ((s >> 40) as f32 / 16777216.0 - 0.5) * 60.0;
    }
    let frames: Vec<&[f32]> = (0..n).map(|k| &buf[k * npix..(k + 1) * npix]).collect();
    let mut out = vec![0f32; npix];
    let op = match which.as_str() {
        "sigclip" => Op::Clip(Clip::sigma(3.0, 3.0)),
        "q" => Op::Quantile(QuantileSpec { form: QForm::Weak, ..QuantileSpec::new(vec![0.1]) }),
        _ => Op::Median,
    };
    let isa = Isa::best();
    let staging = match a.get(4).map(|s| s.as_str()) {
        Some("never") => Staging::Never,
        Some("always") => Staging::Always,
        _ => Staging::Auto,
    };
    let opts = Options { isa: Some(isa), threads: 1, staging, ..Default::default() };
    let mut best = f64::MAX;
    for _ in 0..3 {
        let mut outs: Vec<Out> = vec![Out::F32(&mut out)];
        if let Op::Clip(_) = op {
            outs.push(Out::None);
            outs.push(Out::None);
        }
        let t = Instant::now();
        reduce_frames(&frames, &op, &mut outs, None, &opts).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "{isa} {which} n={n} npix={npix}: {:.3} s {:8.1} ns/px {:6.2} ns/value",
        best,
        best / npix as f64 * 1e9,
        best / (npix * n) as f64 * 1e9
    );
}
