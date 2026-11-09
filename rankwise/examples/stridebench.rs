//! Power-of-two stride experiments with NumPy-like 2 MiB pages.
//!
//!     cargo run --release --example stridebench -- [n] [side] [pad_floats] [auto|never|always|skew] [huge 0|1] [offset_bytes]
use rankwise::*;
use std::alloc::{alloc, Layout};
use std::time::Instant;

#[cfg(target_os = "linux")]
fn advise_huge(p: *mut u8, len: usize) {
    extern "C" {
        fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    unsafe { madvise(p as *mut _, len, 14) }; // MADV_HUGEPAGE
}
#[cfg(not(target_os = "linux"))]
fn advise_huge(_: *mut u8, _: usize) {}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(31);
    let side: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let pad: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let staging = match a.get(4).map(|s| s.as_str()) {
        Some("never") => Staging::Never,
        Some("always") => Staging::Always,
        Some("skew") => Staging::Skew,
        _ => Staging::Auto,
    };
    let huge = a.get(5).map(|s| s != "0").unwrap_or(true);
    // NumPy puts large arrays 16 bytes into a cache line; pass 16 to reproduce that.
    let off: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
    let npix = side * side;
    let stride = npix + pad;
    let len = (n * stride + 16) * 4 + 64;
    let lay = Layout::from_size_align(len, 2 << 20).unwrap();
    let p = unsafe { alloc(lay) };
    if huge {
        advise_huge(p, len);
    }
    let buf: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(p.add(off) as *mut f32, n * stride + 16) };
    let mut s = 12345u64;
    for x in buf.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = 1000.0 + ((s >> 40) as f32 / 16777216.0 - 0.5) * 60.0;
    }
    let frames: Vec<&[f32]> = (0..n).map(|k| &buf[k * stride..k * stride + npix]).collect();
    if let Ok(t) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        for l in t.lines().filter(|l| l.starts_with("AnonHugePages")) {
            println!("{l}");
        }
    }
    let mut out = vec![0f32; npix];
    let isa = resolve_isa(None).unwrap();
    let ops: Vec<(&str, Op)> = vec![("median", Op::Median), ("sigclip", Op::Clip(Clip::sigma(3.0, 3.0)))];
    for (name, op) in &ops {
        let opts = Options { isa: Some(isa), threads: 1, staging, ..Default::default() };
        let mut best = f64::MAX;
        for _ in 0..7 {
            let mut outs: Vec<Out> = vec![Out::F32(&mut out)];
            if let Op::Clip(_) = op {
                outs.push(Out::None);
                outs.push(Out::None);
            }
            let t = Instant::now();
            reduce_frames(&frames, op, &mut outs, None, &opts).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "{} {name:8} n={n} {side}² pad={pad} {staging:?} huge={huge}: {:6.1} ns/px",
            isa.name(),
            best / npix as f64 * 1e9
        );
    }
    // Bandwidth reference: one frame at a time, sequential.
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let mut acc = [0f32; 16];
        for f in &frames {
            for c in f.chunks_exact(16) {
                for i in 0..16 {
                    acc[i] += c[i];
                }
            }
        }
        std::hint::black_box(acc);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "sum (sequential read) {:6.1} ns/px, {:.1} GB/s",
        best / npix as f64 * 1e9,
        (n * npix * 4) as f64 / best / 1e9
    );
}
