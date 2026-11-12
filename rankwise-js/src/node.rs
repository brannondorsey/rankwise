//! Node-API exports (Node, Bun, Deno). The kernels read and write the caller's TypedArrays in
//! place: a request names memory as *segmented addresses*, `(i + 1) << 40 | byte_offset`
//! into `segments[i]`, and every region a call touches is checked against its segment's
//! length before any kernel runs, so a wrong request is an error, never a stray write.

use crate::request::{self, report, set_message, with_message, Fail, R};
use napi::bindgen_prelude::{Float64ArraySlice, Uint8ArraySlice};
use napi_derive::napi;

const SEGMENT_BITS: u32 = 40;

fn segmented(segs: &[(*mut u8, usize)]) -> impl Fn(u64, isize, isize) -> R<*mut u8> + '_ {
    move |addr, lo, hi| {
        let i = (addr >> SEGMENT_BITS) as usize;
        let off = (addr & ((1 << SEGMENT_BITS) - 1)) as usize;
        let (base, len) =
            *segs.get(i.wrapping_sub(1)).ok_or_else(|| Fail::Request(format!("address {addr:#x} names no buffer")))?;
        let (start, end) = (off as isize + lo, off as isize + hi);
        if lo < hi && !(start >= 0 && end as usize <= len) {
            return Err(Fail::Request(format!("bytes {start}..{end} are outside buffer {} ({len} bytes)", i - 1)));
        }
        // empty regions are never dereferenced; keep the pointer in bounds anyway
        Ok(unsafe { base.add(off.min(len)) })
    }
}

fn run(req: &Float64ArraySlice, segments: &mut [Uint8ArraySlice], describe_only: bool) -> R<String> {
    // SAFETY: the slices stay borrowed for the call; JS cannot run on this thread meanwhile.
    let segs: Vec<(*mut u8, usize)> = segments
        .iter_mut()
        .map(|s| {
            let m = unsafe { s.as_mut() };
            (m.as_mut_ptr(), m.len())
        })
        .collect();
    let resolve = segmented(&segs);
    request::execute(req, describe_only, &resolve)
}

/// Run the reduction described by `req`; returns a status (0 = ok, else see `message`).
#[napi]
pub fn reduce(req: Float64ArraySlice, mut segments: Vec<Uint8ArraySlice>) -> i32 {
    report(run(&req, &mut segments, false).map(|_| ()))
}

/// Describe how `req` would run; the text is left in `message()`.
#[napi]
pub fn describe(req: Float64ArraySlice, mut segments: Vec<Uint8ArraySlice>) -> i32 {
    report(run(&req, &mut segments, true).map(set_message))
}

/// The last error message or describe text.
#[napi]
pub fn message() -> String {
    with_message(|m| m.to_string())
}

/// Bit `i` set when ISA code `i` is available (0 scalar, 1 avx2, 2 avx512, 3 neon).
#[napi]
pub fn available_isas() -> u32 {
    request::available_isas()
}

#[napi]
pub fn current_isa() -> i32 {
    request::current_isa()
}

/// Force an ISA by code (-1 restores detection); returns a status.
#[napi]
pub fn set_isa(code: i32) -> i32 {
    request::set_isa(code)
}

#[napi]
pub fn network_size(n: u32, ranks: Option<Vec<u32>>) -> u32 {
    let r: Vec<usize> = ranks.unwrap_or_default().into_iter().map(|x| x as usize).collect();
    request::network_size(n as usize, &r) as u32
}

#[napi]
pub fn max_frames() -> u32 {
    rankwise::MAX_FRAMES as u32
}

#[napi]
pub fn version() -> String {
    rankwise::VERSION.to_string()
}

/// The target this addon was compiled for, e.g. "x86_64-unknown-linux-gnu".
#[napi]
pub fn target() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}
