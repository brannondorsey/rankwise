//! WebAssembly exports. Addresses in requests are linear-memory pointers: the module can only
//! reach its own memory, so they need no further checks.

use crate::request::{self, report, set_message, with_message};
use std::alloc::{alloc, dealloc, Layout};

fn linear(addr: u64, _lo: isize, _hi: isize) -> request::R<*mut u8> {
    Ok(addr as usize as *mut u8)
}

/// Allocate `bytes` with 16-byte alignment (null when `bytes` is 0 or memory is exhausted).
#[no_mangle]
pub extern "C" fn rw_alloc(bytes: usize) -> *mut u8 {
    match Layout::from_size_align(bytes, 16) {
        Ok(l) if bytes > 0 => unsafe { alloc(l) },
        _ => core::ptr::null_mut(),
    }
}

/// Free memory from [`rw_alloc`] (`bytes` as allocated).
#[no_mangle]
pub unsafe extern "C" fn rw_free(ptr: *mut u8, bytes: usize) {
    if !ptr.is_null() && bytes > 0 {
        dealloc(ptr, Layout::from_size_align_unchecked(bytes, 16));
    }
}

/// Run the reduction described by the request at `req` (`len` f64 values).
#[no_mangle]
pub unsafe extern "C" fn rw_reduce(req: *const f64, len: usize) -> i32 {
    let v = core::slice::from_raw_parts(req, len);
    report(request::execute(v, false, &linear).map(|_| ()))
}

/// Describe how a request would run; the text is left in the message buffer.
#[no_mangle]
pub unsafe extern "C" fn rw_describe(req: *const f64, len: usize) -> i32 {
    let v = core::slice::from_raw_parts(req, len);
    report(request::execute(v, true, &linear).map(set_message))
}

/// The last error message or `rw_describe` text (UTF-8).
#[no_mangle]
pub extern "C" fn rw_message_ptr() -> *const u8 {
    with_message(|m| m.as_ptr())
}

#[no_mangle]
pub extern "C" fn rw_message_len() -> usize {
    with_message(|m| m.len())
}

/// Bit `i` set when ISA code `i` is available (0 scalar, 4 simd128).
#[no_mangle]
pub extern "C" fn rw_available_isas() -> u32 {
    request::available_isas()
}

/// The ISA code calls use now.
#[no_mangle]
pub extern "C" fn rw_current_isa() -> i32 {
    request::current_isa()
}

/// Force an ISA by code (-1 restores the default).
#[no_mangle]
pub extern "C" fn rw_set_isa(code: i32) -> i32 {
    request::set_isa(code)
}

/// Comparators in the network for `n` values, pruned to the `nranks` ranks at `ranks` (the
/// full sort when `nranks` is 0).
#[no_mangle]
pub unsafe extern "C" fn rw_network_size(n: usize, ranks: *const u32, nranks: usize) -> usize {
    let r: Vec<usize> = if nranks == 0 {
        vec![]
    } else {
        core::slice::from_raw_parts(ranks, nranks).iter().map(|&x| x as usize).collect()
    };
    request::network_size(n, &r)
}

#[no_mangle]
pub extern "C" fn rw_max_frames() -> usize {
    rankwise::MAX_FRAMES
}

/// Version as `major * 10000 + minor * 100 + patch`.
#[no_mangle]
pub extern "C" fn rw_version() -> u32 {
    let mut it = rankwise::VERSION.split('.').map(|s| s.parse::<u32>().unwrap_or(0));
    let (a, b, c) = (it.next().unwrap_or(0), it.next().unwrap_or(0), it.next().unwrap_or(0));
    a * 10000 + b * 100 + c
}
