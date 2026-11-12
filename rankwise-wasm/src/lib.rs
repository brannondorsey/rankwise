//! WebAssembly exports behind the `rankwise` npm package.
//!
//! No wasm-bindgen: the module exports a handful of plain functions and the JS layer
//! (`js/rankwise.js`) owns NumPy's semantics (argument parsing, dtype rules, output
//! allocation, copying TypedArrays in and out of linear memory), as the Python package does.
//!
//! A call is described by a request: a `Float64Array` the JS layer writes into linear memory
//! (every count, code, pointer and stride is an exact integer in an f64). Its layout:
//!
//! ```text
//! op      kind, kind-specific fields                      (see parse_op)
//! stack   in_dtype, n, ndim, shape[ndim], strides[ndim], frame_addr[n]
//! outs    nouts, { addr, dtype, strides[ndim] } * nouts   (addr 0 = not requested)
//! mask    0 | 1, frame_addr[n], dtype, strides[ndim]
//! opts    omit, nodata (NaN = none), isa (-1 = auto), staging,
//!         list scale, list offset, list weights            (list = len, values; len -1 = none)
//! ```
//!
//! Strides are in bytes. Errors return a nonzero status and leave a message readable with
//! `rw_message_ptr` / `rw_message_len`.

#![allow(clippy::missing_safety_doc)]

use rankwise::{
    presets, reduce_raw, ArgRank, Center, Clip, Combine, DType, Error, Isa, NanPolicy, Normalize, Op, Options,
    OutRounding, Precision, QForm, QuantileMethod, QuantileSpec, RawMask, RawOut, RawStack, Rejection, Spread, Staging,
};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

/// Status codes (`rw_reduce` and friends).
pub const OK: i32 = 0;
pub const ERR_DEPTH: i32 = 1;
pub const ERR_SHAPE: i32 = 2;
pub const ERR_PARAM: i32 = 3;
pub const ERR_DTYPE: i32 = 4;
pub const ERR_ISA: i32 = 5;
pub const ERR_REQUEST: i32 = 6;

thread_local! {
    static MESSAGE: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_message(s: String) {
    MESSAGE.with(|m| *m.borrow_mut() = s);
}

fn status(e: &Error) -> i32 {
    match e {
        Error::BadDepth(_) => ERR_DEPTH,
        Error::Shape(_) => ERR_SHAPE,
        Error::Param(_) => ERR_PARAM,
        Error::DType(_) => ERR_DTYPE,
        Error::UnsupportedIsa(_) | Error::UnknownIsa(_) => ERR_ISA,
    }
}

/// A request-level error: either a core error or a malformed request.
enum Fail {
    Core(Error),
    Request(String),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Core(e)
    }
}

type R<T> = core::result::Result<T, Fail>;

fn param(s: impl Into<String>) -> Fail {
    Fail::Core(Error::Param(s.into()))
}

fn report(r: R<()>) -> i32 {
    match r {
        Ok(()) => OK,
        Err(Fail::Core(e)) => {
            let st = status(&e);
            set_message(e.to_string());
            st
        }
        Err(Fail::Request(s)) => {
            set_message(format!("malformed request: {s}"));
            ERR_REQUEST
        }
    }
}

/// Cursor over the f64 request.
struct Req<'a> {
    v: &'a [f64],
    i: usize,
}

impl Req<'_> {
    fn f(&mut self) -> R<f64> {
        let x = *self.v.get(self.i).ok_or_else(|| Fail::Request("request ended early".into()))?;
        self.i += 1;
        Ok(x)
    }
    fn int(&mut self) -> R<i64> {
        let x = self.f()?;
        if x.fract() != 0.0 || !x.is_finite() || x.abs() > 9007199254740992.0 {
            return Err(Fail::Request(format!("expected an integer, got {x}")));
        }
        Ok(x as i64)
    }
    fn usize(&mut self) -> R<usize> {
        let x = self.int()?;
        usize::try_from(x).map_err(|_| Fail::Request(format!("expected a non-negative integer, got {x}")))
    }
    fn isize(&mut self) -> R<isize> {
        Ok(self.int()? as isize)
    }
    fn flag(&mut self) -> R<bool> {
        Ok(self.int()? != 0)
    }
    /// NaN means "not given".
    fn opt(&mut self) -> R<Option<f64>> {
        let x = self.f()?;
        Ok(if x.is_nan() { None } else { Some(x) })
    }
    fn usizes(&mut self, n: usize) -> R<Vec<usize>> {
        (0..n).map(|_| self.usize()).collect()
    }
    fn isizes(&mut self, n: usize) -> R<Vec<isize>> {
        (0..n).map(|_| self.isize()).collect()
    }
    /// `len, values`; `len = -1` means none.
    fn list(&mut self) -> R<Option<Vec<f64>>> {
        let len = self.int()?;
        if len < 0 {
            return Ok(None);
        }
        Ok(Some((0..len).map(|_| self.f()).collect::<R<_>>()?))
    }
    fn dtype(&mut self) -> R<DType> {
        let c = self.int()?;
        dtype_from_code(c)?.ok_or_else(|| Fail::Request(format!("unknown dtype code {c}")))
    }
    /// A dtype code, or -1 for none.
    fn opt_dtype(&mut self) -> R<Option<DType>> {
        dtype_from_code(self.int()?)
    }
}

fn dtype_from_code(c: i64) -> R<Option<DType>> {
    Ok(Some(match c {
        -1 => return Ok(None),
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::U16,
        3 => DType::I32,
        4 => DType::I64,
        5 => DType::Bool,
        6 => DType::U8,
        c => return Err(Fail::Request(format!("unknown dtype code {c}"))),
    }))
}

fn isa_code(i: Isa) -> i32 {
    i as u8 as i32
}

fn isa_from_code(c: i64) -> R<Option<Isa>> {
    Ok(match c {
        -1 => None,
        0 => Some(Isa::Scalar),
        1 => Some(Isa::Avx2),
        2 => Some(Isa::Avx512),
        3 => Some(Isa::Neon),
        4 => Some(Isa::Simd128),
        c => return Err(Fail::Request(format!("unknown ISA code {c}"))),
    })
}

/// Clip fields: method, low, high, maxiters, preset, cenfunc, stdfunc, combine, precision,
/// mask_infinite, cutoff. Codes follow `js/rankwise.js`.
fn parse_clip(r: &mut Req) -> R<Clip> {
    let method = r.int()?;
    let low = r.f()?;
    let high = r.f()?;
    let maxiters = match r.opt()? {
        Some(m) if m >= 0.0 && m.fract() == 0.0 && m <= u32::MAX as f64 => Some(m as u32),
        Some(m) => return Err(param(format!("maxiters must be a non-negative integer, got {m}"))),
        None => None,
    };
    let preset = r.int()?;
    let cenfunc = r.int()?;
    let stdfunc = r.int()?;
    let combine = r.int()?;
    let precision = r.int()?;
    let mask_infinite = r.flag()?;
    let cutoff = r.opt()?;
    const SIGMA: i64 = 0;
    const MAD: i64 = 1;
    let mut c = match (preset, method) {
        (2, 3) => presets::pixinsight_winsorized(low, high).max_iters(maxiters),
        (2, _) => return Err(param("the pixinsight preset covers method 'winsorized' only")),
        (1, SIGMA) => presets::siril_sigma(low, high),
        (1, MAD) => {
            return Err(param(
                "Siril's k-MAD clipping is not reproduced yet (see docs/presets.md); use method 'mad' without a preset for astropy's mad_std",
            ))
        }
        (_, SIGMA) => Clip::sigma(low, high).max_iters(maxiters),
        (_, MAD) => Clip::mad(low, high).max_iters(maxiters),
        (_, 2) => Clip::median_sigma(low, high),
        (_, 3) => Clip::winsorized(low, high).max_iters(maxiters),
        (_, 4) => Clip::percentile(low, high),
        (_, 5) => Clip::none(),
        (_, m) => return Err(Fail::Request(format!("unknown clip method code {m}"))),
    };
    c = c.center(match cenfunc {
        0 => Center::Median,
        1 => Center::Mean,
        x => return Err(Fail::Request(format!("unknown cenfunc code {x}"))),
    });
    if stdfunc >= 0 {
        let s = if stdfunc == 0 { Spread::Std } else { Spread::MadStd };
        if method == MAD && s != Spread::MadStd {
            return Err(param(
                "method 'mad' clips with mad_std; use method 'sigma' with stdfunc 'std' for that spread",
            ));
        }
        if method != SIGMA && method != MAD {
            return Err(param("stdfunc applies to method 'sigma'"));
        }
        c = c.spread(s);
    }
    c = c.combine(match combine {
        0 => Combine::Mean,
        1 => Combine::Median,
        2 => Combine::Sum,
        x => return Err(Fail::Request(format!("unknown combine code {x}"))),
    });
    c = c.precision(if precision == 1 { Precision::F32 } else { Precision::F64 });
    c.mask_infinite = mask_infinite;
    if let (Some(cut), Rejection::Winsorized { cutoff, .. }) = (cutoff, &mut c.rejection) {
        *cutoff = cut;
    }
    Ok(c)
}

fn parse_op(r: &mut Req) -> R<Op> {
    Ok(match r.int()? {
        0 => {
            let k = r.usize()?;
            Op::SelectRanks(r.usizes(k)?)
        }
        1 => Op::Median,
        2 => {
            let nq = r.usize()?;
            let q = (0..nq).map(|_| r.f()).collect::<R<Vec<f64>>>()?;
            let method = QuantileMethod::ALL
                .get(r.usize()?)
                .copied()
                .ok_or_else(|| Fail::Request("unknown quantile method code".into()))?;
            let form = match r.int()? {
                0 => QForm::Weak,
                1 => QForm::WeakInt,
                2 => QForm::F64,
                3 => QForm::F32,
                x => return Err(Fail::Request(format!("unknown q form code {x}"))),
            };
            let out_dtype = r.opt_dtype()?;
            let out_rounding = if r.flag()? { OutRounding::CastOnce } else { OutRounding::NumpyQuantile };
            Op::Quantile(QuantileSpec { q, method, form, out_dtype, out_rounding })
        }
        3 => Op::Min,
        4 => Op::Max,
        5 => Op::Range,
        6 => Op::ArgMin,
        7 => Op::ArgMax,
        8 => Op::ArgRank(ArgRank::Median),
        9 => Op::ArgRank(ArgRank::Fixed(r.usize()?)),
        10 => Op::TrimMean(r.f()?),
        11 => Op::Iqr { lo: r.f()?, hi: r.f()?, scale: r.f()? },
        12 => Op::Mad { scale: r.f()? },
        13 => Op::Clip(parse_clip(r)?),
        k => return Err(Fail::Request(format!("unknown op code {k}"))),
    })
}

struct Parsed {
    op: Op,
    dtype: DType,
    frames: Vec<*const u8>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    outs: Vec<(*mut u8, DType, Vec<isize>)>,
    mask: Option<(Vec<*mut u8>, DType, Vec<isize>)>,
    opts: Options,
}

fn parse(v: &[f64]) -> R<Parsed> {
    let mut r = Req { v, i: 0 };
    let op = parse_op(&mut r)?;
    let dtype = r.dtype()?;
    let n = r.usize()?;
    if n > 1 << 20 {
        return Err(Fail::Core(Error::BadDepth(n)));
    }
    let ndim = r.usize()?;
    if ndim > 32 {
        return Err(Fail::Request(format!("{ndim} dimensions")));
    }
    let shape = r.usizes(ndim)?;
    let strides = r.isizes(ndim)?;
    let frames = r.usizes(n)?.into_iter().map(|a| a as *const u8).collect();
    let nouts = r.usize()?;
    if nouts > 4096 {
        return Err(Fail::Request(format!("{nouts} outputs")));
    }
    let mut outs = Vec::with_capacity(nouts);
    for _ in 0..nouts {
        let addr = r.usize()? as *mut u8;
        let dt = r.dtype()?;
        outs.push((addr, dt, r.isizes(ndim)?));
    }
    let mask = if r.flag()? {
        let fr = r.usizes(n)?.into_iter().map(|a| a as *mut u8).collect();
        let dt = r.dtype()?;
        Some((fr, dt, r.isizes(ndim)?))
    } else {
        None
    };
    let omit = r.flag()?;
    let nodata = r.opt()?;
    let isa = isa_from_code(r.int()?)?;
    let staging = match r.int()? {
        0 => Staging::Auto,
        1 => Staging::Never,
        2 => Staging::Always,
        3 => Staging::Skew,
        x => return Err(Fail::Request(format!("unknown staging code {x}"))),
    };
    let scale = r.list()?;
    let offset = r.list()?;
    let weights = r.list()?;
    if r.i != v.len() {
        return Err(Fail::Request(format!("{} trailing values", v.len() - r.i)));
    }
    let opts = Options {
        nan: if omit { NanPolicy::Omit } else { NanPolicy::Propagate },
        nodata,
        normalize: if scale.is_some() || offset.is_some() { Some(Normalize { scale, offset }) } else { None },
        weights,
        isa,
        threads: 1,
        staging,
    };
    Ok(Parsed { op, dtype, frames, shape, strides, outs, mask, opts })
}

fn run(p: &Parsed, describe_only: bool) -> R<String> {
    let stack = RawStack { dtype: p.dtype, frames: &p.frames, shape: &p.shape, strides: &p.strides };
    if describe_only {
        return Ok(rankwise::describe(&stack, &p.op, &p.opts)?);
    }
    let outs: Vec<RawOut> = p.outs.iter().map(|o| RawOut { ptr: o.0, dtype: o.1, strides: &o.2 }).collect();
    let rm = p.mask.as_ref().map(|m| RawMask { frames: &m.0, dtype: m.1, strides: &m.2 });
    // SAFETY: the JS layer only passes addresses of live allocations in this module's linear
    // memory, sized from the same shape and strides it sends here.
    unsafe { reduce_raw(&stack, &p.op, &outs, rm.as_ref(), &p.opts)? };
    Ok(String::new())
}

// ------------------------------------------------------------------ exports

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
    report(parse(v).and_then(|p| run(&p, false).map(|_| ())))
}

/// Describe how a request would run; the text is left in the message buffer.
#[no_mangle]
pub unsafe extern "C" fn rw_describe(req: *const f64, len: usize) -> i32 {
    let v = core::slice::from_raw_parts(req, len);
    report(parse(v).and_then(|p| run(&p, true)).map(set_message))
}

/// The last error message or `rw_describe` text (UTF-8).
#[no_mangle]
pub extern "C" fn rw_message_ptr() -> *const u8 {
    MESSAGE.with(|m| m.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn rw_message_len() -> usize {
    MESSAGE.with(|m| m.borrow().len())
}

/// Bit `i` set when ISA code `i` is available (0 scalar, 4 simd128).
#[no_mangle]
pub extern "C" fn rw_available_isas() -> u32 {
    Isa::available().into_iter().fold(0, |b, i| b | (1 << isa_code(i)))
}

/// The ISA code calls use now.
#[no_mangle]
pub extern "C" fn rw_current_isa() -> i32 {
    rankwise::resolve_isa(None).map(isa_code).unwrap_or(-1)
}

/// Force an ISA by code (-1 restores the default).
#[no_mangle]
pub extern "C" fn rw_set_isa(code: i32) -> i32 {
    report(isa_from_code(code as i64).and_then(|i| Ok(rankwise::set_isa_override(i)?)))
}

/// Comparators in the network for `n` values, pruned to `nranks` ranks at `ranks` (or the
/// full sort when `nranks` is 0).
#[no_mangle]
pub unsafe extern "C" fn rw_network_size(n: usize, ranks: *const u32, nranks: usize) -> usize {
    if n == 0 || n > rankwise::MAX_FRAMES {
        return 0;
    }
    if nranks == 0 {
        return rankwise::network_size(n, None);
    }
    let r: Vec<usize> = core::slice::from_raw_parts(ranks, nranks).iter().map(|&x| x as usize).collect();
    if r.iter().any(|&x| x >= n) {
        return 0;
    }
    rankwise::network_size(n, Some(&r))
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
