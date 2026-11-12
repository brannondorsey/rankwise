//! The request format both front ends share.
//!
//! A call is described by a request: a `Float64Array` the JS layer builds (every count, code,
//! address and stride is an exact integer in an f64). Its layout:
//!
//! ```text
//! op      kind, kind-specific fields                      (see parse_op)
//! stack   in_dtype, n, ndim, shape[ndim], strides[ndim], frame_addr[n]
//! outs    nouts, { addr, dtype, strides[ndim] } * nouts   (addr 0 = not requested)
//! mask    0 | 1, frame_addr[n], dtype, strides[ndim]
//! opts    omit, nodata (NaN = none), isa (-1 = auto), staging, threads,
//!         list scale, list offset, list weights            (list = len, values; len -1 = none)
//! ```
//!
//! Strides are in bytes. What an address means is up to the front end: a linear-memory
//! pointer in WebAssembly, a (buffer, offset) pair natively (see [`Resolve`]), where every
//! region a call touches is bounds-checked before any kernel runs. Errors return a nonzero
//! status and leave a message in a thread-local buffer.

use rankwise::{
    presets, reduce_raw, ArgRank, Center, Clip, Combine, DType, Error, Isa, NanPolicy, Normalize, Op, Options,
    OutRounding, Precision, QForm, QuantileMethod, QuantileSpec, RawMask, RawOut, RawStack, Rejection, Spread, Staging,
};
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

pub(crate) fn set_message(s: String) {
    MESSAGE.with(|m| *m.borrow_mut() = s);
}

/// Run `f` with the last error message or describe text.
pub(crate) fn with_message<T>(f: impl FnOnce(&str) -> T) -> T {
    MESSAGE.with(|m| f(&m.borrow()))
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
pub(crate) enum Fail {
    Core(Error),
    Request(String),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Core(e)
    }
}

pub(crate) type R<T> = core::result::Result<T, Fail>;

fn param(s: impl Into<String>) -> Fail {
    Fail::Core(Error::Param(s.into()))
}

pub(crate) fn report(r: R<()>) -> i32 {
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

pub(crate) fn isa_code(i: Isa) -> i32 {
    i as u8 as i32
}

pub(crate) fn isa_from_code(c: i64) -> R<Option<Isa>> {
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

/// Front-end address resolution: turn `addr` into a pointer valid for the byte range
/// `addr + lo .. addr + hi` (an empty range is never dereferenced).
pub(crate) type Resolve<'a> = &'a dyn Fn(u64, isize, isize) -> R<*mut u8>;

struct Parsed {
    op: Op,
    dtype: DType,
    frames: Vec<u64>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    outs: Vec<(u64, DType, Vec<isize>)>,
    mask: Option<(Vec<u64>, DType, Vec<isize>)>,
    opts: Options,
}

fn addrs(r: &mut Req, n: usize) -> R<Vec<u64>> {
    (0..n).map(|_| Ok(r.usize()? as u64)).collect()
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
    let frames = addrs(&mut r, n)?;
    let nouts = r.usize()?;
    if nouts > 4096 {
        return Err(Fail::Request(format!("{nouts} outputs")));
    }
    let mut outs = Vec::with_capacity(nouts);
    for _ in 0..nouts {
        let addr = r.usize()? as u64;
        let dt = r.dtype()?;
        outs.push((addr, dt, r.isizes(ndim)?));
    }
    let mask = if r.flag()? {
        let fr = addrs(&mut r, n)?;
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
    let threads = r.usize()?.clamp(1, 1024);
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
        threads,
        staging,
    };
    Ok(Parsed { op, dtype, frames, shape, strides, outs, mask, opts })
}

/// Byte range `lo..hi` an array with this layout touches from its base, or `None` when it
/// has no elements.
fn extent(shape: &[usize], strides: &[isize], esz: usize) -> R<Option<(isize, isize)>> {
    if shape.contains(&0) {
        return Ok(None);
    }
    let (mut lo, mut hi) = (0i128, esz as i128);
    for (&n, &s) in shape.iter().zip(strides) {
        let span = (n as i128 - 1) * s as i128;
        if span < 0 {
            lo += span;
        } else {
            hi += span;
        }
    }
    let fits = |x: i128| x >= isize::MIN as i128 && x <= isize::MAX as i128;
    if !fits(lo) || !fits(hi) {
        return Err(Fail::Request("layout spans more than the address space".into()));
    }
    Ok(Some((lo as isize, hi as isize)))
}

/// Resolve every address, checking each region against its buffer, before any kernel runs.
fn place(addr: u64, ext: Option<(isize, isize)>, resolve: Resolve) -> R<*mut u8> {
    match ext {
        Some((lo, hi)) => resolve(addr, lo, hi),
        None => resolve(addr, 0, 0),
    }
}

/// Parse and run (or describe) a request.
pub(crate) fn execute(v: &[f64], describe_only: bool, resolve: Resolve) -> R<String> {
    let p = parse(v)?;
    if p.shape.len() != p.strides.len() {
        return Err(Fail::Request("shape and strides differ in length".into()));
    }
    let fext = extent(&p.shape, &p.strides, p.dtype.size())?;
    let frames: Vec<*const u8> =
        p.frames.iter().map(|&a| place(a, fext, resolve).map(|q| q as *const u8)).collect::<R<_>>()?;
    let stack = RawStack { dtype: p.dtype, frames: &frames, shape: &p.shape, strides: &p.strides };
    if describe_only {
        return Ok(rankwise::describe(&stack, &p.op, &p.opts)?);
    }
    let mut outs = Vec::with_capacity(p.outs.len());
    for (a, dt, st) in &p.outs {
        if st.len() != p.shape.len() {
            return Err(Fail::Request("output strides do not match the shape".into()));
        }
        let ptr = if *a == 0 { core::ptr::null_mut() } else { place(*a, extent(&p.shape, st, dt.size())?, resolve)? };
        outs.push(RawOut { ptr, dtype: *dt, strides: st });
    }
    let mptrs: Vec<*mut u8> = match &p.mask {
        Some((fr, dt, st)) => {
            if st.len() != p.shape.len() {
                return Err(Fail::Request("mask strides do not match the shape".into()));
            }
            let ext = extent(&p.shape, st, dt.size())?;
            fr.iter().map(|&a| place(a, ext, resolve)).collect::<R<_>>()?
        }
        None => vec![],
    };
    let rm = p.mask.as_ref().map(|m| RawMask { frames: &mptrs, dtype: m.1, strides: &m.2 });
    // SAFETY: every region was resolved by the front end for exactly the bytes this layout
    // touches (bounds-checked natively; linear memory in WebAssembly).
    unsafe { reduce_raw(&stack, &p.op, &outs, rm.as_ref(), &p.opts)? };
    Ok(String::new())
}

/// Comparators in the network for `n` values, pruned to `ranks` (full sort when empty);
/// 0 for an invalid request.
pub(crate) fn network_size(n: usize, ranks: &[usize]) -> usize {
    if n == 0 || n > rankwise::MAX_FRAMES || ranks.iter().any(|&x| x >= n) {
        return 0;
    }
    rankwise::network_size(n, if ranks.is_empty() { None } else { Some(ranks) })
}

/// Bit `i` set when ISA code `i` is available.
pub(crate) fn available_isas() -> u32 {
    Isa::available().into_iter().fold(0, |b, i| b | (1 << isa_code(i)))
}

pub(crate) fn current_isa() -> i32 {
    rankwise::resolve_isa(None).map(isa_code).unwrap_or(-1)
}

pub(crate) fn set_isa(code: i32) -> i32 {
    report(isa_from_code(code as i64).and_then(|i| Ok(rankwise::set_isa_override(i)?)))
}
