//! Safe API over frame lists: one slice per frame, every slice the same length.

use crate::layout::{RawMask, RawOut, RawStack};
use crate::plan::{reduce_raw, Options};
use crate::spec::{Clip, DType, NanPolicy, Op, QForm, QuantileSpec};
use crate::{Error, Result};

/// Input element types: `f32`, `f64`, `u16`.
pub trait Sample: Copy + Send + Sync + 'static {
    const DTYPE: DType;
}
impl Sample for f32 {
    const DTYPE: DType = DType::F32;
}
impl Sample for f64 {
    const DTYPE: DType = DType::F64;
}
impl Sample for u16 {
    const DTYPE: DType = DType::U16;
}

/// A caller-owned output buffer.
pub enum Out<'a> {
    F32(&'a mut [f32]),
    F64(&'a mut [f64]),
    U16(&'a mut [u16]),
    I32(&'a mut [i32]),
    I64(&'a mut [i64]),
    /// Not requested.
    None,
}

impl Out<'_> {
    fn raw(&mut self) -> (RawOut<'static>, Option<usize>) {
        static S4: [isize; 1] = [4];
        static S8: [isize; 1] = [8];
        static S2: [isize; 1] = [2];
        let (ptr, dtype, strides, len): (*mut u8, DType, &'static [isize], usize) = match self {
            Out::F32(s) => (s.as_mut_ptr() as *mut u8, DType::F32, &S4, s.len()),
            Out::F64(s) => (s.as_mut_ptr() as *mut u8, DType::F64, &S8, s.len()),
            Out::U16(s) => (s.as_mut_ptr() as *mut u8, DType::U16, &S2, s.len()),
            Out::I32(s) => (s.as_mut_ptr() as *mut u8, DType::I32, &S4, s.len()),
            Out::I64(s) => (s.as_mut_ptr() as *mut u8, DType::I64, &S8, s.len()),
            Out::None => return (RawOut::none(), None),
        };
        (RawOut { ptr, dtype, strides }, Some(len))
    }
}

impl<'a> From<&'a mut [f32]> for Out<'a> {
    fn from(s: &'a mut [f32]) -> Self {
        Out::F32(s)
    }
}
impl<'a> From<&'a mut [f64]> for Out<'a> {
    fn from(s: &'a mut [f64]) -> Self {
        Out::F64(s)
    }
}
impl<'a> From<&'a mut [u16]> for Out<'a> {
    fn from(s: &'a mut [u16]) -> Self {
        Out::U16(s)
    }
}
impl<'a> From<&'a mut [i64]> for Out<'a> {
    fn from(s: &'a mut [i64]) -> Self {
        Out::I64(s)
    }
}
impl<'a> From<&'a mut Vec<f32>> for Out<'a> {
    fn from(s: &'a mut Vec<f32>) -> Self {
        Out::F32(s.as_mut_slice())
    }
}

/// Reduce a list of frames with any [`Op`]. `outs` needs [`Op::n_outputs`] entries, each as
/// long as a frame (or [`Out::None`] for optional outputs). `mask`, when given, receives the
/// rejection map (one slice per frame; clipping only).
pub fn reduce_frames<T: Sample>(
    frames: &[&[T]],
    op: &Op,
    outs: &mut [Out],
    mask: Option<&mut [&mut [bool]]>,
    opts: &Options,
) -> Result<()> {
    let n = frames.len();
    if n == 0 {
        return Err(Error::BadDepth(0));
    }
    let npix = frames[0].len();
    if let Some(i) = frames.iter().position(|f| f.len() != npix) {
        return Err(Error::Shape(format!("frame {i} has {} pixels, frame 0 has {npix}", frames[i].len())));
    }
    let ptrs: Vec<*const u8> = frames.iter().map(|f| f.as_ptr() as *const u8).collect();
    let shape = [npix];
    let strides = [core::mem::size_of::<T>() as isize];
    let stack = RawStack { dtype: T::DTYPE, frames: &ptrs, shape: &shape, strides: &strides };
    let mut raws = Vec::with_capacity(outs.len());
    for (j, o) in outs.iter_mut().enumerate() {
        let (r, len) = o.raw();
        if let Some(l) = len {
            if l != npix {
                return Err(Error::Shape(format!("output {j} has {l} elements, frames have {npix}")));
            }
        }
        raws.push(r);
    }
    let mut mptrs: Vec<*mut u8> = vec![];
    let mstrides = [1isize];
    let rm;
    let mask_ref = match mask {
        Some(m) => {
            if m.len() != n || m.iter().any(|s| s.len() != npix) {
                return Err(Error::Shape("rejection map must match the frames".into()));
            }
            mptrs = m.iter_mut().map(|s| s.as_mut_ptr() as *mut u8).collect();
            rm = RawMask { frames: &mptrs, dtype: DType::Bool, strides: &mstrides };
            Some(&rm)
        }
        None => None,
    };
    let _ = &mptrs;
    // SAFETY: slices are valid for npix elements and the borrow checker keeps outputs
    // disjoint from the shared input borrows.
    unsafe { reduce_raw(&stack, op, &raws, mask_ref, opts) }
}

/// `np.median(stack, axis=0)` over a list of frames.
pub fn median_frames(frames: &[&[f32]], out: &mut [f32]) -> Result<()> {
    reduce_frames(frames, &Op::Median, &mut [Out::F32(out)], None, &Options::default())
}

/// `np.nanmedian(stack, axis=0)` over a list of frames.
pub fn nanmedian_frames(frames: &[&[f32]], out: &mut [f32]) -> Result<()> {
    let o = Options { nan: NanPolicy::Omit, ..Options::default() };
    reduce_frames(frames, &Op::Median, &mut [Out::F32(out)], None, &o)
}

/// `np.quantile(stack, q, axis=0)` with `q` a list of floats: interpolates in float64 and
/// writes float64, exactly as NumPy does.
pub fn quantile_frames(frames: &[&[f32]], q: &[f64], outs: &mut [&mut [f64]]) -> Result<()> {
    let op = Op::Quantile(QuantileSpec { form: QForm::F64, ..QuantileSpec::new(q.to_vec()) });
    let mut o: Vec<Out> = outs.iter_mut().map(|s| Out::F64(s)).collect();
    reduce_frames(frames, &op, &mut o, None, &Options::default())
}

/// `np.sort(stack, axis=0)[ranks]` in one pass, keeping only the comparators those ranks need.
pub fn select_ranks_frames<T: Sample>(frames: &[&[T]], ranks: &[usize], outs: &mut [&mut [T]]) -> Result<()>
where
    for<'a> Out<'a>: From<&'a mut [T]>,
{
    let mut o: Vec<Out> = outs.iter_mut().map(|s| Out::from(&mut **s)).collect();
    reduce_frames(frames, &Op::SelectRanks(ranks.to_vec()), &mut o, None, &Options::default())
}

/// Reject outliers with `clip` and combine the survivors into `out`. `low`/`high` receive
/// per-pixel reject counts when given.
pub fn clip_frames(
    frames: &[&[f32]],
    clip: &Clip,
    out: &mut [f32],
    low: Option<&mut [u16]>,
    high: Option<&mut [u16]>,
) -> Result<()> {
    let mut o = [Out::F32(out), low.map(Out::U16).unwrap_or(Out::None), high.map(Out::U16).unwrap_or(Out::None)];
    reduce_frames(frames, &Op::Clip(*clip), &mut o, None, &Options::default())
}

/// astropy-compatible sigma-clipped mean (median center, population std).
pub fn sigma_clip_mean_frames(frames: &[&[f32]], sigma_lower: f64, sigma_upper: f64, max_iters: Option<u32>, out: &mut [f32]) -> Result<()> {
    clip_frames(frames, &Clip::sigma(sigma_lower, sigma_upper).max_iters(max_iters), out, None, None)
}
