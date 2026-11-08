//! `ndarray` integration: reduce any axis of an array view into a caller-owned output.
//!
//! ```
//! use ndarray::{Array2, Array3, Axis};
//! use rankwise::{Clip, Reduce};
//!
//! let stack = Array3::<f32>::from_shape_fn((31, 64, 64), |(k, y, x)| (k * 7 + y * 3 + x) as f32 % 97.0);
//! let mut out = Array2::<f32>::zeros((64, 64));
//! stack.reduce_into(Axis(0), Clip::sigma(3.0, 3.0).max_iters(5).mean(), out.view_mut()).unwrap();
//! ```

use crate::frames::Sample;
use crate::layout::{RawOut, RawStack};
use crate::plan::{reduce_raw, Options};
use crate::spec::{Clip, DType, Op};
use crate::{Error, Result};
use ndarray::{ArrayBase, ArrayViewMut, Axis, Data, Dimension, RemoveAxis};

/// Output element types: `f32`, `f64`, `u16`, `i64`.
pub trait OutSample: Copy {
    const DTYPE: DType;
}
impl OutSample for f32 {
    const DTYPE: DType = DType::F32;
}
impl OutSample for f64 {
    const DTYPE: DType = DType::F64;
}
impl OutSample for u16 {
    const DTYPE: DType = DType::U16;
}
impl OutSample for i64 {
    const DTYPE: DType = DType::I64;
}

/// A reduction plus its options.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub op: Op,
    pub opts: Options,
}

impl From<Op> for Reduction {
    fn from(op: Op) -> Self {
        Reduction { op, opts: Options::default() }
    }
}
impl From<Clip> for Reduction {
    fn from(c: Clip) -> Self {
        Reduction { op: Op::Clip(c), opts: Options::default() }
    }
}

/// Reduce one axis of an array view.
pub trait Reduce<T: Sample, D: Dimension + RemoveAxis> {
    /// Reduce `axis` with a single-output op (median, a clipped mean, ...) into `out`, whose
    /// shape is the input's without `axis`.
    fn reduce_into<O: OutSample>(&self, axis: Axis, r: impl Into<Reduction>, out: ArrayViewMut<O, D::Smaller>) -> Result<()>;
}

impl<S, T, D> Reduce<T, D> for ArrayBase<S, D>
where
    S: Data<Elem = T>,
    T: Sample,
    D: Dimension + RemoveAxis,
{
    fn reduce_into<O: OutSample>(&self, axis: Axis, r: impl Into<Reduction>, mut out: ArrayViewMut<O, D::Smaller>) -> Result<()> {
        let r = r.into();
        let a = axis.index();
        if a >= self.ndim() {
            return Err(Error::Shape(format!("axis {a} out of range for {} dimensions", self.ndim())));
        }
        let shape = self.shape();
        let rest: Vec<usize> = shape.iter().enumerate().filter(|(i, _)| *i != a).map(|(_, &s)| s).collect();
        if out.shape() != rest.as_slice() {
            return Err(Error::Shape(format!("out has shape {:?}, expected {:?}", out.shape(), rest)));
        }
        let es = core::mem::size_of::<T>() as isize;
        let st = self.strides();
        let base = self.as_ptr() as *const u8;
        let frames: Vec<*const u8> =
            (0..shape[a]).map(|k| unsafe { base.offset(k as isize * st[a] * es) }).collect();
        let strides: Vec<isize> = st.iter().enumerate().filter(|(i, _)| *i != a).map(|(_, &s)| s * es).collect();
        let os = core::mem::size_of::<O>() as isize;
        let ostrides: Vec<isize> = out.strides().iter().map(|&s| s * os).collect();
        let n_out = r.op.n_outputs();
        let mut outs = vec![RawOut { ptr: out.as_mut_ptr() as *mut u8, dtype: O::DTYPE, strides: &ostrides }];
        for _ in 1..n_out {
            if !matches!(r.op, Op::Clip(_)) {
                return Err(Error::Param("reduce_into takes single-output ops".into()));
            }
            outs.push(RawOut::none());
        }
        let stack = RawStack { dtype: T::DTYPE, frames: &frames, shape: &rest, strides: &strides };
        // SAFETY: the views are valid for their shapes and strides; `out` is borrowed mutably.
        unsafe { reduce_raw(&stack, &r.op, &outs, None, &r.opts) }
    }
}

impl Reduction {
    /// Attach options (threads, NaN policy, normalization, ...).
    pub fn with(mut self, opts: Options) -> Self {
        self.opts = opts;
        self
    }
}
