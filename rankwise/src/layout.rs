//! Strided problem descriptions and dimension collapsing.
//!
//! A problem is `n` frames that share one strided layout over the non-reduced dimensions
//! (a stack sliced along any axis, or a list of separate buffers), plus outputs laid out over
//! the same dimensions. The engine walks the innermost collapsed dimension as a "row" of
//! pixels and loops over the remaining dimensions.

use crate::spec::DType;
use crate::{Error, Result};

/// One output array over the non-reduced dimensions.
#[derive(Clone, Copy, Debug)]
pub struct RawOut<'a> {
    /// Null means "not requested" (for optional outputs such as reject counts).
    pub ptr: *mut u8,
    pub dtype: DType,
    /// Byte strides, one per non-reduced dimension.
    pub strides: &'a [isize],
}

impl RawOut<'_> {
    pub fn none() -> RawOut<'static> {
        RawOut { ptr: core::ptr::null_mut(), dtype: DType::F32, strides: &[] }
    }
}

/// A per-frame boolean output (the rejection map), laid out like the input.
#[derive(Clone, Copy, Debug)]
pub struct RawMask<'a> {
    pub frames: &'a [*mut u8],
    pub dtype: DType,
    pub strides: &'a [isize],
}

/// A strided input stack.
#[derive(Clone, Copy, Debug)]
pub struct RawStack<'a> {
    pub dtype: DType,
    /// Address of element 0 of every frame.
    pub frames: &'a [*const u8],
    /// Shape of the non-reduced dimensions.
    pub shape: &'a [usize],
    /// Byte strides of the non-reduced dimensions, shared by every frame.
    pub strides: &'a [isize],
}

/// Collapsed iteration space.
#[derive(Clone, Debug)]
pub(crate) struct Space {
    /// Outer dimensions (row loop), outermost first.
    pub outer_shape: Vec<usize>,
    pub outer_in: Vec<isize>,
    pub outer_out: Vec<Vec<isize>>,
    pub outer_mask: Vec<isize>,
    /// Pixels per row and their byte strides.
    pub npix: usize,
    pub in_ps: isize,
    pub out_ps: Vec<isize>,
    pub mask_ps: isize,
}

impl Space {
    pub fn rows(&self) -> usize {
        self.outer_shape.iter().product()
    }

    /// Byte offsets of row `r` for the input, each output and the mask.
    pub fn row_offsets(&self, mut r: usize, outs: &mut [isize]) -> (isize, isize) {
        let mut oi = 0isize;
        let mut om = 0isize;
        for o in outs.iter_mut() {
            *o = 0;
        }
        for d in (0..self.outer_shape.len()).rev() {
            let s = self.outer_shape[d];
            let i = (r % s) as isize;
            r /= s;
            oi += i * self.outer_in[d];
            om += i * self.outer_mask[d];
            for (j, o) in outs.iter_mut().enumerate() {
                *o += i * self.outer_out[j][d];
            }
        }
        (oi, om)
    }
}

pub(crate) fn collapse(
    shape: &[usize],
    in_strides: &[isize],
    out_strides: &[&[isize]],
    mask_strides: Option<&[isize]>,
) -> Result<Space> {
    let nd = shape.len();
    if in_strides.len() != nd {
        return Err(Error::Shape(format!(
            "input has {} strides for {} dimensions",
            in_strides.len(),
            nd
        )));
    }
    for (j, s) in out_strides.iter().enumerate() {
        if s.len() != nd {
            return Err(Error::Shape(format!(
                "output {j} has {} dimensions, expected {nd}",
                s.len()
            )));
        }
    }
    if let Some(m) = mask_strides {
        if m.len() != nd {
            return Err(Error::Shape("rejection map has the wrong number of dimensions".into()));
        }
    }
    // (shape, in, outs.., mask)
    let mut dims: Vec<(usize, isize, Vec<isize>, isize)> = (0..nd)
        .filter(|&d| shape[d] != 1)
        .map(|d| {
            (
                shape[d],
                in_strides[d],
                out_strides.iter().map(|s| s[d]).collect(),
                mask_strides.map(|m| m[d]).unwrap_or(0),
            )
        })
        .collect();
    if dims.iter().any(|d| d.0 == 0) {
        return Ok(Space {
            outer_shape: vec![],
            outer_in: vec![],
            outer_out: vec![vec![]; out_strides.len()],
            outer_mask: vec![],
            npix: 0,
            in_ps: 0,
            out_ps: vec![0; out_strides.len()],
            mask_ps: 0,
        });
    }
    // Innermost = smallest input stride, so frames are read along their contiguous axis.
    dims.sort_by_key(|d| core::cmp::Reverse(d.1.unsigned_abs()));
    // Merge (outer i, inner i+1) when every array is contiguous across the pair.
    let mut merged: Vec<(usize, isize, Vec<isize>, isize)> = Vec::new();
    for d in dims {
        if let Some(last) = merged.last_mut() {
            let ok = last.1 == d.1 * d.0 as isize
                && last.2.iter().zip(d.2.iter()).all(|(a, b)| *a == *b * d.0 as isize)
                && last.3 == d.3 * d.0 as isize;
            if ok {
                last.0 *= d.0;
                last.1 = d.1;
                last.2 = d.2.clone();
                last.3 = d.3;
                continue;
            }
        }
        merged.push(d);
    }
    let inner = merged.pop().unwrap_or((1, 0, vec![0; out_strides.len()], 0));
    let nout = out_strides.len();
    Ok(Space {
        outer_shape: merged.iter().map(|d| d.0).collect(),
        outer_in: merged.iter().map(|d| d.1).collect(),
        outer_out: (0..nout).map(|j| merged.iter().map(|d| d.2[j]).collect()).collect(),
        outer_mask: merged.iter().map(|d| d.3).collect(),
        npix: inner.0,
        in_ps: inner.1,
        out_ps: inner.2,
        mask_ps: inner.3,
    })
}

/// Contiguous (C-order) byte strides for `shape`.
pub fn c_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    let mut s = vec![0isize; shape.len()];
    let mut acc = itemsize as isize;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc *= shape[d] as isize;
    }
    s
}
