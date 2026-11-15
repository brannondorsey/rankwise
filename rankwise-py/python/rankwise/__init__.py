"""rankwise: lane-parallel SIMD order statistics along one axis.

Every SIMD lane holds one pixel's (or grid cell's) slice; a sorting network pruned to the
ranks you ask for sorts 16 slices per instruction on AVX-512. Results are identical to
NumPy (median, quantile with every method, the nan-variants), SciPy (trim_mean, iqr,
median_abs_deviation) and astropy (sigma_clip decisions).

    >>> import numpy as np, rankwise as rw
    >>> stack = np.random.default_rng(0).normal(size=(31, 512, 512)).astype(np.float32)
    >>> bool((rw.median(stack, axis=0) == np.median(stack, axis=0)).all())
    True

Inputs are float32, float64 or uint16 arrays (uint8 is widened), reduced along any axis
without copying, or a list of same-shape frames (memory-mapped files included). Outputs can
be caller-owned (``out=``). Kernels run with the GIL released; ``threads=`` splits the work.
"""

from __future__ import annotations

import contextlib
import operator
import warnings
from typing import Sequence

import numpy as np

from . import _rankwise as _rw

__version__ = _rw.__version__
MAX_FRAMES = _rw.MAX_FRAMES

__all__ = [
    "median", "nanmedian", "quantile", "nanquantile", "percentile", "nanpercentile",
    "min", "max", "amin", "amax", "nanmin", "nanmax", "ptp",
    "argmin", "argmax", "nanargmin", "nanargmax", "argmedian", "nanargmedian", "argselect",
    "select_ranks", "trimmed_mean", "trim_mean", "iqr", "mad", "median_abs_deviation",
    "sigma_clip_mean", "clip_combine",
    "available_isas", "set_isa", "current_isa", "isa_override", "use_isa", "describe", "network_size",
]

_NATIVE = {np.dtype(np.float32): "float32", np.dtype(np.float64): "float64", np.dtype(np.uint16): "uint16"}
_OUT_OK = {
    np.dtype(np.float32): "float32", np.dtype(np.float64): "float64", np.dtype(np.uint16): "uint16",
    np.dtype(np.int32): "int32", np.dtype(np.int64): "int64", np.dtype(np.bool_): "bool",
    np.dtype(np.uint8): "uint8",
}
_ALL_NAN = "All-NaN slice encountered"
_AxisError = getattr(np, "exceptions", np).AxisError  # np.exceptions is NumPy 1.25+


# --------------------------------------------------------------------------- plumbing

class _Stack:
    """A validated input: n frame addresses sharing one strided layout."""

    __slots__ = ("keep", "dtype", "orig_dtype", "n", "frames", "shape", "strides", "axis", "ndim", "first_slice")

    def describe_args(self):
        return self.dtype, self.frames, list(self.shape), list(self.strides)


def _native(arr: np.ndarray) -> np.ndarray:
    dt = arr.dtype
    if dt in _NATIVE and dt.isnative:
        return arr
    if dt.kind == "u" and dt.itemsize == 1:
        return arr.astype(np.uint16)
    nd = dt.newbyteorder("=")
    if nd in _NATIVE:
        # big-endian input (FITS, say): byte-swap into a temporary copy
        return arr.astype(nd)
    raise TypeError(
        f"rankwise reduces float32, float64 and uint16 data (uint8 is widened); got {dt}. "
        "Convert with .astype() first."
    )


def _is_frame_list(a) -> bool:
    return isinstance(a, (list, tuple)) and len(a) > 0 and all(
        isinstance(f, np.ndarray) or hasattr(f, "__array_interface__") for f in a
    )


def _prep(a, axis) -> _Stack:
    s = _Stack()
    if _is_frame_list(a):
        if axis not in (None, 0):
            raise _AxisError(f"a list of frames is reduced along axis 0, got axis={axis}")
        frames = [np.asarray(f) for f in a]
        shape, dt = frames[0].shape, frames[0].dtype
        for i, f in enumerate(frames):
            if f.shape != shape:
                raise ValueError(f"frame {i} has shape {f.shape}, frame 0 has {shape}")
            if f.dtype != dt:
                raise TypeError(f"frame {i} is {f.dtype}, frame 0 is {dt}")
        s.orig_dtype = dt
        frames = [_native(f) for f in frames]
        if any(f.strides != frames[0].strides for f in frames):
            frames = [np.ascontiguousarray(f) for f in frames]
        s.keep = frames
        s.dtype = _NATIVE[frames[0].dtype]
        s.n = len(frames)
        s.frames = [f.__array_interface__["data"][0] for f in frames]
        s.shape, s.strides = tuple(shape), tuple(frames[0].strides)
        s.axis, s.ndim = 0, len(shape) + 1
        s.first_slice = lambda: np.array([f.flat[0] if f.size else np.nan for f in frames])
        return s
    arr = np.asarray(a)
    if arr.ndim == 0:
        raise ValueError("cannot reduce a 0-d array")
    if axis is None:
        if arr.ndim == 1:
            axis = 0
        else:
            raise TypeError(
                "rankwise reduces one axis: pass axis=... (NumPy's axis=None flattens, "
                "which rankwise v0.1 does not do)"
            )
    if isinstance(axis, (tuple, list)):
        if len(axis) != 1:
            raise TypeError("rankwise reduces exactly one axis")
        axis = axis[0]
    axis = operator.index(axis)
    if not -arr.ndim <= axis < arr.ndim:
        raise _AxisError(axis, arr.ndim)
    axis %= arr.ndim
    s.orig_dtype = arr.dtype
    arr = _native(arr)
    s.keep = arr
    s.dtype = _NATIVE[arr.dtype]
    s.n = arr.shape[axis]
    base = arr.__array_interface__["data"][0]
    fs = arr.strides[axis]
    s.frames = [base + k * fs for k in range(s.n)]
    s.shape = arr.shape[:axis] + arr.shape[axis + 1:]
    s.strides = arr.strides[:axis] + arr.strides[axis + 1:]
    s.axis, s.ndim = axis, arr.ndim
    s.first_slice = lambda: _first_slice(arr, axis)
    if s.n == 0:
        raise ValueError("cannot reduce an empty axis")
    return s


def _first_slice(arr, axis):
    idx = [0] * arr.ndim
    idx[axis] = slice(None)
    if any(d == 0 for i, d in enumerate(arr.shape) if i != axis):
        return np.empty(0)
    return arr[tuple(idx)]


def _out_entry(o: np.ndarray | None, rest_ndim: int):
    if o is None:
        return (0, "uint16", [0] * rest_ndim)
    return (o.__array_interface__["data"][0], _OUT_OK[o.dtype], list(o.strides))


def _opts(*, omit=False, nodata=None, threads=1, isa=None, scale=None, offset=None, weights=None, staging=None):
    d = {"omit": bool(omit), "threads": int(threads or 1)}
    if nodata is not None:
        d["nodata"] = float(nodata)
    if isa is not None:
        d["isa"] = str(isa)
    if scale is not None:
        d["scale"] = [float(x) for x in np.asarray(scale, dtype=np.float64).ravel()]
    if offset is not None:
        d["offset"] = [float(x) for x in np.asarray(offset, dtype=np.float64).ravel()]
    if weights is not None:
        d["weights"] = [float(x) for x in np.asarray(weights, dtype=np.float64).ravel()]
    if staging is not None:
        d["staging"] = staging
    return d


def _check_nodata(s: _Stack, nodata):
    if nodata is not None and s.dtype != "uint16":
        raise TypeError("nodata applies to integer input; float data marks missing values with NaN")


def _prepare_out(out, shape, dtype, keepdims, axis, lead=()):
    """Return (array the kernel writes, final array to return, needs_copy)."""
    full = tuple(lead) + tuple(shape)
    if out is None:
        res = np.empty(full, dtype=dtype)
        return res, res, False
    if not isinstance(out, np.ndarray):
        raise TypeError("out must be a numpy array")
    target = out
    if keepdims:
        exp = list(full)
        exp.insert(len(lead) + axis, 1)
        if out.shape != tuple(exp):
            raise ValueError(f"out has shape {out.shape}, expected {tuple(exp)}")
        target = np.squeeze(out, axis=len(lead) + axis)
    elif out.shape != full:
        raise ValueError(f"out has shape {out.shape}, expected {full}")
    if target.dtype in _OUT_OK and target.dtype.isnative and target.flags.writeable:
        return target, out, False
    tmp = np.empty(full, dtype=dtype)
    return tmp, out, True


def _finish(work, final, copy, keepdims, axis, lead_len=0, out_given=False):
    if copy:
        dst = final
        if keepdims:
            dst = np.squeeze(final, axis=lead_len + axis)
        np.copyto(dst, work, casting="unsafe")
        res = final
    elif out_given:
        res = final
    else:
        res = work
        if keepdims:
            res = np.expand_dims(res, lead_len + axis)
    if isinstance(res, np.ndarray) and res.ndim == 0 and not out_given:
        return res[()]
    return res


def _cast_value_back(res, s: _Stack, value_like_input: bool):
    """uint8 input is widened to uint16; outputs in the input dtype go back to uint8."""
    if value_like_input and s.orig_dtype != np.dtype(np.uint16) and s.orig_dtype.kind == "u" and s.orig_dtype.itemsize == 1:
        return np.asarray(res).astype(s.orig_dtype) if isinstance(res, np.ndarray) else s.orig_dtype.type(res)
    return res


def _in_np_dtype(s: _Stack):
    return np.dtype(s.dtype)


def _run(op, s: _Stack, outs, opts, mask=None):
    rest = len(s.shape)
    entries = [_out_entry(o, rest) for o in outs]
    _rw.reduce(op, s.dtype, s.frames, list(s.shape), list(s.strides), entries, mask, opts)


def _simple(op, a, axis, out, keepdims, dtype_fn, *, omit=False, nodata=None, threads=1, isa=None,
            value_like_input=False, warn_nan=False):
    s = _prep(a, axis)
    _check_nodata(s, nodata)
    dt = dtype_fn(s)
    work, final, copy = _prepare_out(out, s.shape, dt, keepdims, s.axis)
    _run(op, s, [work], _opts(omit=omit, nodata=nodata, threads=threads, isa=isa))
    if warn_nan and work.dtype.kind == "f" and np.isnan(work).any():
        warnings.warn(_ALL_NAN, RuntimeWarning, stacklevel=3)
    res = _finish(work, final, copy, keepdims, s.axis, out_given=out is not None)
    return res if out is not None else _cast_value_back(res, s, value_like_input)


def _float_result(s: _Stack):
    return np.dtype(np.float32) if s.dtype == "float32" else np.dtype(np.float64)


# --------------------------------------------------------------------------- medians

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False, *, nodata=None, threads=1, isa=None):
    """``np.median`` along one axis. Identical results and dtype (float32 stays float32;
    integers give float64). A slice containing NaN gives NaN. ``nodata`` marks missing
    values in integer input; they are skipped, like NaN in :func:`nanmedian`."""
    return _simple({"kind": "median"}, a, axis, out, keepdims, _float_result,
                   omit=nodata is not None, nodata=nodata, threads=threads, isa=isa,
                   warn_nan=nodata is not None)


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False, *, nodata=None, threads=1, isa=None):
    """``np.nanmedian`` along one axis: NaN is ignored; an all-NaN slice gives NaN and a
    ``RuntimeWarning``."""
    if isinstance(a, np.ndarray) and a.ndim == 1 and out is None and not keepdims and a.dtype.kind == "f":
        # NumPy's 1-D path compresses out the NaNs and takes a plain median. Validate what
        # the general path would (axis, nodata, dtype) before taking it.
        if axis is not None:
            ax = axis[0] if isinstance(axis, (tuple, list)) and len(axis) == 1 else axis
            if not -1 <= operator.index(ax) < 1:
                raise _AxisError(ax, 1)
        if nodata is not None:
            raise TypeError("nodata applies to integer input; float data marks missing values with NaN")
        a = _native(a)
        x = a[~np.isnan(a)]
        if x.size == 0:
            warnings.warn(_ALL_NAN, RuntimeWarning, stacklevel=2)
            return a.dtype.type(np.nan)
        return median(x, threads=threads, isa=isa)
    return _simple({"kind": "median"}, a, axis, out, keepdims, _float_result,
                   omit=True, nodata=nodata, threads=threads, isa=isa, warn_nan=True)


# --------------------------------------------------------------------------- quantiles

_DISCRETE = {"inverted_cdf", "closest_observation", "lower", "higher", "nearest"}
_METHODS = {
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
    "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher",
    "midpoint", "nearest",
}
_INT_INDEX = {"weibull", "averaged_inverted_cdf", "interpolated_inverted_cdf"}


def _q_form(q, qa: np.ndarray, method: str, weak: bool, percent: bool):
    if weak:
        return "weak_int" if (isinstance(q, int) and not percent) else "weak"
    if qa.dtype == np.float32:
        return "f32"
    if qa.dtype.kind in "iub":
        return "weak_int" if method == "linear" else "f64"
    return "f64"


def _q_dtype(s: _Stack, form: str, method: str):
    data = s.dtype
    if method in _DISCRETE or (form == "weak_int" and method == "linear"):
        return np.dtype(data)
    if data == "float64":
        return np.dtype(np.float64)
    if form in ("weak", "weak_int"):
        return np.dtype(np.float32) if data == "float32" else np.dtype(np.float64)
    if form == "f32":
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _quantile_impl(a, q, axis, out, method, keepdims, *, omit, percent, nodata, threads, isa, interpolation=None):
    if interpolation is not None:
        method = interpolation
    if method not in _METHODS:
        raise ValueError(f"{method!r} is not a valid method. Use one of: {sorted(_METHODS)}")
    weak = type(q) in (int, float)
    if percent:
        # np.percentile's `true_divide(q, 100, out=...)`; out=... needs NumPy 2.3
        qa = np.asanyarray(np.true_divide(q, 100))
    else:
        qa = np.asanyarray(q)
    if qa.dtype.kind == "c":
        raise TypeError("q must be real")
    if qa.ndim > 2:
        raise ValueError("q must be a scalar or 1d")
    flat = qa.astype(np.float64, copy=False).ravel()
    if not ((flat >= 0) & (flat <= 1)).all():
        raise ValueError("Percentiles must be in the range [0, 100]" if percent else "Quantiles must be in the range [0, 1]")
    s = _prep(a, axis)
    _check_nodata(s, nodata)
    form = _q_form(q, qa, method, weak, percent)
    dt = _q_dtype(s, form, method)
    if omit and s.dtype == "float32" and dt == np.float64 and nodata is None:
        # np.nanquantile builds its result with apply_along_axis, whose dtype is the first
        # slice's: an all-NaN first slice yields a float32 array.
        fs = s.first_slice()
        if fs.size and np.isnan(fs).all():
            dt = np.dtype(np.float32)
    work, final, copy = _prepare_out(out, s.shape, dt, keepdims, s.axis, lead=qa.shape)
    # np.quantile's out= changes its arithmetic (the t >= 0.5 branch subtracts in out's
    # dtype); np.nanquantile computes first and casts once.
    op = {"kind": "quantile", "q": [float(x) for x in flat], "method": method, "form": form,
          "out_rounding": "cast_once" if omit else "numpy"}
    if out is not None and not copy:
        op["out_dtype"] = _OUT_OK[work.dtype]
    work2 = work.reshape((flat.size,) + tuple(s.shape))
    # work2[j, ...] is a view even for 1-D input (work2[j] would be a scalar copy, and the
    # kernel would write into a temporary)
    outs = [work2[j, ...] for j in range(flat.size)]
    if flat.size:
        _run(op, s, outs, _opts(omit=omit or nodata is not None, nodata=nodata, threads=threads, isa=isa))
    if omit and work.dtype.kind == "f" and np.isnan(work).any():
        warnings.warn(_ALL_NAN, RuntimeWarning, stacklevel=3)
    res = _finish(work, final, copy, keepdims, s.axis, lead_len=qa.ndim, out_given=out is not None)
    if out is None and (method in _DISCRETE or (form == "weak_int" and method == "linear")):
        res = _cast_value_back(res, s, True)
    elif out is None and qa.dtype.kind in "iub" and not weak and method in _INT_INDEX:
        # integer q: these methods' virtual index stays integer (int64 for bool
        # q) and NumPy's lerp returns its result type with the data; q is 0 or 1, so the
        # values are order statistics either way
        rt = np.result_type(s.orig_dtype, np.int64 if qa.dtype.kind == "b" else qa.dtype)
        res = res.astype(rt) if isinstance(res, np.ndarray) else rt.type(res)
    return res


def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *,
             weights=None, interpolation=None, nodata=None, threads=1, isa=None):
    """``np.quantile`` along one axis, every ``method``, bit-identical including dtype.

    How ``q`` is passed matters, as in NumPy: a Python float interpolates in the data's
    dtype (float32 in, float32 out); a list or float64 array interpolates in float64.
    """
    if weights is not None:
        raise NotImplementedError("weighted quantiles are planned for rankwise v0.2")
    return _quantile_impl(a, q, axis, out, method, keepdims, omit=False, percent=False,
                          nodata=nodata, threads=threads, isa=isa, interpolation=interpolation)


def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *,
                weights=None, interpolation=None, nodata=None, threads=1, isa=None):
    """``np.nanquantile`` along one axis."""
    if weights is not None:
        raise NotImplementedError("weighted quantiles are planned for rankwise v0.2")
    return _quantile_impl(a, q, axis, out, method, keepdims, omit=True, percent=False,
                          nodata=nodata, threads=threads, isa=isa, interpolation=interpolation)


def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *,
               weights=None, interpolation=None, nodata=None, threads=1, isa=None):
    """``np.percentile`` along one axis."""
    if weights is not None:
        raise NotImplementedError("weighted quantiles are planned for rankwise v0.2")
    return _quantile_impl(a, q, axis, out, method, keepdims, omit=False, percent=True,
                          nodata=nodata, threads=threads, isa=isa, interpolation=interpolation)


def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False, *,
                  weights=None, interpolation=None, nodata=None, threads=1, isa=None):
    """``np.nanpercentile`` along one axis."""
    if weights is not None:
        raise NotImplementedError("weighted quantiles are planned for rankwise v0.2")
    return _quantile_impl(a, q, axis, out, method, keepdims, omit=True, percent=True,
                          nodata=nodata, threads=threads, isa=isa, interpolation=interpolation)


# --------------------------------------------------------------------------- min / max / args

def _same(s):
    return np.dtype(s.dtype)


def min(a, axis=None, out=None, keepdims=False, *, nodata=None, threads=1, isa=None):  # noqa: A001
    """``np.min`` along one axis (NaN propagates)."""
    return _simple({"kind": "min"}, a, axis, out, keepdims, _same, omit=nodata is not None, nodata=nodata,
                   threads=threads, isa=isa, value_like_input=True)


def max(a, axis=None, out=None, keepdims=False, *, nodata=None, threads=1, isa=None):  # noqa: A001
    """``np.max`` along one axis (NaN propagates)."""
    return _simple({"kind": "max"}, a, axis, out, keepdims, _same, omit=nodata is not None, nodata=nodata,
                   threads=threads, isa=isa, value_like_input=True)


amin, amax = min, max


def nanmin(a, axis=None, out=None, keepdims=False, *, nodata=None, threads=1, isa=None):
    """``np.nanmin`` along one axis."""
    return _simple({"kind": "min"}, a, axis, out, keepdims, _same, omit=True, nodata=nodata,
                   threads=threads, isa=isa, value_like_input=True, warn_nan=True)


def nanmax(a, axis=None, out=None, keepdims=False, *, nodata=None, threads=1, isa=None):
    """``np.nanmax`` along one axis."""
    return _simple({"kind": "max"}, a, axis, out, keepdims, _same, omit=True, nodata=nodata,
                   threads=threads, isa=isa, value_like_input=True, warn_nan=True)


def ptp(a, axis=None, out=None, keepdims=False, *, nodata=None, threads=1, isa=None):
    """``np.ptp`` (max - min) along one axis, in the input dtype."""
    return _simple({"kind": "range"}, a, axis, out, keepdims, _same, omit=nodata is not None, nodata=nodata,
                   threads=threads, isa=isa, value_like_input=True)


def _intp(s):
    return np.dtype(np.intp)


def _arg(kind, a, axis, out, keepdims, *, omit, strict, nodata, threads, isa, extra=None, s=None):
    """``s``: an already prepared stack for ``a`` (argselect reuses one across ranks)."""
    op = {"kind": kind}
    if extra:
        op.update(extra)
    if s is None:
        s = _prep(a, axis)
    _check_nodata(s, nodata)
    work, final, copy = _prepare_out(out, s.shape, np.dtype(np.intp), keepdims, s.axis)
    _run(op, s, [work], _opts(omit=omit or nodata is not None, nodata=nodata, threads=threads, isa=isa))
    if (omit or nodata is not None) and (work < 0).any():
        if strict:
            raise ValueError(_ALL_NAN)
        warnings.warn(_ALL_NAN + "; returning -1 for those slices", RuntimeWarning, stacklevel=3)
    return _finish(work, final, copy, keepdims, s.axis, out_given=out is not None)


def argmin(a, axis=None, out=None, *, keepdims=False, nodata=None, threads=1, isa=None):
    """``np.argmin`` along one axis: first occurrence; the first NaN wins."""
    return _arg("argmin", a, axis, out, keepdims, omit=False, strict=False, nodata=nodata, threads=threads, isa=isa)


def argmax(a, axis=None, out=None, *, keepdims=False, nodata=None, threads=1, isa=None):
    """``np.argmax`` along one axis: first occurrence; the first NaN wins."""
    return _arg("argmax", a, axis, out, keepdims, omit=False, strict=False, nodata=nodata, threads=threads, isa=isa)


def nanargmin(a, axis=None, out=None, *, keepdims=False, strict=False, nodata=None, threads=1, isa=None):
    """``np.nanargmin`` along one axis. All-NaN slices give -1 and a warning (NumPy raises;
    pass ``strict=True`` for that)."""
    return _arg("argmin", a, axis, out, keepdims, omit=True, strict=strict, nodata=nodata, threads=threads, isa=isa)


def nanargmax(a, axis=None, out=None, *, keepdims=False, strict=False, nodata=None, threads=1, isa=None):
    """``np.nanargmax`` along one axis (see :func:`nanargmin`)."""
    return _arg("argmax", a, axis, out, keepdims, omit=True, strict=strict, nodata=nodata, threads=threads, isa=isa)


def argmedian(a, axis=None, out=None, *, keepdims=False, nodata=None, threads=1, isa=None):
    """Which frame supplied the (lower) median: ``np.argsort(a, axis, kind="stable")`` at
    rank ``(n - 1) // 2``, NaN sorted last."""
    return _arg("argmedian", a, axis, out, keepdims, omit=False, strict=False, nodata=nodata, threads=threads, isa=isa)


def nanargmedian(a, axis=None, out=None, *, keepdims=False, strict=False, nodata=None, threads=1, isa=None):
    """Like :func:`argmedian` over the non-NaN values: rank ``(m - 1) // 2`` of the ``m``
    valid ones. All-NaN slices give -1."""
    return _arg("argmedian", a, axis, out, keepdims, omit=True, strict=strict, nodata=nodata, threads=threads, isa=isa)


def argselect(a, ranks, axis=None, *, threads=1, isa=None):
    """``np.argsort(a, axis, kind="stable")`` at the given ranks (NaN last): which frame
    holds each order statistic."""
    r = np.asarray(ranks)
    if r.dtype.kind not in "iu":
        raise TypeError("ranks must be integers")
    s = _prep(a, axis)  # once: it copies frame lists that need widening, swapping or realigning
    n = s.n
    flat = [int(k) + (n if k < 0 else 0) for k in r.ravel()]
    if any(not 0 <= k < n for k in flat):
        raise ValueError(f"rank out of range for {n} frames")
    res = [_arg("argrank", a, axis, None, False, omit=False, strict=False, nodata=None, threads=threads,
                isa=isa, extra={"rank": k}, s=s) for k in flat]
    if r.ndim == 0:
        return res[0]
    return np.stack([np.asarray(x) for x in res]).reshape(r.shape + np.shape(res[0]))


def select_ranks(a, ranks, axis=None, out=None, *, threads=1, isa=None):
    """``np.sort(a, axis)[ranks]`` in one pass. The network keeps only the comparators the
    requested ranks need. NaN sorts last."""
    r = np.asarray(ranks)
    if r.dtype.kind not in "iu" and r.size:  # [] arrives as float64
        raise TypeError("ranks must be integers")
    s = _prep(a, axis)
    flat = [int(k) + (s.n if k < 0 else 0) for k in r.ravel()]
    if any(not 0 <= k < s.n for k in flat):
        raise ValueError(f"rank out of range for {s.n} frames")
    work, final, copy = _prepare_out(out, s.shape, np.dtype(s.dtype), False, s.axis, lead=r.shape)
    w2 = work.reshape((len(flat),) + tuple(s.shape))
    if flat:  # w2[j, ...]: a view even for 1-D input
        _run({"kind": "select", "ranks": flat}, s, [w2[j, ...] for j in range(len(flat))],
             _opts(threads=threads, isa=isa))
    res = _finish(work, final, copy, False, s.axis, lead_len=r.ndim, out_given=out is not None)
    return res if out is not None else _cast_value_back(res, s, True)


# --------------------------------------------------------------------------- scipy-style

def trimmed_mean(a, proportiontocut, axis=0, *, nan_policy="propagate", keepdims=False, threads=1, isa=None):
    """``scipy.stats.trim_mean``: cut ``int(proportiontocut * n)`` values from each end and
    average the rest. Sums run in float64. With the default ``nan_policy="propagate"`` a
    slice containing NaN gives NaN; ``"omit"`` trims ``int(proportiontocut * m)`` of the
    ``m`` valid values (an all-NaN slice gives NaN); ``"raise"`` raises on NaN."""
    omit = _nan_policy(a, nan_policy)
    return _simple({"kind": "trim_mean", "proportion": float(proportiontocut)}, a, axis, None, keepdims,
                   _float_result, omit=omit, threads=threads, isa=isa)


trim_mean = trimmed_mean


def _nan_policy(x, nan_policy):
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}")
    if nan_policy == "raise":
        arrs = x if _is_frame_list(x) else [x]
        if any(np.isnan(np.asarray(f)).any() for f in arrs if np.asarray(f).dtype.kind == "f"):
            raise ValueError("The input contains nan values")
    return nan_policy == "omit"


def iqr(x, axis=None, rng=(25, 75), scale=1.0, nan_policy="propagate", interpolation="linear", keepdims=False,
        *, threads=1, isa=None):
    """``scipy.stats.iqr`` (linear interpolation), identical results."""
    if interpolation != "linear":
        raise NotImplementedError("rankwise.iqr supports interpolation='linear'")
    if isinstance(scale, str):
        if scale.lower() != "normal":
            raise ValueError(f"{scale} not a valid scale for `iqr`")
        scale = 1.3489795003921636
    if len(rng) != 2:
        raise TypeError("`rng` must be a two element sequence.")
    omit = _nan_policy(x, nan_policy)
    return _simple({"kind": "iqr", "lo": float(rng[0]), "hi": float(rng[1]), "scale": float(scale)}, x, axis, None,
                   keepdims, _float_result, omit=omit, threads=threads, isa=isa)


def median_abs_deviation(x, axis=0, center=None, scale=1.0, nan_policy="propagate", *, keepdims=False, threads=1,
                         isa=None):
    """``scipy.stats.median_abs_deviation`` with the default median center, identical
    results. ``scale="normal"`` divides by 0.6744897501960817."""
    if center is not None:
        raise NotImplementedError("rankwise.mad uses the median as center")
    if isinstance(scale, str):
        if scale.lower() != "normal":
            raise ValueError(f"{scale} is not a valid scale value.")
        scale = 0.6744897501960817
    omit = _nan_policy(x, nan_policy)
    return _simple({"kind": "mad", "scale": float(scale)}, x, axis, None, keepdims, _float_result,
                   omit=omit, threads=threads, isa=isa)


mad = median_abs_deviation


# --------------------------------------------------------------------------- rejection

_SIRIL = {"sigma": "sigma", "mad": "mad", "median_sigma": "median_sigma", "winsorized": "winsorized",
          "percentile": "percentile"}


def clip_combine(a, method="sigma", sigma=3.0, sigma_lower=None, sigma_upper=None, maxiters=5, *, axis=0,
                 cenfunc="median", stdfunc=None, combine="mean", preset=None, out=None, rejected=None,
                 low=None, high=None, mask=None, weights=None, scale=None, offset=None, precision=None,
                 mask_infinite=True, return_counts=False, threads=1, isa=None):
    """Reject outliers along ``axis`` and combine the survivors.

    ``method``: ``"sigma"`` (astropy ``sigma_clip`` semantics), ``"mad"`` (astropy
    ``mad_std`` as the spread), ``"median_sigma"`` (Siril: rejected values replaced by the
    median), ``"winsorized"`` (Siril's Huber-winsorized sigma) or ``"percentile"`` (Siril: one
    pass around the median; ``sigma`` is then the fractional threshold ``(low, high)``).
    ``stdfunc`` (``"std"`` by default, or ``"mad_std"``) applies to ``method="sigma"``.
    ``preset="siril"`` follows Siril 1.2 (sample std, iterate to convergence); ``"pixinsight"``
    follows PCL's documented winsorized clipping. See docs/presets.md for what was verified.

    ``precision`` is the arithmetic of center, spread, bounds and the combine: ``"f64"`` (the
    default, as astropy) or ``"f32"`` (float32 data only). With ``preset="siril"``, f64 gives
    Siril's stacked values bit for bit; f32 resolves exact ties (a value exactly k sigma from
    the center) the way Siril does, with values then within 1 ulp.

    ``scale``/``offset`` normalize frame ``k`` as ``x * scale[k] + offset[k]`` on load;
    ``weights`` give a weighted mean. ``rejected``, ``low`` and ``high`` receive per-pixel
    reject counts; ``mask`` (same shape as the input, bool) receives the rejection map.
    """
    if np.ndim(sigma) == 1:
        lo_s, hi_s = (float(x) for x in sigma)
    else:
        lo_s = hi_s = float(sigma)
    if sigma_lower is not None:
        lo_s = float(sigma_lower)
    if sigma_upper is not None:
        hi_s = float(sigma_upper)
    if maxiters is not None and not np.isfinite(maxiters):
        maxiters = None
    if maxiters == 0 and method in ("sigma", "mad"):
        maxiters = None  # astropy's SigmaClip: `maxiters or np.inf`, so 0 iterates to convergence
    if preset not in (None, "astropy", "siril", "pixinsight"):
        raise ValueError(f"unknown preset {preset!r}")
    op = {"kind": "clip", "method": method, "low": lo_s, "high": hi_s, "cenfunc": cenfunc,
          "combine": combine, "precision": precision, "mask_infinite": bool(mask_infinite)}
    if stdfunc is not None:
        op["stdfunc"] = stdfunc
    if maxiters is not None:
        op["maxiters"] = int(maxiters)
    if preset in ("siril", "pixinsight"):
        op["preset"] = preset
        if preset == "siril" or maxiters == 5:
            op.pop("maxiters", None)  # both tools iterate until nothing changes
    s = _prep(a, axis)
    op["precision"] = "f64" if precision is None else precision
    dt = np.dtype(np.float64) if s.dtype == "float64" else np.dtype(np.float32)
    work, final, copy = _prepare_out(out, s.shape, dt, False, s.axis)
    want_counts = return_counts or rejected is not None
    lo_arr = low if low is not None else (np.empty(s.shape, np.uint16) if want_counts else None)
    hi_arr = high if high is not None else (np.empty(s.shape, np.uint16) if want_counts else None)
    for name, arr in (("low", lo_arr), ("high", hi_arr)):
        if arr is not None and (not isinstance(arr, np.ndarray) or arr.shape != tuple(s.shape)
                                or arr.dtype not in _OUT_OK or not arr.dtype.isnative or not arr.flags.writeable):
            raise ValueError(f"{name} must be a writeable array of shape {tuple(s.shape)} and an integer dtype")
    mspec = None
    if mask is not None:
        if _is_frame_list(a):
            mframes = list(mask)
            if len(mframes) != s.n:
                raise ValueError("mask needs one bool array per frame")
            for f in mframes:
                if not isinstance(f, np.ndarray) or f.shape != tuple(s.shape) or f.dtype not in (np.bool_, np.uint8):
                    raise ValueError("mask frames must be bool arrays shaped like the frames")
                if not f.flags.writeable:
                    raise ValueError("mask frames must be writeable")
            if any(f.strides != mframes[0].strides or f.dtype != mframes[0].dtype for f in mframes):
                # every map frame is written with frame 0's dtype and strides
                raise ValueError("mask frames must share one dtype and memory layout (e.g. all C-contiguous)")
            mspec = ([f.__array_interface__["data"][0] for f in mframes], _OUT_OK[mframes[0].dtype],
                     list(mframes[0].strides))
        else:
            if not isinstance(mask, np.ndarray) or mask.shape != np.shape(s.keep):
                raise ValueError(f"mask must be an array with the input's shape {np.shape(s.keep)}")
            if mask.dtype not in (np.bool_, np.uint8):
                raise TypeError("mask must be bool or uint8")
            if not mask.flags.writeable:
                raise ValueError("mask must be writeable")
            base = mask.__array_interface__["data"][0]
            st = mask.strides[s.axis]
            mspec = ([base + k * st for k in range(s.n)], _OUT_OK[mask.dtype],
                     list(mask.strides[:s.axis] + mask.strides[s.axis + 1:]))
    rest = len(s.shape)
    entries = [_out_entry(work, rest), _out_entry(lo_arr, rest), _out_entry(hi_arr, rest)]
    _rw.reduce(op, s.dtype, s.frames, list(s.shape), list(s.strides), entries, mspec,
               _opts(threads=threads, isa=isa, scale=scale, offset=offset, weights=weights))
    if rejected is not None:
        np.add(lo_arr, hi_arr, out=rejected, casting="unsafe")
    res = _finish(work, final, copy, False, s.axis, out_given=out is not None)
    if return_counts:
        return res, lo_arr, hi_arr
    return res


def sigma_clip_mean(a, axis=0, sigma=3.0, sigma_lower=None, sigma_upper=None, maxiters=5, cenfunc="median",
                    stdfunc="std", *, out=None, rejected=None, low=None, high=None, mask=None, weights=None,
                    scale=None, offset=None, precision="f64", return_counts=False, threads=1, isa=None):
    """Sigma-clipped mean with ``astropy.stats.sigma_clip(..., axis=axis)`` decisions:
    identical masks and kept counts (center, spread and bounds in float64; NaN and ±inf are
    masked first). The mean of the survivors is summed in float64.

    ``sigma`` may be a ``(low, high)`` pair. ``maxiters=0`` or ``None`` iterates until nothing
    changes, as in astropy.
    """
    return clip_combine(a, "sigma", sigma, sigma_lower, sigma_upper, maxiters, axis=axis, cenfunc=cenfunc,
                        stdfunc=stdfunc, out=out, rejected=rejected, low=low, high=high, mask=mask,
                        weights=weights, scale=scale, offset=offset, precision=precision,
                        return_counts=return_counts, threads=threads, isa=isa)


# --------------------------------------------------------------------------- dispatch helpers

def available_isas() -> list[str]:
    """ISAs this CPU can run, fastest first (e.g. ``['avx512', 'avx2', 'scalar']``)."""
    return list(_rw.available_isas())


def current_isa() -> str:
    """The ISA calls use now (after ``set_isa`` and ``RANKWISE_ISA``)."""
    return _rw.current_isa()


def set_isa(name: str | None):
    """Force an ISA for this process (``None`` or ``"auto"`` restores detection)."""
    _rw.set_isa(name)


def isa_override() -> str | None:
    """The ISA forced with :func:`set_isa`, or ``None`` when detection (or ``RANKWISE_ISA``) decides."""
    return _rw.isa_override()


@contextlib.contextmanager
def use_isa(name: str | None):
    """Context manager form of :func:`set_isa`; restores the previous setting on exit."""
    prev = isa_override()
    try:
        _rw.set_isa(name)
        yield
    finally:
        _rw.set_isa(prev)


def network_size(n: int, ranks: Sequence[int] | None = None) -> int:
    """Comparators in the sorting network for ``n`` values, pruned to ``ranks`` if given."""
    return _rw.network_size(int(n), None if ranks is None else [int(r) for r in ranks])


def describe(a, axis=None, op: str = "median", q=None, **kw) -> str:
    """How a call would run: ISA, lanes, comparators and memory path."""
    s = _prep(a, axis)
    if op == "median":
        spec = {"kind": "median"}
    elif op == "quantile":
        spec = {"kind": "quantile", "q": [float(x) for x in np.ravel(q)], "method": "linear", "form": "f64"}
    elif op == "sigma_clip":
        spec = {"kind": "clip", "method": "sigma"}
    else:
        spec = {"kind": op}
    return _rw.describe(spec, s.dtype, s.frames, list(s.shape), list(s.strides), _opts(**kw))
