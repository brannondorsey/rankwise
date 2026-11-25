"""Moving windows against xarray's ``rolling`` (its NumPy path) and the NumPy formula it
evaluates: the NaN-padded array's ``sliding_window_view`` reduced with the ``nan*`` function,
then windows with fewer than ``min_periods`` valid values set to NaN. Grouped reductions
against the named function on each group's frames.
"""

import warnings

import numpy as np
import pytest
import scipy.stats
from numpy.lib.stride_tricks import sliding_window_view

import rankwise as rw
from helpers import identical, make


def windows(a, w, axis, center, step):
    """(windows, counts): NaN-padded windows along the last axis, shape (T', *rest, w)."""
    x = np.moveaxis(np.asarray(a), axis, 0)
    if x.dtype.kind != "f":
        x = x.astype(np.float32)  # xarray pads 1- and 2-byte integers with NaN as float32
    start = w // 2 if center else w - 1
    xp = np.pad(x, [(start, w - 1 - start)] + [(0, 0)] * (x.ndim - 1), constant_values=np.nan)
    v = sliding_window_view(xp, w, axis=0)[::step]
    return v, (~np.isnan(v)).sum(-1)


def apply_first_valid(fn, v, lead=0):
    """``fn(v, axis=-1)`` with a valid slice first: np.nanquantile takes its result dtype from
    the first slice (an all-NaN one would make it float32)."""
    flat = v.reshape(-1, v.shape[-1])
    flat = np.concatenate([np.zeros((1, flat.shape[1]), flat.dtype), flat])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.asarray(fn(flat))
    r = r[..., 1:]
    return r.reshape(r.shape[:lead] + v.shape[:-1])


def reference(a, w, axis, center, step, min_count, fn, lead=0):
    v, cnt = windows(a, w, axis, center, step)
    r = apply_first_valid(fn, v, lead)
    r = np.where(cnt >= min_count, r, np.nan).astype(r.dtype)
    return np.moveaxis(r, lead, lead + axis)


CASES = [(1, 4), (2, 9), (3, 1), (5, 40), (8, 8), (16, 33), (31, 90), (33, 70), (64, 130)]


@pytest.mark.parametrize("w,nf", CASES)
@pytest.mark.parametrize("kind", ["normal", "nan30", "inf", "ties"])
def test_median_quantile_minmax(isa, w, nf, kind):
    x = make(nf, 23, np.float32, kind)
    for center, step, mc in ((False, 1, w), (True, 1, 1), (False, 2, max(1, w // 2)), (True, 3, 1), (False, w + 2, 1)):
        args = dict(axis=0, min_count=mc, center=center, step=step)
        ref = lambda fn, lead=0: reference(x, w, 0, center, step, mc, fn, lead)  # noqa: E731
        identical(rw.rolling(x, w, func="median", **args), ref(lambda f: np.nanmedian(f, axis=-1)))
        for q in (0.3, [0.1, 0.5, 1.0], np.array([0.25, 0.75], np.float32)):
            for m in ("linear", "hazen", "nearest"):
                want = ref(lambda f: np.nanquantile(f, q, axis=-1, method=m), np.ndim(q))
                identical(rw.rolling(x, w, func="quantile", q=q, method=m, **args), want)
        identical(rw.rolling(x, w, func="percentile", q=[5, 95], **args),
                  ref(lambda f: np.nanpercentile(f, [5, 95], axis=-1), 1))
        identical(rw.rolling(x, w, func="min", **args), ref(lambda f: np.nanmin(f, axis=-1)))
        identical(rw.rolling(x, w, func="max", **args), ref(lambda f: np.nanmax(f, axis=-1)))
        identical(rw.rolling(x, w, func="ptp", **args),
                  ref(lambda f: np.nanmax(f, axis=-1) - np.nanmin(f, axis=-1)))
        ranks = [0, w // 2, w - 1]
        identical(rw.rolling(x, w, func="select_ranks", ranks=ranks, **args),
                  ref(lambda f: np.moveaxis(np.sort(f, axis=-1)[..., ranks], -1, 0), 1))


@pytest.mark.parametrize("w,nf", [(3, 12), (9, 50), (31, 90)])
def test_scipy_and_clip(isa, w, nf):
    x = make(nf, 19, np.float64, "nan30")
    for center, mc in ((False, w), (True, 2)):
        args = dict(axis=0, min_count=mc, center=center)
        ref = lambda fn: reference(x, w, 0, center, 1, mc, fn)  # noqa: E731
        identical(rw.rolling(x, w, func="iqr", **args),
                  ref(lambda f: scipy.stats.iqr(f, axis=-1, nan_policy="omit")))
        identical(rw.rolling(x, w, func="mad", scale="normal", **args),
                  ref(lambda f: scipy.stats.median_abs_deviation(f, axis=-1, nan_policy="omit", scale="normal")))
        # rejection: the static kernel on each NaN-padded window (masks identical to astropy's)
        v, cnt = windows(x, w, 0, center, 1)
        want = rw.sigma_clip_mean(v, axis=-1, sigma=2.0)
        want = np.where(cnt >= mc, want, np.nan)  # min_periods counts non-NaN values, ±inf too
        identical(rw.rolling(x, w, func="sigma_clip_mean", sigma=2.0, **args), want)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_axes_layouts_dtypes(isa, axis):
    x = make(40, 6 * 7, np.float32, "nan30").reshape(40, 6, 7)
    xa = np.moveaxis(x, 0, axis)
    for w, center in ((5, False), (6, True)):
        want = reference(xa, w, axis % 3, center, 1, 2, lambda f: np.nanmedian(f, axis=-1))
        identical(rw.rolling(xa, w, axis=axis, func="median", center=center, min_count=2), want)
    # non-contiguous, Fortran order, a list of frames
    xs = np.asfortranarray(x)[:, ::2, :]
    identical(rw.rolling(xs, 4, func="median", min_count=1),
              reference(xs, 4, 0, False, 1, 1, lambda f: np.nanmedian(f, axis=-1)))
    identical(rw.rolling(list(x), 4, func="max", min_count=3),
              reference(x, 4, 0, False, 1, 3, lambda f: np.nanmax(f, axis=-1)))
    # integer input: float64 results, or the input dtype with nodata as the fill
    u = make(30, 20, np.uint16, "ties")
    identical(rw.rolling(u, 5, func="median"), reference(u, 5, 0, False, 1, 5, lambda f: np.nanmedian(f, axis=-1)))
    identical(rw.rolling(u, 5, func="min", min_count=1),
              reference(u, 5, 0, False, 1, 1, lambda f: np.nanmin(f, axis=-1)))
    got = rw.rolling(u, 5, func="min", nodata=0)
    assert got.dtype == np.uint16
    f = np.where(u == 0, np.nan, u.astype(np.float64))
    want = reference(f, 5, 0, False, 1, 5, lambda g: np.nanmin(g, axis=-1))
    identical(got, np.where(np.isnan(want), 0, want).astype(np.uint16))
    u8 = (make(30, 20, np.float32, "ties") * 40).astype(np.uint8)
    identical(rw.rolling(u8, 3, func="median", center=True, min_count=1),
              reference(u8, 3, 0, True, 1, 1, lambda f: np.nanmedian(f, axis=-1)))


def test_matches_xarray(isa):
    xr = pytest.importorskip("xarray")
    # integers: xarray pads them as float32
    u = make(40, 12, np.uint16, "normal")
    du = xr.DataArray(u, dims=("time", "p"))
    with xr.set_options(use_bottleneck=False, use_numbagg=False):
        for w, center, mp in ((5, False, None), (6, True, 2)):
            r = du.rolling(time=w, center=center, min_periods=mp)
            mc = w if mp is None else mp
            identical(rw.rolling(u, w, func="median", center=center, min_count=mc), r.median().values)
            identical(rw.rolling(u, w, func="max", center=center, min_count=mc), r.max().values)
            identical(rw.rolling(u, w, func="quantile", q=0.37, center=center, min_count=mc),
                      r.reduce(lambda v, axis: np.nanquantile(v, 0.37, axis=axis)).values)
        # rejection treats ±inf as missing, but min_periods counts every non-NaN value
        y = np.array([1, 2, np.inf, 3, 4, 5], np.float32)
        dy = xr.DataArray(y, dims=("time",))
        identical(rw.rolling(y, 3, func="sigma_clip_mean", min_count=3),
                  dy.rolling(time=3, min_periods=3).reduce(lambda v, axis: rw.sigma_clip_mean(v, axis=axis)).values)
    x = make(60, 30, np.float32, "nan30").reshape(60, 5, 6)
    da = xr.DataArray(x, dims=("time", "y", "x"))
    with xr.set_options(use_bottleneck=False, use_numbagg=False):
        for w, center, mp in ((7, False, None), (7, True, 3), (12, True, 1)):
            r = da.rolling(time=w, center=center, min_periods=mp)
            mc = w if mp is None else mp
            identical(rw.rolling(x, w, func="median", center=center, min_count=mc), r.median().values)
            identical(rw.rolling(x, w, func="max", center=center, min_count=mc), r.max().values)
            identical(rw.rolling(x, w, func="quantile", q=0.9, center=center, min_count=mc),
                      r.reduce(lambda v, axis: np.nanquantile(v, 0.9, axis=axis)).values)


@pytest.mark.parametrize("n", [1, 2, 6, 50])
def test_one_dimensional(isa, n):
    """1-D input: each window's output is a 0-d view (it used to be a scalar copy)."""
    x = make(n, 3, np.float32, "nan30")[:, 1]
    for w, center, mc in ((3, False, 1), (4, True, 2), (1, False, 1)):
        identical(rw.rolling(x, w, func="median", center=center, min_count=mc),
                  reference(x, w, 0, center, 1, mc, lambda f: np.nanmedian(f, axis=-1)))
        identical(rw.rolling(x, w, func="quantile", q=[0.2, 0.8], center=center, min_count=mc),
                  reference(x, w, 0, center, 1, mc, lambda f: np.nanquantile(f, [0.2, 0.8], axis=-1), 1))


def test_integer_nodata_fill():
    """With nodata, float outputs of short windows are NaN; integer outputs get nodata."""
    x = np.array([[5], [7], [9], [11]], np.uint16)
    identical(rw.rolling(x, 3, func="median", nodata=0), np.array([[np.nan], [np.nan], [7.0], [9.0]]))
    identical(rw.rolling(x, 3, func="iqr", nodata=0)[:2], np.full((2, 1), np.nan))
    identical(rw.rolling(x, 3, func="max", nodata=0), np.array([[0], [0], [9], [11]], np.uint16))


def test_rolling_errors():
    x = make(10, 4, np.float32, "normal")
    with pytest.raises(ValueError):
        rw.rolling(x, 0)
    with pytest.raises(ValueError):
        rw.rolling(x, 3, min_count=4)
    with pytest.raises(ValueError):
        rw.rolling(x, 3, step=0)
    with pytest.raises(ValueError):
        rw.rolling(x, 3, func="argmin")
    with pytest.raises(TypeError):
        rw.rolling(x, 3, func="median", q=0.5)
    with pytest.raises(ValueError):
        rw.rolling(x, 3, func="select_ranks", ranks=[3])


@pytest.mark.parametrize("func,kw", [
    ("median", {}), ("nanmedian", {}), ("quantile", {"q": [0.1, 0.9]}), ("nanpercentile", {"q": 50.0}),
    ("max", {}), ("ptp", {}), ("trimmed_mean", {"proportiontocut": 0.1}), ("mad", {"nan_policy": "omit"}),
    ("sigma_clip_mean", {"sigma": 2.5}), ("select_ranks", {"ranks": [0]}), ("argmax", {}), ("nanargmedian", {}),
    ("argselect", {"ranks": [1]}),
])
def test_groupby_reduce(isa, func, kw):
    rng = np.random.default_rng(1)
    x = make(48, 35, np.float32, "nan30" if func.startswith("nan") or func in ("mad",) else "normal").reshape(48, 5, 7)
    by = rng.integers(0, 6, 48) * 10  # labels 0, 10, ..., 50
    by[:6] = np.arange(6) * 10
    for axis in (0, 1):
        xa = np.moveaxis(x, 0, axis)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res, groups = rw.groupby_reduce(xa, by, axis=axis, func=func, **kw)
            assert list(groups) == [0, 10, 20, 30, 40, 50]
            fn = getattr(rw, func)
            parts = []
            for g in groups:
                sub = np.take(xa, np.flatnonzero(by == g), axis=axis)
                r = np.asarray(fn(sub, axis=axis, **kw))
                if func in ("argmax", "nanargmedian", "argselect"):
                    r = np.where(r >= 0, np.flatnonzero(by == g)[np.clip(r, 0, None)], r)
                parts.append(r)
        lead = parts[0].ndim - (xa.ndim - 1)
        identical(res, np.stack(parts, axis=lead + axis))


def test_groupby_reduce_rejects():
    x = make(8, 4, np.float32, "normal")
    by = np.arange(8) % 2
    with pytest.raises(TypeError):
        rw.groupby_reduce(x, by, func="median", out=np.empty((2, 4), np.float32))
    with pytest.raises(TypeError):
        rw.groupby_reduce(x, by, func=np.median)
    res, _ = rw.groupby_reduce(x, by, func=rw.median)
    identical(res, np.stack([np.median(x[by == g], axis=0) for g in (0, 1)]))
