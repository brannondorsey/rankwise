"""Weighted quantiles against NumPy 2.5.3's ``weights=`` (``method="inverted_cdf"``).

NumPy orders equal values with an unstable argsort, so where the data has ties the weights are
integers (exact sums in any order); continuous data has no ties.
"""

import numpy as np
import pytest

import rankwise as rw
from helpers import identical, make

IC = "inverted_cdf"


def weights_for(n, kind, seed=0):
    rng = np.random.default_rng(seed + n)
    if kind == "random":
        return rng.random(n)
    if kind == "int":
        return rng.integers(0, 4, n).astype(np.float64) + (np.arange(n) == n // 2)
    if kind == "leading_zeros":
        w = rng.random(n) * 10.0 ** rng.integers(-20, 20, n)
        w[: n // 3] = 0.0
        w[-1] += 1.0
        return w
    if kind == "f32":
        return rng.random(n).astype(np.float32)
    raise ValueError(kind)


def q_forms(n):
    rng = np.random.default_rng(n + 7)
    qs = [0.0, 1.0, 0.5, 1 / 3, 0.1, 0.9, float(rng.random())]
    return [
        *qs,
        qs,
        np.array(qs),
        np.array(qs, np.float32),
        np.float32(0.3),
        1,
        0,
    ]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 16, 17, 31, 32, 33, 64, 100, 255, 512])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
@pytest.mark.parametrize("wkind", ["random", "int", "leading_zeros", "f32"])
def test_per_frame_weights(isa, n, dtype, wkind):
    if dtype == np.uint16 and wkind != "int":
        pytest.skip("uint16 stacks have ties: integer weights only")
    x = make(n, 37, dtype, "ties" if wkind == "int" else "normal")
    w = weights_for(n, wkind)
    for q in q_forms(n):
        identical(rw.quantile(x, q, axis=0, method=IC, weights=w), np.quantile(x, q, axis=0, method=IC, weights=w))


@pytest.mark.parametrize("n", [1, 5, 31, 64, 129])
def test_percentile_and_value_families(isa, n):
    for kind in ("normal", "inf", "extreme"):
        x = make(n, 33, np.float32, kind)
        w = weights_for(n, "random")
        for q in (50.0, [5.0, 50.0, 95.0], np.array([12.5, 99.0], np.float32), 100, np.float32(33.3)):
            identical(rw.percentile(x, q, axis=0, method=IC, weights=w),
                      np.percentile(x, q, axis=0, method=IC, weights=w))
    x = make(n, 33, np.float32, "zeros")  # +0 / -0 ties: integer weights
    w = weights_for(n, "int")
    identical(rw.quantile(x, [0.2, 0.5], axis=0, method=IC, weights=w),
              np.quantile(x, [0.2, 0.5], axis=0, method=IC, weights=w))


@pytest.mark.parametrize("n", [1, 3, 16, 33, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weight_maps(isa, n, dtype):
    """One weight per value, any axis, any layout, float32 or float64 weights."""
    rng = np.random.default_rng(n)
    x = make(n, 6 * 7, dtype, "normal").reshape(n, 6, 7)
    for wdt in (np.float64, np.float32, np.int64):
        wm = (rng.random(x.shape) * 3).astype(wdt)
        wm[0] += 1  # every stack weighs something
        wm[: n // 2, ..., 0] = 0  # leading zeros along some pixels' stacks
        wm[-1, ..., 0] = 1
        for axis in (0, 1, 2, -1):
            xa = np.moveaxis(x, 0, axis)
            wa = np.moveaxis(wm, 0, axis)
            identical(rw.quantile(xa, [0.1, 0.5, 0.9], axis=axis, method=IC, weights=wa),
                      np.quantile(xa, [0.1, 0.5, 0.9], axis=axis, method=IC, weights=wa))
    # non-contiguous data and weights; Fortran order
    xs = np.asfortranarray(x)[:, ::2, :]
    ws = (rng.random((n, 12, 7)))[:, ::4, :][:, :3, :]
    identical(rw.quantile(xs, 0.7, axis=0, method=IC, weights=ws), np.quantile(xs, 0.7, axis=0, method=IC, weights=ws))
    # a list of frames with per-frame or per-value weights
    frames = list(x)
    w = weights_for(n, "random")
    identical(rw.quantile(frames, 0.5, method=IC, weights=w), np.quantile(x, 0.5, axis=0, method=IC, weights=w))
    identical(rw.quantile(frames, 0.5, method=IC, weights=list(wm.astype(np.float64))),
              np.quantile(x, 0.5, axis=0, method=IC, weights=wm.astype(np.float64)))


@pytest.mark.parametrize("n", [1, 2, 7, 31, 64])
@pytest.mark.parametrize("kind", ["nan_one", "nan30", "nan_all"])
def test_nan(isa, n, kind):
    x = make(n, 29, np.float32, kind)
    w = weights_for(n, "random")
    wb = np.broadcast_to(w[:, None], x.shape)  # np.nanquantile's 1-D weights fail on 2-D input
    for q in (0.5, [0.0, 0.25, 1.0], np.array([0.3], np.float32)):
        identical(rw.quantile(x, q, axis=0, method=IC, weights=w), np.quantile(x, q, axis=0, method=IC, weights=w))
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning) if kind == "nan_all" else _nowarn():
            got = rw.nanquantile(x, q, axis=0, method=IC, weights=w)
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning) if kind == "nan_all" else _nowarn():
            want = np.nanquantile(x, q, axis=0, method=IC, weights=wb)
        identical(got, want)
        identical(rw.nanquantile(x, q, axis=0, method=IC, weights=wb), want)


class _nowarn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_dtypes_keepdims_out(isa):
    x = make(9, 20, np.float64, "normal").reshape(9, 4, 5)
    w = weights_for(9, "random")
    identical(rw.quantile(x, 0.4, axis=1, method=IC, weights=np.ones(4), keepdims=True),
              np.quantile(x, 0.4, axis=1, method=IC, weights=np.ones(4), keepdims=True))
    out = np.empty((2, 4, 5), np.float32)
    r = rw.quantile(x, [0.2, 0.8], axis=0, method=IC, weights=w, out=out)
    assert r is out
    identical(out, np.quantile(x, [0.2, 0.8], axis=0, method=IC, weights=w).astype(np.float32))
    u8 = (make(9, 20, np.float32, "ties") * 50).astype(np.uint8)
    identical(rw.quantile(u8, 0.5, axis=0, method=IC, weights=weights_for(9, "int")),
              np.quantile(u8, 0.5, axis=0, method=IC, weights=weights_for(9, "int")))
    identical(rw.quantile(x[:, 0, 0], 0.5, method=IC, weights=w), np.quantile(x[:, 0, 0], 0.5, method=IC, weights=w))


def test_weighted_median(isa):
    x = make(31, 50, np.float32, "normal")
    w = weights_for(31, "random")
    identical(rw.weighted_median(x, w, axis=0), np.quantile(x, 0.5, axis=0, method=IC, weights=w))
    x[3, :10] = np.nan
    identical(rw.weighted_median(x, w, axis=0), np.quantile(x, 0.5, axis=0, method=IC, weights=w))
    identical(rw.weighted_median(x, w, axis=0, nan_policy="omit"),
              np.nanquantile(x, 0.5, axis=0, method=IC, weights=np.broadcast_to(w[:, None], x.shape)))
    # equal weights: the lower middle value
    e = np.arange(6, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    identical(rw.weighted_median(e, np.ones(6), axis=0), np.full(3, 2.0, np.float32))


def test_errors():
    x = make(5, 4, np.float32, "normal")
    w = np.ones(5)
    with pytest.raises(ValueError, match="Only method 'inverted_cdf' supports weights. Got: linear."):
        rw.quantile(x, 0.5, axis=0, weights=w)
    with pytest.raises(ValueError, match="Weights must be non-negative."):
        rw.quantile(x, 0.5, axis=0, method=IC, weights=-w)
    with pytest.raises(ValueError, match="Shape of weights"):
        rw.quantile(x, 0.5, axis=0, method=IC, weights=np.ones(4))
    with pytest.raises(ValueError, match="Weights included NaN, inf or were all zero."):
        rw.quantile(x, 0.5, axis=0, method=IC, weights=np.zeros(5))
    with pytest.raises(ValueError, match="Weights included NaN, inf or were all zero."):
        rw.quantile(x, 0.5, axis=0, method=IC, weights=np.array([1, np.inf, 1, 1, 1]))
    with pytest.raises(ValueError, match="Weights included NaN, inf or were all zero."):
        rw.quantile(x, 0.5, axis=0, method=IC, weights=np.array([1, np.nan, 1, 1, 1]))
    # NaN data whose valid values weigh nothing: np.nanquantile raises too
    y = x.copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="Weights included NaN, inf or were all zero."):
        rw.nanquantile(y, 0.5, axis=0, method=IC, weights=np.array([1.0, 0, 0, 0, 0]))
    with pytest.raises(ValueError, match="Quantiles must be in the range"):
        rw.quantile(x, 1.5, axis=0, method=IC, weights=w)


def test_review_regressions(isa):
    # uint8 out= holds values, not flags (weighted and unweighted paths)
    x = np.random.default_rng(3).integers(0, 200, (9, 6)).astype(np.uint8)
    w = weights_for(9, "int")
    o = np.empty(6, np.uint8)
    rw.quantile(x, 0.5, axis=0, method=IC, weights=w, out=o)
    identical(o, np.quantile(x, 0.5, axis=0, method=IC, weights=w))
    o2 = np.empty(6, np.uint8)
    rw.max(x, axis=0, out=o2)
    identical(o2, np.max(x, axis=0))
    o3 = np.empty((2, 6), np.uint8)
    rw.select_ranks(x, [0, 4], axis=0, out=o3)
    identical(o3, np.sort(x, axis=0)[[0, 4]])
    # an empty q gives an empty result
    y = make(5, 4, np.float32, "normal")
    identical(rw.quantile(y, [], axis=0, method=IC, weights=np.ones(5)),
              np.quantile(y, [], axis=0, method=IC, weights=np.ones(5)))
    # subnormal q: the quotient near q is subnormal too
    a = np.array([[0.0], [1.0], [2.0]])
    for wv, q in (([1e-45, 0.5, 0.5], np.float32(1e-45)), ([6e-308, 1e16, 1e16], 5e-324),
                  ([1e-40, 1.0, 1.0], np.array([1e-39, 2e-39], np.float32))):
        identical(rw.quantile(a, q, axis=0, method=IC, weights=np.array(wv)),
                  np.quantile(a, q, axis=0, method=IC, weights=np.array(wv)))
    # a weight map over many short rows (one tile per row)
    z = make(31, 4000, np.float32, "normal").reshape(31, 1000, 4).transpose(1, 0, 2)
    wm = np.random.default_rng(4).random(z.shape)
    identical(rw.quantile(z, [0.2, 0.7], axis=1, method=IC, weights=wm),
              np.quantile(z, [0.2, 0.7], axis=1, method=IC, weights=wm))
