"""Regressions found in the second v0.1 audit: each failed before its fix."""

import warnings

import numpy as np
import pytest
from scipy import stats

import rankwise as rw
from helpers import identical, make


def same_scalar_or_array(got, want):
    assert type(got) is type(want), f"{type(got)} != {type(want)}"
    identical(got, want)


@pytest.mark.parametrize("n", [1, 2, 5, 17, 31, 128])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_one_d_quantiles_and_ranks(isa, n, dtype):
    # 1-D input: the per-q output was a NumPy scalar copy, so the kernel wrote into a
    # temporary and the result was uninitialized memory.
    kind = "normal" if dtype == np.uint16 else "nan_one"
    x = make(n, 1, dtype, "normal")[:, 0]
    xn = make(n, 1, dtype, kind)[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q in (0.3, [0.1, 0.5, 0.9], np.array([0.25], np.float32), 1):
            same_scalar_or_array(rw.quantile(x, q), np.quantile(x, q))
            same_scalar_or_array(rw.nanquantile(xn, q), np.nanquantile(xn, q))
        for p in (30, [10, 90]):
            same_scalar_or_array(rw.percentile(x, p), np.percentile(x, p))
            same_scalar_or_array(rw.nanpercentile(xn, p), np.nanpercentile(xn, p))
    identical(rw.select_ranks(x, [0, -1, n // 2]), np.sort(x)[[0, -1, n // 2]])
    # a list of 0-d frames reduces to a 0-d result the same way
    frames = [np.asarray(v) for v in x]
    identical(rw.quantile(frames, [0.2, 0.8]), np.quantile(x, [0.2, 0.8]))


def test_empty_q_and_ranks():
    x = make(5, 8)
    identical(rw.quantile(x, [], axis=0), np.quantile(x, [], axis=0))
    identical(rw.nanpercentile(x, [], axis=0), np.nanpercentile(x, [], axis=0))
    identical(rw.select_ranks(x, [], axis=0), np.sort(x, axis=0)[[]])


def test_select_ranks_out_of_range():
    x = make(6, 8)
    for r in (6, -7):
        with pytest.raises(ValueError, match="out of range"):
            rw.select_ranks(x, [r], axis=0)


def test_integer_q_dtype():
    # Integer q with these methods keeps an integer virtual index: NumPy returns the result
    # type of the data and q (int64 for uint16 data and a list of ints), not float64.
    u = make(5, 6, np.uint16, "ties")
    for data in (u, u.astype(np.uint8), u.astype(np.float32), u.astype(np.float64)):
        for q in ([0, 1], np.array([1, 0], np.int8), np.array([0, 1], np.int16), np.array([0, 1], np.int32),
                  np.int64(1), [True, False]):
            for method in ("weibull", "averaged_inverted_cdf", "interpolated_inverted_cdf", "hazen", "linear"):
                identical(rw.quantile(data, q, axis=0, method=method), np.quantile(data, q, axis=0, method=method))
                identical(rw.nanquantile(data, q, axis=0, method=method),
                          np.nanquantile(data, q, axis=0, method=method))


@pytest.mark.parametrize("n", [1, 2, 3, 10, 31, 64, 200])
def test_trimmed_mean_nan_policy(isa, n):
    for dtype in (np.float32, np.float64):
        for kind in ("nan_one", "nan30", "nan_all", "normal"):
            x = make(n, 61, dtype, kind).reshape(n, 61)
            for p in (0.0, 0.1, 0.25, 0.5):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = rw.trimmed_mean(x, p, axis=0, nan_policy="omit")
                    want = stats.trim_mean(x, p, axis=0, nan_policy="omit")
                assert got.dtype == want.dtype
                identical(np.isnan(got), np.isnan(want))
                # SciPy sums float32 in float32 and in partition order; rankwise in float64, sorted
                tol = 1e-5 if dtype == np.float32 else 1e-12
                np.testing.assert_allclose(got, want, rtol=tol, atol=tol * np.nanmax(np.abs(x), initial=1.0))
    x = make(n, 12)
    want = stats.trim_mean(x, 0.1, axis=0, keepdims=True)
    got = rw.trimmed_mean(x, 0.1, axis=0, keepdims=True)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5)
    if n > 1:
        y = make(n, 12, kind="nan_one")
        with pytest.raises(ValueError, match="nan"):
            rw.trimmed_mean(y, 0.1, axis=0, nan_policy="raise")


def test_mad_keepdims(isa):
    x = make(9, 30).reshape(9, 5, 6)
    for axis in (0, 1, 2):
        identical(rw.mad(x, axis=axis, keepdims=True), stats.median_abs_deviation(x, axis=axis, keepdims=True))


def test_median_sigma_refuses_other_combines_and_weights():
    # Both were silently wrong: combine= was ignored (the mean came back) and weights= made
    # it a mean of the kept values only, so weights=ones changed the result.
    x = make(15, 40)
    with pytest.raises(ValueError, match="mean only"):
        rw.clip_combine(x, "median_sigma", combine="median")
    with pytest.raises(ValueError, match="weights"):
        rw.clip_combine(x, "median_sigma", weights=np.ones(15))
    rw.clip_combine(x, "median_sigma", combine="mean")


def test_nanmedian_one_d_validates_like_the_general_path():
    x = make(7, 1)[:, 0]
    with pytest.raises(np.exceptions.AxisError):
        rw.nanmedian(x, axis=3)
    identical(rw.nanmedian(x, axis=-1), np.nanmedian(x, axis=-1))
    with pytest.raises(TypeError, match="nodata"):
        rw.nanmedian(x, nodata=0)
    with pytest.raises(TypeError, match="float16"):
        rw.nanmedian(np.full(3, np.nan, np.float16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = rw.nanmedian(np.full(3, np.nan, ">f4"))
    assert type(got) is np.float32 and np.isnan(got)
