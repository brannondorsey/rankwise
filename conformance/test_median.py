import warnings

import numpy as np
import pytest

import rankwise as rw
from helpers import N_MATRIX, NPIX, identical, make

KINDS = ["normal", "ties", "nan_one", "nan30", "nan_all", "inf", "zeros", "extreme"]


@pytest.mark.parametrize("n", N_MATRIX)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_median_matrix(isa, n, dtype):
    for kind in KINDS:
        for npix in (7, 33, 1001):
            x = make(n, npix, dtype, kind)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                identical(rw.median(x, axis=0), np.median(x, axis=0))
                identical(rw.nanmedian(x, axis=0), np.nanmedian(x, axis=0))


@pytest.mark.parametrize("npix", NPIX)
def test_median_partial_vectors(isa, npix):
    for n in (1, 2, 3, 31, 64, 129):
        x = make(n, npix, np.float32, "nan_one")
        identical(rw.median(x, axis=0), np.median(x, axis=0))
        identical(rw.nanmedian(x, axis=0), np.nanmedian(x, axis=0))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 31, 40, 255, 512])
def test_median_uint16(isa, n):
    for kind in ("normal", "ties"):
        x = make(n, 1001, np.uint16, kind)
        identical(rw.median(x, axis=0), np.median(x, axis=0))


@pytest.mark.parametrize("n", [1, 3, 8, 31, 90, 300])
def test_median_uint16_nodata(isa, n):
    x = make(n, 1001, np.uint16, "ties")
    ref = np.where(x == 0, np.nan, x.astype(np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        identical(rw.median(x, axis=0, nodata=0), np.nanmedian(ref, axis=0))


def test_all_nan_warns():
    x = np.full((5, 3), np.nan, np.float32)
    with pytest.warns(RuntimeWarning, match="All-NaN slice"):
        r = rw.nanmedian(x, axis=0)
    assert np.isnan(r).all() and r.dtype == np.float32


def test_signed_zero_compares_equal(isa):
    x = np.array([[-0.0], [0.0], [1.0]], np.float32)
    r = rw.median(x, axis=0)
    assert r[0] == 0.0  # sign of zero may differ from NumPy's +0; values are equal


def test_overflow_like_numpy(isa):
    m = np.finfo(np.float32).max
    x = np.array([[m], [m]], np.float32)
    identical(rw.median(x, axis=0), np.median(x, axis=0))  # (a + b) * 0.5 -> inf
