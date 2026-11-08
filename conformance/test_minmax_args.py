import warnings

import numpy as np
import pytest

import rankwise as rw
from helpers import N_MATRIX, identical, make


@pytest.mark.parametrize("n", N_MATRIX)
def test_min_max(isa, n):
    for dtype in (np.float32, np.float64):
        for kind in ("normal", "ties", "nan_one", "nan_all", "inf"):
            x = make(n, 77, dtype, kind)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                identical(rw.min(x, axis=0), np.min(x, axis=0))
                identical(rw.max(x, axis=0), np.max(x, axis=0))
                identical(rw.nanmin(x, axis=0), np.nanmin(x, axis=0))
                identical(rw.nanmax(x, axis=0), np.nanmax(x, axis=0))
                identical(rw.ptp(x, axis=0), np.ptp(x, axis=0))
    x = make(n, 77, np.uint16)
    identical(rw.min(x, axis=0), np.min(x, axis=0))
    identical(rw.max(x, axis=0), np.max(x, axis=0))
    identical(rw.ptp(x, axis=0), np.ptp(x, axis=0))


@pytest.mark.parametrize("n", N_MATRIX)
def test_argmin_argmax(isa, n):
    for kind in ("normal", "ties", "nan_one", "inf"):
        x = make(n, 77, np.float32, kind)
        identical(rw.argmin(x, axis=0), np.argmin(x, axis=0))
        identical(rw.argmax(x, axis=0), np.argmax(x, axis=0))
    x = make(n, 77, np.float32, "nan30")
    ok = ~np.isnan(x).all(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got_mn, got_mx = rw.nanargmin(x, axis=0), rw.nanargmax(x, axis=0)
    identical(got_mn[ok], np.nanargmin(x[:, ok], axis=0))
    identical(got_mx[ok], np.nanargmax(x[:, ok], axis=0))
    assert (got_mn[~ok] == -1).all()
    u = make(n, 77, np.uint16, "ties")
    identical(rw.argmin(u, axis=0), np.argmin(u, axis=0))
    identical(rw.argmax(u, axis=0), np.argmax(u, axis=0))


def test_nanarg_strict():
    x = np.full((4, 2), np.nan, np.float32)
    with pytest.raises(ValueError, match="All-NaN"):
        rw.nanargmin(x, axis=0, strict=True)
    with pytest.warns(RuntimeWarning):
        assert (rw.nanargmax(x, axis=0) == -1).all()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 16, 31, 32, 64, 127, 300, 512])
def test_argmedian_and_argselect(isa, n):
    for kind in ("normal", "ties", "nan_one", "inf"):
        x = make(n, 77, np.float32, kind)
        order = np.argsort(x, axis=0, kind="stable")
        identical(rw.argmedian(x, axis=0), order[(n - 1) // 2])
        ranks = sorted({0, n // 3, n - 1})
        identical(rw.argselect(x, ranks, axis=0), order[ranks])
    # nanargmedian: rank (m - 1) // 2 of the valid values
    x = make(n, 77, np.float32, "nan30")
    order = np.argsort(x, axis=0, kind="stable")
    m = (~np.isnan(x)).sum(0)
    want = np.where(m > 0, np.take_along_axis(order, np.maximum((m - 1) // 2, 0)[None], 0)[0], -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        identical(rw.nanargmedian(x, axis=0), want)


@pytest.mark.parametrize("n", [1, 2, 7, 31, 33, 64, 129, 512])
def test_select_ranks(isa, n):
    for dtype in (np.float32, np.float64, np.uint16):
        x = make(n, 77, dtype, "nan_one" if dtype != np.uint16 else "ties")
        s = np.sort(x, axis=0)
        ranks = sorted({0, 3 % n, n // 2, n - 1})
        identical(rw.select_ranks(x, ranks, axis=0), s[ranks])
        identical(rw.select_ranks(x, n // 2, axis=0), s[n // 2])


def test_greenest_pixel_snippet():
    """The Landsat snippet: argmax of NDVI, then gather RGB from that scene."""
    rng = np.random.default_rng(3)
    ndvi = rng.random((90, 40, 50)).astype(np.float32)
    ndvi[rng.random(ndvi.shape) < 0.4] = np.nan
    ndvi[:, 0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best = rw.nanargmax(ndvi, axis=0)
    ok = best >= 0
    identical(best[ok], np.nanargmax(ndvi.reshape(90, -1)[:, ok.ravel()], axis=0))
    bands = rng.integers(0, 65535, (3, 90, 40, 50)).astype(np.uint16)
    rgb = np.take_along_axis(bands, np.maximum(best, 0)[None, None], axis=1)[:, 0]
    assert rgb.shape == (3, 40, 50)
