import warnings

import numpy as np
import pytest
from scipy import stats

import rankwise as rw
from helpers import identical, make


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 17, 31, 40, 64, 127, 255, 512])
def test_trim_mean(isa, n):
    for dtype in (np.float32, np.float64, np.uint16):
        x = make(n, 77, dtype, "normal")
        for p in (0.0, 0.1, 0.25, 0.4):
            if int(p * n) > n - int(p * n):
                continue
            got = rw.trimmed_mean(x, p, axis=0)
            want = stats.trim_mean(x, p, axis=0)
            assert got.dtype == want.dtype
            # f64 reference: exact mean of the kept order statistics
            s = np.sort(x.astype(np.float64), axis=0)
            lo = int(p * n)
            ref = s[lo:n - lo].mean(axis=0).astype(got.dtype)
            if dtype == np.float32:
                ulp = np.spacing(np.abs(ref).astype(np.float32))
                assert (np.abs(got.astype(np.float64) - ref) <= ulp).all()
            # SciPy sums float32 in float32; with heavy cancellation its own error exceeds
            # 1e-6 relative, so the tolerance scales with the data's magnitude.
            scale = np.abs(x.astype(np.float64)).max(axis=0)
            assert (np.abs(got.astype(np.float64) - want) <= 1e-6 * np.maximum(np.abs(want), scale) + 1e-30).all()


def test_trim_mean_too_big():
    with pytest.raises(ValueError, match="Proportion too big"):
        rw.trimmed_mean(np.zeros((5, 3), np.float32), 0.6, axis=0)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 15, 31, 40, 64, 127, 255, 512])
def test_iqr(isa, n):
    for dtype in (np.float32, np.float64, np.uint16):
        x = make(n, 77, dtype, "normal")
        identical(rw.iqr(x, axis=0), stats.iqr(x, axis=0))
        identical(rw.iqr(x, axis=0, rng=(10, 90), scale="normal"), stats.iqr(x, axis=0, rng=(10, 90), scale="normal"))
    x = make(n, 77, np.float32, "nan30")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        identical(rw.iqr(x, axis=0), stats.iqr(x, axis=0))
        identical(rw.iqr(x, axis=0, nan_policy="omit"), stats.iqr(x, axis=0, nan_policy="omit"))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 15, 16, 31, 32, 33, 40, 64, 127, 255, 512])
def test_mad(isa, n):
    for dtype in (np.float32, np.float64, np.uint16):
        for kind in ("normal", "ties"):
            x = make(n, 77, dtype, kind)
            identical(rw.mad(x, axis=0), stats.median_abs_deviation(x, axis=0))
            identical(rw.mad(x, axis=0, scale="normal"), stats.median_abs_deviation(x, axis=0, scale="normal"))
    x = make(n, 77, np.float32, "nan30")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        identical(rw.mad(x, axis=0), stats.median_abs_deviation(x, axis=0))
        identical(rw.mad(x, axis=0, nan_policy="omit"), stats.median_abs_deviation(x, axis=0, nan_policy="omit"))
