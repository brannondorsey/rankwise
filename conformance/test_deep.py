"""Stacks deeper than 512 frames (the blocked bitonic network) against NumPy, SciPy and astropy."""

import numpy as np
import pytest
import scipy.stats
from astropy.stats import sigma_clip

import rankwise as rw
from helpers import identical, make

DEEP = [513, 1000, 1024, 1025, 2049, 4096]


@pytest.mark.parametrize("n", DEEP)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_median_quantile_select(isa, n, dtype):
    for kind in ("normal", "ties") if dtype != np.uint16 else ("normal", "ties"):
        x = make(n, 21, dtype, kind)
        identical(rw.median(x, axis=0), np.median(x, axis=0))
        for q in (0.5, [0.0, 0.1, 0.9, 1.0], np.array([0.33], np.float32)):
            identical(rw.quantile(x, q, axis=0), np.quantile(x, q, axis=0))
        identical(rw.quantile(x, [0.2, 0.7], axis=0, method="nearest"), np.quantile(x, [0.2, 0.7], axis=0, method="nearest"))
        ranks = [0, 1, n // 3, n // 2, n - 2, n - 1]
        identical(rw.select_ranks(x, ranks, axis=0), np.sort(x, axis=0)[ranks])
        identical(rw.argmedian(x, axis=0), np.argsort(x, axis=0, kind="stable")[(n - 1) // 2])
        identical(rw.max(x, axis=0), np.max(x, axis=0))


@pytest.mark.parametrize("n", DEEP)
@pytest.mark.parametrize("kind", ["nan30", "nan_all", "inf", "zeros"])
def test_nan_and_special_values(isa, n, kind):
    x = make(n, 17, np.float32, kind)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning) if kind == "nan_all" else _quiet():
        got = rw.nanmedian(x, axis=0)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning) if kind == "nan_all" else _quiet():
        want = np.nanmedian(x, axis=0)
    identical(got, want)
    identical(rw.median(x, axis=0), np.median(x, axis=0))
    with np.errstate(all="ignore"), _quiet():
        identical(rw.nanquantile(x, [0.25, 0.75], axis=0), np.nanquantile(x, [0.25, 0.75], axis=0))


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


@pytest.mark.parametrize("n", [513, 1025, 2049])
def test_scipy_and_astropy(isa, n):
    x = make(n, 13, np.float64, "normal")
    x[np.random.default_rng(n).random(x.shape) < 0.02] += 1e4
    identical(rw.iqr(x, axis=0), scipy.stats.iqr(x, axis=0))
    identical(rw.mad(x, axis=0), scipy.stats.median_abs_deviation(x, axis=0))
    for stdfunc in ("std", "mad_std"):
        mask = np.empty(x.shape, bool)
        rw.sigma_clip_mean(x, axis=0, sigma=3.0, stdfunc=stdfunc, mask=mask)
        sc = sigma_clip(x, sigma=3.0, maxiters=5, cenfunc="median", stdfunc=stdfunc, axis=0)
        identical(mask, sc.mask)
    # the PixInsight preset's Sn start on a deep stack (was quadratic per pixel)
    out = rw.clip_combine(x.astype(np.float32), method="winsorized", preset="pixinsight", sigma=3.0)
    assert np.isfinite(out).all()


def test_limits():
    assert rw.MAX_FRAMES == 65535
    x = np.zeros((rw.MAX_FRAMES, 2), np.uint16)
    x[-1] = 7
    identical(rw.max(x, axis=0), np.full(2, 7, np.uint16))
    identical(rw.median(x, axis=0), np.zeros(2))
    with pytest.raises(ValueError, match="65535"):
        rw.median(np.zeros((rw.MAX_FRAMES + 1, 1), np.float32), axis=0)
