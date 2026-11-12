"""Regressions found in the v0.1 audit: each failed before its fix."""

import warnings

import numpy as np
import pytest
from astropy.stats import sigma_clip
from scipy import stats

import rankwise as rw
from helpers import identical, make


@pytest.mark.parametrize("n", [3, 5, 10, 31, 64])
def test_trim_mean_propagates_nan(isa, n):
    # SciPy's default nan_policy="propagate" gives NaN even when the NaN would be trimmed.
    for dtype in (np.float32, np.float64):
        x = make(n, 77, dtype, "nan_one")
        x[:, ::2] = make(n, 39, dtype, "normal")  # half the pixels clean
        for p in (0.0, 0.1, 0.25, 0.4):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                got, want = rw.trimmed_mean(x, p, axis=0), stats.trim_mean(x, p, axis=0)
            assert got.dtype == want.dtype
            identical(np.isnan(got), np.isnan(want))
            # values: SciPy sums float32 in float32, rankwise in float64 (docs/conformance.md)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * np.nanmax(np.abs(x)))


def test_sigma_clip_maxiters_zero_is_astropys():
    # astropy's SigmaClip stores `maxiters or np.inf`: maxiters=0 iterates to convergence.
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 40, 100], np.float64)[:, None]
    for method, stdfunc in (("sigma", "std"), ("mad", "mad_std")):
        sc = sigma_clip(x, axis=0, sigma=2, maxiters=0, stdfunc=stdfunc)
        mask = np.empty(x.shape, bool)
        rw.clip_combine(x, method, sigma=2, maxiters=0, mask=mask)
        identical(mask, np.ma.getmaskarray(sc))
    assert np.ma.getmaskarray(sigma_clip(x, axis=0, sigma=2, maxiters=0)).sum() == 2


def test_use_isa_restores_previous_override():
    first, last = rw.available_isas()[0], rw.available_isas()[-1]
    rw.set_isa(last)
    try:
        with rw.use_isa(first):
            assert rw.current_isa() == first
        assert rw.current_isa() == last
    finally:
        rw.set_isa(None)


def test_argselect_negative_ranks(isa):
    x = make(9, 77, np.float32, "ties")
    order = np.argsort(x, axis=0, kind="stable")
    identical(rw.argselect(x, -1, axis=0), order[-1].astype(np.intp))
    identical(rw.argselect(x, [0, -2], axis=0), order[[0, -2]].astype(np.intp))
    with pytest.raises(ValueError):
        rw.argselect(x, -10, axis=0)


def test_isa_override_reflects_set_isa():
    assert rw.isa_override() is None
    last = rw.available_isas()[-1]
    rw.set_isa(last)
    try:
        assert rw.isa_override() == last
        with rw.use_isa(None):
            assert rw.isa_override() is None
        assert rw.isa_override() == last
    finally:
        rw.set_isa(None)
    assert rw.isa_override() is None


@pytest.mark.parametrize("dtype", [np.uint8, np.dtype(">f4"), np.dtype(">u2")])
def test_argselect_frame_list_needing_a_copy(isa, dtype):
    # uint8 is widened and big-endian swapped into a copy; argselect prepares it once for all ranks
    x = make(7, 77, np.float32, "ties").astype(dtype)
    frames = list(x)
    order = np.argsort(x, axis=0, kind="stable").astype(np.intp)
    identical(rw.argselect(frames, -1), order[-1])
    identical(rw.argselect(frames, [0, -3, -7]), order[[0, -3, -7]])
    with pytest.raises(ValueError):
        rw.argselect(frames, -8)


@pytest.mark.parametrize("method", ["median_sigma", "winsorized", "percentile"])
def test_cenfunc_rejected_where_it_does_not_apply(method):
    with pytest.raises(ValueError, match="cenfunc"):
        rw.clip_combine(make(9, 7), method, cenfunc="mean")


def test_mask_frames_with_different_layouts_are_refused():
    # Map frames were all written with frame 0's strides: a compact frame after a strided one
    # was written past its end (heap corruption).
    rng = np.random.default_rng(0)
    frames = [rng.normal(size=(5, 7)).astype(np.float32) for _ in range(9)]
    masks = [np.zeros((10, 14), bool)[::2, ::2]] + [np.zeros((5, 7), bool) for _ in range(8)]
    with pytest.raises(ValueError, match="layout"):
        rw.sigma_clip_mean(frames, mask=masks)
    masks = [np.zeros((5, 7), bool) for _ in range(9)]
    frames[4][:] = 100
    rw.sigma_clip_mean(frames, mask=masks)
    assert masks[4].all() and not masks[0].any()


def test_read_only_outputs_are_refused():
    x = make(9, 50)
    ro = np.zeros(50, np.uint16)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        rw.clip_combine(x, low=ro)
    m = np.zeros(x.shape, bool)
    m.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        rw.clip_combine(x, mask=m)
