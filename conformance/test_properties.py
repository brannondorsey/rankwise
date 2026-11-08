"""Property tests: random shapes, depths, NaN placement and q, against NumPy."""

import warnings

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

import rankwise as rw
from helpers import identical

floats32 = st.floats(width=32, allow_nan=True, allow_infinity=True, allow_subnormal=True)


@settings(max_examples=300, deadline=None)
@given(hnp.arrays(np.float32, hnp.array_shapes(min_dims=2, max_dims=3, min_side=1, max_side=40), elements=floats32),
       st.integers(0, 2), st.floats(0, 1))
def test_median_quantile_random(x, axis, q):
    axis = axis % x.ndim
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        identical(rw.median(x, axis=axis), np.median(x, axis=axis))
        identical(rw.nanmedian(x, axis=axis), np.nanmedian(x, axis=axis))
        identical(rw.quantile(x, q, axis=axis), np.quantile(x, q, axis=axis))
        identical(rw.quantile(x, [q], axis=axis), np.quantile(x, [q], axis=axis))
        identical(rw.nanquantile(x, [q, 0.5], axis=axis), np.nanquantile(x, [q, 0.5], axis=axis))


@settings(max_examples=200, deadline=None)
@given(hnp.arrays(np.uint16, hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=60)), st.floats(0, 1),
       st.sampled_from(["linear", "lower", "nearest", "hazen", "median_unbiased"]))
def test_uint16_random(x, q, method):
    identical(rw.median(x, axis=0), np.median(x, axis=0))
    identical(rw.quantile(x, q, axis=0, method=method), np.quantile(x, q, axis=0, method=method))
    identical(rw.quantile(x, [q], axis=0, method=method), np.quantile(x, [q], axis=0, method=method))
