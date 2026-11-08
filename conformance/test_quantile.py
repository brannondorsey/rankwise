import warnings

import numpy as np
import pytest

import rankwise as rw
from helpers import identical, make

METHODS = ["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
           "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher",
           "midpoint", "nearest"]


def q_values(n):
    rng = np.random.default_rng(n)
    qs = [0.0, 1.0, 0.5, 1 / 3, 0.1, 0.9] + list(rng.random(3))
    if n > 1:
        qs += [k / (n - 1) for k in (1, (n - 1) // 2)]
    return [float(q) for q in qs]


def forms(q):
    """Every way NumPy distinguishes q."""
    return {
        "float": float(q),
        "list": [float(q), 0.25],
        "f64array": np.array([float(q), 0.75]),
        "f32array": np.array([q, 0.6], np.float32),
        "np.float64": np.float64(q),
        "np.float32": np.float32(q),
    }


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16, 17, 31, 32, 40, 63, 64, 127, 255, 512])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_quantile_linear_forms(isa, n, dtype):
    x = make(n, 77, dtype, "normal")
    for q in q_values(n):
        for name, qf in forms(q).items():
            identical(rw.quantile(x, qf, axis=0), np.quantile(x, qf, axis=0))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_quantile_methods(isa, method, dtype):
    for n in (1, 2, 3, 7, 31, 40, 129):
        x = make(n, 33, dtype, "normal")
        for q in q_values(n):
            for name in ("float", "list", "f32array"):
                qf = forms(q)[name]
                identical(rw.quantile(x, qf, axis=0, method=method), np.quantile(x, qf, axis=0, method=method))


@pytest.mark.parametrize("method", ["linear", "hazen", "lower", "midpoint", "averaged_inverted_cdf"])
def test_quantile_with_out(isa, method):
    """np.quantile's out= changes the arithmetic of the t >= 0.5 branch."""
    for n in (5, 31, 40):
        x = make(n, 101, np.float32, "normal")
        for q in q_values(n):
            for qf in (float(q), [float(q), 0.7], np.array([q, 0.6], np.float32)):
                for odt in (np.float32, np.float64):
                    shape = np.shape(qf) + (101,)
                    o1, o2 = np.empty(shape, odt), np.empty(shape, odt)
                    r1 = rw.quantile(x, qf, axis=0, method=method, out=o1)
                    np.quantile(x, qf, axis=0, method=method, out=o2)
                    assert r1 is o1
                    identical(o1, o2)


@pytest.mark.parametrize("n", [1, 2, 3, 31, 40, 127])
def test_nanquantile(isa, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for kind in ("nan_one", "nan30", "nan_all", "inf"):
            x = make(n, 61, np.float32, kind)
            for q in q_values(n):
                for qf in (float(q), [float(q), 0.3], np.array([q], np.float32)):
                    identical(rw.quantile(x, qf, axis=0), np.quantile(x, qf, axis=0))
                    identical(rw.nanquantile(x, qf, axis=0), np.nanquantile(x, qf, axis=0))
                    o1, o2 = np.empty(np.shape(qf) + (61,), np.float32), np.empty(np.shape(qf) + (61,), np.float32)
                    rw.nanquantile(x, qf, axis=0, out=o1)
                    np.nanquantile(x, qf, axis=0, out=o2)
                    identical(o1, o2)


@pytest.mark.parametrize("n", [1, 4, 31, 100])
def test_percentile(isa, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = make(n, 45, np.float32, "nan_one")
        y = make(n, 45, np.float32, "normal")
        for p in (0, 10, 33.3, 50, 90, 100, [10, 50, 90], np.array([5.0, 95.0], np.float32)):
            identical(rw.percentile(y, p, axis=0), np.percentile(y, p, axis=0))
            identical(rw.nanpercentile(x, p, axis=0), np.nanpercentile(x, p, axis=0))


def test_quantile_int_q_takes(isa):
    x = make(9, 50, np.uint16, "normal")
    for q in (0, 1, [0, 1]):
        identical(rw.quantile(x, q, axis=0), np.quantile(x, q, axis=0))


def test_quantile_validation():
    x = np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError, match="Quantiles must be in the range"):
        rw.quantile(x, 1.5, axis=0)
    with pytest.raises(ValueError, match="Percentiles must be in the range"):
        rw.percentile(x, 101, axis=0)
    with pytest.raises(ValueError, match="not a valid method"):
        rw.quantile(x, 0.5, axis=0, method="bogus")


def test_ensemble_shape_bit_identical():
    """The doc's GEFS shape: p10/p50/p90 over 31 members."""
    rng = np.random.default_rng(7)
    x = (288 + 3 * rng.standard_t(5, (31, 91, 180))).astype(np.float32)
    identical(rw.quantile(x, [0.1, 0.5, 0.9], axis=0), np.quantile(x, [0.1, 0.5, 0.9], axis=0))
    out = np.empty((3, 91, 180), np.float32)
    rw.quantile(x, [0.1, 0.5, 0.9], axis=0, out=out)
    ref = np.empty_like(out)
    np.quantile(x, [0.1, 0.5, 0.9], axis=0, out=ref)
    identical(out, ref)
