import numpy as np

N_MATRIX = list(range(1, 18)) + [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 511, 512]
NPIX = [1, 7, 15, 16, 17, 33, 1001]


def identical(got, want, *, check_dtype=True):
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    if check_dtype:
        assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
    eq = (got == want) | (np.isnan(got) & np.isnan(want)) if got.dtype.kind == "f" else (got == want)
    if not eq.all():
        idx = np.argwhere(~eq)[:5]
        detail = [(tuple(i), got[tuple(i)], want[tuple(i)]) for i in idx]
        raise AssertionError(f"{(~eq).sum()} of {eq.size} values differ, e.g. {detail}")


def make(n, npix, dtype=np.float32, kind="normal", seed=0):
    """A (n, npix) stack of one of several value families."""
    rng = np.random.default_rng(seed + 1000 * n + npix)
    if dtype == np.uint16:
        if kind == "ties":
            return rng.integers(0, 4, (n, npix)).astype(np.uint16)
        return rng.integers(0, 65536, (n, npix)).astype(np.uint16)
    x = rng.standard_t(4, (n, npix)) * rng.choice([1e-3, 1.0, 300.0, 1e6]) + rng.choice([0.0, 288.0, -5.0])
    x = x.astype(dtype)
    if kind == "ties":
        x = rng.integers(0, 4, (n, npix)).astype(dtype)
    elif kind == "nan_one":
        x[rng.integers(0, n, npix), np.arange(npix)] = np.nan
    elif kind == "nan30":
        x[rng.random((n, npix)) < 0.3] = np.nan
    elif kind == "nan_all":
        x[rng.random((n, npix)) < 0.3] = np.nan
        x[:, ::3] = np.nan
    elif kind == "inf":
        x[rng.random((n, npix)) < 0.1] = np.inf
        x[rng.random((n, npix)) < 0.1] = -np.inf
    elif kind == "zeros":
        x[rng.random((n, npix)) < 0.4] = 0.0
        x[rng.random((n, npix)) < 0.4] = -0.0
    elif kind == "extreme":
        fi = np.finfo(dtype)
        pick = rng.random((n, npix))
        x[pick < 0.15] = fi.max
        x[(pick >= 0.15) & (pick < 0.3)] = -fi.max
        x[(pick >= 0.3) & (pick < 0.45)] = fi.smallest_subnormal * rng.integers(1, 100, ((pick >= 0.3) & (pick < 0.45)).sum())
    return x
