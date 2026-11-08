import numpy as np
import pytest

import rankwise as rw
from helpers import identical


def data(shape, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(500, 40, shape).astype(dtype)
    x[rng.random(shape) < 0.05] = np.nan
    return x


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_any_axis(isa, axis):
    x = data((9, 17, 31))
    identical(rw.median(x, axis=axis), np.median(x, axis=axis))
    identical(rw.nanmedian(x, axis=axis), np.nanmedian(x, axis=axis))
    identical(rw.quantile(x, [0.2, 0.8], axis=axis), np.quantile(x, [0.2, 0.8], axis=axis))


def test_strided_views(isa):
    x = data((31, 40, 66))
    for v in (x[:, ::2, 1::3], x[:, ::-1, :], x[::-1], x[:, :, ::-2], np.asfortranarray(x), x.transpose(1, 0, 2)):
        identical(rw.median(v, axis=0), np.median(v, axis=0))
        identical(rw.nanquantile(v, 0.3, axis=0), np.nanquantile(v, 0.3, axis=0))


def test_power_of_two_and_padded_strides(isa):
    base = data((17, 64, 64))
    padded = np.empty((17, 64 * 64 + 16), np.float32)
    padded[:, :4096] = base.reshape(17, -1)
    view = padded[:, :4096].reshape(17, 64, 64)
    identical(rw.median(view, axis=0), np.median(base, axis=0))
    identical(rw.median(base, axis=0), np.median(base, axis=0))


def test_frame_lists_and_memmaps(isa, tmp_path):
    x = data((12, 30, 20))
    paths = []
    for k in range(12):
        p = tmp_path / f"f{k}.npy"
        np.save(p, x[k])
        paths.append(p)
    frames = [np.load(p, mmap_mode="r") for p in paths]
    identical(rw.median(frames), np.median(x, axis=0))
    identical(rw.quantile(frames, 0.9), np.quantile(x, 0.9, axis=0))


def test_out_and_keepdims(isa):
    x = data((7, 5, 6))
    out = np.empty((5, 6), np.float32)
    r = rw.median(x, axis=0, out=out)
    assert r is out
    identical(out, np.median(x, axis=0))
    identical(rw.median(x, axis=1, keepdims=True), np.median(x, axis=1, keepdims=True))
    o = np.empty((1, 5, 6), np.float32)
    rw.median(x, axis=0, keepdims=True, out=o)
    identical(o, np.median(x, axis=0, keepdims=True))
    # non-contiguous out
    big = np.zeros((5, 12), np.float32)
    rw.median(x, axis=0, out=big[:, ::2])
    identical(big[:, ::2], np.median(x, axis=0))


def test_uint8_and_big_endian(isa):
    rng = np.random.default_rng(1)
    u8 = rng.integers(0, 255, (9, 50)).astype(np.uint8)
    identical(rw.median(u8, axis=0), np.median(u8, axis=0))
    identical(rw.select_ranks(u8, [0, 8], axis=0), np.sort(u8, axis=0)[[0, 8]])
    be = data((9, 50)).astype(">f4")
    identical(rw.median(be, axis=0), np.median(be, axis=0).astype(np.float32))


def test_threads_identical(isa):
    x = data((31, 300, 301))
    identical(rw.median(x, axis=0, threads=4), rw.median(x, axis=0, threads=1))
    identical(rw.sigma_clip_mean(x, threads=3), rw.sigma_clip_mean(x, threads=1))
    identical(rw.median(x, axis=2, threads=4), np.median(x, axis=2))


def test_one_d_and_errors():
    identical(rw.median(np.array([3.0, 1.0, 2.0], np.float32)), np.median(np.array([3.0, 1.0, 2.0], np.float32)))
    with pytest.raises(TypeError, match="axis"):
        rw.median(np.zeros((3, 3), np.float32))
    with pytest.raises(ValueError, match="out of range|512"):
        rw.median(np.zeros((513, 2), np.float32), axis=0)
    with pytest.raises(TypeError):
        rw.median(np.zeros((3, 2), np.int64), axis=0)


def test_xarray_dask_apply_ufunc():
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    rng = np.random.default_rng(2)
    tas = xr.DataArray(rng.normal(280, 5, (40, 30, 31)).astype(np.float32), dims=("lat", "lon", "member")).chunk({"lat": 10})
    p90 = xr.apply_ufunc(rw.quantile, tas, 0.9, input_core_dims=[["member"], []], kwargs={"axis": -1},
                         dask="parallelized", output_dtypes=[np.float32])
    ref = xr.apply_ufunc(np.quantile, tas, 0.9, input_core_dims=[["member"], []], kwargs={"axis": -1},
                         dask="parallelized", output_dtypes=[np.float32])
    identical(p90.compute().values, ref.compute().values)


def test_conflicting_strides_take_the_staged_path(isa):
    """63 frames exactly 128 KiB apart start in the same L2 set: rankwise stages them."""
    x = data((63, 128, 256))
    assert "path=staged" in rw.describe(x, axis=0)
    identical(rw.median(x, axis=0), np.median(x, axis=0))
    identical(rw.nanquantile(x, [0.1, 0.9], axis=0), np.nanquantile(x, [0.1, 0.9], axis=0))
    out, lo, hi = rw.sigma_clip_mean(x, axis=0, return_counts=True)
    xs = x.reshape(63, -1)[:, ::7].copy().reshape(63, -1)  # a non-conflicting copy of some pixels
    o2, l2, h2 = rw.sigma_clip_mean(xs, axis=0, return_counts=True)
    identical(out.reshape(-1)[::7], o2)
    identical(lo.reshape(-1)[::7], l2)
