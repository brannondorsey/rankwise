import warnings

import numpy as np
import pytest
from astropy.stats import sigma_clip

import rankwise as rw
from helpers import identical


def astro_stack(n, npix, seed=0, nan=0.0, inf=0.0, ties=False):
    rng = np.random.default_rng(seed + n)
    x = rng.normal(1000, 30, (n, npix))
    if ties:
        x = np.round(x / 20) * 20
    hits = rng.random(x.shape) < 0.04
    x[hits] += rng.uniform(300, 30000, hits.sum())
    x = x.astype(np.float32)
    x[rng.random(x.shape) < nan] = np.nan
    x[rng.random(x.shape) < inf] = np.inf
    return x


def check(x, sig=(3.0, 3.0), maxiters=5, cenfunc="median", stdfunc="std"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = sigma_clip(x, sigma_lower=sig[0], sigma_upper=sig[1], maxiters=maxiters, cenfunc=cenfunc,
                        stdfunc=stdfunc, axis=0)
    n = x.shape[0]
    mask = np.empty(x.shape, bool)
    out, lo, hi = rw.sigma_clip_mean(x, axis=0, sigma=sig, maxiters=maxiters, cenfunc=cenfunc,
                                     stdfunc=stdfunc, mask=mask, return_counts=True)
    identical(mask, sc.mask)
    finite = np.isfinite(x).sum(0)
    identical((finite - lo - hi).astype(np.int64), (~sc.mask).sum(0).astype(np.int64))
    # mean of the survivors within 1 ulp of an exact float64 reference
    kept = np.where(sc.mask, 0.0, x.astype(np.float64))
    cnt = (~sc.mask).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = (kept.sum(0) / cnt).astype(np.float32)
    both = np.isnan(ref) & np.isnan(out)
    ulp = np.spacing(np.abs(ref))
    assert ((np.abs(out.astype(np.float64) - ref) <= ulp) | both).all()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512])
def test_sigma_clip_matches_astropy(isa, n):
    check(astro_stack(n, 301))
    check(astro_stack(n, 301, seed=1, ties=True))
    check(astro_stack(n, 301, seed=2, nan=0.1, inf=0.05))


@pytest.mark.parametrize("n", [4, 9, 31, 64, 127])
def test_sigma_clip_variants(isa, n):
    x = astro_stack(n, 257, seed=5)
    check(x, sig=(2.0, 2.5), maxiters=None)
    check(x, maxiters=1)
    check(x, cenfunc="mean")
    check(x, stdfunc="mad_std")
    check(x, cenfunc="mean", stdfunc="mad_std", maxiters=None)


def test_sigma_clip_list_and_uint16(isa):
    x = astro_stack(23, 500, seed=9)
    frames = [f.reshape(20, 25) for f in x]
    a = rw.sigma_clip_mean(frames, sigma=(3, 3))
    b = rw.sigma_clip_mean(x.reshape(23, 20, 25), axis=0)
    identical(a, b)
    u = np.clip(x, 0, 65535).astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = sigma_clip(u, sigma=3, maxiters=5, axis=0)
    _, lo, hi = rw.sigma_clip_mean(u, axis=0, return_counts=True)
    identical((23 - lo - hi).astype(np.int64), (~sc.mask).sum(0).astype(np.int64))


def test_weights_and_normalization(isa):
    rng = np.random.default_rng(4)
    x = astro_stack(15, 200, seed=4).astype(np.float64)
    s = rng.uniform(0.9, 1.1, 15)
    o = rng.uniform(-10, 10, 15)
    w = rng.uniform(0.5, 2, 15)
    xn = x * s[:, None] + o[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = sigma_clip(xn, sigma=3, maxiters=5, axis=0)
    got = rw.sigma_clip_mean(x, axis=0, scale=s, offset=o, weights=w)
    keep = ~sc.mask
    ref = (np.where(keep, xn, 0) * w[:, None]).sum(0) / np.where(keep, w[:, None], 0).sum(0)
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_rejected_count_snippet():
    """The doc's NumPy snippet: caller-owned out and rejected arrays."""
    x = astro_stack(40, 120, seed=3).reshape(40, 10, 12)
    out = np.empty(x.shape[1:], np.float32)
    rej = np.empty(x.shape[1:], np.uint16)
    r = rw.sigma_clip_mean(x, axis=0, sigma=(3, 3), maxiters=5, out=out, rejected=rej)
    assert r is out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = sigma_clip(x, sigma=3, maxiters=5, axis=0)
    identical(rej.astype(np.int64), sc.mask.sum(0).astype(np.int64))


def test_isas_agree_on_other_rejections():
    x = astro_stack(31, 999, seed=11)
    results = {}
    for isa in rw.available_isas():
        with rw.use_isa(isa):
            results[isa] = [
                rw.clip_combine(x, "winsorized", 3.0),
                rw.clip_combine(x, "winsorized", 3.0, preset="pixinsight"),
                rw.clip_combine(x, "percentile", (0.2, 0.1)),
                rw.clip_combine(x, "median_sigma", 3.0),
                rw.clip_combine(x, "mad", 3.0),
                rw.clip_combine(x, "sigma", 3.0, combine="median"),
                rw.sigma_clip_mean(x, precision="f32"),
            ]
    base = results[rw.available_isas()[0]]
    for isa, rs in results.items():
        for a, b in zip(base, rs):
            identical(a, b)


@pytest.mark.parametrize("maxiters", [1, 5])
def test_clip_combine_mad_is_mad_std(isa, maxiters):
    """method="mad" clips with astropy's mad_std (a default stdfunc once turned it into std)."""
    rng = np.random.default_rng(maxiters)
    x = np.round(rng.normal(1000, 10, (10, 60, 70)))
    hit = rng.random(x.shape) < 0.03
    x[hit] += rng.uniform(30, 300, hit.sum())
    mask = np.empty(x.shape, bool)
    rw.clip_combine(x, "mad", sigma=5, maxiters=maxiters, mask=mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = sigma_clip(x, sigma=5, maxiters=maxiters, cenfunc="median", stdfunc="mad_std", axis=0)
    identical(mask, sc.mask)
    if maxiters == 1:  # ccdproc's Combiner.sigma_clipping: callables, astropy's Python path, one pass
        from astropy.stats import mad_std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = sigma_clip(x, sigma=5, maxiters=1, cenfunc=np.ma.median, stdfunc=mad_std, axis=0, masked=True)
        identical(mask, py.mask)
    with pytest.raises(ValueError, match="mad_std"):
        rw.clip_combine(x, "mad", stdfunc="std")
