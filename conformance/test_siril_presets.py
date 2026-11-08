"""Siril presets against siril-cli itself (skipped when Siril is not installed)."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rankwise as rw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "siril"))


@pytest.mark.skipif(shutil.which("siril-cli") is None, reason="needs siril-cli")
@pytest.mark.parametrize("n", [5, 15, 31])
def test_siril_presets_identical(tmp_path, n):
    from astropy.io import fits
    from compare_presets import METHODS, make, run_siril

    x = make(n, 24, 32, seed=100 + n)
    run_siril(tmp_path, x)
    for name in ("sigma", "median_sigma", "percentile"):
        _, method, kw = METHODS[name]
        rd = lambda s: np.round(np.asarray(fits.getdata(tmp_path / "seq" / f"{name}_{s}_rejmap.fit"), np.float64) * n).astype(int)
        val = np.asarray(fits.getdata(tmp_path / "seq" / f"{name}.fit"), np.float32)
        out, lo, hi = rw.clip_combine(x, method, preset="siril", return_counts=True, **kw)
        assert np.array_equal(lo, rd("low")), name
        assert np.array_equal(hi, rd("high")), name
        assert np.array_equal(out, val), name
