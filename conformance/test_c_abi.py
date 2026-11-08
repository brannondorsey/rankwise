"""The C example builds and matches Python on the same data (milestone M6)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import rankwise as rw

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("cc") is None or shutil.which("make") is None, reason="needs a C toolchain")
def test_c_example_matches_python(tmp_path):
    subprocess.run(["make", "-s", "-C", str(ROOT / "examples" / "c")], check=True)
    rng = np.random.default_rng(5)
    n, h, w = 31, 64, 75
    x = rng.normal(1000, 30, (n, h, w)).astype(np.float32)
    x[rng.random(x.shape) < 0.03] += 8000
    x.tofile(tmp_path / "frames.f32")
    r = subprocess.run([str(ROOT / "examples" / "c" / "stack"), str(tmp_path / "frames.f32"), str(n), str(h * w),
                        str(tmp_path / "m.f32"), str(tmp_path / "c.f32"), str(tmp_path / "r.u16")],
                       check=True, capture_output=True, text=True)
    assert "integrated" in r.stdout
    med = np.fromfile(tmp_path / "m.f32", np.float32).reshape(h, w)
    clipped = np.fromfile(tmp_path / "c.f32", np.float32).reshape(h, w)
    rej = np.fromfile(tmp_path / "r.u16", np.uint16).reshape(h, w)
    assert np.array_equal(med, np.median(x, axis=0))
    out, lo, hi = rw.sigma_clip_mean(x, axis=0, sigma=(3, 3), maxiters=5, return_counts=True)
    assert np.array_equal(clipped, out)
    assert np.array_equal(rej, lo + hi)
