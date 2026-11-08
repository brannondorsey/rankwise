"""Shared benchmark plumbing: one-thread pinning, timing, machine info, synthetic data.

Every script imports this module first so thread-pool environment variables are set before
NumPy, numba or rayon start. Run scripts under ``taskset -c 1`` (Linux) as well, so a tool
that ignores those variables still gets one core.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import platform
import subprocess
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def best_of(fn, reps=3, warmup=True):
    """Best wall time of ``reps`` runs after one warm-up (JIT compilation, page faults)."""
    if warmup:
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def versions(*pkgs):
    out = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def machine():
    import rankwise as rw
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity": affinity,
        "isas": rw.available_isas(),
        "python": platform.python_version(),
        "date": time.strftime("%Y-%m-%d"),
    }


def save(name, payload):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"wrote {path}")


def astro_stack(n, h, w, seed=0, pad=0):
    """Astro-like light frames (the prototype's generator): sky + gradient + Moffat stars,
    shot and read noise, cosmic rays and satellite trails. ``pad`` extra floats per frame
    break power-of-two frame strides; the returned array is a view of the padded buffer."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    truth = 600 + 120 * (xx / w) + 60 * (yy / h)
    pts = np.zeros((h, w), np.float32)
    ns = int(h * w / 1500)
    pts[rng.integers(0, h, ns), rng.integers(0, w, ns)] = (rng.pareto(1.2, ns) + 1) * 300
    r = np.arange(-12, 13, dtype=np.float32)
    r2 = r[:, None] ** 2 + r[None, :] ** 2
    alpha = 2.6 / (2 * np.sqrt(2 ** (1 / 3) - 1))
    psf = (1 + r2 / alpha ** 2) ** -3
    psf /= psf.sum()
    truth += fftconvolve(pts, psf, mode="same").astype(np.float32)
    buf = np.empty((n, h * w + pad), np.float32)
    st = buf[:, : h * w].reshape(n, h, w)
    for i in range(n):
        f = truth * rng.uniform(0.97, 1.03) + rng.uniform(-15, 15)
        st[i] = f + rng.standard_normal((h, w), dtype=np.float32) * np.sqrt(f + 25.0)
        hits = rng.random((h, w)) < 2e-4
        st[i][hits] += rng.uniform(2000, 20000, hits.sum())
    for i in rng.choice(n, max(1, n // 12), replace=False):
        x0, y0, ang = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, np.pi)
        d = np.abs((xx - x0) * np.sin(ang) - (yy - y0) * np.cos(ang))
        st[i] += (3000 * np.exp(-0.5 * (d / 1.2) ** 2)).astype(np.float32)
    return st


def ensemble(n=31, h=721, w=1440, seed=7, nan_frac=0.0):
    """GEFS-shaped synthetic 2 m temperature: n heavy-tailed members on a 0.25° grid (K)."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(90, -90, h, dtype=np.float32)[:, None]
    lon = np.linspace(0, 360, w, endpoint=False, dtype=np.float32)[None, :]
    base = 288 - 40 * np.abs(np.sin(np.radians(lat))) ** 1.5 + 6 * np.sin(np.radians(3 * lon)) * np.cos(np.radians(lat))
    spread = 1.0 + 3.0 * np.abs(np.sin(np.radians(lat)))
    st = (base + spread * rng.standard_t(5, size=(n, h, w))).astype(np.float32)
    if nan_frac:
        st[rng.random(st.shape) < nan_frac] = np.nan
    return st
