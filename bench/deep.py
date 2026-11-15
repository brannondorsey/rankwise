"""Deep stacks: median and sigma-clipped mean from 256 to 16384 frames, one thread.

    taskset -c 1 python bench/deep.py

About 256 MiB of float32 per depth (fewer pixels as the stack deepens), against np.median
(introselect per pixel), bottleneck's median and astropy's sigma_clip.
"""

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, machine, save, versions

rng = np.random.default_rng(11)
rows = []
for n in (256, 511, 512, 513, 1000, 1024, 2048, 4096, 8192, 16384):
    npix = (64 << 20) // n
    x = rng.normal(1000, 30, (n, npix)).astype(np.float32)
    ref = np.median(x, axis=0)
    cells = n * npix

    def record(tool, op, fn, reps, family="competitor", want=ref):
        t = best_of(fn, reps, warmup=reps > 1)
        ident = bool(np.array_equal(np.asarray(fn()), want)) if want is not None else None
        rows.append(dict(n=n, npix=npix, tool=tool, op=op, family=family, seconds=t, ns_per_value=t / cells * 1e9,
                         identical=ident))
        print(f"n={n:6d} {op:8s} {tool:24s} {t:7.3f} s  {t / cells * 1e9:6.2f} ns/value  identical={ident}", flush=True)

    for isa in rw.available_isas():
        if isa == "scalar" and n > 4096:
            continue
        label = {"avx512": "AVX-512", "avx2": "AVX2", "scalar": "scalar", "neon": "NEON"}[isa]
        record(f"rankwise · {label}", "median", lambda: rw.median(x, axis=0, isa=isa), 3, family="rankwise")
    record("NumPy 2.5", "median", lambda: np.median(x, axis=0), 2)
    try:
        import bottleneck as bn
        record(f"bottleneck {bn.__version__}", "median", lambda: bn.median(x, axis=0), 2)
    except ImportError:
        pass
    if n in (513, 2048, 8192):
        from astropy.stats import sigma_clip
        sub = x[:, :4096]
        sc = sigma_clip(sub, sigma=3, maxiters=5, cenfunc="median", stdfunc="std", axis=0)
        counts = (~sc.mask).sum(0)
        _, lo, hi = rw.sigma_clip_mean(sub, axis=0, return_counts=True)
        same = bool(np.array_equal(n - lo - hi, counts))
        t_rw = best_of(lambda: rw.sigma_clip_mean(x, axis=0), 3)
        t_ap = best_of(lambda: sigma_clip(sub, sigma=3, maxiters=5, cenfunc="median", stdfunc="std", axis=0), 1,
                       warmup=False) * npix / 4096
        for tool, t, fam in (("rankwise · " + {"avx512": "AVX-512"}.get(rw.current_isa(), rw.current_isa()), t_rw,
                              "rankwise"), ("astropy 8.0 (scaled)", t_ap, "competitor")):
            rows.append(dict(n=n, npix=npix, tool=tool, op="sigclip", family=fam, seconds=t,
                             ns_per_value=t / cells * 1e9, identical=same if fam == "rankwise" else None))
            print(f"n={n:6d} sigclip  {tool:24s} {t:7.3f} s  {t / cells * 1e9:6.2f} ns/value", flush=True)

save("deep", dict(machine=machine(), versions=versions("numpy", "bottleneck", "astropy", "rankwise"), rows=rows))
