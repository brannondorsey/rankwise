"""Power-of-two versus padded frame strides (handoff §4.6), one thread.

    taskset -c 1 python bench/stride.py
"""

import common  # noqa: F401
import numpy as np

import rankwise as rw
from common import best_of, machine, save

rng = np.random.default_rng(0)
rows = []
for (h, w) in [(1024, 1024), (2048, 2048), (721, 1440)]:
    npix = h * w
    for pad in (0, 16, 1024):  # extra floats per frame: none, one cache line, one 4 KiB page
        buf = rng.normal(1000, 30, (31, npix + pad)).astype(np.float32)
        st = buf[:, :npix].reshape(31, h, w)
        out = np.empty((h, w), np.float32)
        r = dict(frame=f"{h}x{w}", stride_bytes=(npix + pad) * 4, pad_bytes=pad * 4)
        r["median_ns_px"] = best_of(lambda: rw.median(st, axis=0, out=out), 5) / npix * 1e9
        r["sigclip_ns_px"] = best_of(lambda: rw.sigma_clip_mean(st, axis=0, out=out), 3) / npix * 1e9
        rows.append(r)
        print(r, flush=True)
        del buf, st
save("stride", dict(machine=machine(), unit="ns per output pixel", n=31, threads=1, rows=rows))
