# rankwise

Lane-parallel SIMD order statistics for NumPy arrays: medians, quantiles, ranks and sigma
clipping along one axis, with results identical to NumPy, SciPy and astropy, and many times
faster on short axes (stacks of frames, ensemble members, scene time series).

```python
import numpy as np, rankwise as rw

stack = np.load("lights.npy", mmap_mode="r")          # (40, H, W) float32, never copied
clean = rw.sigma_clip_mean(stack, axis=0, sigma=(3, 3), maxiters=5)   # astropy's decisions
med = rw.median(stack, axis=0)                          # == np.median, dtype included
p10, p90 = rw.quantile(stack, [0.1, 0.9], axis=0)       # == np.quantile, every method
```

Runs AVX-512, AVX2, NEON or a portable kernel, chosen at runtime. See the
[project README](https://github.com/brannondorsey/rankwise) for benchmarks, the design and the
conformance suite. MIT OR Apache-2.0.
