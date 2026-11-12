# rankwise for JavaScript

rankwise's lane-parallel order statistics compiled to WebAssembly, for Node, Bun, Deno and browsers.
It covers medians, quantiles, ranks and sigma clipping along one axis of a stack, with the same
results as NumPy, SciPy and astropy. One ES module, no dependencies, and synchronous calls once it is loaded.

```js
import { init } from "rankwise";            // in this repository: rankwise-wasm/js/rankwise.js

const rw = await init();                    // the simd128 build, or the portable one without WebAssembly SIMD

const master = rw.median(frames);           // frames: Float32Array[] of equal length → Float32Array
const [p10, p90] = rw.quantile(cube, [0.1, 0.9], { shape: [31, 721, 1440] });  // one array, axis 0
const clean = rw.sigmaClipMean(subs, { sigma: 3, maxiters: 5 });               // astropy's decisions
const { value, rejected, mask } = rw.sigmaClipMean(subs, { mask: true, returnCounts: true });
const composite = rw.median(scenes, { nodata: 0 });                             // uint16, 0 = missing
```

The other options and functions follow NumPy's names, written in camelCase. It reduces **float32, float64 and uint16** directly.
Uint8Array and ImageData's Uint8ClampedArray are widened to uint16 as they are copied in.

> The package is not on npm yet (its name is unregistered). Build it from this repository:
> `node rankwise-wasm/scripts/build.mjs` (needs Rust and `rustup target add wasm32-unknown-unknown`),
> then import `rankwise-wasm/js/rankwise.js`.

## Inputs

| Form | Reduces | Result |
|---|---|---|
| `[f0, f1, …]` (same-length TypedArrays) | across the list | one array the length of a frame |
| `x` with `{ shape, axis }` (C order, `axis` default 0) | along `axis` | the other dimensions, flattened in C order |
| `{ data, shape }` (ndarray-style) | along `axis` | as above |
| a 1-D TypedArray with no `shape` | the whole array | a number |

Functions with several results (`quantile` with an array `q`, `selectRanks`, `argselect`) return an
array of TypedArrays. Pass `out` (an array, or one array per result) to write into memory you own.

## API

| Function | Same results as |
|---|---|
| `median`, `nanmedian` | `np.median`, `np.nanmedian` (float32 stays float32; integers give float64) |
| `quantile`, `nanquantile`, `percentile`, `nanpercentile` (`method`: all 13) | NumPy, dtype included |
| `min`, `max`, `nanmin`, `nanmax`, `ptp` | NumPy, in the input's type |
| `argmin`, `argmax`, `nanargmin`, `nanargmax`, `argmedian`, `nanargmedian`, `argselect` | NumPy, as Int32Array |
| `selectRanks(x, ranks)` | `np.sort(x, axis)[ranks]` |
| `trimmedMean`, `iqr`, `mad` | `scipy.stats.trim_mean`, `iqr`, `median_abs_deviation` |
| `sigmaClipMean`, `clipCombine({ method })` | `astropy.stats.sigma_clip(...).mean()`, plus MAD, median-sigma, winsorized, percentile and the Siril / PixInsight presets |
| `nodata` option (uint16) | missing values skipped, like NaN in the `nan*` functions |
| `scale`, `offset`, `weights` (clipping) | per-frame normalization and weights in the same pass |
| `availableIsas()`, `setIsa()`, `isa`, `build`, `describe()`, `networkSize()` | dispatch control and introspection |

How `q` is passed selects NumPy's arithmetic, as it does in Python. A number behaves like a Python
float, so float32 data interpolates in float32. An Array or Float64Array behaves like a list and
interpolates in float64. A Float32Array behaves like `np.float32` values.

## Memory

A WebAssembly module computes only on its own linear memory, so inputs are copied in. Frame lists, and
arrays reduced along axis 0, are copied in 1 MiB chunks (`rw.chunkBytes`), each processed right after it is
copied. That bounds memory and keeps the copied chunk in cache for the kernel. Reducing an inner axis copies
the whole array once.

To skip the copy, allocate the data in rankwise's memory and fill it there:

```js
const stack = rw.alloc("float32", n * h * w);   // a WasmArray
decodeInto(stack.array);                        // fill it (stack.array is a Float32Array view)
const out = rw.alloc("float32", h * w);
rw.median(stack.array, { shape: [n, h * w], out: out.array });
// ...
stack.free(); out.free();
```

Memory growth detaches old views, so read `.array` again after calls rather than keeping it.
WebAssembly memory is limited to 4 GiB.

## Threads

Each instance runs on the thread that calls it. To use more cores, run one instance per
[worker_thread](../examples/js/node-parallel.mjs) or Web Worker. Give each one a strip of pixels and share the
frames through SharedArrayBuffers. That runs 3–4× faster on 4 cores, with identical results.

## Builds and engines

`dist/rankwise-simd128.wasm` (400 KB gzipped) needs WebAssembly SIMD, which is in Chrome 91+, Firefox 89+, Safari
16.4+, Node 16.4+, Bun and Deno. `dist/rankwise-portable.wasm` (220 KB gzipped) runs anywhere. `init()` picks the
SIMD build when `WebAssembly.validate` accepts a SIMD instruction. Pass `init({ simd: false })` to force the
portable build, or `init({ wasm })` to load bytes, a URL, a Response or a compiled module yourself (for
bundlers without `new URL(..., import.meta.url)` support, Cloudflare Workers, and similar). `initSync(bytes)` loads
synchronously.

Loading takes 15–45 ms. The first few calls run while the engine is still optimizing the kernels and take up
to about twice as long as later ones. See [docs/wasm.md](../docs/wasm.md) for benchmarks, design notes and the known differences.

## Development

```bash
node scripts/build.mjs [--dev]                     # both .wasm files into dist/ (--dev: no LTO, faster)
pip install -r ../conformance/requirements.txt
python test/make_fixtures.py                       # NumPy / SciPy / astropy references
node test/api.test.mjs && node test/conformance.test.mjs     # and the same with bun
npm install && node bench/bench.mjs                # benchmarks (d3-array is a dev dependency)
```
