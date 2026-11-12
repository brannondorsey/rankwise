# rankwise for JavaScript

rankwise's lane-parallel order statistics for Node, Bun, Deno and browsers: medians, quantiles,
ranks and sigma clipping along one axis of a stack, with the same results as NumPy, SciPy and
astropy. On a server it runs the native library (AVX-512, AVX2 or NEON, and threads) through
Node-API; everywhere else it runs WebAssembly. The API and results are identical either way, and
calls are synchronous once it is loaded.

```js
import { init } from "rankwise";

const rw = await init();                    // native addon if this platform has one, else WebAssembly
rw.backend;                                 // "native" | "wasm"
rw.isa;                                     // "avx512", "avx2", "neon", "simd128", ...

const master = rw.median(frames);           // frames: Float32Array[] of equal length → Float32Array
const [p10, p90] = rw.quantile(cube, [0.1, 0.9], { shape: [31, 721, 1440] });  // one array, axis 0
const clean = rw.sigmaClipMean(subs, { sigma: 3, maxiters: 5 });               // astropy's decisions
const { value, rejected, mask } = rw.sigmaClipMean(subs, { mask: true, returnCounts: true });
const composite = rw.median(scenes, { nodata: 0, threads: 8 });                 // uint16, 0 = missing
```

The other options and functions follow NumPy's names, written in camelCase. It reduces **float32, float64 and uint16** directly.
Uint8Array and ImageData's Uint8ClampedArray are widened to uint16.

> The packages are not on npm yet (the names are unregistered). Build from this repository:
> `node rankwise-js/scripts/build.mjs --all` (needs Rust, and `rustup target add wasm32-unknown-unknown`),
> then import `rankwise-js/js/node.js` (Node, Bun, Deno) or `rankwise-js/js/rankwise.js` (browsers).

## Backends: native or WebAssembly

The choice is made in two places, each by the tool that owns it.

**At install time, by the package manager.** `rankwise` lists one package per platform as an
optional dependency (`@rankwise/linux-x64-gnu`, `linux-arm64-gnu`, `darwin-arm64`, `darwin-x64`,
`win32-x64-msvc`), each with `os`, `cpu` and `libc` set. npm, pnpm, yarn and bun install only the one
that matches the machine, about 1.7 MB. This is how esbuild, SWC and Lightning CSS ship binaries:
no install scripts, no downloads at runtime, and lockfiles stay portable.

**At load time, by the export conditions.** Node, Bun and Deno resolve `rankwise` through the
`"node"` condition to `js/node.js`. It loads the platform's addon and falls back to WebAssembly when
the addon is missing or cannot load (`--omit=optional`, an unsupported platform such as musl,
Deno without `--allow-ffi`). Browsers and bundlers resolve the `"browser"` condition to
`js/rankwise.js`, which never references the addon, so bundles stay WebAssembly-only.
`rankwise/wasm` selects that entry explicitly anywhere.

The choice is visible and can be overridden:

| | |
|---|---|
| `rw.backend`, `rw.build`, `rw.isa` | what runs: `"native"`, `"linux-x64-gnu"`, `"avx512"` |
| `rw.fallbackReason` | why the addon was not used, e.g. `no native addon for linux-x64-musl could be loaded (...)` |
| `init({ backend: "native" })` | the addon or an error: never a silent fallback |
| `init({ backend: "wasm" })`, `RANKWISE_BACKEND=wasm` | WebAssembly even where the addon is installed |
| `RANKWISE_NATIVE_PATH=/path/rankwise.node` | an addon from a file (custom builds) |
| `RANKWISE_ISA=avx2` | the native ISA, as for the Rust, C and Python libraries |
| `initSync()` | the same choice, synchronously (Node, Bun, Deno) |

`test/package.test.mjs` checks all of this end to end. It serves the release tarballs from a
local npm registry, then installs with npm and bun as users would. Exactly one addon lands on
disk, matching the machine: the test checks each other platform with `--os`/`--cpu`/`--libc`.
Node and Bun then run it natively. `--omit=optional` gives WebAssembly with its reason, and an
esbuild browser bundle contains no native loader.

The native addon reads and writes the caller's TypedArrays in place. Inputs are never copied,
`threads` splits large calls across cores, and every region a call touches is bounds-checked
before any kernel runs. A wrong request is an error, never a stray write in the server process.

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

The native addon works on your arrays in place. A WebAssembly module computes only on its own
linear memory, so there inputs are copied in. Frame lists, and
arrays reduced along axis 0, are copied in 1 MiB chunks (`rw.chunkBytes`), each processed right after it is
copied. That bounds memory and keeps the copied chunk in cache for the kernel. Reducing an inner axis copies
the whole array once.

To skip the copy, allocate the data with `rw.alloc` and fill it there (natively this returns an ordinary
TypedArray, so the same code runs on both backends):

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

Natively, `{ threads: n }` splits one call across `n` cores (rayon), with identical results.
WebAssembly runs on the thread that calls it. To use more cores there, run one instance per
[worker_thread](../examples/js/node-parallel.mjs) or Web Worker, give each a strip of pixels, and share the
frames through SharedArrayBuffers: 3–4× faster on 4 cores, identical results. Either way a call is
synchronous; on a busy server, run large reductions in a worker so they don't block the event loop.

## WebAssembly builds and engines

`dist/rankwise-simd128.wasm` (400 KB gzipped) needs WebAssembly SIMD, which is in Chrome 91+, Firefox 89+, Safari
16.4+, Node 16.4+, Bun and Deno. `dist/rankwise-portable.wasm` (220 KB gzipped) runs anywhere. `init()` picks the
SIMD build when `WebAssembly.validate` accepts a SIMD instruction. Pass `init({ simd: false })` to force the
portable build, or `init({ wasm })` to load bytes, a URL, a Response or a compiled module yourself (for
bundlers without `new URL(..., import.meta.url)` support, Cloudflare Workers, and similar). `initSync(bytes)` loads
synchronously.

Loading takes 15–45 ms. In WebAssembly, the first few calls run while the engine is still optimizing the
kernels and take up to about twice as long as later ones; the native addon runs at full speed from the first call. See [docs/javascript.md](../docs/javascript.md) for benchmarks, design notes and the known differences.

## Development

```bash
node scripts/build.mjs --all [--dev]               # dist/*.wasm and npm/<platform>/ (--dev: no LTO, faster)
pip install -r ../conformance/requirements.txt
python test/make_fixtures.py                       # NumPy / SciPy / astropy references
node test/api.test.mjs && node test/conformance.test.mjs     # also bun, and deno run -A
node test/package.test.mjs                         # pack, serve from a local registry, install, run
node scripts/pack.mjs                              # release tarballs into release/
npm install && node bench/bench.mjs                # benchmarks (d3-array is a dev dependency)
```

Debug (unoptimized) builds of the addon need more than Node's 8 MB main-thread stack; the build
script always builds release.
