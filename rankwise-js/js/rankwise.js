// rankwise for JavaScript: lane-parallel SIMD order statistics, identical to NumPy, SciPy
// and astropy. This is the WebAssembly entry point (browsers, bundlers, workers, and the
// `rankwise/wasm` export); Node, Bun and Deno resolve `rankwise` to node.js, which prefers
// the native addon.
//
//   import { init } from "rankwise";
//   const rw = await init();
//   const master = rw.median(frames);                          // frames: Float32Array[]
//   const [p10, p90] = rw.quantile(cube, [0.1, 0.9], { shape: [31, 721, 1440] });
//
// Inputs are a list of same-length frames (TypedArrays) or one TypedArray with `shape`
// (C order) and `axis` (default 0). float32, float64 and uint16 are reduced directly;
// uint8 (including ImageData's Uint8ClampedArray) is widened to uint16.

export { Rankwise, RankwiseError, ZeroCopyArray } from "./core.js";
export { init, initSync, simdSupported, WasmArray } from "./wasm.js";
export { init as default } from "./wasm.js";
