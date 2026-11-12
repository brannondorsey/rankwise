/** Arrays rankwise reduces. Uint8Array and Uint8ClampedArray are widened to uint16. */
export type Input = Float32Array | Float64Array | Uint16Array | Uint8Array | Uint8ClampedArray;

/** An output array (`out`). */
export type Output = Float32Array | Float64Array | Uint16Array | Uint8Array | Uint8ClampedArray | Int32Array | BigInt64Array;

/** An array with its shape, as ndarray-style libraries hold it (C order). */
export interface Shaped<T extends Input = Input> {
  data: T;
  shape: readonly number[];
}

/**
 * A stack to reduce: a list of same-length frames (reduced across the list), a TypedArray with
 * `shape` (reduced along `axis`), a `{ data, shape }` object or a {@link WasmArray}.
 */
export type Stack<T extends Input = Input> = readonly (T | ZeroCopyArray<T>)[] | T | Shaped<T> | ZeroCopyArray<T>;

export type Isa = "simd128" | "scalar" | "avx2" | "avx512" | "neon";

export interface CommonOptions {
  /** C-order shape of a single TypedArray input (or of each frame, for frame lists). */
  shape?: readonly number[];
  /** Axis to reduce (default 0; negative counts from the end). */
  axis?: number;
  /** Force an ISA for this call. */
  isa?: Isa | "auto";
  /** Worker threads for this call (native backend; WebAssembly runs on the calling thread). Default 1. */
  threads?: number;
}

export interface ReduceOptions<O extends Output = Output> extends CommonOptions {
  /** Write here instead of allocating (its type picks the output dtype). */
  out?: O;
  /** uint16 value marking missing data; skipped like NaN in the nan* functions. */
  nodata?: number;
}

export type QuantileMethod =
  | "inverted_cdf" | "averaged_inverted_cdf" | "closest_observation" | "interpolated_inverted_cdf" | "hazen"
  | "weibull" | "linear" | "median_unbiased" | "normal_unbiased" | "lower" | "higher" | "midpoint" | "nearest";

export interface QuantileOptions extends CommonOptions {
  /** One of NumPy's 13 methods (default "linear"). */
  method?: QuantileMethod;
  /** One array per quantile (or one array for a scalar q). */
  out?: Output | Output[];
  nodata?: number;
}

export type NanPolicy = "propagate" | "omit" | "raise";

export interface ClipOptions extends CommonOptions {
  /** "sigma" (astropy), "mad", "median_sigma", "winsorized", "percentile" (Siril) or "none". */
  method?: "sigma" | "mad" | "median_sigma" | "winsorized" | "percentile" | "none";
  /** Sigma (or `[low, high]`); for "percentile", the fractional thresholds. Default 3. */
  sigma?: number | readonly [number, number];
  sigmaLower?: number;
  sigmaUpper?: number;
  /** Rejection passes (default 5; `null` = until nothing changes). */
  maxiters?: number | null;
  cenfunc?: "median" | "mean";
  stdfunc?: "std" | "mad_std";
  combine?: "mean" | "median" | "sum";
  preset?: "astropy" | "siril" | "pixinsight" | null;
  /** Arithmetic of center, spread and bounds: "f64" (default, as astropy) or "f32". */
  precision?: "f64" | "f32";
  /** Treat ±inf as missing as well as NaN (default true, as astropy). */
  maskInfinite?: boolean;
  /** Winsorization cutoff (PixInsight), 0 to disable. */
  cutoff?: number;
  /** Per-frame normalization `x * scale[k] + offset[k]`, applied on load. */
  scale?: ArrayLike<number>;
  offset?: ArrayLike<number>;
  /** Per-frame weights for the mean. */
  weights?: ArrayLike<number>;
  out?: Float32Array | Float64Array;
  /** Return `{ value, low, high, rejected }`. */
  returnCounts?: boolean;
  low?: Uint16Array | true;
  high?: Uint16Array | true;
  rejected?: Uint16Array | true;
  /**
   * Return the rejection map in `mask` (1 = rejected): one Uint8Array per frame for frame
   * lists, else one Uint8Array shaped like the input.
   */
  mask?: true | Uint8Array | Uint8Array[];
}

export interface ClipResult<V> {
  value: V;
  low?: Uint16Array | number;
  high?: Uint16Array | number;
  rejected?: Uint16Array | number;
  mask?: Uint8Array | Uint8Array[];
}

/** Errors from the kernels. */
export class RankwiseError extends Error {
  readonly kind: "depth" | "shape" | "param" | "dtype" | "isa" | "request";
}

/**
 * An array the backend reads and writes without copying (`rw.alloc`): in WebAssembly memory
 * for the wasm backend, an ordinary TypedArray natively.
 */
export class ZeroCopyArray<T extends Output = Output> {
  readonly length: number;
  /** A fresh view (WebAssembly memory growth detaches old ones: read it again after calls). */
  readonly array: T;
  free(): void;
}

/** A {@link ZeroCopyArray} in WebAssembly memory. */
export class WasmArray<T extends Output = Output> extends ZeroCopyArray<T> {
  readonly ptr: number;
  readonly bytes: number;
}

type Float<T> = T extends Float32Array ? Float32Array : Float64Array;
type Same<T> = T extends Uint8Array ? Uint8Array : T extends Uint8ClampedArray ? Uint8ClampedArray : T;
/** float32 stays float32; float64 and integers give float64. Numbers for 1-D input. */
type Res<A> = A | number;

export class Rankwise {
  /** Which engine runs the kernels: the Node-API addon or WebAssembly. */
  readonly backend: "native" | "wasm";
  /** The binary: "simd128" or "portable" for wasm, the platform (e.g. "linux-x64-gnu") for native. */
  readonly build: string;
  /** Why `init()` did not use the native addon when it fell back to WebAssembly; else null. */
  fallbackReason: string | null;
  readonly version: string;
  readonly maxFrames: number;
  /** The ISA calls use now. */
  readonly isa: Isa;
  /** The module's memory (wasm backend; undefined natively). */
  readonly memory: WebAssembly.Memory | undefined;
  /** Bytes copied into WebAssembly memory per kernel call when streaming (wasm backend, default 1 MiB). */
  chunkBytes: number;

  availableIsas(): Isa[];
  setIsa(name: Isa | "auto" | null): void;
  networkSize(n: number, ranks?: ArrayLike<number>): number;

  alloc(dtype: "float32", length: number): ZeroCopyArray<Float32Array>;
  alloc(dtype: "float64", length: number): ZeroCopyArray<Float64Array>;
  alloc(dtype: "uint16", length: number): ZeroCopyArray<Uint16Array>;
  alloc(dtype: "uint8", length: number): ZeroCopyArray<Uint8Array>;
  alloc(dtype: "int32", length: number): ZeroCopyArray<Int32Array>;
  alloc(dtype: "int64", length: number): ZeroCopyArray<BigInt64Array>;
  /** Free the streaming buffer kept between calls. */
  releaseScratch(): void;

  /** `np.median` along one axis. */
  median<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Float<T>>;
  /** `np.nanmedian` along one axis. */
  nanmedian<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Float<T>>;

  /**
   * `np.quantile` along one axis, every method. A number `q` behaves like a Python float
   * (float32 data interpolates in float32); an array like a list (float64); a Float32Array
   * like a float32 array.
   */
  quantile<T extends Input>(x: Stack<T>, q: number, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>;
  quantile<T extends Input>(x: Stack<T>, q: ArrayLike<number>, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>[];
  nanquantile<T extends Input>(x: Stack<T>, q: number, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>;
  nanquantile<T extends Input>(x: Stack<T>, q: ArrayLike<number>, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>[];
  percentile<T extends Input>(x: Stack<T>, q: number, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>;
  percentile<T extends Input>(x: Stack<T>, q: ArrayLike<number>, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>[];
  nanpercentile<T extends Input>(x: Stack<T>, q: number, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>;
  nanpercentile<T extends Input>(x: Stack<T>, q: ArrayLike<number>, opts?: QuantileOptions): Res<Float32Array | Float64Array | Same<T>>[];

  min<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Same<T>>;
  max<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Same<T>>;
  nanmin<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Same<T>>;
  nanmax<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Same<T>>;
  /** max - min in the input dtype. */
  ptp<T extends Input>(x: Stack<T>, opts?: ReduceOptions): Res<Same<T>>;

  /** Frame indices (first occurrence) as Int32Array. */
  argmin(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  argmax(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  /** All-NaN slices give -1. */
  nanargmin(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  nanargmax(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  /** Which frame supplied the (lower) median. */
  argmedian(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  nanargmedian(x: Stack, opts?: ReduceOptions): Res<Int32Array>;
  /** `np.argsort(kind="stable")` at each rank. */
  argselect(x: Stack, ranks: number, opts?: CommonOptions): Res<Int32Array>;
  argselect(x: Stack, ranks: ArrayLike<number>, opts?: CommonOptions): Res<Int32Array>[];
  /** `np.sort(x, axis)[ranks]` in one pass; negative ranks count from the end. */
  selectRanks<T extends Input>(x: Stack<T>, ranks: number, opts?: ReduceOptions): Res<Same<T>>;
  selectRanks<T extends Input>(x: Stack<T>, ranks: ArrayLike<number>, opts?: { out?: Output[] } & CommonOptions): Res<Same<T>>[];

  /** `scipy.stats.trim_mean`. */
  trimmedMean<T extends Input>(x: Stack<T>, proportion: number, opts?: CommonOptions): Res<Float<T>>;
  /** `scipy.stats.iqr`. */
  iqr<T extends Input>(x: Stack<T>, opts?: CommonOptions & { rng?: [number, number]; scale?: number | "normal"; nanPolicy?: NanPolicy; out?: Output }): Res<Float<T>>;
  /** `scipy.stats.median_abs_deviation` (median center). */
  mad<T extends Input>(x: Stack<T>, opts?: CommonOptions & { scale?: number | "normal"; nanPolicy?: NanPolicy; out?: Output }): Res<Float<T>>;

  /** Reject outliers and combine the survivors. */
  clipCombine<T extends Input>(x: Stack<T>, opts?: ClipOptions): Res<Float<T>> | ClipResult<Res<Float<T>>>;
  /** Sigma-clipped mean with astropy's `sigma_clip` decisions. */
  sigmaClipMean<T extends Input>(x: Stack<T>, opts?: ClipOptions): Res<Float<T>> | ClipResult<Res<Float<T>>>;

  /** How a call would run: ISA, lanes, comparators, memory path. */
  describe(x: Stack, opts?: CommonOptions & { op?: "median" | "quantile" | "sigma_clip"; q?: number }): string;
}

export interface InitOptions {
  /**
   * Node, Bun and Deno: "auto" (default; the native addon if it loads, else WebAssembly),
   * "native" (throw if it does not load) or "wasm". Defaults to RANKWISE_BACKEND. Browsers
   * and the `rankwise/wasm` entry always use WebAssembly.
   */
  backend?: "auto" | "native" | "wasm";
  /** WebAssembly build: true, false or "auto" (default). */
  simd?: boolean | "auto";
  /** Where the WebAssembly module comes from: a URL or path, a Response, bytes or a compiled module. */
  wasm?: string | URL | Response | BufferSource | WebAssembly.Module;
}

/** Load rankwise (cached per choice). */
export function init(options?: InitOptions): Promise<Rankwise>;
/** Synchronous load from WebAssembly bytes or a compiled module. */
export function initSync(wasm: BufferSource | WebAssembly.Module): Rankwise;
/** Node, Bun and Deno: the same choice as `init()`, synchronously. */
export function initSync(options?: Omit<InitOptions, "wasm"> & { wasm?: BufferSource }): Rankwise;
/** Whether this engine supports WebAssembly SIMD. */
export function simdSupported(): boolean;

export default init;
