// rankwise for JavaScript: lane-parallel SIMD order statistics compiled to WebAssembly.
//
// One ES module for Node, Bun, Deno, browsers and bundlers. `init()` loads the simd128
// build where the engine supports WebAssembly SIMD (every current browser, Node >= 16.4,
// Bun, Deno) and the portable build elsewhere. Results are bit-identical between the two,
// and to the native library's (NumPy's median/quantile, SciPy's trim_mean/iqr/MAD,
// astropy's sigma_clip decisions).
//
//   import { init } from "rankwise";
//   const rw = await init();
//   const master = rw.median(frames);                          // frames: Float32Array[]
//   const [p10, p90] = rw.quantile(cube, [0.1, 0.9], { shape: [31, 721, 1440] });
//
// Inputs are a list of same-length frames (TypedArrays) or one TypedArray with `shape`
// (C order) and `axis` (default 0). float32, float64 and uint16 are reduced directly;
// uint8 (including ImageData's Uint8ClampedArray) is widened to uint16 as it is copied in.

const DTYPES = {
  float32: { code: 0, size: 4, ctor: Float32Array },
  float64: { code: 1, size: 8, ctor: Float64Array },
  uint16: { code: 2, size: 2, ctor: Uint16Array },
  int32: { code: 3, size: 4, ctor: Int32Array },
  int64: { code: 4, size: 8, ctor: BigInt64Array },
  bool: { code: 5, size: 1, ctor: Uint8Array },
  uint8: { code: 6, size: 1, ctor: Uint8Array },
};

const ISAS = ["scalar", "avx2", "avx512", "neon", "simd128"];
const STAGING = { auto: 0, never: 1, always: 2, skew: 3 };
const METHODS = [
  "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
  "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest",
];
const DISCRETE = new Set(["inverted_cdf", "closest_observation", "lower", "higher", "nearest"]);
const CLIP_METHODS = { sigma: 0, mad: 1, median_sigma: 2, winsorized: 3, percentile: 4, none: 5 };
const STATUS = ["ok", "depth", "shape", "param", "dtype", "isa", "request"];


/** Errors from the kernels: `kind` is "depth", "shape", "param", "dtype", "isa" or "request". */
export class RankwiseError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "RankwiseError";
    this.kind = kind;
  }
}

// A 30-byte module using one SIMD instruction: validates only where simd128 is supported.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

/** Whether this engine runs WebAssembly SIMD (and so the simd128 build). */
export function simdSupported() {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

function defaultUrl(simd) {
  // Literal `new URL(..., import.meta.url)` so bundlers (Vite, webpack, Parcel, esbuild with
  // the file loader) see and emit both files.
  return simd
    ? new URL("../dist/rankwise-simd128.wasm", import.meta.url)
    : new URL("../dist/rankwise-portable.wasm", import.meta.url);
}

async function compileFrom(src) {
  if (src instanceof WebAssembly.Module) return src;
  if (src instanceof ArrayBuffer || ArrayBuffer.isView(src)) return WebAssembly.compile(src);
  if (typeof Response !== "undefined" && src instanceof Response) return compileResponse(src);
  const url = src instanceof URL ? src : new URL(String(src), import.meta.url);
  if (url.protocol === "file:") {
    // Node, Bun and Deno: read from disk (Node's fetch does not do file: URLs)
    const { readFile } = await import("node:fs/promises");
    return WebAssembly.compile(await readFile(url));
  }
  return compileResponse(await fetch(url));
}

async function compileResponse(res) {
  if (!res.ok) throw new Error(`rankwise: fetching ${res.url} failed with HTTP ${res.status}`);
  const type = res.headers.get("content-type") || "";
  if (WebAssembly.compileStreaming && type.startsWith("application/wasm")) return WebAssembly.compileStreaming(res);
  return WebAssembly.compile(await res.arrayBuffer());
}

const cache = new Map();

/**
 * Load rankwise. Resolves to a {@link Rankwise} instance (cached per build).
 *
 * Options: `simd` (true / false / "auto", default "auto") picks the build; `wasm` overrides
 * where it comes from (a URL or path, a Response, bytes or a compiled WebAssembly.Module).
 */
export function init(options = {}) {
  const simd = options.simd === undefined || options.simd === "auto" ? simdSupported() : Boolean(options.simd);
  if (simd && !simdSupported()) {
    return Promise.reject(new Error("rankwise: this engine does not support WebAssembly SIMD; use init({ simd: false })"));
  }
  const build = simd ? "simd128" : "portable";
  if (options.wasm !== undefined) return compileFrom(options.wasm).then(instantiate);
  if (!cache.has(build)) {
    const p = compileFrom(defaultUrl(simd)).then(instantiate);
    p.catch(() => cache.delete(build));
    cache.set(build, p);
  }
  return cache.get(build);
}

/**
 * Synchronous form of {@link init} for code that already holds the module bytes or a compiled
 * `WebAssembly.Module` (workers, serverless bundles).
 */
export function initSync(wasm) {
  const m = wasm instanceof WebAssembly.Module ? wasm : new WebAssembly.Module(wasm);
  return new Rankwise(new WebAssembly.Instance(m, {}));
}

async function instantiate(module) {
  return new Rankwise(await WebAssembly.instantiate(module, {}));
}

// ------------------------------------------------------------------ helpers

function dtypeOf(a) {
  if (a instanceof Float32Array) return "float32";
  if (a instanceof Float64Array) return "float64";
  if (a instanceof Uint16Array) return "uint16";
  if (a instanceof Uint8Array || a instanceof Uint8ClampedArray) return "uint8";
  if (a instanceof Int32Array) return "int32";
  if (typeof BigInt64Array !== "undefined" && a instanceof BigInt64Array) return "int64";
  return null;
}

function isTyped(a) {
  return ArrayBuffer.isView(a) && !(a instanceof DataView);
}

function prod(s) {
  let p = 1;
  for (const d of s) p *= d;
  return p;
}

function cStrides(shape, size) {
  const s = new Array(shape.length);
  let acc = size;
  for (let d = shape.length - 1; d >= 0; d--) {
    s[d] = acc;
    acc *= shape[d];
  }
  return s;
}

function checkShape(shape, len) {
  if (!Array.isArray(shape) && !isTyped(shape)) throw new TypeError("shape must be an array of integers");
  const s = Array.from(shape, Number);
  for (const d of s) if (!Number.isInteger(d) || d < 0) throw new TypeError(`invalid shape [${s}]`);
  if (prod(s) !== len) throw new RangeError(`shape [${s}] holds ${prod(s)} elements, the array has ${len}`);
  return s;
}

function floatResult(dt) {
  return dt === "float32" ? "float32" : "float64";
}

/**
 * A TypedArray living inside rankwise's WebAssembly memory, so calls read it without
 * copying. Fill `.array`, pass `.array` (or the handle itself) as input or `out`, `free()` it
 * when done. `.array` re-creates the view whenever the memory has grown (growth detaches old
 * views), so read it again after each call rather than keeping it.
 */
export class WasmArray {
  #rw;
  #ctor;
  constructor(rw, ctor, ptr, length, bytes) {
    this.#rw = rw;
    this.#ctor = ctor;
    this.ptr = ptr;
    this.length = length;
    this.bytes = bytes;
  }
  get array() {
    if (!this.ptr) throw new Error("rankwise: this WasmArray was freed");
    return new this.#ctor(this.#rw.memory.buffer, this.ptr, this.length);
  }
  free() {
    if (this.ptr) this.#rw._free(this.ptr, this.bytes);
    this.ptr = 0;
  }
}

// ------------------------------------------------------------------ the instance

/** A loaded rankwise module. Every reduction method is synchronous. */
export class Rankwise {
  #x;
  #scratch = 0;
  #scratchBytes = 0;
  /** "simd128" or "portable": which build is loaded. */
  build;
  /** Deepest stack one call reduces. */
  maxFrames;
  /** Library version, e.g. "0.1.0". */
  version;
  /**
   * Bytes (all frames together) copied into WebAssembly memory per kernel call when inputs
   * live outside it. Small enough that the copy is still in cache when the kernel reads it,
   * large enough to amortize the call.
   */
  chunkBytes = 1 << 20;

  constructor(instance) {
    this.#x = instance.exports;
    this.build = this.#x.rw_available_isas() & (1 << 4) ? "simd128" : "portable";
    this.maxFrames = this.#x.rw_max_frames();
    const v = this.#x.rw_version();
    this.version = `${Math.floor(v / 10000)}.${Math.floor(v / 100) % 100}.${v % 100}`;
  }

  /** The module's WebAssembly.Memory. */
  get memory() {
    return this.#x.memory;
  }

  // ---------------------------------------------------------------- dispatch

  /** ISAs this module can run, fastest first: `["simd128", "scalar"]` or `["scalar"]`. */
  availableIsas() {
    const bits = this.#x.rw_available_isas();
    return [4, 2, 3, 1, 0].filter((c) => bits & (1 << c)).map((c) => ISAS[c]);
  }

  /** The ISA calls use now. */
  get isa() {
    return ISAS[this.#x.rw_current_isa()];
  }

  /** Force an ISA for later calls (`null` or "auto" restores the default). */
  setIsa(name) {
    const code = name == null || name === "auto" ? -1 : ISAS.indexOf(name);
    if (code === -1 && name != null && name !== "auto") throw new RangeError(`unknown ISA ${JSON.stringify(name)}`);
    this.#check(this.#x.rw_set_isa(code));
  }

  /** Comparators in the sorting network for `n` values, pruned to `ranks` if given. */
  networkSize(n, ranks) {
    if (!ranks || ranks.length === 0) return this.#x.rw_network_size(n, 0, 0);
    const r = Uint32Array.from(ranks);
    const p = this._alloc(r.byteLength);
    try {
      new Uint32Array(this.memory.buffer, p, r.length).set(r);
      return this.#x.rw_network_size(n, p, r.length);
    } finally {
      this._free(p, r.byteLength);
    }
  }

  // ---------------------------------------------------------------- memory

  /**
   * Allocate a zero-copy array in WebAssembly memory: `dtype` is "float32", "float64",
   * "uint16", "uint8", "int32" or "int64". Returns a {@link WasmArray}.
   */
  alloc(dtype, length) {
    const d = DTYPES[dtype];
    if (!d) throw new TypeError(`unknown dtype ${JSON.stringify(dtype)}`);
    const bytes = Math.max(1, length * d.size);
    const p = this._alloc(bytes);
    new Uint8Array(this.memory.buffer, p, bytes).fill(0);
    return new WasmArray(this, d.ctor, p, length, bytes);
  }

  _alloc(bytes) {
    const p = this.#x.rw_alloc(bytes);
    if (!p) throw new RangeError(`rankwise: cannot allocate ${bytes} bytes of WebAssembly memory (4 GiB at most)`);
    return p;
  }

  _free(p, bytes) {
    this.#x.rw_free(p, bytes);
  }

  /** Release the scratch buffer kept between calls. */
  releaseScratch() {
    if (this.#scratch) this._free(this.#scratch, this.#scratchBytes);
    this.#scratch = 0;
    this.#scratchBytes = 0;
  }

  #reserve(bytes) {
    bytes = Math.ceil(bytes / 64) * 64 + 64;
    if (bytes > this.#scratchBytes) {
      this.releaseScratch();
      this.#scratch = this._alloc(bytes);
      this.#scratchBytes = bytes;
    }
    return this.#scratch;
  }

  #resident(a) {
    return a.buffer === this.memory.buffer;
  }

  #check(st) {
    if (st === 0) return;
    const msg = new TextDecoder().decode(new Uint8Array(this.memory.buffer, this.#x.rw_message_ptr(), this.#x.rw_message_len()));
    throw new RankwiseError(STATUS[st] || "unknown", msg);
  }

  // ---------------------------------------------------------------- input

  #prep(x, opts = {}) {
    if (x instanceof WasmArray) x = x.array;
    if (Array.isArray(x) && x.length > 0 && x.every((f) => isTyped(f) || f instanceof WasmArray)) {
      if (opts.axis !== undefined && opts.axis !== 0) {
        throw new RangeError(`a list of frames is reduced along axis 0, got axis ${opts.axis}`);
      }
      const frames = x.map((f) => (f instanceof WasmArray ? f.array : f));
      const src = dtypeOf(frames[0]);
      const len = frames[0].length;
      frames.forEach((f, i) => {
        if (dtypeOf(f) !== src || f.constructor !== frames[0].constructor) {
          throw new TypeError(`frame ${i} is a ${f.constructor.name}, frame 0 is a ${frames[0].constructor.name}`);
        }
        if (f.length !== len) throw new RangeError(`frame ${i} has ${f.length} elements, frame 0 has ${len}`);
      });
      const st = this.#base(src, frames[0]);
      const shape = opts.shape ? checkShape(opts.shape, len) : [len];
      return { ...st, kind: "frames", frames, n: frames.length, outShape: shape, npix: len, scalar: false };
    }
    if (Array.isArray(x)) {
      if (x.length === 0) throw new RangeError("cannot reduce an empty list of frames");
      throw new TypeError("expected a TypedArray, or an array of TypedArray frames");
    }
    if (x && !isTyped(x) && isTyped(x.data) && x.shape) {
      // ndarray-like { data, shape }
      opts = { ...opts, shape: x.shape };
      x = x.data;
    }
    if (!isTyped(x)) throw new TypeError("expected a TypedArray, or an array of TypedArray frames");
    const src = dtypeOf(x);
    const st = this.#base(src, x);
    const shape = opts.shape ? checkShape(opts.shape, x.length) : [x.length];
    if (shape.length === 0) throw new RangeError("cannot reduce a 0-d array");
    let axis = opts.axis ?? 0;
    if (!Number.isInteger(axis) || axis < -shape.length || axis >= shape.length) {
      throw new RangeError(`axis ${axis} is out of bounds for an array of dimension ${shape.length}`);
    }
    axis = (axis + shape.length) % shape.length;
    const n = shape[axis];
    if (n === 0) throw new RangeError("cannot reduce an empty axis");
    const outShape = shape.filter((_, d) => d !== axis);
    const npix = prod(outShape);
    if (axis === 0 || prod(shape.slice(0, axis)) === 1) {
      // frames are contiguous blocks of the array
      const frames = Array.from({ length: n }, (_, k) => x.subarray(k * npix, (k + 1) * npix));
      return { ...st, kind: "frames", frames, n, outShape, npix, scalar: outShape.length === 0, flat: x };
    }
    return { ...st, kind: "strided", data: x, shape, axis, n, outShape, npix, scalar: false };
  }

  #base(src, sample) {
    if (!["float32", "float64", "uint16", "uint8"].includes(src)) {
      throw new TypeError(
        `rankwise reduces Float32Array, Float64Array and Uint16Array data (Uint8Array is widened); got ${sample.constructor.name}`,
      );
    }
    return { src, dtype: src === "uint8" ? "uint16" : src, srcCtor: sample.constructor };
  }

  // ---------------------------------------------------------------- the call

  /**
   * Run one reduction. `outs`: [{ dtype, out?, none? }] (out: caller array to write; none: not
   * requested). `mask`: true or caller arrays for the rejection map.
   * Returns { values: (TypedArray | null)[], mask }.
   */
  #run(s, op, outs, opts, mask = null) {
    const nout = outs.length;
    const esz = DTYPES[s.dtype].size;
    const In = DTYPES[s.dtype].ctor;
    // The kernel's uint8 output is boolean; uint8 values (min/max/select of uint8 input into a
    // Uint8Array `out`) are written as uint16 and narrowed on the copy out.
    const odt = outs.map((o) => DTYPES[o.dtype === "uint8" ? "uint16" : o.dtype]);
    // Result arrays (the caller's `out` or new ones), made before any WebAssembly allocation.
    const results = outs.map((o) => {
      if (o.none) return null;
      if (o.out) {
        if (o.out.length !== s.npix) throw new RangeError(`out has ${o.out.length} elements, expected ${s.npix}`);
        return o.out;
      }
      return new DTYPES[o.dtype].ctor(s.npix);
    });
    const maskRes = mask ? this.#maskArrays(s, mask) : null;
    const tail = this.#optsTail(s, opts);
    const head = [...op, DTYPES[s.dtype].code, s.n];

    const native = s.src === s.dtype;
    const residentFrames = s.kind === "frames" && native && s.frames.every((f) => this.#resident(f));
    const residentData = s.kind === "strided" && native && this.#resident(s.data);

    if (s.kind === "frames" && !residentFrames) {
      // Stream pixel chunks of every frame through linear memory: bounded memory, and the
      // kernel reads each chunk from cache right after it was copied.
      const chunk = Math.max(64, Math.min(Math.ceil(s.npix / 64) * 64, Math.floor(this.chunkBytes / (s.n * esz) / 64) * 64));
      const fstride = Math.ceil((chunk * esz) / 64) * 64 + 64; // +1 line: no 2^k cache aliasing
      const ostride = odt.map((d, j) => (results[j] ? Math.ceil((chunk * d.size) / 64) * 64 : 0));
      const mstride = mask ? Math.ceil(chunk / 64) * 64 : 0;
      const reqMax = head.length + 8 + 2 * s.n + 3 * nout + tail.length;
      const base = this.#reserve(s.n * fstride + ostride.reduce((a, b) => a + b, 0) + s.n * mstride + reqMax * 8 + 8);
      const faddr = Array.from({ length: s.n }, (_, k) => base + k * fstride);
      let p = base + s.n * fstride;
      const oaddr = ostride.map((b) => ((p += b), p - b));
      const maddr = Array.from({ length: s.n }, (_, k) => p + k * mstride);
      p += s.n * mstride;
      const raddr = Math.ceil(p / 8) * 8;
      let heap, views, oviews, mviews;
      const mkViews = () => {
        heap = this.memory.buffer;
        views = faddr.map((a) => new In(heap, a, chunk));
        oviews = odt.map((d, j) => (results[j] ? new d.ctor(heap, oaddr[j], chunk) : null));
        mviews = mask ? maddr.map((a) => new Uint8Array(heap, a, chunk)) : null;
      };
      mkViews();
      let p0 = 0;
      do {
        const len = Math.min(chunk, s.npix - p0);
        for (let k = 0; k < s.n; k++) {
          const f = s.frames[k];
          views[k].set(p0 === 0 && len === f.length ? f : f.subarray(p0, p0 + len));
        }
        const req = [...head, 1, len, esz, ...faddr, nout];
        for (let j = 0; j < nout; j++) req.push(results[j] ? oaddr[j] : 0, odt[j].code, odt[j].size);
        if (mask) req.push(1, ...maddr, DTYPES.bool.code, 1);
        else req.push(0);
        req.push(...tail);
        new Float64Array(heap, raddr, req.length).set(req);
        this.#check(this.#x.rw_reduce(raddr, req.length));
        if (this.memory.buffer !== heap) mkViews(); // the kernel grew memory
        for (let j = 0; j < nout; j++) {
          if (results[j]) results[j].set(len === chunk ? oviews[j] : oviews[j].subarray(0, len), p0);
        }
        if (mask) for (let k = 0; k < s.n; k++) maskRes.frames[k].set(mviews[k].subarray(0, len), p0);
        p0 += chunk;
      } while (p0 < s.npix);
      return { values: results, mask: maskRes?.result ?? null };
    }

    // One call over the whole array: frames already in WebAssembly memory, or a strided
    // layout (an inner axis reduced), which is copied in whole.
    let shape, strides, frameOffsets, inBytes = 0;
    if (s.kind === "frames") {
      shape = [s.npix];
      strides = [esz];
      frameOffsets = s.frames.map((f) => f.byteOffset);
    } else {
      const st = cStrides(s.shape, esz);
      shape = s.outShape;
      strides = st.filter((_, d) => d !== s.axis);
      frameOffsets = Array.from({ length: s.n }, (_, k) => k * st[s.axis]);
      if (!residentData) inBytes = Math.ceil((s.data.length * esz) / 64) * 64;
    }
    const outResident = results.map((r, j) => r !== null && this.#resident(r) && r instanceof odt[j].ctor);
    const obytes = results.map((r, j) => (r === null || outResident[j] ? 0 : Math.ceil((s.npix * odt[j].size) / 64) * 64));
    const mbytes = mask ? Math.ceil((s.n * s.npix) / 64) * 64 : 0;
    const reqMax = head.length + 4 + 3 * shape.length + 2 * s.n + nout * (2 + shape.length) + tail.length;
    const base = this.#reserve(inBytes + obytes.reduce((a, b) => a + b, 0) + mbytes + reqMax * 8 + 8);
    let p = base;
    let frameAddrs = frameOffsets;
    if (s.kind === "strided") {
      let dataAddr = s.data.byteOffset;
      if (!residentData) {
        dataAddr = p;
        p += inBytes;
        new In(this.memory.buffer, dataAddr, s.data.length).set(s.data);
      }
      frameAddrs = frameOffsets.map((o) => dataAddr + o);
    }
    const oaddr = results.map((r, j) => (r === null ? 0 : outResident[j] ? r.byteOffset : ((p += obytes[j]), p - obytes[j])));
    const maddr = p;
    p += mbytes;
    const raddr = Math.ceil(p / 8) * 8;
    const req = [...head, shape.length, ...shape, ...strides, ...frameAddrs, nout];
    for (let j = 0; j < nout; j++) req.push(oaddr[j], odt[j].code, ...cStrides(shape, odt[j].size));
    if (mask) {
      // the map has the input's layout, in bytes
      const full = s.kind === "frames" ? [s.n, s.npix] : s.shape;
      const ax = s.kind === "frames" ? 0 : s.axis;
      const ms = cStrides(full, 1);
      req.push(1);
      for (let k = 0; k < s.n; k++) req.push(maddr + k * ms[ax]);
      req.push(DTYPES.bool.code, ...ms.filter((_, d) => d !== ax));
    } else req.push(0);
    req.push(...tail);
    new Float64Array(this.memory.buffer, raddr, req.length).set(req);
    this.#check(this.#x.rw_reduce(raddr, req.length));
    const heap = this.memory.buffer;
    for (let j = 0; j < nout; j++) {
      if (results[j] && !outResident[j]) results[j].set(new odt[j].ctor(heap, oaddr[j], s.npix));
    }
    if (mask) {
      const m = new Uint8Array(heap, maddr, s.n * s.npix);
      if (maskRes.flat) maskRes.flat.set(m);
      else for (let k = 0; k < s.n; k++) maskRes.frames[k].set(m.subarray(k * s.npix, (k + 1) * s.npix));
    }
    return { values: results, mask: maskRes?.result ?? null };
  }

  #maskArrays(s, mask) {
    // Frame lists get one Uint8Array per frame; a single array gets one Uint8Array shaped
    // like the input.
    const asList = !s.flat && s.kind === "frames";
    if (asList) {
      const frames = mask === true ? s.frames.map(() => new Uint8Array(s.npix)) : mask;
      if (!Array.isArray(frames) || frames.length !== s.n || frames.some((f) => !(f instanceof Uint8Array) || f.length !== s.npix)) {
        throw new RangeError(`mask needs ${s.n} Uint8Arrays of ${s.npix} elements`);
      }
      return { frames, result: frames };
    }
    const total = s.n * s.npix;
    const flat = mask === true ? new Uint8Array(total) : mask;
    if (!(flat instanceof Uint8Array) || flat.length !== total) throw new RangeError(`mask must be a Uint8Array of ${total} elements`);
    const frames = s.kind === "frames" ? Array.from({ length: s.n }, (_, k) => flat.subarray(k * s.npix, (k + 1) * s.npix)) : null;
    return { frames, flat, result: flat };
  }

  #optsTail(s, o) {
    const list = (v, name) => {
      if (v == null) return [-1];
      const a = Array.from(v, Number);
      if (a.length !== s.n) throw new RangeError(`${name} needs ${s.n} values, one per frame; got ${a.length}`);
      return [a.length, ...a];
    };
    let isa = -1;
    if (o.isa != null && o.isa !== "auto") {
      isa = ISAS.indexOf(o.isa);
      if (isa < 0) throw new RangeError(`unknown ISA ${JSON.stringify(o.isa)}`);
    }
    if (o.nodata != null && s.dtype !== "uint16") {
      throw new TypeError("nodata applies to integer input; float data marks missing values with NaN");
    }
    // per-frame normalization and weights belong to clipping (iqr/mad's `scale` is a factor)
    const clip = o.clip === true;
    return [
      o.omit || o.nodata != null ? 1 : 0,
      o.nodata == null ? NaN : Number(o.nodata),
      isa,
      STAGING[o.staging ?? "auto"] ?? 0,
      ...list(clip ? o.scale : null, "scale"),
      ...list(clip ? o.offset : null, "offset"),
      ...list(clip ? o.weights : null, "weights"),
    ];
  }

  // Shape the results: a number for 1-D input, uint8 back for value-like ops.
  #finish(s, arrs, { valueLike = false, given = false } = {}) {
    if (given) return arrs;
    let out = arrs;
    if (valueLike && s.src === "uint8") out = out.map((a) => new s.srcCtor(a));
    if (s.scalar) out = out.map((a) => a[0]);
    return out;
  }

  #simple(op, x, opts, dtypeFn, { omit = false, valueLike = false } = {}) {
    const s = this.#prep(x, opts);
    const dt = opts.out ? dtypeOf(opts.out) : dtypeFn(s);
    if (opts.out && !DTYPES[dt]) throw new TypeError(`unsupported out ${opts.out.constructor.name}`);
    const { values } = this.#run(s, op, [{ dtype: dt, out: opts.out }], { ...opts, omit });
    return this.#finish(s, values, { valueLike, given: !!opts.out })[0];
  }

  // ---------------------------------------------------------------- medians

  /**
   * `np.median` along one axis: float32 stays float32, uint16 gives float64. A slice with
   * NaN gives NaN. `nodata` (uint16) marks missing values, which are skipped.
   */
  median(x, opts = {}) {
    return this.#simple([1], x, opts, (s) => floatResult(s.dtype), { omit: opts.nodata != null });
  }

  /** `np.nanmedian` along one axis: NaN is ignored; an all-NaN slice gives NaN. */
  nanmedian(x, opts = {}) {
    return this.#simple([1], x, opts, (s) => floatResult(s.dtype), { omit: true });
  }

  // ---------------------------------------------------------------- quantiles

  #quantile(x, q, opts, { omit, percent }) {
    const method = opts.method ?? "linear";
    const mi = METHODS.indexOf(method);
    if (mi < 0) throw new RangeError(`${JSON.stringify(method)} is not a valid method. Use one of: ${METHODS.join(", ")}`);
    // How q is passed picks NumPy's arithmetic: a number behaves like a Python float (float32
    // data interpolates in float32), an Array or Float64Array like a list (float64), a
    // Float32Array like a float32 array.
    const scalarQ = typeof q === "number";
    let qs = scalarQ ? [q] : Array.from(q, Number);
    const f32 = q instanceof Float32Array;
    if (percent) qs = qs.map((v) => (f32 ? Math.fround(Math.fround(v) / 100) : v / 100));
    if (qs.length === 0) throw new RangeError("no quantiles requested");
    if (!qs.every((v) => v >= 0 && v <= 1)) {
      throw new RangeError(percent ? "Percentiles must be in the range [0, 100]" : "Quantiles must be in the range [0, 1]");
    }
    const form = scalarQ ? 0 : f32 ? 3 : 2; // weak, f64, f32
    const s = this.#prep(x, opts);
    let dt;
    if (DISCRETE.has(method)) dt = s.dtype;
    else if (s.dtype === "float64") dt = "float64";
    else if (form === 0) dt = s.dtype === "float32" ? "float32" : "float64";
    else if (form === 3) dt = "float32";
    else dt = "float64";
    const outs = scalarQ ? [opts.out] : opts.out ?? qs.map(() => undefined);
    if (!Array.isArray(outs) || outs.length !== qs.length) throw new RangeError(`out needs ${qs.length} arrays`);
    const given = outs[0] !== undefined;
    const outDt = given ? dtypeOf(outs[0]) : dt;
    const op = [2, qs.length, ...qs, mi, form, given ? DTYPES[outDt].code : -1, omit ? 1 : 0];
    const { values } = this.#run(s, op, outs.map((o) => ({ dtype: given ? dtypeOf(o) : dt, out: o })), {
      ...opts,
      omit: omit || opts.nodata != null,
    });
    const res = this.#finish(s, values, { valueLike: DISCRETE.has(method), given });
    return scalarQ ? res[0] : res;
  }

  /**
   * `np.quantile` along one axis, all 13 methods, bit-identical including dtype. `q` is a
   * number (one result) or a list (an array of results). Option `method` (default "linear").
   */
  quantile(x, q, opts = {}) {
    return this.#quantile(x, q, opts, { omit: false, percent: false });
  }

  /** `np.nanquantile` along one axis. */
  nanquantile(x, q, opts = {}) {
    return this.#quantile(x, q, opts, { omit: true, percent: false });
  }

  /** `np.percentile` along one axis (`q` in percent). */
  percentile(x, q, opts = {}) {
    return this.#quantile(x, q, opts, { omit: false, percent: true });
  }

  /** `np.nanpercentile` along one axis. */
  nanpercentile(x, q, opts = {}) {
    return this.#quantile(x, q, opts, { omit: true, percent: true });
  }

  // ---------------------------------------------------------------- min / max / args

  /** `np.min` along one axis (NaN propagates). */
  min(x, opts = {}) {
    return this.#simple([3], x, opts, (s) => s.dtype, { omit: opts.nodata != null, valueLike: true });
  }

  /** `np.max` along one axis (NaN propagates). */
  max(x, opts = {}) {
    return this.#simple([4], x, opts, (s) => s.dtype, { omit: opts.nodata != null, valueLike: true });
  }

  /** `np.nanmin` along one axis. */
  nanmin(x, opts = {}) {
    return this.#simple([3], x, opts, (s) => s.dtype, { omit: true, valueLike: true });
  }

  /** `np.nanmax` along one axis. */
  nanmax(x, opts = {}) {
    return this.#simple([4], x, opts, (s) => s.dtype, { omit: true, valueLike: true });
  }

  /** `np.ptp` (max - min) along one axis, in the input dtype. */
  ptp(x, opts = {}) {
    return this.#simple([5], x, opts, (s) => s.dtype, { omit: opts.nodata != null, valueLike: true });
  }

  #arg(op, x, opts, omit) {
    return this.#simple(op, x, opts, () => "int32", { omit: omit || opts.nodata != null });
  }

  /** `np.argmin` along one axis (first occurrence; the first NaN wins). Int32Array. */
  argmin(x, opts = {}) {
    return this.#arg([6], x, opts, false);
  }

  /** `np.argmax` along one axis. */
  argmax(x, opts = {}) {
    return this.#arg([7], x, opts, false);
  }

  /** `np.nanargmin` along one axis; all-NaN slices give -1. */
  nanargmin(x, opts = {}) {
    return this.#arg([6], x, opts, true);
  }

  /** `np.nanargmax` along one axis; all-NaN slices give -1. */
  nanargmax(x, opts = {}) {
    return this.#arg([7], x, opts, true);
  }

  /** Which frame supplied the (lower) median: `np.argsort(kind="stable")` at rank (n - 1) / 2. */
  argmedian(x, opts = {}) {
    return this.#arg([8], x, opts, false);
  }

  /** {@link argmedian} over the non-NaN values; all-NaN slices give -1. */
  nanargmedian(x, opts = {}) {
    return this.#arg([8], x, opts, true);
  }

  /** `np.argsort(kind="stable")` at each rank: which frame holds that order statistic. */
  argselect(x, ranks, opts = {}) {
    const one = typeof ranks === "number";
    const res = (one ? [ranks] : Array.from(ranks)).map((r) => this.#arg([9, r], x, opts, false));
    return one ? res[0] : res;
  }

  /**
   * `np.sort(x, axis)[ranks]` in one pass, keeping only the comparators those ranks need.
   * Negative ranks count from the end. NaN sorts last.
   */
  selectRanks(x, ranks, opts = {}) {
    const one = typeof ranks === "number";
    const s = this.#prep(x, opts);
    const rs = (one ? [ranks] : Array.from(ranks, Number)).map((r) => (r < 0 ? r + s.n : r));
    if (!rs.every(Number.isInteger)) throw new TypeError("ranks must be integers");
    const outs = one ? [opts.out] : opts.out ?? rs.map(() => undefined);
    const given = outs[0] !== undefined;
    const { values } = this.#run(s, [0, rs.length, ...rs], outs.map((o) => ({ dtype: given ? dtypeOf(o) : s.dtype, out: o })), opts);
    const res = this.#finish(s, values, { valueLike: true, given });
    return one ? res[0] : res;
  }

  // ---------------------------------------------------------------- scipy-style

  /** `scipy.stats.trim_mean`: cut `floor(proportion * n)` values from each end, average the rest. */
  trimmedMean(x, proportion, opts = {}) {
    return this.#simple([10, Number(proportion)], x, opts, (s) => floatResult(s.dtype));
  }

  #nanPolicy(x, policy = "propagate") {
    if (!["propagate", "omit", "raise"].includes(policy)) throw new RangeError("nanPolicy must be 'propagate', 'omit' or 'raise'");
    if (policy === "raise") {
      const arrs = Array.isArray(x) ? x : [x?.data ?? x];
      for (const a of arrs) {
        const v = a instanceof WasmArray ? a.array : a;
        if ((v instanceof Float32Array || v instanceof Float64Array) && v.some(Number.isNaN)) {
          throw new RangeError("The input contains nan values");
        }
      }
    }
    return policy === "omit";
  }

  /**
   * `scipy.stats.iqr`. Options: `rng` (default [25, 75]), `scale` (a number or "normal"),
   * `nanPolicy` ("propagate", "omit" or "raise").
   */
  iqr(x, opts = {}) {
    const rng = opts.rng ?? [25, 75];
    if (rng.length !== 2) throw new TypeError("`rng` must be a two element sequence.");
    let scale = opts.scale ?? 1;
    if (typeof scale === "string") {
      if (scale.toLowerCase() !== "normal") throw new RangeError(`${scale} not a valid scale for \`iqr\``);
      scale = 1.3489795003921636;
    }
    const omit = this.#nanPolicy(x, opts.nanPolicy);
    return this.#simple([11, Number(rng[0]), Number(rng[1]), Number(scale)], x, opts, (s) => floatResult(s.dtype), { omit });
  }

  /**
   * `scipy.stats.median_abs_deviation` with the median as center. Options: `scale` (a number
   * or "normal"), `nanPolicy`.
   */
  mad(x, opts = {}) {
    let scale = opts.scale ?? 1;
    if (typeof scale === "string") {
      if (scale.toLowerCase() !== "normal") throw new RangeError(`${scale} is not a valid scale value.`);
      scale = 0.6744897501960817;
    }
    const omit = this.#nanPolicy(x, opts.nanPolicy);
    return this.#simple([12, Number(scale)], x, opts, (s) => floatResult(s.dtype), { omit });
  }

  // ---------------------------------------------------------------- rejection

  /**
   * Reject outliers along the axis and combine the survivors.
   *
   * `method`: "sigma" (astropy `sigma_clip`), "mad" (astropy `mad_std` spread),
   * "median_sigma", "winsorized" or "percentile" (Siril's; `sigma` is then the fractional
   * threshold), or "none". `sigma` (default 3) may be `[low, high]`; `sigmaLower` /
   * `sigmaUpper` override. `maxiters` (default 5; `null` = until nothing changes),
   * `cenfunc` ("median" | "mean"), `stdfunc` ("std" | "mad_std"), `combine` ("mean" |
   * "median" | "sum"), `preset` ("astropy" | "siril" | "pixinsight"), `precision` ("f64" |
   * "f32"), `scale` / `offset` (per-frame normalization `x * scale[k] + offset[k]`), `weights`.
   *
   * Returns the combined array. With `returnCounts`, `rejected`, `low`, `high` or `mask`,
   * returns `{ value, low, high, rejected, mask }` instead (Uint16Array counts; `mask` is the
   * rejection map: 1 where a value was rejected).
   */
  clipCombine(x, opts = {}) {
    const method = opts.method ?? "sigma";
    if (!(method in CLIP_METHODS)) throw new RangeError(`unknown rejection method ${JSON.stringify(method)}`);
    let [lo, hi] = Array.isArray(opts.sigma) ? opts.sigma.map(Number) : [opts.sigma ?? 3, opts.sigma ?? 3].map(Number);
    if (opts.sigmaLower != null) lo = Number(opts.sigmaLower);
    if (opts.sigmaUpper != null) hi = Number(opts.sigmaUpper);
    const preset = opts.preset ?? null;
    if (![null, "astropy", "siril", "pixinsight"].includes(preset)) throw new RangeError(`unknown preset ${JSON.stringify(preset)}`);
    let maxiters = opts.maxiters === undefined ? 5 : opts.maxiters;
    if (maxiters != null && !Number.isFinite(maxiters)) maxiters = null;
    // both tools iterate until nothing changes
    if (preset === "siril" || (preset === "pixinsight" && maxiters === 5)) maxiters = null;
    const cen = { median: 0, mean: 1 }[opts.cenfunc ?? "median"];
    if (cen === undefined) throw new RangeError(`${opts.cenfunc} is an invalid cenfunc.`);
    const std = opts.stdfunc == null ? -1 : { std: 0, mad_std: 1 }[opts.stdfunc];
    if (std === undefined) throw new RangeError(`${opts.stdfunc} is an invalid stdfunc.`);
    const comb = { mean: 0, median: 1, sum: 2 }[opts.combine ?? "mean"];
    if (comb === undefined) throw new RangeError(`unknown combine ${JSON.stringify(opts.combine)}`);
    const prec = { f64: 0, float64: 0, f32: 1, float32: 1 }[opts.precision ?? "f64"];
    if (prec === undefined) throw new RangeError(`unknown precision ${JSON.stringify(opts.precision)}`);
    const op = [
      13, CLIP_METHODS[method], lo, hi, maxiters ?? NaN, { siril: 1, pixinsight: 2 }[preset] ?? 0, cen, std, comb, prec,
      opts.maskInfinite === false ? 0 : 1, opts.cutoff ?? NaN,
    ];
    const s = this.#prep(x, opts);
    const dt = s.dtype === "float64" ? "float64" : "float32";
    const extras = opts.returnCounts || opts.rejected || opts.low || opts.high || opts.mask;
    const counts = extras && (opts.returnCounts || opts.rejected || opts.low || opts.high);
    const arr = (v) => (v === true || v == null ? undefined : v);
    const outs = [{ dtype: opts.out ? dtypeOf(opts.out) : dt, out: opts.out }];
    if (counts) {
      outs.push({ dtype: opts.low && opts.low !== true ? dtypeOf(opts.low) : "uint16", out: arr(opts.low) });
      outs.push({ dtype: opts.high && opts.high !== true ? dtypeOf(opts.high) : "uint16", out: arr(opts.high) });
    }
    // The kernel always writes three outputs; unrequested counts are not written.
    while (outs.length < 3) outs.push({ dtype: "uint16", none: true });
    const run = this.#run(s, op, outs, { ...opts, clip: true }, opts.mask || null);
    const value = this.#finish(s, [run.values[0]], { given: !!opts.out })[0];
    if (!extras) return value;
    const res = { value };
    if (counts) {
      res.low = run.values[1];
      res.high = run.values[2];
      const rej = opts.rejected && opts.rejected !== true ? opts.rejected : new Uint16Array(s.npix);
      for (let i = 0; i < s.npix; i++) rej[i] = res.low[i] + res.high[i];
      res.rejected = rej;
      if (s.scalar) for (const k of ["low", "high", "rejected"]) res[k] = res[k][0];
    }
    if (opts.mask) res.mask = run.mask;
    return res;
  }

  /**
   * Sigma-clipped mean with `astropy.stats.sigma_clip` decisions (median center, population
   * std, bounds in float64). Same options as {@link clipCombine}.
   */
  sigmaClipMean(x, opts = {}) {
    return this.clipCombine(x, { stdfunc: "std", ...opts, method: "sigma" });
  }

  // ---------------------------------------------------------------- introspection

  /** How a call would run: ISA, lanes, comparators and memory path. `op`: "median" (default), "quantile", "sigma_clip". */
  describe(x, opts = {}) {
    const s = this.#prep(x, opts);
    const op = opts.op === "quantile" ? [2, 1, opts.q ?? 0.5, 6, 2, -1, 0] : opts.op === "sigma_clip" ? [13, 0, 3, 3, 5, 0, 0, -1, 0, 0, 1, NaN] : [1];
    const esz = DTYPES[s.dtype].size;
    const req = [...op, DTYPES[s.dtype].code, s.n, 1, s.npix, esz];
    // addresses only matter for the cache-set heuristics; describe a streamed call
    const fstride = Math.ceil((Math.max(64, Math.min(s.npix, this.chunkBytes / (s.n * esz))) * esz) / 64) * 64 + 64;
    for (let k = 0; k < s.n; k++) req.push(65536 + k * fstride);
    req.push(0, 0, ...this.#optsTail(s, opts));
    const p = this._alloc(req.length * 8);
    try {
      new Float64Array(this.memory.buffer, p, req.length).set(req);
      this.#check(this.#x.rw_describe(p, req.length));
      return new TextDecoder().decode(new Uint8Array(this.memory.buffer, this.#x.rw_message_ptr(), this.#x.rw_message_len()));
    } finally {
      this._free(p, req.length * 8);
    }
  }
}

export default init;
