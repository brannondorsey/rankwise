// The WebAssembly engine: loads the simd128 or portable module, and runs requests on its
// linear memory, streaming inputs in cache-sized chunks when they live outside it.

import { DTYPES, Rankwise, ZeroCopyArray, check, cStrides } from "./core.js";

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

/** Where the default modules live. */
export function wasmUrl(simd) {
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

/** Whether `options.simd` ("auto" by default) means the simd128 build here. */
export function chooseSimd(options) {
  const simd = options.simd === undefined || options.simd === "auto" ? simdSupported() : Boolean(options.simd);
  if (simd && !simdSupported()) throw new Error("rankwise: this engine does not support WebAssembly SIMD; use { simd: false }");
  return simd;
}

const engines = new Map();

/** Load (once per build) and instantiate a WebAssembly engine. */
export function wasmEngine(options = {}) {
  let simd;
  try {
    simd = chooseSimd(options);
  } catch (e) {
    return Promise.reject(e);
  }
  if (options.wasm !== undefined) return compileFrom(options.wasm).then(instantiate);
  const key = simd ? "simd128" : "portable";
  if (!engines.has(key)) {
    const p = compileFrom(wasmUrl(simd)).then(instantiate);
    p.catch(() => engines.delete(key));
    engines.set(key, p);
  }
  return engines.get(key);
}

async function instantiate(module) {
  return new WasmEngine(await WebAssembly.instantiate(module, {}));
}

/** A WebAssembly engine from module bytes or a compiled module, synchronously. */
export function wasmEngineSync(wasm) {
  const m = wasm instanceof WebAssembly.Module ? wasm : new WebAssembly.Module(wasm);
  return new WasmEngine(new WebAssembly.Instance(m, {}));
}

const cache = new Map();

/**
 * Load rankwise's WebAssembly build. Resolves to a {@link Rankwise} (cached per build).
 *
 * Options: `simd` (true / false / "auto", default "auto") picks the build; `wasm` overrides
 * where it comes from (a URL or path, a Response, bytes or a compiled WebAssembly.Module).
 */
export function init(options = {}) {
  if (options.backend === "native") {
    return Promise.reject(new Error("rankwise: the native backend is for Node, Bun and Deno; this entry point loads WebAssembly"));
  }
  if (options.wasm !== undefined) return wasmEngine(options).then((e) => new Rankwise(e));
  let key;
  try {
    key = chooseSimd(options) ? "simd128" : "portable";
  } catch (e) {
    return Promise.reject(e);
  }
  if (!cache.has(key)) {
    const p = wasmEngine(options).then((e) => new Rankwise(e));
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return cache.get(key);
}

/**
 * Synchronous form of {@link init} for code that already holds the module bytes or a compiled
 * `WebAssembly.Module` (workers, serverless bundles).
 */
export function initSync(wasm) {
  return new Rankwise(wasmEngineSync(wasm));
}

/** A TypedArray inside the module's memory: calls read and write it without copying. */
export class WasmArray extends ZeroCopyArray {
  #engine;
  #ctor;
  constructor(engine, ctor, ptr, length, bytes) {
    super();
    this.#engine = engine;
    this.#ctor = ctor;
    this.ptr = ptr;
    this.length = length;
    this.bytes = bytes;
  }
  get array() {
    if (!this.ptr) throw new Error("rankwise: this array was freed");
    return new this.#ctor(this.#engine.memory.buffer, this.ptr, this.length);
  }
  free() {
    if (this.ptr) this.#engine.free(this.ptr, this.bytes);
    this.ptr = 0;
  }
}

export class WasmEngine {
  #x;
  #scratch = 0;
  #scratchBytes = 0;
  backend = "wasm";
  chunkBytes = 1 << 20;

  constructor(instance) {
    this.#x = instance.exports;
    this.build = this.#x.rw_available_isas() & (1 << 4) ? "simd128" : "portable";
    this.maxFrames = this.#x.rw_max_frames();
    const v = this.#x.rw_version();
    this.version = `${Math.floor(v / 10000)}.${Math.floor(v / 100) % 100}.${v % 100}`;
  }

  get memory() {
    return this.#x.memory;
  }

  #message = () =>
    new TextDecoder().decode(new Uint8Array(this.memory.buffer, this.#x.rw_message_ptr(), this.#x.rw_message_len()));

  availableIsas() {
    return this.#x.rw_available_isas();
  }
  currentIsa() {
    return this.#x.rw_current_isa();
  }
  setIsa(code) {
    check(this.#x.rw_set_isa(code), this.#message);
  }

  networkSize(n, ranks) {
    if (ranks.length === 0) return this.#x.rw_network_size(n, 0, 0);
    const r = Uint32Array.from(ranks);
    const p = this.malloc(r.byteLength);
    try {
      new Uint32Array(this.memory.buffer, p, r.length).set(r);
      return this.#x.rw_network_size(n, p, r.length);
    } finally {
      this.free(p, r.byteLength);
    }
  }

  alloc(d, length) {
    const bytes = Math.max(1, length * d.size);
    const p = this.malloc(bytes);
    new Uint8Array(this.memory.buffer, p, bytes).fill(0);
    return new WasmArray(this, d.ctor, p, length, bytes);
  }

  malloc(bytes) {
    const p = this.#x.rw_alloc(bytes);
    if (!p) throw new RangeError(`rankwise: cannot allocate ${bytes} bytes of WebAssembly memory (4 GiB at most)`);
    return p;
  }

  free(p, bytes) {
    this.#x.rw_free(p, bytes);
  }

  releaseScratch() {
    if (this.#scratch) this.free(this.#scratch, this.#scratchBytes);
    this.#scratch = 0;
    this.#scratchBytes = 0;
  }

  #reserve(bytes) {
    bytes = Math.ceil(bytes / 64) * 64 + 64;
    if (bytes > this.#scratchBytes) {
      this.releaseScratch();
      this.#scratch = this.malloc(bytes);
      this.#scratchBytes = bytes;
    }
    return this.#scratch;
  }

  #resident(a) {
    return a.buffer === this.memory.buffer;
  }

  /** Run one request. Results are written into `job.outs[j].result` and `job.mask`. */
  run({ s, head, tail, outs, mask }) {
    const nout = outs.length;
    const esz = DTYPES[s.dtype].size;
    const In = DTYPES[s.dtype].ctor;
    const results = outs.map((o) => o.result);
    const odt = outs.map((o) => o.kdt);
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
        check(this.#x.rw_reduce(raddr, req.length), this.#message);
        if (this.memory.buffer !== heap) mkViews(); // the kernel grew memory
        for (let j = 0; j < nout; j++) {
          if (results[j]) results[j].set(len === chunk ? oviews[j] : oviews[j].subarray(0, len), p0);
        }
        if (mask) for (let k = 0; k < s.n; k++) mask.frames[k].set(mviews[k].subarray(0, len), p0);
        p0 += chunk;
      } while (p0 < s.npix);
      return;
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
    check(this.#x.rw_reduce(raddr, req.length), this.#message);
    const heap = this.memory.buffer;
    for (let j = 0; j < nout; j++) {
      if (results[j] && !outResident[j]) results[j].set(new odt[j].ctor(heap, oaddr[j], s.npix));
    }
    if (mask) {
      const m = new Uint8Array(heap, maddr, s.n * s.npix);
      if (mask.flat) mask.flat.set(m);
      else for (let k = 0; k < s.n; k++) mask.frames[k].set(m.subarray(k * s.npix, (k + 1) * s.npix));
    }
  }

  /** Describe how a request would run (addresses only steer the cache-set heuristics). */
  describe({ s, head, tail }) {
    const esz = DTYPES[s.dtype].size;
    const fstride = Math.ceil((Math.max(64, Math.min(s.npix, this.chunkBytes / (s.n * esz))) * esz) / 64) * 64 + 64;
    const req = [...head, 1, s.npix, esz];
    for (let k = 0; k < s.n; k++) req.push(65536 + k * fstride);
    req.push(0, 0, ...tail);
    const p = this.malloc(req.length * 8);
    try {
      new Float64Array(this.memory.buffer, p, req.length).set(req);
      check(this.#x.rw_describe(p, req.length), this.#message);
      return this.#message();
    } finally {
      this.free(p, req.length * 8);
    }
  }
}
