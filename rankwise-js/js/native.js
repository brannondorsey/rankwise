// The native engine: the Node-API addon reads and writes the caller's TypedArrays in place.
// Each array the request touches becomes a segment, and addresses are `(i + 1) * 2^40 +
// byte offset` into segment i; the addon bounds-checks every region before running.

import { DTYPES, ZeroCopyArray, check, cStrides } from "./core.js";

const SEGMENT = 2 ** 40;
const MAX_SEGMENTS = 8190; // (i + 1) * 2^40 stays an exact integer in a double

/** An ordinary TypedArray: the native addon never copies its inputs. */
class HostArray extends ZeroCopyArray {
  #a;
  constructor(a) {
    super();
    this.#a = a;
    this.length = a.length;
  }
  get array() {
    if (!this.#a) throw new Error("rankwise: this array was freed");
    return this.#a;
  }
  free() {
    this.#a = null;
  }
}

const bytes = (v) => (v instanceof Uint8Array ? v : new Uint8Array(v.buffer, v.byteOffset, v.byteLength));

export class NativeEngine {
  #a;
  backend = "native";

  /** `addon`: the loaded `.node` module; `build`: its platform, e.g. "linux-x64-gnu". */
  constructor(addon, build) {
    this.#a = addon;
    this.build = build;
    this.maxFrames = addon.maxFrames();
    this.version = addon.version();
  }

  #message = () => this.#a.message();

  availableIsas() {
    return this.#a.availableIsas();
  }
  currentIsa() {
    return this.#a.currentIsa();
  }
  setIsa(code) {
    check(this.#a.setIsa(code), this.#message);
  }
  networkSize(n, ranks) {
    return this.#a.networkSize(n, ranks.length ? ranks : null);
  }
  alloc(d, length) {
    return new HostArray(new d.ctor(length));
  }

  #request({ s, head, tail, outs, mask }) {
    const segs = [];
    const index = new Map();
    const at = (view, offset = 0) => {
      let i = index.get(view);
      if (i === undefined) {
        i = segs.length;
        if (i >= MAX_SEGMENTS) throw new RangeError(`rankwise: one call can touch at most ${MAX_SEGMENTS} arrays`);
        segs.push(bytes(view));
        index.set(view, i);
      }
      return (i + 1) * SEGMENT + offset;
    };
    const esz = DTYPES[s.dtype].size;
    // uint8 input (ImageData included) is widened once; the kernels read uint16
    const widen = s.src !== s.dtype;
    let shape, strides, frameAddrs;
    if (s.kind === "frames") {
      shape = [s.npix];
      strides = [esz];
      frameAddrs = s.frames.map((f) => at(widen ? Uint16Array.from(f) : f));
    } else {
      const st = cStrides(s.shape, esz);
      shape = s.outShape;
      strides = st.filter((_, d) => d !== s.axis);
      const data = widen ? Uint16Array.from(s.data) : s.data;
      frameAddrs = Array.from({ length: s.n }, (_, k) => at(data, k * st[s.axis]));
    }
    const req = [...head, shape.length, ...shape, ...strides, ...frameAddrs, outs.length];
    const narrow = [];
    for (const { result, kdt } of outs ?? []) {
      let addr = 0;
      if (result) {
        let target = result;
        if (!(target instanceof kdt.ctor)) {
          // the kernel writes uint16 where the caller asked for uint8 values
          target = new kdt.ctor(result.length);
          narrow.push([result, target]);
        }
        addr = at(target);
      }
      req.push(addr, kdt.code, ...cStrides(shape, kdt.size));
    }
    if (mask) {
      req.push(1);
      if (s.kind === "frames") {
        for (const f of mask.frames) req.push(at(f));
        req.push(DTYPES.bool.code, 1);
      } else {
        const ms = cStrides(s.shape, 1);
        for (let k = 0; k < s.n; k++) req.push(at(mask.flat, k * ms[s.axis]));
        req.push(DTYPES.bool.code, ...ms.filter((_, d) => d !== s.axis));
      }
    } else req.push(0);
    req.push(...tail);
    return { req: Float64Array.from(req), segs, narrow };
  }

  /** Run one request, writing into `job.outs[j].result` and `job.mask` in place. */
  run(job) {
    const { req, segs, narrow } = this.#request(job);
    check(this.#a.reduce(req, segs), this.#message);
    for (const [dst, src] of narrow) dst.set(src);
  }

  describe(job) {
    const { req, segs } = this.#request({ ...job, outs: [], mask: null });
    check(this.#a.describe(req, segs), this.#message);
    return this.#a.message();
  }
}
