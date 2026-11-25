// API behaviour that the NumPy fixtures do not cover: loading and backend selection, errors,
// zero-copy arrays, memory growth, ImageData input, ISA control, threads. Runs under node,
// bun and deno; the shared tests run on every backend that loads here.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as nodeEntry from "../js/node.js";
import { addonFile, platformKey } from "../js/platforms.js";
import * as wasmEntry from "../js/rankwise.js";

const { RankwiseError, simdSupported } = wasmEntry;
const tests = [];
const test = (name, f) => tests.push([name, f]);

const frames = (n, len, f) => Array.from({ length: n }, (_, k) => Float32Array.from({ length: len }, (_, i) => f(k, i)));
const key = platformKey();
const addonPath = fileURLToPath(new URL(`../npm/${key}/${addonFile(key)}`, import.meta.url));
const hasNative = existsSync(addonPath);

// One process per scenario: the backend is chosen once per process.
function select(options = {}, env = {}) {
  const args = typeof Deno !== "undefined" ? ["run", "-A"] : [];
  const child = fileURLToPath(new URL("./select.child.mjs", import.meta.url));
  const r = spawnSync(process.execPath, [...args, child, JSON.stringify(options)], {
    env: { ...process.env, RANKWISE_BACKEND: "", RANKWISE_NATIVE_PATH: "", ...env },
    encoding: "utf8",
  });
  if (r.status !== 0) return { error: (r.stderr || r.stdout).split("\n").find((l) => /Error: rankwise/.test(l)) ?? r.stderr };
  return JSON.parse(r.stdout);
}

test("the WebAssembly entry picks the SIMD build and caches it", async () => {
  const a = await wasmEntry.init();
  const b = await wasmEntry.init();
  assert.equal(a, b);
  assert.equal(a.backend, "wasm");
  assert.equal(a.build, simdSupported() ? "simd128" : "portable");
  assert.deepEqual(a.availableIsas(), simdSupported() ? ["simd128", "scalar"] : ["scalar"]);
  assert.equal(a.isa, a.availableIsas()[0]);
  assert.match(a.version, /^\d+\.\d+\.\d+$/);
  assert.equal(a.maxFrames, 512);
  await assert.rejects(wasmEntry.init({ backend: "native" }), /Node, Bun and Deno/);
});

test("initSync and init({ wasm }) take bytes", async () => {
  const bytes = await readFile(new URL("../dist/rankwise-portable.wasm", import.meta.url));
  for (const entry of [wasmEntry, nodeEntry]) {
    const s = entry.initSync(bytes);
    assert.equal(s.build, "portable");
    const m = await entry.init({ wasm: bytes });
    assert.equal(m.build, "portable");
    assert.deepEqual(s.median([Float32Array.of(1, 5), Float32Array.of(2, 6), Float32Array.of(3, 4)]), Float32Array.of(2, 5));
  }
});

test("Node entry: native when the addon loads, WebAssembly (with the reason) when not", async () => {
  const auto = select();
  assert.equal(auto.backend, hasNative ? "native" : "wasm");
  if (hasNative) {
    assert.equal(auto.build, key);
    assert.equal(auto.fallbackReason, null);
    assert.ok(auto.isas.includes(auto.isa) && !auto.isas.includes("simd128"));
  } else {
    assert.match(auto.fallbackReason, /no native addon/);
  }
  assert.deepEqual(auto.median, [2, 5, 8]);
  // forced WebAssembly, by option and by environment
  assert.equal(select({ backend: "wasm" }).backend, "wasm");
  assert.equal(select({}, { RANKWISE_BACKEND: "wasm" }).backend, "wasm");
  assert.equal(select({ backend: "wasm" }).fallbackReason, null);
  // an addon that is not there: auto falls back and says why; "native" refuses
  const gone = select({}, { RANKWISE_NATIVE_PATH: "/nonexistent/rankwise.node" });
  assert.equal(gone.backend, "wasm");
  assert.match(gone.fallbackReason, /nonexistent/);
  assert.match(select({ backend: "native" }, { RANKWISE_NATIVE_PATH: "/nonexistent/rankwise.node" }).error, /no native addon/);
  assert.match(select({ backend: "fast" }).error, /backend must be/);
  // the synchronous form makes the same choice
  assert.equal(select({ sync: true }).backend, auto.backend);
  if (hasNative) {
    // an explicit addon path, and the option beating the environment
    assert.equal(select({}, { RANKWISE_NATIVE_PATH: addonPath }).backend, "native");
    assert.equal(select({ backend: "native" }, { RANKWISE_BACKEND: "wasm" }).backend, "native");
  }
});

test("Node entry: an explicit native backend ignores `wasm` instead of loading WebAssembly", async () => {
  const wasm = fileURLToPath(new URL("../dist/rankwise-portable.wasm", import.meta.url));
  const missing = { RANKWISE_NATIVE_PATH: "/nonexistent/rankwise.node" };
  // "native" is the addon or an error, by option or by environment, async or sync
  assert.match(select({ backend: "native", wasm }, missing).error, /no native addon/);
  assert.match(select({ wasm }, { ...missing, RANKWISE_BACKEND: "native" }).error, /no native addon/);
  assert.match(select({ backend: "native", wasm, sync: true }, missing).error, /no native addon/);
  if (hasNative) {
    assert.equal(select({ backend: "native", wasm }).backend, "native");
    assert.equal(select({ wasm }, { RANKWISE_BACKEND: "native" }).backend, "native");
  }
  // auto and wasm still take the module they are given
  for (const backend of ["auto", "wasm"]) {
    const r = select({ backend, wasm });
    assert.equal(r.backend, "wasm");
    assert.equal(r.build, "portable");
    assert.equal(r.fallbackReason, null);
  }
});

if (hasNative) {
  test("the native addon refuses requests that reach outside their arrays", async () => {
    const addon = createRequire(import.meta.url)(addonPath);
    const S = 2 ** 40;
    const f = [Float32Array.of(1, 5, 9), Float32Array.of(2, 6, 7), Float32Array.of(3, 4, 8)];
    const out = new Float32Array(3);
    const segs = [...f, out].map((v) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
    // median of 3 frames of 3 float32 into `out`
    const req = [1, 0, 3, 1, 3, 4, 1 * S, 2 * S, 3 * S, 1, 4 * S, 0, 4, 0, 0, NaN, -1, 0, 1, -1, -1, -1];
    assert.equal(addon.reduce(Float64Array.from(req), segs), 0);
    assert.deepEqual(out, Float32Array.of(2, 5, 8));
    const bad = (i, v) => {
      const r = req.slice();
      r[i] = v;
      out.fill(-1);
      const st = addon.reduce(Float64Array.from(r), segs);
      assert.deepEqual(out, Float32Array.of(-1, -1, -1)); // nothing written
      return [st, addon.message()];
    };
    assert.deepEqual(bad(4, 4), [6, "malformed request: bytes 0..16 are outside buffer 0 (12 bytes)"]); // 4 pixels
    assert.match(bad(5, 8)[1], /outside buffer 0/); // stride 8
    assert.match(bad(10, 4 * S + 4)[1], /outside buffer 3/); // output offset
    assert.match(bad(8, 9 * S)[1], /names no buffer/);
    assert.match(bad(10, 5)[1], /names no buffer/); // a raw pointer is not an address here
  });
}

const backends = [["wasm", () => nodeEntry.init({ backend: "wasm" })]];
if (hasNative) backends.push(["native", () => nodeEntry.init({ backend: "native" })]);
const shared = (name, f) => {
  for (const [b, load] of backends) test(`[${b}] ${name}`, async () => f(await load()));
};

shared("errors carry a kind and leave the instance usable", async (rw) => {
  const x = frames(5, 10, (k, i) => k + i);
  assert.throws(() => rw.quantile(x, 1.5), RangeError);
  assert.throws(() => rw.median(new Int16Array(10)), TypeError);
  assert.throws(() => rw.median([new Float32Array(3), new Float32Array(4)]), RangeError);
  assert.throws(() => rw.median(new Float32Array(12), { shape: [3, 5] }), RangeError);
  assert.throws(() => rw.median(new Float32Array(12), { shape: [3, 4], axis: 2 }), RangeError);
  assert.throws(() => rw.median(x, { nodata: 0 }), TypeError);
  assert.throws(
    () => rw.selectRanks(x, [7]),
    (e) => e instanceof RankwiseError && e.kind === "param" && /rank 7 out of range/.test(e.message),
  );
  assert.throws(
    () => rw.median(frames(513, 4, () => 0)),
    (e) => e instanceof RankwiseError && e.kind === "depth",
  );
  assert.deepEqual(rw.min(x), Float32Array.from({ length: 10 }, (_, i) => i));
});

shared("shape and axis: every axis of a 3-D array", async (rw) => {
  const shape = [4, 3, 5];
  const data = Float64Array.from({ length: 60 }, (_, i) => (i * 37) % 11);
  const at = (a, b, c) => data[a * 15 + b * 5 + c];
  for (const axis of [0, 1, 2, -1]) {
    const ax = (axis + 3) % 3;
    const rest = shape.filter((_, d) => d !== ax);
    const got = rw.max(data, { shape, axis });
    assert.equal(got.length, rest[0] * rest[1]);
    for (let i = 0; i < rest[0]; i++) {
      for (let j = 0; j < rest[1]; j++) {
        let m = -Infinity;
        for (let k = 0; k < shape[ax]; k++) {
          const idx = [i, j];
          idx.splice(ax, 0, k);
          m = Math.max(m, at(...idx));
        }
        assert.equal(got[i * rest[1] + j], m);
      }
    }
  }
  // ndarray-like objects
  assert.deepEqual(rw.max({ data, shape }), rw.max(data, { shape }));
});

test("zero-copy WebAssembly arrays survive memory growth", async () => {
  const rw = await wasmEntry.init();
  const n = 9;
  const npix = 1000;
  const stack = rw.alloc("float32", n * npix);
  const out = rw.alloc("float32", npix);
  stack.array.forEach((_, i, a) => (a[i] = (i * 7919) % 1009));
  const before = rw.memory.buffer.byteLength;
  const big = rw.alloc("float64", 8 << 20); // 64 MiB: forces growth, detaching old views
  assert.ok(rw.memory.buffer.byteLength > before);
  const r = rw.median(stack.array, { shape: [n, npix], out: out.array });
  assert.equal(r.buffer, rw.memory.buffer); // written in place
  const ref = rw.median(Float32Array.from(stack.array), { shape: [n, npix] });
  assert.deepEqual(Float32Array.from(out.array), ref);
  assert.deepEqual(rw.median(stack, { shape: [n, npix] }), ref); // the handle works too
  big.free();
  stack.free();
  out.free();
  assert.throws(() => stack.array, /freed/);
});

// Compares large arrays without assert's diff (which can take gigabytes on a mismatch).
const sameArray = (a, b, msg = "arrays differ") => {
  const i = a.length !== b?.length ? -1 : a.findIndex((v, k) => !Object.is(v, b[k]));
  if (a.length !== b?.length || i >= 0) assert.fail(`${msg}: ${a.length !== b?.length ? `length ${a.length} vs ${b?.length}` : `at ${i}: ${a[i]} vs ${b[i]}`}`);
};
// A fresh instance (small memory), so the scratch buffer a call reserves has to grow it.
const freshWasm = async () =>
  wasmEntry.init({ wasm: await readFile(new URL(`../dist/rankwise-${simdSupported() ? "simd128" : "portable"}.wasm`, import.meta.url)) });
const growing = (rw, f) => {
  const heap = rw.memory.buffer;
  const r = f();
  assert.notEqual(rw.memory.buffer, heap, "the call was meant to grow memory");
  return r;
};

test("arrays in WebAssembly memory stay correct when the call itself grows memory", async () => {
  const ref = await wasmEntry.init();
  const n = 9;
  const npix = 1 << 18;
  const val = (i) => ((i * 7919) % 1009) + (i % 97 === 0 ? 1e6 : 0);
  const fill = (h) => (h.array.forEach((_, i, a) => (a[i] = val(i) % (a instanceof Uint8Array ? 251 : Infinity))), h);
  {
    // frames read in place, out written in place; the mask's scratch grows memory
    const rw = await freshWasm();
    const stack = fill(rw.alloc("float32", n * npix));
    const want = ref.clipCombine(Float32Array.from(stack.array), { shape: [n, npix], mask: true });
    const out = rw.alloc("float32", npix);
    const r = growing(rw, () => rw.clipCombine(stack.array, { shape: [n, npix], out: out.array, mask: true }));
    sameArray(out.array, want.value);
    sameArray(r.mask, want.mask);
    sameArray(stack.array, Float32Array.from({ length: n * npix }, (_, i) => val(i))); // input untouched
  }
  {
    // a mask in memory too (written through a fresh view, and returned as one)
    const rw = await freshWasm();
    const stack = fill(rw.alloc("float32", n * npix));
    const want = ref.clipCombine(Float32Array.from(stack.array), { shape: [n, npix], mask: true });
    const m = rw.alloc("uint8", n * npix);
    const r = growing(rw, () => rw.clipCombine(stack.array, { shape: [n, npix], mask: m.array }));
    sameArray(r.value, want.value);
    sameArray(m.array, want.mask);
    sameArray(r.mask, want.mask);
  }
  {
    // low/high counts in memory: `rejected` is summed from them after the call
    const rw = await freshWasm();
    const stack = fill(rw.alloc("float32", n * npix));
    const want = ref.clipCombine(Float32Array.from(stack.array), { shape: [n, npix], mask: true, rejected: true });
    const low = rw.alloc("uint16", npix);
    const high = rw.alloc("uint16", npix);
    const r = growing(rw, () =>
      rw.clipCombine(stack.array, { shape: [n, npix], mask: true, low: low.array, high: high.array, rejected: true }),
    );
    sameArray(low.array, want.low);
    sameArray(high.array, want.high);
    sameArray(r.low, want.low);
    sameArray(r.rejected, want.rejected);
    assert.ok(want.rejected.some((v) => v > 0));
  }
  {
    // `rejected` in memory: written after the call, through a fresh view
    const rw = await freshWasm();
    const stack = fill(rw.alloc("float32", n * npix));
    const want = ref.clipCombine(Float32Array.from(stack.array), { shape: [n, npix], mask: true, rejected: true });
    const rej = rw.alloc("uint16", npix);
    const r = growing(rw, () => rw.clipCombine(stack.array, { shape: [n, npix], mask: true, rejected: rej.array }));
    sameArray(rej.array, want.rejected);
    sameArray(r.rejected, want.rejected);
  }
  {
    // a mask given as one in-memory array per frame comes back as fresh views
    const rw = await freshWasm();
    const frames = Array.from({ length: n }, (_, k) => {
      const h = rw.alloc("float32", npix);
      h.array.forEach((_, i, a) => (a[i] = val(k * npix + i)));
      return h;
    });
    const want = ref.clipCombine(frames.map((h) => Float32Array.from(h.array)), { mask: true });
    const m = frames.map(() => rw.alloc("uint8", npix));
    const r = growing(rw, () => rw.clipCombine(frames.map((h) => h.array), { mask: m.map((h) => h.array) }));
    sameArray(r.value, want.value);
    for (let k = 0; k < n; k++) {
      sameArray(m[k].array, want.mask[k]);
      sameArray(r.mask[k], want.mask[k]);
    }
  }
  {
    // argselect runs one request per rank on one prepared input; the first grows memory
    for (const list of [false, true]) {
      const rw = await freshWasm();
      const stack = fill(rw.alloc("float32", n * npix));
      const input = () => (list ? Array.from({ length: n }, (_, k) => stack.array.subarray(k * npix, (k + 1) * npix)) : stack.array);
      const opts = list ? {} : { shape: [n, npix] };
      const want = ref.argselect(Float32Array.from(stack.array), [0, 4, -1], { shape: [n, npix] });
      const r = growing(rw, () => rw.argselect(input(), [0, 4, -1], opts));
      r.forEach((a, j) => sameArray(a, want[j]));
    }
  }
  {
    // a strided layout (inner axis) read in place
    const rw = await freshWasm();
    const shape = [4, n, npix / 4];
    const data = fill(rw.alloc("float32", n * npix));
    const want = ref.clipCombine(Float32Array.from(data.array), { shape, axis: 1, mask: true });
    const r = growing(rw, () => rw.clipCombine(data.array, { shape, axis: 1, mask: true }));
    sameArray(r.value, want.value);
    sameArray(r.mask, want.mask);
  }
  {
    // a strided uint8 layout in memory is widened into scratch, which grows memory
    const rw = await freshWasm();
    const shape = [4, n, npix / 4];
    const data = fill(rw.alloc("uint8", n * npix));
    const want = ref.max(Uint8Array.from(data.array), { shape, axis: 1 });
    const r = growing(rw, () => rw.max(data.array, { shape, axis: 1 }));
    sameArray(r, want);
  }
  {
    // uint8 frames in memory are streamed (widened); out in memory is copied into
    const rw = await freshWasm();
    rw.chunkBytes = 1 << 30; // one chunk: the whole stack in scratch
    const u = fill(rw.alloc("uint8", n * npix));
    const list = () => Array.from({ length: n }, (_, k) => u.array.subarray(k * npix, (k + 1) * npix));
    const want = ref.median(list().map((f) => Uint8Array.from(f)));
    const out = rw.alloc("float64", npix);
    growing(rw, () => rw.median(list(), { out: out.array }));
    sameArray(out.array, want);
  }
  {
    // uint16 frames in place, a Uint8Array out in memory (narrowed by a copy)
    const rw = await freshWasm();
    const stack = fill(rw.alloc("uint16", n * npix));
    stack.array.forEach((v, i, a) => (a[i] = v % 256));
    const want = ref.max(Uint16Array.from(stack.array), { shape: [n, npix], out: new Uint8Array(npix) });
    const out = rw.alloc("uint8", npix);
    growing(rw, () => rw.max(stack.array, { shape: [n, npix], out: out.array }));
    sameArray(out.array, want);
  }
});

test("alloc() past 4 GiB is refused, not wrapped around to a small allocation", async () => {
  const rw = await freshWasm();
  for (const len of [2 ** 32 + 16, 2 ** 33, 2 ** 52]) assert.throws(() => rw.alloc("uint8", len), /cannot allocate .* \(4 GiB at most\)/);
  assert.throws(() => rw.alloc("float32", 2 ** 30 + 4), /4 GiB at most/);
});

test("a WebAssembly trap discards the instance, and init() then loads a new one", async () => {
  // A fresh copy of the loader (its own caches), whose module traps in rw_reduce until `armed`
  // is cleared: a Rust panic compiles to `unreachable`, which traps like this.
  let armed = true;
  const real = WebAssembly.instantiate;
  WebAssembly.instantiate = async (...a) => {
    const inst = await real(...a);
    const exports = { ...inst.exports };
    const reduce = exports.rw_reduce;
    exports.rw_reduce = (...b) => (armed ? reduce(0x7fffffff, 1e9) : reduce(...b)); // faults: out of bounds
    return { exports };
  };
  let w, rw;
  try {
    w = await import(`../js/wasm.js?trap=${Date.now()}`);
    rw = await w.init({ simd: false });
  } finally {
    WebAssembly.instantiate = real;
  }
  const x = new Float32Array([3, 1, 2]);
  const engine = await w.wasmEngine({ simd: false }); // the engine behind rw
  const h = rw.alloc("float32", 4);
  assert.throws(() => rw.median(x), (e) => /trapped/.test(e.message) && e.cause instanceof WebAssembly.RuntimeError);
  armed = false;
  assert.throws(() => rw.median(x), /trapped earlier and cannot be used again/); // even once the fault is gone
  assert.throws(() => rw.alloc("float32", 4), /trapped earlier/);
  h.free(); // cleanup does not throw over the trap
  let late = false;
  engine.onTrap(() => (late = true)); // a hook added after the trap (a loader still settling) runs at once
  assert.ok(late);
  const again = await w.init({ simd: false });
  assert.notEqual(again, rw);
  assert.equal(again.median(x), 2);
  assert.equal(await w.init({ simd: false }), again); // cached again
});

test("init({ wasm }) takes paths relative to the working directory, absolute paths and file URLs", async () => {
  const file = fileURLToPath(new URL("../dist/rankwise-portable.wasm", import.meta.url));
  for (const wasm of [relative(process.cwd(), file), file, pathToFileURL(file).href, pathToFileURL(file)]) {
    assert.equal((await wasmEntry.init({ wasm })).build, "portable", String(wasm));
  }
  // a Windows path is a path, not a URL with the scheme "c:"
  await assert.rejects(wasmEntry.init({ wasm: "C:\\nonexistent\\rankwise.wasm" }), (e) => e.code === "ENOENT");
});

shared("ImageData-style RGBA Uint8ClampedArray frames: value ops keep the type", async (rw) => {
  const imgs = Array.from({ length: 7 }, (_, k) => Uint8ClampedArray.from({ length: 4 * 64 }, (_, i) => (i * 13 + k * 71) % 256));
  const med = rw.median(imgs);
  assert.ok(med instanceof Float64Array); // np.median of integers is float64
  const mx = rw.max(imgs);
  assert.ok(mx instanceof Uint8ClampedArray);
  for (let i = 0; i < mx.length; i++) {
    const col = imgs.map((f) => f[i]).sort((a, b) => a - b);
    assert.equal(mx[i], col[6]);
    assert.equal(med[i], col[3]);
  }
});

shared("1-D input reduces to a number; counts come back for clipping", async (rw) => {
  assert.equal(rw.median(Float64Array.of(5, 1, 4, 2, 3)), 3);
  assert.equal(rw.quantile(Float32Array.of(1, 2, 3, 4), 0.5), 2.5);
  assert.deepEqual(rw.quantile(Float64Array.of(1, 2, 3, 4), [0, 1]), [1, 4]);
  const x = Float32Array.of(10, 10.1, 9.9, 10, 10.2, 9.8, 10, 500);
  const r = rw.sigmaClipMean(x, { returnCounts: true });
  assert.equal(r.high, 1);
  assert.equal(r.low, 0);
  assert.equal(r.rejected, 1);
  assert.ok(Math.abs(r.value - 10) < 1e-6);
});

shared("weights, normalization and the rejection map on frame lists", async (rw) => {
  const x = frames(6, 50, (k, i) => (k === 5 && i % 10 === 0 ? 1e6 : 100 + k + (i % 3)));
  const r = rw.clipCombine(x, { method: "sigma", sigma: 2, mask: true, returnCounts: true });
  assert.equal(r.mask.length, 6);
  for (let i = 0; i < 50; i++) assert.equal(r.mask[5][i], i % 10 === 0 ? 1 : 0);
  const w = rw.clipCombine(x, { method: "none", weights: [1, 1, 1, 1, 1, 0] });
  const s = rw.clipCombine(x, { method: "none", scale: [2, 2, 2, 2, 2, 2], offset: [1, 1, 1, 1, 1, 1] });
  for (let i = 1; i < 10; i++) {
    const mean5 = x.slice(0, 5).reduce((a, f) => a + f[i], 0) / 5;
    assert.ok(Math.abs(w[i] - mean5) < 1e-4);
    const meanAll = x.reduce((a, f) => a + 2 * f[i] + 1, 0) / 6;
    assert.ok(Math.abs(s[i] - meanAll) < 1e-3);
  }
});

shared("setIsa, describe and networkSize", async (rw) => {
  const x = frames(31, 1000, (k, i) => (k * 31 + i) % 97);
  rw.setIsa("scalar");
  assert.equal(rw.isa, "scalar");
  const a = rw.median(x);
  rw.setIsa(null);
  assert.deepEqual(rw.median(x), a);
  const missing = ["avx512", "neon", "simd128"].find((i) => !rw.availableIsas().includes(i));
  assert.throws(() => rw.setIsa(missing), RankwiseError);
  assert.throws(() => rw.setIsa("nope"), RangeError);
  assert.match(rw.describe(x), new RegExp(`^backend=${rw.backend} isa=${rw.isa} .*n=31 comparators\\(fast\\)=152 comparators\\(full\\)=186`));
  assert.equal(rw.networkSize(31, [15]), 152);
  assert.equal(rw.networkSize(31), 186);
});

shared("networkSize checks n and takes ranks as selectRanks does", async (rw) => {
  assert.equal(rw.networkSize(31, [-16]), rw.networkSize(31, [15]));
  assert.equal(rw.networkSize(9, [-1, 0]), rw.networkSize(9, [8, 0]));
  assert.equal(rw.networkSize(9n), rw.networkSize(9));
  for (const n of [0, -1, 1.5, NaN, 2 ** 32 + 9, rw.maxFrames + 1, "9"]) {
    assert.throws(() => rw.networkSize(n), /n must be an integer from 1 to/, String(n));
  }
  assert.throws(() => rw.networkSize(9, [1.5]), /ranks must be integers/);
  assert.throws(() => rw.networkSize(9, [9]), /rank 9 out of range for 9 frames/);
  assert.throws(() => rw.networkSize(9, [-10]), /rank -10 out of range/);
});

shared("alloc() takes a non-negative integer length", async (rw) => {
  for (const len of [-1, 1.5, NaN, Infinity, "4", 2 ** 53]) {
    assert.throws(() => rw.alloc("float32", len), /length must be a non-negative integer/, String(len));
  }
  const h = rw.alloc("float32", 0);
  assert.equal(h.array.length, 0);
  h.free?.();
});

shared("clipCombine takes sigma as a number or a [low, high] Array or TypedArray", async (rw) => {
  const x = frames(15, 400, (k, i) => ((k * 37 + i * 11) % 101) + (k === 3 ? 500 : 0));
  const want = rw.clipCombine(x, { sigma: [2, 3], returnCounts: true });
  for (const sigma of [Float64Array.of(2, 3), Float32Array.of(2, 3)]) {
    const r = rw.clipCombine(x, { sigma, returnCounts: true });
    assert.deepEqual(r.value, want.value);
    assert.deepEqual(r.low, want.low);
  }
  assert.deepEqual(rw.clipCombine(x, { sigma: [2, 3] }), rw.clipCombine(x, { sigmaLower: 2, sigmaUpper: 3 }));
  assert.throws(() => rw.clipCombine(x, { sigma: [2] }), /sigma needs a number or \[low, high\]; got 1 values/);
  assert.throws(() => rw.clipCombine(x, { sigma: Float64Array.of(1, 2, 3) }), /got 3 values/);
  for (const sigma of ["x", {}, [2, "x"]]) assert.throws(() => rw.clipCombine(x, { sigma }), TypeError);
  assert.throws(() => rw.clipCombine(x, { sigmaLower: "x" }), TypeError);
});

shared("alloc() arrays are read and written in place", async (rw) => {
  const n = 7;
  const npix = 3000;
  const stack = rw.alloc("float32", n * npix);
  const out = rw.alloc("float32", npix);
  stack.array.forEach((_, i, a) => (a[i] = (i * 7919) % 1009));
  const r = rw.median(stack.array, { shape: [n, npix], out: out.array });
  assert.ok(r.buffer === out.array.buffer && r.byteOffset === out.array.byteOffset); // written in place
  assert.deepEqual(Float32Array.from(out.array), rw.median(Float32Array.from(stack.array), { shape: [n, npix] }));
  stack.free();
  out.free();
});

shared("threads give identical results", async (rw) => {
  const x = frames(31, 50_000, (k, i) => ((k * 7919 + i * 104729) % 10007) / 7);
  for (const f of [(o) => rw.median(x, o), (o) => rw.sigmaClipMean(x, o), (o) => rw.quantile(x, [0.1, 0.9], o)[1]]) {
    assert.deepEqual(f({ threads: 4 }), f({}));
  }
  assert.throws(() => rw.median(x, { threads: 0 }), RangeError);
});

shared("maxiters 0 clips until nothing changes for 'sigma' and 'mad', as astropy and the Python binding", async (rw) => {
  const x = Float64Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 40, 60, 200);
  const rej = (o) => rw.clipCombine(x, { ...o, returnCounts: true }).rejected;
  // these thresholds need several passes, so one pass rejects less than convergence
  for (const [method, sigma, once, all] of [["sigma", 2, 1, 3], ["mad", 1, 6, 20]]) {
    assert.equal(rej({ method, sigma, maxiters: 1 }), once, method);
    assert.equal(rej({ method, sigma, maxiters: null }), all, method);
    assert.equal(rej({ method, sigma, maxiters: 0 }), all, method);
    assert.deepEqual(rw.clipCombine(x, { method, sigma, maxiters: 0 }), rw.clipCombine(x, { method, sigma, maxiters: null }));
  }
  assert.equal(rw.sigmaClipMean(x, { sigma: 2, maxiters: 0, returnCounts: true }).rejected, 3);
});

shared("argselect and selectRanks count negative ranks from the end and check the range", async (rw) => {
  const x = frames(5, 6, (k, i) => ((k * 7 + i * 3) % 5) + k / 10);
  const n = x.length;
  for (let r = -n; r < 0; r++) {
    assert.deepEqual(rw.argselect(x, r), rw.argselect(x, r + n), `rank ${r}`);
    assert.deepEqual(rw.selectRanks(x, r), rw.selectRanks(x, r + n), `rank ${r}`);
  }
  assert.deepEqual(rw.argselect(x, -1), rw.argmax(x)); // the frame holding the max
  assert.deepEqual(rw.argselect(x, [-1, 0]), [rw.argselect(x, n - 1), rw.argselect(x, 0)]);
  assert.equal(rw.argselect(Float64Array.of(3, 9, 1, 4), -1), 1); // 1-D input gives a number
  for (const f of [(r) => rw.argselect(x, r), (r) => rw.argselect(x, [0, r]), (r) => rw.selectRanks(x, [r])]) {
    for (const r of [n, -n - 1, 100]) {
      assert.throws(
        () => f(r),
        (e) => e instanceof RankwiseError && e.kind === "param" && e.message === `invalid parameter: rank ${r} out of range for ${n} frames`,
        `rank ${r}`,
      );
    }
    for (const r of [1.5, NaN, "1"]) assert.throws(() => f(r), (e) => e instanceof TypeError && e.message === "ranks must be integers");
  }
});

shared("an unknown staging is a RangeError", async (rw) => {
  const x = frames(5, 10, (k, i) => k + i);
  const want = rw.median(x);
  for (const staging of ["auto", "never", "always", "skew", null, undefined]) assert.deepEqual(rw.median(x, { staging }), want);
  for (const staging of ["Never", "bogus", 1]) {
    assert.throws(() => rw.median(x, { staging }), (e) => e instanceof RangeError && e.message === `unknown staging ${JSON.stringify(staging)}`);
  }
});

shared("cenfunc 'mean' applies to method 'sigma' or 'mad' only, as in the Python binding", async (rw) => {
  const x = frames(9, 4, (k, i) => (k * 13 + i * 7) % 11);
  for (const method of ["sigma", "mad"]) assert.ok(rw.clipCombine(x, { method, cenfunc: "mean" }) instanceof Float32Array);
  for (const method of ["median_sigma", "winsorized", "percentile", "none"]) {
    rw.clipCombine(x, { method, cenfunc: "median" }); // the default center is accepted everywhere
    assert.throws(
      () => rw.clipCombine(x, { method, cenfunc: "mean" }),
      (e) => e instanceof RankwiseError && e.kind === "param" && /cenfunc applies to method 'sigma' or 'mad'/.test(e.message),
      method,
    );
  }
});

let failed = 0;
for (const [name, f] of tests) {
  try {
    await f();
    console.log(`ok   ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${name}\n${e.stack}`);
  }
}
console.log(`${tests.length - failed}/${tests.length} API tests passed`);
if (failed) process.exit(1);
