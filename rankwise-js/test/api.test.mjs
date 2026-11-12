// API behaviour that the NumPy fixtures do not cover: loading and backend selection, errors,
// zero-copy arrays, memory growth, ImageData input, ISA control, threads. Runs under node,
// bun and deno; the shared tests run on every backend that loads here.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
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
