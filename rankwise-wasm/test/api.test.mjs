// API behaviour that the NumPy fixtures do not cover: loading, errors, zero-copy arrays,
// memory growth, ImageData input, ISA control. Runs under node, bun and deno.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { init, initSync, RankwiseError, simdSupported } from "../js/rankwise.js";

const tests = [];
const test = (name, f) => tests.push([name, f]);

const frames = (n, len, f) => Array.from({ length: n }, (_, k) => Float32Array.from({ length: len }, (_, i) => f(k, i)));

test("init picks the SIMD build where supported and caches it", async () => {
  const a = await init();
  const b = await init();
  assert.equal(a, b);
  assert.equal(a.build, simdSupported() ? "simd128" : "portable");
  assert.deepEqual(a.availableIsas(), simdSupported() ? ["simd128", "scalar"] : ["scalar"]);
  assert.equal(a.isa, a.availableIsas()[0]);
  assert.match(a.version, /^\d+\.\d+\.\d+$/);
  assert.equal(a.maxFrames, 512);
});

test("initSync and init({ wasm }) take bytes", async () => {
  const bytes = await readFile(new URL("../dist/rankwise-portable.wasm", import.meta.url));
  const s = initSync(bytes);
  assert.equal(s.build, "portable");
  const m = await init({ wasm: bytes });
  assert.equal(m.build, "portable");
  assert.deepEqual(s.median([Float32Array.of(1, 5), Float32Array.of(2, 6), Float32Array.of(3, 4)]), Float32Array.of(2, 5));
});

test("errors carry a kind and leave the instance usable", async () => {
  const rw = await init();
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

test("shape and axis: every axis of a 3-D array", async () => {
  const rw = await init();
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
  const rw = await init();
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

test("ImageData-style RGBA Uint8ClampedArray frames: value ops keep the type", async () => {
  const rw = await init();
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

test("1-D input reduces to a number; counts come back for clipping", async () => {
  const rw = await init();
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

test("weights, normalization and the rejection map on frame lists", async () => {
  const rw = await init();
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

test("setIsa, describe and networkSize", async () => {
  const rw = await init();
  const x = frames(31, 1000, (k, i) => (k * 31 + i) % 97);
  rw.setIsa("scalar");
  assert.equal(rw.isa, "scalar");
  const a = rw.median(x);
  rw.setIsa(null);
  assert.deepEqual(rw.median(x), a);
  assert.throws(() => rw.setIsa("avx512"), RankwiseError);
  assert.throws(() => rw.setIsa("nope"), RangeError);
  assert.match(rw.describe(x), /n=31 comparators\(fast\)=152 comparators\(full\)=186/);
  assert.equal(rw.networkSize(31, [15]), 152);
  assert.equal(rw.networkSize(31), 186);
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
