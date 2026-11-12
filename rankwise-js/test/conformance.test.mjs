// Every fixture from make_fixtures.py (NumPy / SciPy / astropy references), on both
// WebAssembly builds and the native addon (where it loads)
// and on every ISA each build has, through every input path: streamed chunks, frame lists,
// zero-copy WebAssembly arrays and caller-supplied outputs.
//
//   python test/make_fixtures.py && node test/conformance.test.mjs   (or bun / deno run -A)
//   in a browser: serve the repository root and open rankwise-js/test/browser.html
import { init, simdSupported } from "../js/rankwise.js";

const dir = new URL("./fixtures/", import.meta.url);
async function readFile(url) {
  if (url.protocol === "file:") return (await import("node:fs/promises")).readFile(url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}
const manifest = JSON.parse(new TextDecoder().decode(await readFile(new URL("manifest.json", dir))));
const CTORS = { float32: Float32Array, float64: Float64Array, uint16: Uint16Array, uint8: Uint8Array, int32: Int32Array };

async function load(spec) {
  const buf = await readFile(new URL(spec.file, dir));
  const C = CTORS[spec.dtype];
  return { data: new C(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)), ...spec };
}

const same = (a, b) => a === b || (Number.isNaN(a) && Number.isNaN(b));
const DTYPE_OF = new Map([[Float32Array, "float32"], [Float64Array, "float64"], [Uint16Array, "uint16"], [Uint8Array, "uint8"], [Int32Array, "int32"]]);

function compareExact(got, want, label) {
  if (want.shape.length === 0) {
    if (typeof got !== "number") throw new Error(`${label}: expected a number, got ${got?.constructor?.name}`);
    if (!same(got, want.data[0])) throw new Error(`${label}: ${got} != ${want.data[0]}`);
    return;
  }
  const dt = DTYPE_OF.get(got.constructor);
  if (dt !== want.dtype) throw new Error(`${label}: dtype ${dt} != ${want.dtype}`);
  if (got.length !== want.data.length) throw new Error(`${label}: length ${got.length} != ${want.data.length}`);
  for (let i = 0; i < got.length; i++) {
    if (!same(got[i], want.data[i])) throw new Error(`${label}: [${i}] ${got[i]} != ${want.data[i]}`);
  }
}

function compareUlp(got, want, label) {
  for (let i = 0; i < got.length; i++) {
    const g = got[i];
    const w = want.data[i];
    if (Number.isNaN(g) && Number.isNaN(w)) continue;
    const ulp = Math.abs(w) === 0 ? 1.4e-45 : Math.abs(Math.fround(Math.abs(w) * (1 + 2 ** -23)) - Math.abs(w));
    if (!(Math.abs(g - w) <= ulp)) throw new Error(`${label}: [${i}] ${g} vs ${w} (more than 1 ulp)`);
  }
}

function frameList(x) {
  const [n] = x.shape;
  const len = x.data.length / n;
  return Array.from({ length: n }, (_, k) => x.data.slice(k * len, (k + 1) * len));
}

function args(c) {
  if (c.fn.includes("quantile") || c.fn.includes("percentile")) {
    const q = c.args[0];
    if (c.qform === "number") return [q];
    if (c.qform === "f32") return [Float32Array.from(q)];
    return [q];
  }
  return c.args;
}

function call(rw, c, x, extra = {}) {
  const input = c.frames ? frameList(x) : x.data;
  const opts = { ...c.opts, ...extra };
  if (!c.frames) opts.shape = x.shape;
  return rw[c.fn](input, ...args(c), opts);
}

function check(c, got, expect, label) {
  if (c.compare === "clip") {
    compareUlp(got.value, expect[0], `${label} value`);
    compareExact(got.mask, { ...expect[1], shape: [expect[1].data.length] }, `${label} mask`);
    compareExact(got.rejected, expect[2], `${label} rejected`);
    return;
  }
  const outs = expect.length === 1 && !Array.isArray(got) ? [got] : got;
  if (outs.length !== expect.length) throw new Error(`${label}: ${outs.length} outputs, expected ${expect.length}`);
  outs.forEach((o, j) => compareExact(o, expect[j], `${label} out ${j}`));
}

// Every backend that loads here: the native addon (Node, Bun, Deno, once built), then both
// WebAssembly builds.
const loaders = [];
if (typeof window === "undefined") {
  const node = await import("../js/node.js");
  loaders.push(() => node.init({ backend: "native" }).catch(() => null));
}
if (simdSupported()) loaders.push(() => init({ simd: true }));
loaders.push(() => init({ simd: false }));
let ran = 0;
let failed = 0;
const fail = (label, e) => {
  failed++;
  if (failed <= 20) console.error(`FAIL ${label}\n  ${e.message}`);
};

const cases = await Promise.all(
  manifest.cases.map(async (c) => ({ ...c, x: await load(c.input), want: await Promise.all(c.expect.map(load)) })),
);

const covered = [];
for (const load of loaders) {
  const rw = await load();
  if (!rw) continue;
  covered.push(`${rw.backend} ${rw.build} (${rw.availableIsas().join(", ")})`);
  for (const isa of rw.availableIsas()) {
    rw.setIsa(isa);
    for (const c of cases) {
      const label = `[${rw.backend} ${rw.build}/${isa}] ${c.name}`;
      try {
        check(c, call(rw, c, c.x), c.want, label);
        ran++;
      } catch (e) {
        fail(label, e);
      }
    }
    rw.setIsa(null);
  }

  // Zero-copy inputs and outputs, a tiny chunk budget (many ragged chunks), caller `out`s.
  const paths = {
    resident(c, x) {
      if (x.dtype === "uint8") return null;
      const w = rw.alloc(x.dtype, x.data.length);
      w.array.set(x.data);
      try {
        return rw[c.fn](w.array, ...args(c), { ...c.opts, shape: x.shape, axis: c.frames ? 0 : c.opts.axis });
      } finally {
        w.free();
      }
    },
    chunked(c, x) {
      if (rw.backend !== "wasm") return null; // the addon reads inputs in place
      const prev = rw.chunkBytes;
      rw.chunkBytes = 4096;
      try {
        return call(rw, c, x);
      } finally {
        rw.chunkBytes = prev;
      }
    },
    out(c, x) {
      if (c.compare === "clip" || c.want.some((w) => w.shape.length === 0)) return null;
      const outs = c.want.map((w) => new CTORS[w.dtype](w.data.length));
      const single = c.want.length === 1 && !(c.args[0] instanceof Array) && !["argselect", "selectRanks"].includes(c.fn);
      const r = call(rw, c, x, { out: single ? outs[0] : outs });
      if (single ? r !== outs[0] : r.some((o, j) => o !== outs[j])) throw new Error("out was not returned");
      return single ? outs[0] : outs;
    },
  };
  for (const [name, f] of Object.entries(paths)) {
    for (const c of cases) {
      if (name === "out" && c.fn === "argselect") continue; // argselect takes no out
      const label = `[${rw.backend} ${rw.build}/${name}] ${c.name}`;
      try {
        const got = f(c, c.x);
        if (got === null) continue;
        check(c, got, c.want, label);
        ran++;
      } catch (e) {
        fail(label, e);
      }
    }
  }
}

const runtime =
  typeof Bun !== "undefined" ? `Bun ${Bun.version}`
  : typeof Deno !== "undefined" ? `Deno ${Deno.version.deno}`
  : typeof process !== "undefined" ? `Node ${process.version}`
  : navigator.userAgent.match(/(Chrome|Firefox|Safari)\/[\d.]+/)?.[0] ?? "browser";
const summary = `${runtime}: ${ran} checks passed, ${failed} failed (${manifest.cases.length} cases, NumPy ${manifest.numpy}, SciPy ${manifest.scipy}) on ${covered.join("; ")}`;
console.log(summary);
globalThis.conformance = { ran, failed, summary };
if (failed && typeof process !== "undefined") process.exit(1);
