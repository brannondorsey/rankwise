// rankwise in WebAssembly against the JavaScript ways of doing the same reductions, one
// thread each, on the workloads of the native benchmarks (bench/*.py) at JS-friendly sizes.
//
//   node bench/bench.mjs [--quick] [--only median,clip,...] [--json path]
//   bun bench/bench.mjs
//   taskset -c 1 node bench/bench.mjs          # what docs/wasm.md reports
//
// Every result is checked against rankwise's (exact for order statistics, counts for
// clipping) so no competitor is timed doing less work. Times are the best of repeated runs
// after a warm-up: 7 for rankwise, 2 for the slower JS baselines.
import { writeFile } from "node:fs/promises";
import { median as d3median, quantile as d3quantile } from "d3-array";
import { init } from "../js/rankwise.js";

const args = process.argv.slice(2);
const quick = args.includes("--quick");
const only = args.includes("--only") ? args[args.indexOf("--only") + 1].split(",") : null;
const jsonPath = args.includes("--json") ? args[args.indexOf("--json") + 1] : null;
const runtime =
  typeof Bun !== "undefined" ? `Bun ${Bun.version}` : typeof Deno !== "undefined" ? `Deno ${Deno.version.deno}` : `Node ${process.version}`;

// ------------------------------------------------------------------ data

// deterministic fill
function filler(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

/** Astro-like frames: sky ~ N(1000, 30) with 1/1024 cosmic-ray hits (as quickbench.rs). */
function astroFrames(n, npix, seed = 1) {
  const r = filler(seed);
  return Array.from({ length: n }, () => {
    const f = new Float32Array(npix);
    for (let i = 0; i < npix; i++) {
      const g = r() + r() + r() + r() - 2; // ~N(0, 0.58)
      f[i] = 1000 + g * 52;
      if (r() < 1 / 1024) f[i] += 5000 * r();
    }
    return f;
  });
}

/** Ensemble-like members: temperature field plus member spread. */
function ensembleMembers(n, npix, seed = 2) {
  const r = filler(seed);
  const base = Float32Array.from({ length: npix }, (_, i) => 280 + 20 * Math.sin(i / 997));
  return Array.from({ length: n }, () => Float32Array.from(base, (b) => b + (r() + r() + r() - 1.5) * 4));
}

/** Landsat-like uint16 scenes, 65% nodata (0). */
function scenes(n, npix, seed = 3) {
  const r = filler(seed);
  return Array.from({ length: n }, () => Uint16Array.from({ length: npix }, () => (r() < 0.65 ? 0 : 7000 + Math.floor(r() * 9000))));
}

// ------------------------------------------------------------------ JS baselines

function selectK(a, k, lo, hi) {
  // Wirth / Hoare selection on a Float32Array or Uint16Array, monomorphic and allocation-free
  while (hi > lo) {
    const x = a[k];
    let i = lo;
    let j = hi;
    do {
      while (a[i] < x) i++;
      while (x < a[j]) j--;
      if (i <= j) {
        const t = a[i];
        a[i] = a[j];
        a[j] = t;
        i++;
        j--;
      }
    } while (i <= j);
    if (j < k) lo = i;
    if (k < i) hi = j;
  }
  return a[k];
}

const js = {
  sortMedian(frames) {
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float32Array(n);
    const out = new Float32Array(npix);
    const m = n >> 1;
    for (let i = 0; i < npix; i++) {
      for (let k = 0; k < n; k++) col[k] = frames[k][i];
      col.sort();
      out[i] = n & 1 ? col[m] : Math.fround((col[m - 1] + col[m]) / 2);
    }
    return out;
  },
  selectMedian(frames) {
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float32Array(n);
    const out = new Float32Array(npix);
    const m = n >> 1;
    for (let i = 0; i < npix; i++) {
      for (let k = 0; k < n; k++) col[k] = frames[k][i];
      const hi = selectK(col, m, 0, n - 1);
      if (n & 1) out[i] = hi;
      else {
        let lo = col[0];
        for (let k = 1; k < m; k++) if (col[k] > lo) lo = col[k];
        out[i] = Math.fround((lo + hi) / 2);
      }
    }
    return out;
  },
  d3Median(frames) {
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float32Array(n);
    const out = new Float32Array(npix);
    for (let i = 0; i < npix; i++) {
      for (let k = 0; k < n; k++) col[k] = frames[k][i];
      out[i] = d3median(col);
    }
    return out;
  },
  sortQuantiles(frames, qs) {
    // NumPy's linear method on a sorted column, float64 lerp as np.quantile(q=list) does
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float32Array(n);
    const outs = qs.map(() => new Float64Array(npix));
    const idx = qs.map((q) => q * (n - 1));
    for (let i = 0; i < npix; i++) {
      for (let k = 0; k < n; k++) col[k] = frames[k][i];
      col.sort();
      for (let j = 0; j < qs.length; j++) {
        const v = idx[j];
        const i0 = Math.floor(v);
        const t = v - i0;
        const a = col[i0];
        const b = col[Math.min(i0 + 1, n - 1)];
        const d = b - a;
        outs[j][i] = t >= 0.5 ? b - d * (1 - t) : a + d * t;
      }
    }
    return outs;
  },
  d3Quantiles(frames, qs) {
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float32Array(n);
    const outs = qs.map(() => new Float64Array(npix));
    for (let i = 0; i < npix; i++) {
      for (let k = 0; k < n; k++) col[k] = frames[k][i];
      for (let j = 0; j < qs.length; j++) outs[j][i] = d3quantile(col, qs[j]);
    }
    return outs;
  },
  sigmaClipMean(frames, sigma = 3, maxiters = 5) {
    // astropy's algorithm on each sorted column: median center, population std, survivors
    // shrink to a contiguous range; the last bounds are applied to every value.
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Float64Array(n);
    const out = new Float32Array(npix);
    const rej = new Uint16Array(npix);
    for (let i = 0; i < npix; i++) {
      let m = 0;
      for (let k = 0; k < n; k++) {
        const v = frames[k][i];
        if (Number.isFinite(v)) col[m++] = v;
      }
      const s = col.subarray(0, m).sort();
      let a = 0;
      let b = m;
      let lo = -Infinity;
      let hi = Infinity;
      for (let it = 0; it < maxiters && b > a; it++) {
        const c = b - a;
        const h = a + (c >> 1);
        const med = c & 1 ? s[h] : (s[h - 1] + s[h]) / 2;
        let sum = 0;
        for (let k = a; k < b; k++) sum += s[k];
        const mean = sum / c;
        let ss = 0;
        for (let k = a; k < b; k++) ss += (s[k] - mean) ** 2;
        const std = Math.sqrt(ss / c);
        lo = med - sigma * std;
        hi = med + sigma * std;
        let na = a;
        let nb = b;
        while (na < nb && s[na] < lo) na++;
        while (nb > na && s[nb - 1] > hi) nb--;
        if (na === a && nb === b) break;
        a = na;
        b = nb;
      }
      let sum = 0;
      let kept = 0;
      for (let k = 0; k < m; k++) {
        if (s[k] >= lo && s[k] <= hi) {
          sum += s[k];
          kept++;
        }
      }
      out[i] = sum / kept;
      rej[i] = m - kept;
    }
    return { value: out, rejected: rej };
  },
  nodataMedian(frames) {
    const n = frames.length;
    const npix = frames[0].length;
    const col = new Uint16Array(n);
    const out = new Float64Array(npix);
    for (let i = 0; i < npix; i++) {
      let m = 0;
      for (let k = 0; k < n; k++) {
        const v = frames[k][i];
        if (v !== 0) col[m++] = v;
      }
      if (m === 0) {
        out[i] = NaN;
        continue;
      }
      const h = m >> 1;
      const hi = selectK(col, h, 0, m - 1);
      if (m & 1) out[i] = hi;
      else {
        let lo = col[0];
        for (let k = 1; k < h; k++) if (col[k] > lo) lo = col[k];
        out[i] = (lo + hi) / 2;
      }
    }
    return out;
  },
};

// ------------------------------------------------------------------ harness

function time(f, reps) {
  f(); // warm-up (and JIT)
  let best = Infinity;
  let res;
  for (let r = 0; r < reps; r++) {
    const t = performance.now();
    res = f();
    best = Math.min(best, performance.now() - t);
  }
  return { ms: best, res };
}

function maxDiff(a, b) {
  let d = 0;
  let bad = 0;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i]) && Number.isNaN(b[i])) continue;
    const e = Math.abs(a[i] - b[i]);
    if (!(e <= 0)) bad++;
    if (e > d || Number.isNaN(e)) d = Number.isNaN(e) ? Infinity : e;
  }
  return { maxAbs: d, differ: bad };
}

const simd = await init({ simd: true });
const portable = await init({ simd: false });

function rankwiseRows(name, f, reference) {
  // streamed from JS arrays (the default), zero-copy from WebAssembly memory, portable build
  return [
    { tool: "rankwise wasm simd128", run: () => f(simd, "stream"), reps: 7 },
    { tool: "rankwise wasm simd128, zero-copy", run: () => f(simd, "resident"), reps: 7, resident: true },
    { tool: "rankwise wasm portable", run: () => f(portable, "stream"), reps: 3 },
  ].map((r) => ({ ...r, name, reference }));
}

function resident(rw, frames) {
  // the same frames, held in WebAssembly memory (as an app that allocates there would)
  const n = frames.length;
  const npix = frames[0].length;
  const C = frames[0].constructor;
  const dt = C === Float32Array ? "float32" : C === Float64Array ? "float64" : "uint16";
  const w = rw.alloc(dt, n * npix);
  const a = w.array;
  frames.forEach((f, k) => a.set(f, k * npix));
  return w;
}

const results = [];

async function workload(key, title, detail, frames, rankwise, baselines, check) {
  if (only && !only.includes(key)) return;
  const npix = frames[0].length;
  console.log(`\n## ${title}\n${detail}`);
  const handles = new Map([[simd, resident(simd, frames)], [portable, null]]);
  const run = (rw, mode) => {
    if (mode === "resident") {
      const w = handles.get(rw);
      return rankwise(rw, w, { shape: [frames.length, npix] });
    }
    return rankwise(rw, frames, {});
  };
  const rows = [];
  let ref = null;
  for (const r of rankwiseRows(key, run)) {
    const { ms, res } = time(r.run, quick ? 2 : r.reps);
    ref ??= res;
    rows.push({ tool: r.tool, ms, check: check(res, ref) });
  }
  for (const [tool, f, reps] of baselines) {
    const { ms, res } = time(() => f(frames), quick ? 1 : reps ?? 2);
    rows.push({ tool, ms, check: check(res, ref) });
  }
  handles.get(simd)?.free();
  const fastestJs = Math.min(...rows.filter((r) => !r.tool.startsWith("rankwise")).map((r) => r.ms));
  for (const r of rows) {
    r.vsFastestJs = fastestJs / r.ms;
    console.log(
      `${r.tool.padEnd(36)} ${r.ms.toFixed(1).padStart(9)} ms  ${(fastestJs / r.ms).toFixed(1).padStart(6)}x  ${r.check}`,
    );
  }
  results.push({ key, title, detail, n: frames.length, npix, rows });
}

const exact = (res, ref) => {
  const a = Array.isArray(res) ? res : [res];
  const b = Array.isArray(ref) ? ref : [ref];
  const d = a.map((x, j) => maxDiff(x, b[j]));
  const differ = d.reduce((s, x) => s + x.differ, 0);
  return differ === 0 ? "identical" : `${differ} values differ (max ${Math.max(...d.map((x) => x.maxAbs)).toExponential(1)})`;
};

const side = quick ? 256 : 1024;
const npix = side * side;

{
  const frames = astroFrames(31, npix);
  await workload(
    "median",
    "Median of 31 frames",
    `31 × ${side}×${side} float32, like stacking registered subs (bench/astro.py)`,
    frames,
    (rw, x, o) => rw.median(x, o),
    [
      ["JS quickselect (hand-written)", js.selectMedian],
      ["JS TypedArray.sort per pixel", js.sortMedian],
      ["d3-array median per pixel", js.d3Median],
    ],
    exact,
  );
  await workload(
    "clip",
    "Sigma-clipped mean of 31 frames",
    `31 × ${side}×${side} float32, 3σ, 5 iterations, astropy's decisions`,
    frames,
    (rw, x, o) => rw.sigmaClipMean(x, { ...o, returnCounts: true }),
    [["JS sort + astropy iteration", (f) => js.sigmaClipMean(f)]],
    (res, ref) => {
      const c = maxDiff(res.rejected, ref.rejected);
      const v = maxDiff(res.value, ref.value);
      return c.differ === 0 ? `same rejections, mean within ${v.maxAbs.toExponential(1)}` : `${c.differ} reject counts differ`;
    },
  );
}
{
  const ny = quick ? 181 : 721;
  const nx = quick ? 360 : 1440;
  const members = ensembleMembers(31, ny * nx);
  const qs = [0.1, 0.5, 0.9];
  await workload(
    "quantiles",
    "p10 / p50 / p90 of a 31-member ensemble",
    `31 × ${ny}×${nx} float32 (a 0.25° global grid, as bench/ensemble.py), q as a list: float64 like np.quantile`,
    members,
    (rw, x, o) => rw.quantile(x, qs, o),
    [
      ["JS TypedArray.sort per cell", (f) => js.sortQuantiles(f, qs)],
      ["d3-array quantile per cell", (f) => js.d3Quantiles(f, qs)],
    ],
    exact,
  );
}
{
  const s = quick ? 128 : 512;
  const sc = scenes(90, s * s);
  await workload(
    "nodata",
    "Cloud-free median of 90 uint16 scenes",
    `90 × ${s}×${s} uint16, 65% nodata (= 0), as bench/landsat.py`,
    sc,
    (rw, x, o) => rw.median(x, { ...o, nodata: 0 }),
    [["JS filter + quickselect", js.nodataMedian]],
    exact,
  );
}
if (!only || only.includes("depth")) {
  // stack depth sweep at a fixed number of values: where rankwise's networks pay off
  console.log(`\n## Median by stack depth (${quick ? "4M" : "32M"} values each)`);
  const rows = [];
  for (const n of [4, 8, 16, 31, 64, 127, 255, 511]) {
    const np = Math.floor((quick ? 4 : 32) * 2 ** 20 / n / 64) * 64;
    const frames = astroFrames(n, np, n);
    const a = time(() => simd.median(frames), quick ? 2 : 5);
    const p = time(() => portable.median(frames), quick ? 1 : 2);
    const b = time(() => js.selectMedian(frames), quick ? 1 : 2);
    const ok = exact(b.res, a.res);
    rows.push({ n, npix: np, simd: a.ms, portable: p.ms, jsSelect: b.ms, check: ok });
    console.log(`n=${String(n).padStart(3)}  simd128 ${a.ms.toFixed(1).padStart(7)} ms  portable ${p.ms.toFixed(1).padStart(7)} ms  JS quickselect ${b.ms.toFixed(1).padStart(7)} ms  ${(b.ms / a.ms).toFixed(1)}x  ${ok}`);
  }
  results.push({ key: "depth", title: "Median by stack depth", rows });
}

const report = { runtime, quick, date: new Date().toISOString(), arch: typeof process !== "undefined" ? process.arch : "?", results };
if (jsonPath) {
  await writeFile(jsonPath, JSON.stringify(report, null, 1));
  console.log(`\nwrote ${jsonPath}`);
}
