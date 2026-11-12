// Stack the website's 40 synthetic astro subs with rankwise in WebAssembly.
// Serve the repository root (python3 -m http.server) and open /examples/js/browser/.
import { init } from "../../../rankwise-js/js/rankwise.js";

const $ = (s) => document.querySelector(s);
const DATA = "../../../site/public/data/";

const t0 = performance.now();
const [rw, meta, img] = await Promise.all([
  init(),
  fetch(DATA + "astro.json").then((r) => r.json()),
  new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = rej;
    i.src = DATA + "astro_subs.jpg";
  }),
]);
const loadMs = performance.now() - t0;

// ---- decode the sprite: 8-bit codes -> electrons (the site's encoding), RGB interleaved
const N = meta.frames;
const T = meta.tile;
const [X1, C1, MAX] = meta.enc;
const dec = new Float32Array(256).map((_, c) => (c < C1 ? (c / C1) * X1 : X1 * Math.exp(((c - C1) / (255 - C1)) * Math.log(MAX / X1))));
const cvs = new OffscreenCanvas(img.width, img.height);
const ctx = cvs.getContext("2d", { willReadFrequently: true });
ctx.drawImage(img, 0, 0);
const frames = Array.from({ length: N }, (_, k) => {
  const { data } = ctx.getImageData((k % meta.cols) * T, Math.floor(k / meta.cols) * T, T, T);
  const f = new Float32Array(T * T * 3);
  for (let p = 0, q = 0; p < data.length; p += 4, q += 3) {
    f[q] = dec[data[p]];
    f[q + 1] = dec[data[p + 1]];
    f[q + 2] = dec[data[p + 2]];
  }
  return f;
});

$("#status").textContent =
  `rankwise ${rw.version} · ${rw.build} build · ${rw.isa} kernels · ${N} × ${T}×${T}×3 float32 · loaded in ${loadMs.toFixed(0)} ms`;

// ---- display: the site's per-channel stretch and saturation
const view = $("#view").getContext("2d");
const out = view.createImageData(T, T);
const st = meta.stretch;
const mtf = (m, x) => ((m - 1) * x) / ((2 * m - 1) * x - m);
function show(rgb, rejected = null) {
  const o = out.data;
  for (let p = 0, q = 0; q < rgb.length; p += 4, q += 3) {
    const c = [0, 1, 2].map((ch) => mtf(st[ch][2], Math.min(1, Math.max(0, (rgb[q + ch] - st[ch][0]) / (st[ch][1] - st[ch][0])))));
    const lum = (c[0] + c[1] + c[2]) / 3;
    if (rejected) {
      const r = rejected[q / 3];
      const l = r < 400 ? 0 : Math.min(1, Math.log(r) / Math.log(40000));
      o[p] = 255 * (lum * 0.3 + l);
      o[p + 1] = 255 * (lum * 0.3 + l * 0.63);
      o[p + 2] = 255 * (lum * 0.3 + l * 0.16);
    } else {
      for (let ch = 0; ch < 3; ch++) o[p + ch] = 255 * Math.min(1, Math.max(0, lum + (c[ch] - lum) * meta.sat));
    }
    o[p + 3] = 255;
  }
  view.putImageData(out, 0, 0);
}

// ---- the reductions, rankwise and plain JS
function jsMedian(fs) {
  const n = fs.length;
  const col = new Float32Array(n);
  const res = new Float32Array(fs[0].length);
  for (let i = 0; i < res.length; i++) {
    for (let k = 0; k < n; k++) col[k] = fs[k][i];
    col.sort();
    res[i] = n & 1 ? col[n >> 1] : Math.fround((col[n / 2 - 1] + col[n / 2]) / 2);
  }
  return res;
}
function jsClip(fs, sigma) {
  const n = fs.length;
  const s = new Float64Array(n);
  const res = new Float32Array(fs[0].length);
  for (let i = 0; i < res.length; i++) {
    for (let k = 0; k < n; k++) s[k] = fs[k][i];
    s.sort();
    let a = 0, b = n, lo = -Infinity, hi = Infinity;
    for (let it = 0; it < 5 && b > a; it++) {
      const c = b - a, h = a + (c >> 1);
      const med = c & 1 ? s[h] : (s[h - 1] + s[h]) / 2;
      let sum = 0;
      for (let k = a; k < b; k++) sum += s[k];
      const mean = sum / c;
      let ss = 0;
      for (let k = a; k < b; k++) ss += (s[k] - mean) ** 2;
      const sd = Math.sqrt(ss / c);
      lo = med - sigma * sd;
      hi = med + sigma * sd;
      let na = a, nb = b;
      while (na < nb && s[na] < lo) na++;
      while (nb > na && s[nb - 1] > hi) nb--;
      if (na === a && nb === b) break;
      a = na;
      b = nb;
    }
    let sum = 0, kept = 0;
    for (let k = 0; k < n; k++) if (s[k] >= lo && s[k] <= hi) { sum += s[k]; kept++; }
    res[i] = sum / kept;
  }
  return res;
}
const mean = (fs) => {
  const r = new Float32Array(fs[0].length);
  for (const f of fs) for (let i = 0; i < r.length; i++) r[i] += f[i] / fs.length;
  return r;
};

let mode = "clip";
let sigma = 3;
let timings = {};
const label = { sub: "sub 20 of 40", mean: "mean of 40", median: "median of 40", clip: "σ-clipped mean of 40", rejected: "removed by σ-clipping" };

function time(f) {
  const t = performance.now();
  const r = f();
  return [r, performance.now() - t];
}

function render() {
  let ms = null;
  if (mode === "sub") show(frames[19]);
  else if (mode === "mean") show(mean(frames));
  else if (mode === "median") {
    let r;
    [r, ms] = time(() => rw.median(frames));
    show(r);
  } else {
    let r;
    [r, ms] = time(() => rw.sigmaClipMean(frames, { sigma, mask: mode === "rejected" }));
    if (mode === "rejected") {
      // the largest value removed above the stacked one, per pixel
      // (frame lists get one rejection map per frame)
      const removed = new Float32Array(T * T);
      r.mask.forEach((m, k) => {
        for (let j = 0; j < m.length; j++) {
          if (!m[j]) continue;
          const d = frames[k][j] - r.value[j];
          if (d > removed[(j / 3) | 0]) removed[(j / 3) | 0] = d;
        }
      });
      show(r.value, removed);
    } else show(r);
  }
  $("#tag").textContent = mode === "clip" || mode === "rejected" ? `${label[mode]} · ${sigma.toFixed(1)}σ` : label[mode];
  timings.rw = ms;
  timings.js = timings.mode === mode && timings.sigma === sigma ? timings.js : null;
  timings.mode = mode;
  timings.sigma = sigma;
  bars();
}

function bars() {
  const el = $("#bars");
  el.innerHTML = "";
  $("#compare").disabled = timings.rw == null;
  if (timings.rw == null) {
    el.innerHTML = `<p class="note">Pick median or σ-clipping to time it.</p>`;
    $("#speedup").textContent = "";
    return;
  }
  const rows = [["rankwise (wasm SIMD)", timings.rw, ""]];
  if (timings.js != null) rows.push(["plain JavaScript", timings.js, "js"]);
  const max = Math.max(...rows.map((r) => r[1]));
  for (const [name, ms, cls] of rows) {
    el.insertAdjacentHTML("beforeend", `<div class="row"><span>${name}</span><span>${ms.toFixed(1)} ms</span></div><div class="bar ${cls}" style="width:${(100 * ms) / max}%"></div>`);
  }
  $("#speedup").textContent = timings.js != null ? `${(timings.js / timings.rw).toFixed(1)}× faster` : "";
}

$("#modes").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  mode = b.dataset.mode;
  for (const x of document.querySelectorAll("#modes button")) x.setAttribute("aria-pressed", String(x === b));
  render();
});
$("#sigma").addEventListener("input", (e) => {
  sigma = Number(e.target.value);
  $("#sigma-out").textContent = sigma.toFixed(1);
  if (mode === "clip" || mode === "rejected") render();
});
$("#compare").addEventListener("click", () => {
  const b = $("#compare");
  b.disabled = true;
  b.textContent = "Running plain JavaScript…";
  setTimeout(() => {
    // rankwise again, warmed up, so both numbers are steady-state
    const f = mode === "median" ? () => rw.median(frames) : () => rw.sigmaClipMean(frames, { sigma });
    let best = Infinity;
    for (let i = 0; i < 3; i++) best = Math.min(best, time(f)[1]);
    timings.rw = best;
    const [, ms] = time(() => (mode === "median" ? jsMedian(frames) : jsClip(frames, sigma)));
    timings.js = ms;
    b.textContent = "Time plain JavaScript too";
    bars();
  }, 30);
});

render();
window.rankwiseReady = true;
