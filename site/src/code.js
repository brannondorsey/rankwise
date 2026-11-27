// 05 · Code, from the README.
import { $, segmented } from "./util.js";

const SNIPPETS = {
  py: { lang: "py", code: `# pip install rankwise

import numpy as np
import rankwise as rw

stack = np.load("lights.npy", mmap_mode="r")   # (40, 1200, 1800, 3) float32, never copied

clean = rw.sigma_clip_mean(stack, axis=0, sigma=(3, 3), maxiters=5)  # astropy's decisions
med   = rw.median(stack, axis=0)                    # identical to np.median
lo, hi = rw.quantile(stack, [0.1, 0.9], axis=0)     # identical to np.quantile, every method
comp  = rw.median(scenes, axis=0, nodata=0)         # uint16, clouds skipped
k     = rw.select_ranks(stack, [0, 3, 36], axis=0)  # any ranks, one pass` },
  js: { lang: "js", code: `// npm install @rankwise-js/rankwise

import { init } from "@rankwise-js/rankwise";

// init() picks the fastest backend the runtime has. Same API, identical results:
//   native    Node, Bun, Deno: the Rust kernels (AVX-512, AVX2, NEON) via Node-API
//   simd128   browsers, or Node with no native addon for the platform: WebAssembly SIMD
//   portable  fallback for engines without WebAssembly SIMD: plain WebAssembly
const rw = await init();
console.log(rw.backend, rw.isa);   // "native" "avx512" · "wasm" "simd128" · "wasm" "scalar"

const clean = rw.sigmaClipMean(subs, { sigma: 3, maxiters: 5 });  // subs: Float32Array[]
const med   = rw.median(subs);                                     // identical to np.median
const [lo, hi] = rw.quantile(cube, [0.1, 0.9], { shape: [31, 721, 1440] });  // along axis 0
const comp  = rw.median(scenes, { nodata: 0 });                    // Uint16Array[], clouds skipped` },
  rs: { lang: "rs", code: `// cargo add rankwise --features ndarray

use ndarray::{Array2, ArrayView3, Axis};
use rankwise::{Clip, Reduce};

fn integrate(stack: ArrayView3<f32>) -> rankwise::Result<Array2<f32>> {
    let mut out = Array2::zeros((stack.dim().1, stack.dim().2));
    stack.reduce_into(Axis(0), Clip::sigma(3.0, 3.0).max_iters(5).mean(), out.view_mut())?;
    Ok(out)
}

// plain slices, one per frame: no ndarray needed
let frames: Vec<&[f32]> = subs.iter().map(|s| s.as_slice()).collect();
rankwise::median_frames(&frames, &mut master_dark)?;` },
  c: { lang: "c", code: `/* build: cargo build --release -p rankwise-capi */
/* link:  cc stack.c -Irankwise-capi/include -Ltarget/release -lrankwise */

#include <rankwise.h>

const float *frames[40];                  /* one pointer per frame, H*W floats each */
rw_clip clip = rw_clip_sigma(3.0f, 3.0f, /*maxiters*/ 5);
rw_status st = rw_clip_mean_f32(frames, 40, H * W, &clip, out, /*rejected*/ NULL);` },
  build: { lang: "sh", code: `# Rust core
cargo test --release -p rankwise --features ndarray

# C library: librankwise.so / .dylib / .dll + rankwise.h
cargo build --release -p rankwise-capi

# Python: abi3 wheel, Python >= 3.9
pip install maturin && maturin build --release -m rankwise-py/Cargo.toml -o dist
pip install --find-links dist rankwise

# JavaScript: both WebAssembly builds (simd128, portable) + this machine's native addon
rustup target add wasm32-unknown-unknown
node rankwise-js/scripts/build.mjs --all` },
};

const KW = {
  py: new Set(["import", "as", "from", "def", "return", "for", "in", "if", "None", "True", "False"]),
  js: new Set(["import", "from", "const", "let", "await", "new", "return"]),
  rs: new Set(["use", "fn", "let", "mut", "pub", "struct", "impl", "for", "in", "as"]),
  c: new Set(["const", "float", "NULL"]),
  sh: new Set(["cargo", "pip", "maturin", "rustup", "node"]),
};

function highlight(src, lang) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const out = [];
  for (const line of src.split("\n")) {
    // split off the comment
    const cm = lang === "c" ? line.indexOf("/*") : lang === "rs" || lang === "js" ? line.indexOf("//") : line.indexOf("#");
    const isInclude = lang === "c" && line.startsWith("#include");
    let code = cm >= 0 && !isInclude && !(lang === "sh" && cm > 0) ? line.slice(0, cm) : line;
    let com = cm >= 0 && !isInclude && !(lang === "sh" && cm > 0) ? line.slice(cm) : "";
    if (lang === "c" && cm >= 0) { const end = line.indexOf("*/", cm); if (end > 0 && end + 2 < line.length) { code = line.slice(0, cm); com = line.slice(cm, end + 2); code += ""; com += ""; out.push(hl(code, lang) + `<span class="c">${esc(com)}</span>` + hl(line.slice(end + 2), lang)); continue; } }
    out.push(hl(code, lang) + (com ? `<span class="c">${esc(com)}</span>` : ""));
  }
  return out.join("\n");
  function hl(s, l) {
    const tokens = [];
    const re = /("[^"]*"|<[a-z.]+>)|(\b\d+(?:\.\d+)?f?\b)|(\b(?:rw|rankwise|np)\b(?=\.|::))|(\b[A-Za-z_]\w*(?=\())|([A-Za-z_]\w*)/g;
    let last = 0, m;
    while ((m = re.exec(s))) {
      tokens.push(esc(s.slice(last, m.index)));
      if (m[1]) tokens.push(`<span class="s">${esc(m[1])}</span>`);
      else if (m[2]) tokens.push(`<span class="n">${m[2]}</span>`);
      else if (m[3]) tokens.push(`<span class="t">${m[3]}</span>`);
      else if (m[4]) tokens.push(`<span class="f">${m[4]}</span>`);
      else {
        const w = m[5];
        tokens.push(KW[l].has(w) ? `<span class="k">${w}</span>` : /^[A-Z]\w*$/.test(w) && l !== "sh" ? `<span class="t">${w}</span>` : w);
      }
      last = re.lastIndex;
    }
    tokens.push(esc(s.slice(last)));
    return tokens.join("").replace("#include", '<span class="k">#include</span>');
  }
}

// The operations, grouped, with the library each one matches result for result.
const API = [
  { h: "Medians & quantiles", f: ["median", "nanmedian", "quantile", "nanquantile", "percentile"], m: "NumPy, all 13 quantile methods" },
  { h: "Extremes & positions", f: ["min", "max", "ptp", "argmin", "argmax", "argmedian"], m: "NumPy" },
  { h: "Any ranks", f: ["select_ranks", "argselect"], m: "np.sort(x)[ranks], in one pass" },
  { h: "Robust statistics", f: ["trimmed_mean", "iqr", "mad"], m: "scipy.stats" },
  { h: "Outlier rejection", f: ["sigma_clip_mean", "clip_combine"], m: "astropy sigma_clip, plus MAD, winsorized, percentile" },
  { h: "Options, on every operation", opt: [
    ["nodata", "A value that means missing, skipped like NaN. Landsat marks cloud pixels 0, so nodata=0 takes the median of the clear passes only."],
    ["weights", "One weight per frame, e.g. by exposure time or seeing."],
    ["scale", "Per-frame normalization (with offset=), fused into the same pass."],
    ["threads", "Split a large call across up to this many cores."],
  ] },
];
const camel = (n) => n.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

export function initCode() {
  const body = $("#code-body"), api = $("#api");
  let cur = "py";
  const names = (k) => {
    const js = k === "js";
    $("#api-k").textContent = js ? "JavaScript names" : k === "py" ? "Python names" : "Python names · camelCase in JavaScript";
    api.querySelectorAll("code").forEach((c) => { const n = c.dataset.n; c.textContent = js ? camel(n) : n; });
  };
  const show = (k) => { cur = k; body.innerHTML = highlight(SNIPPETS[k].code, SNIPPETS[k].lang); names(k); };
  api.innerHTML = API.map((g) => g.opt
    ? `<div class="api-g opt"><h3>${g.h}</h3><dl class="api-o">${g.opt.map(([n, d]) => `<div><dt><code data-n="${n}">${n}</code></dt><dd>${d}</dd></div>`).join("")}</dl></div>`
    : `<div class="api-g"><h3>${g.h}</h3><div class="api-f">${g.f.map((n) => `<code data-n="${n}">${n}</code>`).join("")}</div><p>Same results as ${g.m}</p></div>`).join("");
  const setTab = segmented($("#code-tabs"), (d) => d.k && show(d.k));
  // the language names in the hero chip open their snippet (the link itself scrolls to #code)
  document.querySelectorAll("[data-code]").forEach((a) => a.addEventListener("click", () => { setTab((d) => d.k === a.dataset.code); show(a.dataset.code); }));
  // the copy button lives in the tab bar; keep the selected tab highlighted
  const copy = $("#code-copy");
  copy.addEventListener("click", async (e) => {
    e.stopPropagation();
    $("#code-tabs").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === cur));
    try { await navigator.clipboard.writeText(SNIPPETS[cur].code); copy.textContent = "copied"; } catch { copy.textContent = "select + copy"; }
    setTimeout(() => (copy.textContent = "copy"), 1400);
  });
  show("py");
}
