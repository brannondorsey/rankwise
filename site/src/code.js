// Code samples, shortened from the README.
import { $, segmented } from "./util.js";

const SNIPPETS = {
  py: { lang: "py", code: `import numpy as np
import rankwise as rw

stack = np.load("frames.npy")   # 40 exposures stacked on axis 0: shape (40, 1200, 1800)

med = rw.median(stack, axis=0)                      # same result as np.median
p10, p90 = rw.quantile(stack, [0.1, 0.9], axis=0)   # same result as np.quantile
clean = rw.sigma_clip_mean(stack, axis=0, sigma=3)  # same pixels rejected as astropy` },
  rs: { lang: "rs", code: `// one slice per frame, each with one value per pixel
let frames: Vec<&[f32]> = exposures.iter().map(|f| f.as_slice()).collect();

let mut med = vec![0.0f32; npix];
rankwise::median_frames(&frames, &mut med)?;   // same result as np.median(stack, axis=0)` },
  c: { lang: "c", code: `#include <rankwise.h>

const float *frames[40];   /* 40 exposures, npix floats each */
float *med = malloc(npix * sizeof(float));

rw_status st = rw_median_f32(frames, 40, npix, med);` },
  build: { lang: "sh", code: `# Python: abi3 wheel, Python >= 3.9
pip install maturin && maturin build --release -m rankwise-py/Cargo.toml -o dist
pip install --find-links dist rankwise

# Rust
cargo add rankwise --git https://github.com/brannondorsey/rankwise

# C library: librankwise.so / .dylib / .dll + rankwise.h
cargo build --release -p rankwise-capi` },
};

const KW = {
  py: new Set(["import", "as", "from", "def", "return", "for", "in", "if", "None", "True", "False"]),
  rs: new Set(["use", "fn", "let", "mut", "pub", "struct", "impl", "for", "in", "as"]),
  c: new Set(["const", "float", "NULL"]),
  sh: new Set(["cargo", "pip", "maturin"]),
};

function highlight(src, lang) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const out = [];
  for (const line of src.split("\n")) {
    // split off the comment
    const cm = lang === "c" ? line.indexOf("/*") : lang === "rs" ? line.indexOf("//") : line.indexOf("#");
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

const API = ["median", "nanmedian", "quantile", "nanquantile", "percentile", "min", "max", "ptp", "argmin", "argmax", "argmedian", "select_ranks", "trimmed_mean", "iqr", "mad", "sigma_clip_mean", "clip_combine", "nodata=", "weights=", "rejected=", "threads="];

export function initCode() {
  const body = $("#code-body");
  let cur = "py";
  const show = (k) => { cur = k; body.innerHTML = highlight(SNIPPETS[k].code, SNIPPETS[k].lang); };
  segmented($("#code-tabs"), (d) => d.k && show(d.k));
  // the copy button lives in the tab bar; keep the selected tab highlighted
  const copy = $("#code-copy");
  copy.addEventListener("click", async (e) => {
    e.stopPropagation();
    $("#code-tabs").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === cur));
    try { await navigator.clipboard.writeText(SNIPPETS[cur].code); copy.textContent = "copied"; } catch { copy.textContent = "select + copy"; }
    setTimeout(() => (copy.textContent = "copy"), 1400);
  });
  show("py");
  $("#api").innerHTML = API.map((a) => `<span>${a.endsWith("=") ? a : "rw." + a}</span>`).join("");
}
