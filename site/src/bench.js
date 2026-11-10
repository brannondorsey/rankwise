// 06 · Benchmarks from bench/results/*.json, replayed in real time, plus the depth sweep.
import B from "./bench.json";
import { $, el, loop, reduced, secs, segmented, whenNear } from "./util.js";

const TABS = {
  median: { rows: B.median, sub: "median · 31 frames × 2048 × 2048 float32 · identical to np.median" },
  sigclip: { rows: B.sigclip, sub: "sigma-clipped mean · 3σ, ≤ 5 iterations · astropy's exact masks" },
  gefs: { rows: B.gefs, sub: "p10 · p50 · p90 · real NOAA GEFS, 31 × 81 × 121 × 257 · identical to np.quantile" },
  landsat: { rows: B.landsat, sub: "median · real Landsat, 90 × 2048² uint16, 65% clouds as nodata · identical to np.nanmedian" },
};

const clean = (t) => t.replace(" · scalar", " · portable").replace(/ \((uint16, nodata=0|float32)\)/, "").replace(" (nanmedian)", " nanmedian");

export function initBench() {
  $("#bench-machine").textContent = `${B.machine} · one core per tool · best of repeated runs after a warm-up · every result checked against its reference`;
  race();
  sweep();
}

function race() {
  const box = $("#race"), clock = $("#race-clock"), sub = $("#race-sub"), xax = $("#race-x");
  let rows = [], els = [], t = 0, running = false, fastestOther = 1, max = 1;
  function setup(k) {
    const d = TABS[k];
    rows = d.rows.map((r) => ({ ...r, tool: clean(r.tool), rw: r.tool.startsWith("rankwise") })).sort((a, b) => a.s - b.s);
    fastestOther = Math.min(...rows.filter((r) => !r.rw).map((r) => r.s));
    const best = rows[0].s;
    max = Math.max(...rows.map((r) => r.s));
    sub.textContent = d.sub;
    box.innerHTML = rows.map((r) => `<div class="race-row${r.rw ? " rw" : ""}"><div class="nm">${r.tool}</div><div class="tr"><i></i></div><div class="tm">—</div></div>`).join("");
    els = [...box.children].map((e) => ({ e, i: e.querySelector("i"), tm: e.querySelector(".tm") }));
    rows.forEach((r) => (r.x = r.s / best));
    xax.innerHTML = `<span>0 s</span><span class="hl">${(fastestOther / best).toFixed(0)}× faster than the fastest alternative</span><span>${secs(max)}</span>`;
    t = 0;
    running = true;
  }
  setup("median");
  running = false;
  segmented($("#race-tabs"), (d) => setup(d.k));
  $("#race-run").addEventListener("click", () => { t = 0; running = true; });
  whenNear(box, () => (running = true), "-20%");
  loop(box, (dt) => {
    if (!running) return;
    t += dt * (reduced ? 4 : 1);
    clock.textContent = `${Math.min(t, max).toFixed(2)} s`;
    rows.forEach((r, k) => {
      const f = Math.min(1, t / r.s), { e, i, tm } = els[k];
      i.style.width = (Math.min(t, r.s) / max) * 100 + "%";
      if (f >= 1 && !e.classList.contains("done")) {
        e.classList.add("done");
        tm.innerHTML = `${r.rw ? '<span class="ok">✓</span>' : ""}${secs(r.s)}<span class="x">${r.x < 1.05 ? "1×" : r.x.toFixed(r.x < 10 ? 1 : 0) + "×"}</span>`;
      } else if (f < 1) tm.textContent = "…";
    });
    if (t > max + 0.2) running = false;
  });
}

function sweep() {
  const svg = $("#sweep");
  svg.removeAttribute("viewBox");
  let op = "median", last = -1;
  const data = B.sweep;
  const series = {
    median: [["rankwise", "rankwise_median"], ["reducers", "reducers_median"], ["imcombiners", "imcombiners_median"], ["NumPy", "numpy_median"]],
    sigclip: [["rankwise", "rankwise_sigclip"], ["reducers", "reducers_sigclip"], ["imcombiners", "imcombiners_sigclip"]],
  };
  const greys = ["#9ba5b7", "#6c7689", "#4b5466"];
  function speedups(o) {
    return data.map((r) => Math.min(r[`reducers_${o}`], r[`imcombiners_${o}`]) / r[`rankwise_${o}`]);
  }
  function draw() {
    const W = Math.max(300, svg.clientWidth || 800), H = W < 560 ? 280 : 360;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.height = H + "px";
    svg.innerHTML = `<defs><linearGradient id="sweepGrad" x1="0" x2="1"><stop offset="0" stop-color="#5ee6ff"/><stop offset="0.5" stop-color="#4f8dff"/><stop offset="1" stop-color="#8a72ff"/></linearGradient><filter id="glow" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
    const L = 44, R = W < 560 ? 70 : 96, T = 10, Bm = 28;
    const lx = (n) => Math.log2(n), x0 = lx(8), x1 = lx(511);
    const X = (n) => L + ((lx(n) - x0) / (x1 - x0)) * (W - L - R);
    const y0 = Math.log10(2), y1 = Math.log10(30000);
    const Y = (v) => T + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - T - Bm);
    for (const v of [10, 100, 1000, 10000]) {
      el("line", { class: "grid", x1: L, x2: W - R, y1: Y(v), y2: Y(v) }, svg);
      el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }, svg).textContent = v >= 1000 ? `${v / 1000}k` : v;
    }
    for (const r of data) el("text", { x: X(r.n), y: H - 8, "text-anchor": "middle" }, svg).textContent = r.n;
    el("text", { x: L, y: T + 2, "font-size": 10 }, svg).textContent = "";
    const s = series[op];
    const ends = [];
    [...s].reverse().forEach(([name, key], j) => {
      const isRw = name === "rankwise";
      const d = data.map((r, i) => `${i ? "L" : "M"}${X(r.n).toFixed(1)},${Y(r[key]).toFixed(1)}`).join("");
      const p = el("path", { class: "ln" + (isRw ? " rw" : ""), d, stroke: isRw ? undefined : greys[s.length - 2 - j] ?? greys[0] }, svg);
      if (isRw) p.removeAttribute("stroke");
      const len = p.getTotalLength?.() || 1000;
      if (!reduced) {
        p.style.strokeDasharray = len;
        p.style.strokeDashoffset = len;
        p.getBoundingClientRect();
        p.style.transition = `stroke-dashoffset ${isRw ? 1.6 : 1.1}s cubic-bezier(.2,.7,.1,1) ${isRw ? 0.35 : 0}s`;
        requestAnimationFrame(() => (p.style.strokeDashoffset = 0));
      }
      for (const r of data) el("circle", { class: "pt", cx: X(r.n), cy: Y(r[key]), r: isRw ? 3.5 : 2.5, fill: isRw ? "#5ee6ff" : greys[s.length - 2 - j] ?? greys[0] }, svg);
      const lr = data[data.length - 1];
      ends.push({ y: Y(lr[key]), name, isRw, c: isRw ? "#5ee6ff" : greys[s.length - 2 - j] ?? greys[0] });
    });
    // end labels, pushed apart
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
    for (const e of ends) el("text", { class: "lbl", x: W - R + 8, y: e.y + 4, fill: e.c, style: `fill:${e.c}` }, svg).textContent = e.name;
    el("text", { x: W - R + 16, y: H - 8, "text-anchor": "start" }, svg).textContent = "frames";
    el("text", { x: L + 6, y: Y(10000) - 8, "text-anchor": "start" }, svg).textContent = "ns per pixel · log";
    // crosshair
    const ch = el("g", { style: "opacity:0" }, svg);
    const vl = el("line", { y1: T, y2: H - Bm, stroke: "rgba(255,255,255,0.25)", "stroke-dasharray": "3 3" }, ch);
    const box = el("rect", { rx: 8, fill: "rgba(4,6,10,0.92)", stroke: "rgba(255,255,255,0.12)" }, ch);
    const tx = el("text", { style: "fill:#eef2f8;font-size:11.5px" }, ch);
    const sp = speedups(op);
    const move = (e) => {
      const r = svg.getBoundingClientRect();
      const mx = ((e.clientX - r.left) / r.width) * W;
      let bi = 0, bd = 1e9;
      data.forEach((d, i) => { const dd = Math.abs(X(d.n) - mx); if (dd < bd) { bd = dd; bi = i; } });
      const d = data[bi], x = X(d.n);
      ch.style.opacity = 1;
      vl.setAttribute("x1", x); vl.setAttribute("x2", x);
      const lines = [`${d.n} frames`, ...s.map(([nm, k]) => `${nm} ${d[k].toFixed(d[k] < 100 ? 1 : 0)} ns`), `→ ${sp[bi].toFixed(1)}× faster`];
      tx.innerHTML = "";
      lines.forEach((t, i) => { const ts = el("tspan", { x: 0, dy: i ? 15 : 0, style: i === lines.length - 1 ? "fill:#5ee6ff" : i === 1 ? "fill:#5ee6ff" : i ? "fill:#9ba5b7" : "" }, tx); ts.textContent = t; });
      const bw = 168, bh = lines.length * 15 + 12;
      const bx = x + 12 + bw > W - 4 ? x - 12 - bw : x + 12;
      box.setAttribute("x", bx); box.setAttribute("y", T + 4); box.setAttribute("width", bw); box.setAttribute("height", bh);
      tx.setAttribute("transform", `translate(${bx + 12},${T + 22})`);
      tx.querySelectorAll("tspan").forEach((t) => t.setAttribute("x", 0));
    };
    svg.onpointermove = move;
    svg.onpointerdown = move;
    svg.onpointerleave = () => (ch.style.opacity = 0);
    $("#sweep-big").textContent = `≥ ${Math.min(...sp).toFixed(1)}×`;
  }
  segmented($("#sweep-op"), (d) => { op = d.k; draw(); });
  whenNear(svg, draw, "150px");
  new ResizeObserver(() => {
    const w = svg.clientWidth;
    if (Math.abs(w - last) > 20 && last > 0) draw();
    last = w;
  }).observe(svg);
}
