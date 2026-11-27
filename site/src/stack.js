// 06 Examples · Sigma clipping. rankwise's sigma-clipped stack of the 40 synthetic subs, wiped against
// one sub, the plain mean or the median (all four precomputed by site/scripts/build_assets.py, so the
// page only moves a clip-path), and the 40 values of one pixel on a trail: the satellite drags the mean
// up, sigma clipping rejects it.
import { $, $$, asset, clamp, easeOut, el, fmt, getJSON, loop, reduced, segmented, whenNear } from "./util.js";
import { sigmaClip } from "./clip.js";
import { miniRace } from "./weather.js";

const MODES = {
  single: { lbl: "1 sub · 30 s", cap: "<b>Left</b>, one 30 s sub: grainy, with a satellite across it. <b>Right</b>, rankwise's sigma-clipped stack of all 40." },
  mean: { lbl: "Mean of 40", cap: "<b>Left</b>, the plain mean: smooth, but every satellite, the airplane and the meteor survive as streaks. <b>Right</b>, sigma clipping rejects just those outliers." },
  median: { lbl: "Median of 40", cap: "<b>Left</b>, the median: no streaks, but 25% noisier than a mean. <b>Right</b>, sigma clipping: no streaks, and as smooth as the mean." },
};
const ROWS = [["single", "1 sub"], ["mean", "Mean"], ["median", "Median"], ["sig", "σ-clip"]];

export function initStack() {
  const root = $("#stack");
  whenNear(root, () => start(root), "600px");
}

async function start(root) {
  const fig = $("#ab", root), a = $("#ab-a", root), b = $("#ab-b", root), tags = $("#ab-tags", root);
  const ex = (await getJSON("data/astro.json")).ex;
  const src = (m) => asset(`data/astro_${m}.webp`);
  let mode = "mean";
  b.src = src("sigclip");
  a.src = src(mode);
  await Promise.all([a, b].map((i) => i.decode().catch(() => {})));
  $(".vp-loading", fig).classList.add("done");
  for (const m of ["single", "median"]) new Image().src = src(m); // warm the cache for the other modes

  // ---- the wipe: σ-clip on the right of the divider, the chosen method on the left
  let x = reduced ? 0.5 : 1;
  const div = $("#ab-div", root), lblL = $("#ab-l", root), lblR = $(".ab-r", root);
  const setX = (v) => {
    x = clamp(v, 0, 1);
    fig.style.setProperty("--x", (x * 100).toFixed(2) + "%");
    div.setAttribute("aria-valuenow", Math.round(x * 100));
    lblL.style.opacity = x > 0.22 ? 1 : 0;   // each label only while its side is wide enough to hold it
    lblR.style.opacity = x < 0.78 ? 1 : 0;
  };
  setX(x);
  let intro = reduced ? -1 : 0, held = false, goal = -1;   // goal: where the divider is gliding to
  const fromEvent = (e) => { const r = fig.getBoundingClientRect(); setX((e.clientX - r.left) / r.width); };
  fig.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    held = true; intro = goal = -1;
    fig.setPointerCapture(e.pointerId);
    fig.classList.add("dragging");
    fromEvent(e);
  });
  fig.addEventListener("pointermove", (e) => held && fromEvent(e));
  const up = () => { held = false; fig.classList.remove("dragging"); };
  fig.addEventListener("pointerup", up);
  fig.addEventListener("pointercancel", up);
  div.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -0.04, ArrowRight: 0.04, Home: -1, End: 1 }[e.key];
    if (step === undefined) return;
    e.preventDefault();
    intro = goal = -1;
    setX(x + step);
  });
  // first view: the clean stack wipes in from the right
  let started = false;
  whenNear(fig, () => (started = true), "-25%");
  const glide = loop(fig, (dt) => {
    if (goal >= 0) {
      setX(x + (goal - x) * (reduced ? 1 : 1 - Math.exp(-dt * 7)));
      if (Math.abs(goal - x) < 0.002) setX(goal), (goal = -1);
    }
    if (intro < 0 || !started) return;
    intro = Math.min(1, intro + dt / 1.8);
    setX(1 - 0.5 * easeOut(intro));
    if (intro >= 1) intro = -1;
  });

  // ---- what is where: labels on the trails, on the left side only
  tags.innerHTML = ex.trails.map((t) => `<span class="ab-tag mono" data-k="${t.kind}" data-sub="${t.sub}" style="left:${(t.x * 100).toFixed(2)}%;top:${(t.y * 100).toFixed(2)}%"><i></i>${t.kind}</span>`).join("");
  const tagEls = $$(".ab-tag", tags);

  // ---- the scorecard
  const max = Math.max(...ROWS.map(([k]) => ex.noise[k]));
  $("#ab-score", root).innerHTML = `<div class="sc-h mono"><span></span><span>noise, e⁻ · lower is better</span><span>streaks</span></div>` + ROWS.map(([k, name]) =>
    `<div class="sc-row mono${k === "sig" ? " rw" : ""}" data-k="${k}"><span class="nm">${name}<em>${k === "sig" ? "right" : "left"}</em></span>` +
    `<span class="nz"><span class="trk"><i style="width:${(ex.noise[k] / max * 100).toFixed(1)}%"></i></span><b>${ex.noise[k].toFixed(1)}</b></span>` +
    `<span class="tr ${ex.streaks[k] ? "bad" : "ok"}">${ex.streaks[k] || "0 ✓"}</span></div>`).join("");

  const cap = $("#ab-cap", root);
  function show(m) {
    mode = m;
    a.src = src(m);
    lblL.textContent = MODES[m].lbl;
    cap.innerHTML = MODES[m].cap;
    $$(".sc-row", root).forEach((r) => r.classList.toggle("on", r.dataset.k === m));
    // the streaks the left image actually has
    tagEls.forEach((t) => t.classList.toggle("on", m === "mean" || (m === "single" && +t.dataset.sub === ex.single)));
  }
  segmented($("#ab-mode", root), (d) => show(d.m));
  show(mode);

  // ---- one pixel on a trail: its 40 values, what sigma clipping keeps, and where each method lands
  const svg = $("#px-chart", root);
  const pick = $("#px-pick", root);
  pick.innerHTML = ex.pixels.map((p, i) => `<button data-i="${i}"${i ? "" : ' class="on"'}>${p.kind}</button>`).join("");
  let cur = 0;
  segmented(pick, (d) => {
    cur = +d.i;
    draw();
    // slide the divider so the picked pixel's trail is in view on the left
    const t = ex.trails.find((s) => s.sub === ex.pixels[cur].sub);
    if (t && mode === "mean" && x < t.x + 0.12) { intro = -1; goal = Math.min(0.94, t.x + 0.18); glide.kick(); }
  });
  function draw() {
    const p = ex.pixels[cur], v = p.v, n = v.length;
    // what rankwise kept and returned (from the build); the ±3σ bounds of its last iteration from clip.js
    const r = sigmaClip(v, n, 3, 5), kept = p.kept, clipped = p.sigclip;
    const mean = v.reduce((s, x) => s + x, 0) / n;
    $("#px-t", root).textContent = `one pixel on the ${p.kind} · 40 subs`;
    $("#px-r", root).textContent = `${n - kept.reduce((s, k) => s + k, 0)} rejected`;
    tagEls.forEach((t) => t.classList.toggle("sel", +t.dataset.sub === p.sub));
    const W = Math.max(240, svg.clientWidth || 320), H = 176, L = 6, R = 84, T = 12, B = 16;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    // broken axis: the kept values, the ±3σ band and both results on a linear scale below the break,
    // outliers far above it on a log scale
    const keptV = v.filter((_, i) => kept[i]);
    const a0 = Math.min(...keptV, r.lo), a1 = Math.max(...keptV, r.hi, mean), pad = (a1 - a0) * 0.12;
    const lo0 = a0 - pad, hi0 = a1 + pad, top = Math.max(...v), broken = top > hi0;
    const yb = broken ? T + 0.3 * (H - T - B) : T - 6;
    const y = (u) => u <= hi0 ? yb + 6 + (1 - (u - lo0) / (hi0 - lo0)) * (H - B - yb - 6)
      : T + (1 - Math.log(u / hi0) / Math.log((top * 1.02) / hi0)) * (yb - 6 - T);
    if (broken) for (const bx of [L, W - R]) el("path", { d: `M${bx - 4} ${yb + 4}l8 -6M${bx - 4} ${yb + 8}l8 -6`, stroke: "rgba(255,255,255,0.35)", "stroke-width": 1.2 }, svg);
    const xs = (i) => L + (i + 0.5) * ((W - L - R) / n);
    el("rect", { x: L, width: W - L - R, y: y(r.hi), height: Math.max(2, y(r.lo) - y(r.hi)), rx: 3, fill: "rgba(79,141,255,0.16)", stroke: "rgba(79,141,255,0.4)", "stroke-dasharray": "3 3" }, svg);
    const line = (u, color, dash, text) => {
      el("line", { x1: L, x2: W - R + 4, y1: y(u), y2: y(u), stroke: color, "stroke-width": 1.5, "stroke-dasharray": dash }, svg);
      const t = el("text", { x: W - R + 10, y: y(u) + 3.5, fill: color, "font-size": 10.5, "font-family": "JetBrains Mono Variable, monospace" }, svg);
      t.textContent = text;
    };
    // labels must not collide: push the mean's label clear of the σ-clip one
    line(mean, "#ff9f43", "5 4", "mean");
    line(clipped, "#5ee6ff", "", "σ-clip");
    const lbls = svg.querySelectorAll("text");
    const dy = +lbls[1].getAttribute("y") - +lbls[0].getAttribute("y");
    if (Math.abs(dy) < 13) lbls[0].setAttribute("y", +lbls[1].getAttribute("y") - 13);
    const g = el("g", { filter: "url(#pxg)" }, svg);
    const defs = el("defs", {}, svg);
    defs.innerHTML = `<filter id="pxg" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
    for (let i = 0; i < n; i++) {
      const rej = !kept[i];
      if (rej) el("line", { x1: xs(i), x2: xs(i), y1: y(v[i]) + 5, y2: H - B + 4, stroke: "rgba(255,159,67,0.35)" }, svg);
      el("circle", { cx: xs(i), cy: y(v[i]), r: rej ? 4.6 : 2.8, fill: rej ? "#ff9f43" : "#5ee6ff" }, g);
      if (rej && i === p.sub) {
        const t = el("text", { x: xs(i) + (i > n / 2 ? -9 : 9), y: y(v[i]) + 4, fill: "#ff9f43", "font-size": 10.5, "font-family": "JetBrains Mono Variable, monospace", "text-anchor": i > n / 2 ? "end" : "start" }, svg);
        t.textContent = `${p.kind} · sub ${i + 1}`;
      }
    }
    const ax = el("text", { x: L, y: H - 2, fill: "rgba(255,255,255,0.4)", "font-size": 10, "font-family": "JetBrains Mono Variable, monospace" }, svg);
    ax.textContent = "sub 1";
    const ax2 = el("text", { x: W - R, y: H - 2, fill: "rgba(255,255,255,0.4)", "font-size": 10, "font-family": "JetBrains Mono Variable, monospace", "text-anchor": "end" }, svg);
    ax2.textContent = "40";
    $("#px-vals", root).innerHTML = `the ${p.kind} adds <b class="o">${fmt(mean - clipped)} e⁻</b> to the mean · σ-clip rejects it`;
  }
  new ResizeObserver(draw).observe(svg);
  draw();

  miniRace($("#ab-race", root), [
    { tool: "rankwise · AVX-512", s: 0.23, rw: true },
    { tool: "astropy sigma_clip", s: 4.82 },
    { tool: "ccdproc Combiner · 1 pass", s: 9.54 },
  ], "σ-clip of 31 × 2048² · 3σ, ≤ 5 iterations");
}
