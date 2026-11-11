// 03 · Batcher's odd-even merge network (ported from rankwise/src/netgen.rs), pruned to the ranks
// you ask for. A column of values sweeps through; only the live comparators fire.
import { $, el, loop, rangeFill, reduced, segmented } from "./util.js";
import { batcher, layout, medianRanks, pruneMask, quantileRanks } from "./netgen.js";

const DY = 14, DX = 13, LEFT = 30, RIGHT = 54, TOP = 12;
const ramp = ["#14307a", "#2a6cff", "#5ee6ff", "#e8f6ff"];
function hex(c) { return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)); }
const rampRGB = ramp.map(hex);
function color(v) {
  const t = Math.min(0.9999, Math.max(0, v)) * (ramp.length - 1), i = Math.floor(t), f = t - i;
  const a = rampRGB[i], b = rampRGB[i + 1];
  return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`;
}

export function initNetwork() {
  const svg = $("#net-svg");
  const nIn = $("#net-n"), nOut = $("#net-n-out");
  const fill = rangeFill(nIn);
  let preset = "median", n = +nIn.value, net, lay, needed, keep, W, H;
  let cmpEls = [], wireEls = [], outEls = [], dots = [], vals = [], order = [], scan = -1, hold = 0, ci = 0;

  function presetRanks() {
    if (preset === "sort") return [...Array(n).keys()];
    if (preset === "median") return medianRanks(n);
    if (preset === "q") return quantileRanks(n, [0.1, 0.5, 0.9]);
    return [0, n - 1];
  }

  function build() {
    net = batcher(n);
    lay = layout(net);
    W = LEFT + lay.width * DX + RIGHT;
    H = TOP * 2 + (n - 1) * DY;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const defs = el("defs", {}, svg);
    defs.innerHTML = `<filter id="ng" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="2.4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
    const y = (w) => TOP + w * DY;
    wireEls = [];
    for (let w = 0; w < n; w++) wireEls.push(el("line", { class: "wire", x1: LEFT - 8, x2: W - RIGHT + 8, y1: y(w), y2: y(w) }, svg));
    const g = el("g", {}, svg);
    cmpEls = net.map(([a, b], i) => {
      const x = LEFT + (lay.pos[i] + 0.5) * DX;
      const c = el("g", { class: "cmp" }, g);
      el("line", { x1: x, x2: x, y1: y(a), y2: y(b) }, c);
      el("circle", { cx: x, cy: y(a), r: 2.6 }, c);
      el("circle", { cx: x, cy: y(b), r: 2.6 }, c);
      c._x = x;
      return c;
    });
    order = net.map((_, i) => i).sort((p, q) => lay.pos[p] - lay.pos[q]);
    // inputs and outputs
    outEls = [];
    for (let w = 0; w < n; w++) {
      el("rect", { class: "val", x: LEFT - 22, y: y(w) - 4.5, width: 9, height: 9, rx: 2, fill: "rgba(255,255,255,0.15)" }, svg);
      const o = el("g", { class: "out" }, svg);
      el("rect", { class: "hit", x: W - RIGHT + 2, y: y(w) - DY / 2, width: RIGHT - 2, height: DY }, o);
      const sq = el("rect", { class: "val", x: W - RIGHT + 13, y: y(w) - 4.5, width: 9, height: 9, rx: 2, fill: "rgba(255,255,255,0.08)" }, o);
      const t = el("text", { x: W - RIGHT + 27, y: y(w) + 3.5, "font-size": 9.5, "font-family": "JetBrains Mono Variable, monospace", fill: "#5ee6ff" }, o);
      o.addEventListener("click", () => {
        const s = new Set(needed);
        s.has(w) ? s.delete(w) : s.add(w);
        needed = [...s].sort((a, b) => a - b);
        preset = "custom";
        setPreset((d) => false);
        prune();
      });
      outEls.push({ sq, t });
    }
    dots = [];
    const dg = el("g", { filter: "url(#ng)" }, svg);
    for (let w = 0; w < n; w++) dots.push(el("circle", { r: 4, cx: -20, cy: y(w), fill: "#fff" }, dg));
    needed = presetRanks();
    prune();
    restart();
  }

  function prune() {
    keep = pruneMask(net, n, needed);
    const k = keep.filter(Boolean).length;
    cmpEls.forEach((c, i) => c.setAttribute("class", "cmp " + (keep[i] ? "keep" : "drop")));
    wireEls.forEach((w, i) => w.setAttribute("class", "wire" + (needed.includes(i) ? " need" : "")));
    outEls.forEach((o, i) => { o.t.textContent = needed.includes(i) ? `r${i}` : ""; });
    $("#net-kept").textContent = k;
    $("#net-total").textContent = net.length;
    const lbl = needed.length === n ? "every rank" : needed.length === 0 ? "no ranks" : needed.length > 4 ? `${needed.length} ranks` : `rank${needed.length > 1 ? "s" : ""} ${needed.join(", ")}`;
    $("#net-ranks").textContent = lbl;
    restart();
  }

  function restart() {
    vals = Array.from({ length: n }, () => Math.random());
    const y = (w) => TOP + w * DY;
    svg.querySelectorAll(":scope > rect.val").forEach((r, w) => r.setAttribute("fill", color(vals[w])));
    outEls.forEach((o) => { o.sq.setAttribute("fill", "rgba(255,255,255,0.08)"); });
    dots.forEach((d, w) => { d.setAttribute("cy", y(w)); d.setAttribute("fill", color(vals[w])); });
    scan = LEFT - 10;
    ci = 0;
    hold = 0;
  }

  const setPreset = segmented($("#net-preset"), (d) => {
    preset = d.p;
    needed = presetRanks();
    prune();
  });
  nIn.addEventListener("input", () => {
    n = +nIn.value;
    nOut.textContent = n;
    fill();
    if (preset === "custom") preset = "median", setPreset((d) => d.p === "median");
    build();
  });
  build();

  const speed = () => (W - LEFT - RIGHT) / 2.6;
  loop(svg, (dt) => {
    if (scan < 0) return;
    if (scan < W - RIGHT + 6) {
      scan += dt * speed() * (reduced ? 3 : 1);
      while (ci < order.length && cmpEls[order[ci]]._x <= scan) {
        const i = order[ci++];
        if (!keep[i]) continue;
        const [a, b] = net[i];
        if (vals[a] > vals[b]) [vals[a], vals[b]] = [vals[b], vals[a]];
        const c = cmpEls[i];
        c.classList.add("flash");
        setTimeout(() => c.classList.remove("flash"), 140);
        dots[a].setAttribute("fill", color(vals[a]));
        dots[b].setAttribute("fill", color(vals[b]));
      }
      dots.forEach((d) => d.setAttribute("cx", Math.min(scan, W - RIGHT + 6)));
      if (scan >= W - RIGHT + 6) {
        // land: needed ranks are exact, the others are whatever the pruned network left there
        const sorted = vals.slice().sort((p, q) => p - q);
        outEls.forEach((o, w) => {
          const ok = needed.includes(w);
          o.sq.setAttribute("fill", ok ? color(sorted[w]) : "rgba(255,255,255,0.08)");
        });
        dots.forEach((d, w) => d.setAttribute("opacity", needed.includes(w) ? 1 : 0.15));
      }
    } else {
      hold += dt;
      if (hold > 1.6) {
        dots.forEach((d) => d.setAttribute("opacity", 1));
        restart();
      }
    }
  });
}
