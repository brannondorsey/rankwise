// A small SVG chart of one slice: one dot per frame, kept vs rejected, the ±σ band and the result.
import { el } from "./util.js";

export function sliceChart(svg, { n, h = 150 }) {
  svg.removeAttribute("preserveAspectRatio");
  let W = 320, last = null;
  const defs = el("defs", {}, svg);
  defs.innerHTML = `<filter id="sg" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
  const band = el("rect", { fill: "rgba(79,141,255,0.16)", stroke: "rgba(79,141,255,0.45)", "stroke-dasharray": "3 3", rx: 3 }, svg);
  const base = el("line", { stroke: "rgba(255,255,255,0.08)" }, svg);
  const line = el("line", { stroke: "#5ee6ff", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
  const lineLbl = el("text", { fill: "#5ee6ff", "font-size": 10, "font-family": "JetBrains Mono Variable, monospace", "text-anchor": "start" }, svg);
  const g = el("g", { filter: "url(#sg)" }, svg);
  const dots = [];
  for (let i = 0; i < n; i++) dots.push(el("circle", { r: 3 }, g));
  const stems = el("g", {}, svg);
  svg.prepend(stems);

  function draw(d) {
    last = d;
    W = Math.max(200, svg.clientWidth || 320);
    svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
    const { codes, kept, band: b, line: ln, cur } = d;
    const m = d.n ?? n;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, codes[i]); hi = Math.max(hi, codes[i]); }
    if (b) { lo = Math.min(lo, b[0]); hi = Math.max(hi, b[1]); }
    const pad = Math.max(4, (hi - lo) * 0.12);
    lo -= pad; hi += pad;
    const y = (v) => 8 + (1 - (v - lo) / (hi - lo)) * (h - 16);
    const x = (i) => 8 + (i + 0.5) * ((W - 16) / n);
    base.setAttribute("x1", 8); base.setAttribute("x2", W - 8);
    base.setAttribute("y1", h - 1); base.setAttribute("y2", h - 1);
    if (b) {
      band.setAttribute("x", 6); band.setAttribute("width", W - 12);
      band.setAttribute("y", y(b[1])); band.setAttribute("height", Math.max(2, y(b[0]) - y(b[1])));
      band.style.opacity = 1;
    } else band.style.opacity = 0;
    if (ln != null) {
      line.setAttribute("x1", 6); line.setAttribute("x2", W - 6);
      line.setAttribute("y1", y(ln)); line.setAttribute("y2", y(ln));
      lineLbl.setAttribute("x", 8); lineLbl.setAttribute("y", y(ln) + 14);
      lineLbl.textContent = "result";
      line.style.opacity = lineLbl.style.opacity = 1;
    } else line.style.opacity = lineLbl.style.opacity = 0;
    stems.innerHTML = "";
    for (let i = 0; i < n; i++) {
      const c = dots[i];
      const inUse = i < m;
      const rej = kept && inUse && !kept[i];
      c.setAttribute("cx", x(i));
      c.setAttribute("cy", y(codes[i]));
      c.setAttribute("r", rej ? 4.6 : i === cur ? 5 : 3);
      c.setAttribute("fill", !inUse ? "rgba(255,255,255,0.14)" : rej ? "#ff9f43" : i === cur ? "#ffffff" : "#5ee6ff");
      if (rej) el("line", { x1: x(i), x2: x(i), y1: y(codes[i]) + 5, y2: h - 2, stroke: "rgba(255,159,67,0.35)", "stroke-width": 1 }, stems);
    }
  }
  new ResizeObserver(() => last && draw(last)).observe(svg);
  return { draw };
}
