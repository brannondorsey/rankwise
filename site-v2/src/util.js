export const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
export const coarse = matchMedia("(pointer: coarse)").matches;
export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOut = (t) => 1 - Math.pow(1 - t, 3);
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];

export const base = import.meta.env.BASE_URL;
export const asset = (p) => base + p;

export function loadImage(src) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = asset(src);
  });
}

export const getJSON = (p) => fetch(asset(p)).then((r) => r.json());

/** Call cb once when el comes within margin of the viewport. */
export function whenNear(el, cb, margin = "400px") {
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) {
      io.disconnect();
      cb();
    }
  }, { rootMargin: margin });
  io.observe(el);
}

/** Track visibility of el; returns a getter. */
export function visibility(el, margin = "100px") {
  const st = { v: false, listeners: [] };
  new IntersectionObserver((es) => {
    for (const e of es) st.v = e.isIntersecting;
    st.listeners.forEach((f) => f(st.v));
  }, { rootMargin: margin }).observe(el);
  return st;
}

/** requestAnimationFrame loop that only runs while el is on screen and the tab is visible. */
export function loop(el, fn) {
  const vis = visibility(el);
  let raf = 0, last = 0;
  const tick = (t) => {
    raf = 0;
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 0.016;
    last = t;
    fn(dt, t / 1000);
    schedule();
  };
  const schedule = () => {
    if (!raf && vis.v && !document.hidden) raf = requestAnimationFrame(tick);
    else if (!vis.v || document.hidden) last = 0;
  };
  vis.listeners.push(schedule);
  document.addEventListener("visibilitychange", schedule);
  return { kick: schedule, get visible() { return vis.v; } };
}

/** Keep a renderer sized to its canvas' CSS box. */
export function autosize(renderer, canvas, onResize, maxDpr = 2) {
  const fit = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, maxDpr));
    renderer.setSize(w, h, false);
    onResize?.(w, h);
  };
  new ResizeObserver(fit).observe(canvas);
  fit();
  return fit;
}

/** Wire a range input so its filled track follows its value. */
export function rangeFill(input) {
  const f = () => {
    const p = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--p", p + "%");
  };
  input.addEventListener("input", f);
  f();
  return f;
}

export function segmented(root, onPick) {
  const bs = $$("button", root);
  bs.forEach((b) => b.addEventListener("click", () => {
    bs.forEach((x) => x.classList.toggle("on", x === b));
    onPick(b.dataset, b);
  }));
  return (pred) => bs.forEach((x) => x.classList.toggle("on", pred(x.dataset)));
}

export const fmt = (x, d = 0) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export function secs(s) {
  if (s < 0.1) return (s * 1000).toFixed(0) + " ms";
  if (s < 10) return s.toFixed(2) + " s";
  return s.toFixed(1) + " s";
}

export const svgNS = "http://www.w3.org/2000/svg";
export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

export const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
