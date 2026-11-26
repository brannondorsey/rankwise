// 04 · 1,057 conformance tests light up.
import { $, clamp, easeOut, fmt, loop, reduced, whenNear } from "./util.js";

const TESTS = 1057, COLS = 41, ROWS = 26;

export function initCorrect() {
  const cv = $("#tests"), card = cv.parentElement;
  const ctx = cv.getContext("2d");
  const order = [...Array(TESTS).keys()];
  for (let i = order.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }
  const lit = new Float32Array(TESTS);
  let t = reduced ? 10 : 0, started = false, W = 0, H = 0, dpr = 1;
  const fit = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
  };
  new ResizeObserver(fit).observe(cv);
  fit();
  whenNear(card, () => (started = true), "-20%");
  const n = $("#tests-n");
  loop(card, (dt, now) => {
    if (!started) return;
    t += dt;
    const done = Math.min(TESTS, Math.floor(easeOut(clamp(t / 2.4)) * TESTS));
    for (let k = 0; k < done; k++) lit[order[k]] = Math.min(1, lit[order[k]] + dt * 4);
    n.textContent = fmt(done);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const cw = W / COLS, ch = H / ROWS, s = Math.min(cw, ch) * 0.62;
    for (let i = 0; i < TESTS; i++) {
      const x = (i % COLS) * cw + (cw - s) / 2, y = Math.floor(i / COLS) * ch + (ch - s) / 2;
      const a = lit[i];
      const pulse = a >= 1 ? 0.85 + 0.15 * Math.sin(now * 2 + i * 0.37) : 1;
      ctx.fillStyle = a > 0 ? `rgba(${Math.round(94 + (79 - 94) * (i / TESTS))},${Math.round(230 - 90 * (i / TESTS))},255,${(0.25 + 0.75 * a) * pulse})` : "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, y, s, s, s * 0.25) : ctx.rect(x, y, s, s);
      ctx.fill();
    }
  });
}
