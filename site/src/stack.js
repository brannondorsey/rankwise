// 01 · Live stacking of the 40 synthetic subs on the GPU: mean, median and astropy-style sigma
// clipping per pixel and channel, the same rank-bounds algorithm rankwise runs per SIMD lane.
import * as THREE from "three";
import { $, autosize, clamp, coarse, easeOut, getJSON, loadImage, loop, rangeFill, reduced, segmented, whenNear } from "./util.js";
import { sigmaClip } from "./clip.js";
import { sliceChart } from "./slice.js";

const FRAG = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uAtlas;
uniform int uN, uMode, uFrame, uCols, uTile;
uniform float uSigma, uSat, uRej;
uniform vec3 uEnc;           // X1, C1, MAX
uniform vec3 uLo, uHi, uM;   // display stretch per channel
in vec2 vUv;
out vec4 outColor;

vec3 dec(vec3 c) {           // 8-bit code -> electrons
  c *= 255.0;
  vec3 lin = c / uEnc.y * uEnc.x;
  vec3 lg = uEnc.x * exp((c - uEnc.y) / (255.0 - uEnc.y) * log(uEnc.z / uEnc.x));
  return mix(lin, lg, step(uEnc.y, c));
}
vec3 frameAt(ivec2 p, int i) {
  ivec2 o = ivec2(i % uCols, i / uCols) * uTile;
  return texelFetch(uAtlas, o + p, 0).rgb;
}
vec3 mtf(vec3 m, vec3 x) { return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m); }

vec3 cd[40];   // this pixel's codes, 0..255

// k-th smallest (0-based, per channel) among codes in [lo, hi], by bisection over the 256 codes
vec3 kth(vec3 k, vec3 lo, vec3 hi) {
  vec3 a = vec3(0.0), b = vec3(255.0);
  for (int s = 0; s < 8; s++) {
    vec3 mid = floor((a + b) * 0.5);
    vec3 cnt = vec3(0.0);
    for (int i = 0; i < 40; i++) {
      if (i >= uN) break;
      vec3 c = cd[i];
      cnt += step(c, mid) * step(lo, c) * step(c, hi);
    }
    vec3 ge = step(k + 0.5, cnt);
    b = mix(b, mid, ge);
    a = mix(mid + 1.0, a, ge);
  }
  return a;
}

void main() {
  ivec2 p = ivec2(floor(vUv * float(uTile)));
  p.y = uTile - 1 - p.y;
  vec3 v;
  float removed = 0.0;
  if (uMode == 0) {
    v = dec(frameAt(p, uFrame));
  } else {
    for (int i = 0; i < 40; i++) { if (i >= uN) break; cd[i] = floor(frameAt(p, i) * 255.0 + 0.5); }
    float n = float(uN);
    if (uMode == 1) {
      v = vec3(0.0);
      for (int i = 0; i < 40; i++) { if (i >= uN) break; v += dec(cd[i] / 255.0); }
      v /= n;
    } else if (uMode == 2) {
      vec3 lo = vec3(-1.0), hi = vec3(256.0);
      vec3 a = dec(kth(vec3(floor((n - 1.0) * 0.5)), lo, hi) / 255.0);
      vec3 b = dec(kth(vec3(floor(n * 0.5)), lo, hi) / 255.0);
      v = 0.5 * (a + b);
    } else {
      // sigma clipping: kept values are always a contiguous code range [lo, hi]
      vec3 lo = vec3(-1.0), hi = vec3(256.0), cnt = vec3(n);
      for (int it = 0; it < 5; it++) {
        vec3 a = dec(kth(floor((cnt - 1.0) * 0.5), lo, hi) / 255.0);
        vec3 b = dec(kth(floor(cnt * 0.5), lo, hi) / 255.0);
        vec3 center = 0.5 * (a + b);
        vec3 s1 = vec3(0.0);
        for (int i = 0; i < 40; i++) { if (i >= uN) break; vec3 w = step(lo, cd[i]) * step(cd[i], hi); s1 += w * dec(cd[i] / 255.0); }
        vec3 mean = s1 / max(cnt, 1.0);
        vec3 s2 = vec3(0.0);
        for (int i = 0; i < 40; i++) { if (i >= uN) break; vec3 w = step(lo, cd[i]) * step(cd[i], hi); vec3 d = dec(cd[i] / 255.0) - mean; s2 += w * d * d; }
        vec3 sd = sqrt(s2 / max(cnt, 1.0));
        vec3 bl = center - uSigma * sd, bh = center + uSigma * sd;
        vec3 nlo = vec3(256.0), nhi = vec3(-1.0), nc = vec3(0.0);
        for (int i = 0; i < 40; i++) {
          if (i >= uN) break;
          vec3 x = dec(cd[i] / 255.0);
          vec3 w = step(bl, x) * step(x, bh);
          nc += w;
          nlo = mix(nlo, min(nlo, cd[i]), w);
          nhi = mix(nhi, max(nhi, cd[i]), w);
        }
        bool same = all(equal(nc, cnt));
        lo = nlo; hi = nhi; cnt = nc;
        if (same) break;
      }
      vec3 s = vec3(0.0);
      for (int i = 0; i < 40; i++) { if (i >= uN) break; vec3 w = step(lo, cd[i]) * step(cd[i], hi); s += w * dec(cd[i] / 255.0); }
      v = s / max(cnt, 1.0);
      if (uRej > 0.5) {
        for (int i = 0; i < 40; i++) {
          if (i >= uN) break;
          vec3 w = 1.0 - step(lo, cd[i]) * step(cd[i], hi);
          vec3 d = max(dec(cd[i] / 255.0) - v, 0.0) * w;
          removed = max(removed, max(d.r, max(d.g, d.b)));
        }
      }
    }
  }
  vec3 x = clamp((v - uLo) / (uHi - uLo), 0.0, 1.0);
  vec3 o = mtf(uM, x);
  float lum = (o.r + o.g + o.b) / 3.0;
  o = clamp(lum + (o - lum) * uSat, 0.0, 1.0);
  if (uRej > 0.5 && uMode == 3) {
    float l = removed < 400.0 ? 0.0 : clamp(log(removed) / log(40000.0), 0.0, 1.0);
    o = vec3(lum * 0.3) + l * vec3(1.0, 0.63, 0.16);
  }
  outColor = vec4(o, 1.0);
}`;

const VERT = /* glsl */ `
in vec3 position; in vec2 uv; out vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export function initStack() {
  const root = $("#stack");
  whenNear(root, () => start(root));
}

async function start(root) {
  const view = $("#stack-view", root), canvas = $("canvas", view);
  const meta = await getJSON("data/astro.json");
  const img = await loadImage("data/astro_subs.jpg");
  $(".vp-loading", view).classList.add("done");
  const N = meta.frames, T = meta.tile, cols = meta.cols, [X1, C1, MAX] = meta.enc;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // the shader already writes display values
  const tex = new THREE.Texture(img);
  Object.assign(tex, { flipY: false, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, colorSpace: THREE.NoColorSpace });
  tex.needsUpdate = true;

  const rt = new THREE.WebGLRenderTarget(T, T, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false });
  const st = meta.stretch;
  const uniforms = {
    uAtlas: { value: tex }, uN: { value: N }, uMode: { value: 3 }, uFrame: { value: 19 % N }, uCols: { value: cols }, uTile: { value: T },
    uSigma: { value: 3 }, uSat: { value: meta.sat }, uRej: { value: 0 }, uEnc: { value: new THREE.Vector3(X1, C1, MAX) },
    uLo: { value: new THREE.Vector3(st[0][0], st[1][0], st[2][0]) }, uHi: { value: new THREE.Vector3(st[0][1], st[1][1], st[2][1]) },
    uM: { value: new THREE.Vector3(st[0][2], st[1][2], st[2][2]) },
  };
  const quad = new THREE.PlaneGeometry(2, 2);
  const compute = new THREE.Mesh(quad, new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, uniforms, vertexShader: VERT, fragmentShader: FRAG }));
  const show = new THREE.Mesh(quad, new THREE.MeshBasicMaterial({ map: rt.texture }));
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const scA = new THREE.Scene(), scB = new THREE.Scene();
  scA.add(compute);
  scB.add(show);
  let dirty = true;
  autosize(renderer, canvas, () => (dirty = true), 2);

  // ---- controls
  const nIn = $("#stack-n"), sIn = $("#stack-sigma"), nOut = $("#stack-n-out"), sOut = $("#stack-sigma-out"), rejIn = $("#stack-rej");
  const fillN = rangeFill(nIn), fillS = rangeFill(sIn);
  const tag = $("#stack-tag"), nLabel = $("#stack-n-label"), sRow = $("#stack-sigma-row");
  const names = ["1 sub", "mean", "median", "σ-clip"];
  function sync() {
    const m = uniforms.uMode.value;
    const n = +nIn.value;
    if (m === 0) uniforms.uFrame.value = n - 1;
    else uniforms.uN.value = n;
    uniforms.uSigma.value = +sIn.value;
    uniforms.uRej.value = rejIn.checked ? 1 : 0;
    nOut.textContent = n;
    sOut.textContent = (+sIn.value).toFixed(1);
    nLabel.textContent = m === 0 ? "Sub #" : "Frames";
    sIn.disabled = m !== 3;
    rejIn.disabled = m !== 3;
    sRow.style.opacity = m === 3 ? 1 : 0.4;
    tag.textContent = m === 0 ? `sub ${n} of ${N} · 30 s` : `${names[m]} · ${n} frame${n > 1 ? "s" : ""}${m === 3 ? ` · ${(+sIn.value).toFixed(1)}σ` : ""}`;
    fillN();
    fillS();
    dirty = true;
    probe();
  }
  segmented($("#stack-mode"), (d) => {
    const m = +d.mode;
    const prev = uniforms.uMode.value;
    uniforms.uMode.value = m;
    if (m === 0 && prev !== 0) nIn.value = uniforms.uFrame.value + 1;
    if (m !== 0 && prev === 0) nIn.value = uniforms.uN.value;
    sync();
  });
  nIn.addEventListener("input", () => { stopIntro(); sync(); });
  sIn.addEventListener("input", sync);
  rejIn.addEventListener("change", sync);

  // ---- probe: decode the pixel's codes from a CPU copy of the tile column we need
  const cvs = document.createElement("canvas");
  cvs.width = img.width; cvs.height = img.height;
  const c2 = cvs.getContext("2d", { willReadFrequently: true });
  c2.drawImage(img, 0, 0);
  const dec = (c) => (c < C1 ? (c / C1) * X1 : X1 * Math.exp(((c - C1) / (255 - C1)) * Math.log(MAX / X1)));
  const chart = sliceChart($("#stack-slice"), { n: N, enc: (x) => x, max: 255 });
  let px = meta.probe[0], py = meta.probe[1];
  const cross = $(".probe-cross", view);
  const vals = new Float64Array(N), codes = new Float64Array(N);
  function probe() {
    for (let i = 0; i < N; i++) {
      const d = c2.getImageData((i % cols) * T + px, Math.floor(i / cols) * T + py, 1, 1).data;
      codes[i] = d[1];
      vals[i] = dec(d[1]);
    }
    const m = uniforms.uMode.value, n = m === 0 ? N : uniforms.uN.value;
    const r = sigmaClip(vals, n, uniforms.uSigma.value, 5);
    const toCode = (x) => (x < X1 ? (x / X1) * C1 : C1 + (255 - C1) * Math.log(Math.max(x, X1) / X1) / Math.log(MAX / X1));
    chart.draw({
      codes, n, kept: m === 3 ? r.kept : null, cur: m === 0 ? uniforms.uFrame.value : -1,
      band: m === 3 ? [toCode(Math.max(0, r.lo)), toCode(r.hi)] : null,
      line: m === 0 ? null : toCode(m === 1 ? avg(vals, n) : m === 2 ? r.center : r.mean),
    });
    $("#stack-px").textContent = `${px + meta.crop[1]}, ${py + meta.crop[0]}`;
    $("#stack-px-rej").textContent = m === 3 && r.rejected ? `${r.rejected} rejected` : "";
    cross.hidden = false;
    cross.style.left = ((px + 0.5) / T) * 100 + "%";
    cross.style.top = ((py + 0.5) / T) * 100 + "%";
  }
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    px = clamp(Math.floor(((e.clientX - r.left) / r.width) * T), 0, T - 1);
    py = clamp(Math.floor(((e.clientY - r.top) / r.height) * T), 0, T - 1);
    probe();
  };
  let down = false;
  canvas.addEventListener("pointerdown", (e) => { down = true; canvas.setPointerCapture(e.pointerId); pick(e); });
  canvas.addEventListener("pointermove", (e) => { if (down || (!coarse && e.buttons === 0 && e.shiftKey)) pick(e); });
  canvas.addEventListener("pointerup", () => (down = false));
  canvas.style.touchAction = "none";

  // ---- intro: frames 1 → 40 the first time it is seen
  let intro = reduced ? -1 : 0;
  function stopIntro() { intro = -1; }
  if (intro === 0) nIn.value = 1;
  sync();

  loop(view, (dt) => {
    if (intro >= 0) {
      const vis = view.getBoundingClientRect();
      if (vis.top < innerHeight * 0.75) {
        intro += dt / 3.2;
        const n = Math.max(1, Math.round(1 + easeOut(clamp(intro)) * (N - 1)));
        if (n !== +nIn.value) { nIn.value = n; sync(); }
        if (intro >= 1) intro = -1;
      }
    }
    if (!dirty) return;
    dirty = false;
    renderer.setRenderTarget(rt);
    renderer.render(scA, cam);
    renderer.setRenderTarget(null);
    renderer.render(scB, cam);
  });
}

function avg(a, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[i];
  return s / n;
}
