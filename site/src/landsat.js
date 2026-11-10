// 05 · Every Landsat 8/9 pass over Seattle and Tacoma in 2023–2024 (90 scenes, real data via
// Microsoft Planetary Computer). Medians run per pixel on the GPU: with clouds kept (np.median) or
// with QA-flagged clouds and shadows as nodata (rw.median(..., nodata=0)).
import * as THREE from "three";
import { $, autosize, clamp, easeOut, getJSON, loadImage, loop, rangeFill, reduced, segmented } from "./util.js";
import { miniRace } from "./weather.js";

const FRAG = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uRGB, uMask;
uniform int uN, uMode, uFrame, uCols, uTile;
in vec2 vUv;
out vec4 outColor;
ivec2 at(ivec2 p, int i) { return ivec2(i % uCols, i / uCols) * uTile + p; }

// k-th smallest code (0-based, per channel) over the first uN scenes, skipping nodata if asked
vec3 kth(ivec2 p, vec3 k, bool skip) {
  vec3 a = vec3(0.0), b = vec3(255.0);
  for (int s = 0; s < 8; s++) {
    vec3 mid = floor((a + b) * 0.5);
    vec3 cnt = vec3(0.0);
    for (int i = 0; i < 90; i++) {
      if (i >= uN) break;
      ivec2 q = at(p, i);
      float ok = skip ? 1.0 - step(0.5, texelFetch(uMask, q, 0).r) : 1.0;
      vec3 c = floor(texelFetch(uRGB, q, 0).rgb * 255.0 + 0.5);
      cnt += ok * step(c, mid);
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
  vec3 o;
  if (uMode == 0) {
    ivec2 q = at(p, uFrame);
    o = texelFetch(uRGB, q, 0).rgb;
  } else {
    bool skip = uMode == 2;
    float n = 0.0;
    for (int i = 0; i < 90; i++) { if (i >= uN) break; n += skip ? 1.0 - step(0.5, texelFetch(uMask, at(p, i), 0).r) : 1.0; }
    if (n < 0.5) {
      float h = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 8.0));
      o = vec3(0.03 + 0.03 * h);
    } else {
      vec3 lo = kth(p, vec3(floor((n - 1.0) * 0.5)), skip);
      vec3 hi = mod(n, 2.0) > 0.5 ? lo : kth(p, vec3(floor(n * 0.5)), skip);
      o = (lo + hi) * 0.5 / 255.0;
    }
  }
  outColor = vec4(o, 1.0);
}`;

const VERT = /* glsl */ `in vec3 position; in vec2 uv; out vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export async function initLandsat() {
  const view = $("#ls-view"), canvas = $("canvas", view);
  const meta = await getJSON("data/landsat.json");
  const [rgbImg, maskImg] = await Promise.all([loadImage("data/landsat_scenes.webp"), loadImage("data/landsat_masks.png")]);
  $(".vp-loading", view).classList.add("done");
  const N = meta.scenes, T = meta.tile;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const mk = (img) => {
    const t = new THREE.Texture(img);
    Object.assign(t, { flipY: false, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, colorSpace: THREE.NoColorSpace });
    t.needsUpdate = true;
    return t;
  };
  const rt = new THREE.WebGLRenderTarget(T, T, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false });
  const uniforms = {
    uRGB: { value: mk(rgbImg) }, uMask: { value: mk(maskImg) }, uN: { value: N }, uMode: { value: 2 },
    uFrame: { value: 0 }, uCols: { value: meta.cols }, uTile: { value: T },
  };
  const quad = new THREE.PlaneGeometry(2, 2);
  const scA = new THREE.Scene(), scB = new THREE.Scene(), cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  scA.add(new THREE.Mesh(quad, new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, uniforms, vertexShader: VERT, fragmentShader: FRAG })));
  scB.add(new THREE.Mesh(quad, new THREE.MeshBasicMaterial({ map: rt.texture })));
  let dirty = true;
  autosize(renderer, canvas, () => (dirty = true), 2);

  // timeline of cloud cover
  const tl = $("#ls-timeline");
  tl.innerHTML = meta.cloud.map((c) => `<i style="height:${Math.max(4, c * 100)}%"></i>`).join("");
  const bars = [...tl.children];
  const nIn = $("#ls-n"), nOut = $("#ls-n-out"), tag = $("#ls-tag"), date = $("#ls-date"), nLabel = $("#ls-n-label");
  const fill = rangeFill(nIn);
  const fmtD = (s) => new Date(s + "T00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  function sync() {
    const m = uniforms.uMode.value, n = +nIn.value;
    if (m === 0) uniforms.uFrame.value = n - 1; else uniforms.uN.value = n;
    nOut.textContent = n;
    nLabel.textContent = m === 0 ? "Pass #" : "Scenes";
    bars.forEach((b, i) => { b.classList.toggle("in", m !== 0 && i < n); b.classList.toggle("cur", m === 0 && i === n - 1); });
    if (m === 0) {
      tag.textContent = `${fmtD(meta.dates[n - 1])} · ${Math.round(meta.cloud[n - 1] * 100)}% cloud`;
      date.textContent = fmtD(meta.dates[n - 1]);
    } else {
      tag.textContent = `${m === 1 ? "np.median" : "rw.median(nodata=0)"} · ${n} scene${n > 1 ? "s" : ""}`;
      date.textContent = `${meta.dates[0].slice(0, 7)} → ${meta.dates[n - 1].slice(0, 7)}`;
    }
    fill();
    dirty = true;
  }
  segmented($("#ls-mode"), (d) => {
    const m = +d.mode, prev = uniforms.uMode.value;
    uniforms.uMode.value = m;
    if (m === 0 && prev !== 0) nIn.value = 31; // a cloudy November pass
    if (m !== 0 && prev === 0) nIn.value = uniforms.uN.value;
    intro = -1;
    sync();
  });
  nIn.addEventListener("input", () => { intro = -1; sync(); });
  const pickTl = (e) => {
    const r = tl.getBoundingClientRect();
    nIn.value = clamp(Math.ceil(((e.clientX - r.left) / r.width) * N), 1, N);
    intro = -1;
    sync();
  };
  let dragTl = false;
  tl.addEventListener("pointerdown", (e) => { dragTl = true; tl.setPointerCapture(e.pointerId); pickTl(e); });
  tl.addEventListener("pointermove", (e) => dragTl && pickTl(e));
  tl.addEventListener("pointerup", () => (dragTl = false));

  let intro = reduced ? -1 : 0;
  if (intro === 0) nIn.value = 1;
  sync();
  loop(view, (dt) => {
    if (intro >= 0 && view.getBoundingClientRect().top < innerHeight * 0.75) {
      intro += dt / 4.5;
      const n = Math.max(1, Math.round(1 + easeOut(clamp(intro)) * (N - 1)));
      if (n !== +nIn.value) { nIn.value = n; sync(); }
      if (intro >= 1) intro = -1;
    }
    if (!dirty) return;
    dirty = false;
    renderer.setRenderTarget(rt);
    renderer.render(scA, cam);
    renderer.setRenderTarget(null);
    renderer.render(scB, cam);
  });

  miniRace($("#ls-race"), [
    { tool: "rankwise · uint16, nodata=0", s: 0.231, rw: true },
    { tool: "Bottleneck nanmedian", s: 4.91 },
    { tool: "NumPy nanmedian", s: 20.6 },
  ], "median of 90 × 2048² · 65% cloud");
}
