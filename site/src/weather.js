// 04 · Real NOAA GEFS 2 m temperature, 31 members, init 2026-11-01 00Z. p10/p50/p90 per cell come
// from rankwise.quantile (identical to np.quantile). The surface's height is the p90 − p10 spread,
// its colour the median; ridgelines run along each latitude row.
import * as THREE from "three";
import { $, autosize, clamp, el, getJSON, loadImage, loop, rangeFill, reduced, secs, whenNear } from "./util.js";

const CITIES = [
  ["Seattle", 47.61, -122.33], ["San Francisco", 37.77, -122.42], ["Los Angeles", 34.05, -118.24],
  ["Denver", 39.74, -104.99], ["Dallas", 32.78, -96.8], ["Chicago", 41.88, -87.63],
  ["Atlanta", 33.75, -84.39], ["New York", 40.71, -74.0], ["Miami", 25.76, -80.19],
];

const VERT = /* glsl */ `
precision highp float;
precision highp int;
uniform sampler2D uData;
uniform float uLead, uSpreadMax, uTmin, uTmax, uH;
uniform int uLeads, uRows, uCols;
uniform mat4 modelViewMatrix, projectionMatrix;
in vec3 position; in vec2 uv;
out float vT; out float vS; out vec2 vUv; out float vRow;
vec3 at(int lead, ivec2 c) { return texelFetch(uData, ivec2(c.x, lead * uRows + c.y), 0).rgb; }
void main() {
  ivec2 c = ivec2(int(round(uv.x * float(uCols - 1))), int(round((1.0 - uv.y) * float(uRows - 1))));
  int l0 = int(floor(uLead)); int l1 = min(l0 + 1, uLeads - 1); float f = uLead - float(l0);
  vec3 q = mix(at(l0, c), at(l1, c), f) * (uTmax - uTmin) + uTmin;  // p10, p50, p90 in °C
  float spread = max(q.b - q.r, 0.0);
  vS = spread / uSpreadMax; vT = q.g; vUv = uv; vRow = float(c.y);
  vec3 p = position;
  p.z = pow(vS, 0.85) * uH;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp float;
uniform float uLine; uniform int uRows;
in float vT; in float vS; in vec2 vUv; in float vRow;
out vec4 outColor;
vec3 ramp(float t) {     // -10 … 30 °C
  vec3 c0 = vec3(0.17, 0.24, 1.0), c1 = vec3(0.16, 0.55, 1.0), c2 = vec3(0.28, 0.88, 1.0),
       c3 = vec3(0.91, 0.98, 1.0), c4 = vec3(1.0, 0.83, 0.42), c5 = vec3(1.0, 0.54, 0.24), c6 = vec3(1.0, 0.24, 0.43);
  float x = clamp((t + 10.0) / 40.0, 0.0, 1.0) * 6.0;
  if (x < 1.0) return mix(c0, c1, x);
  if (x < 2.0) return mix(c1, c2, x - 1.0);
  if (x < 3.0) return mix(c2, c3, x - 2.0);
  if (x < 4.0) return mix(c3, c4, x - 3.0);
  if (x < 5.0) return mix(c4, c5, x - 4.0);
  return mix(c5, c6, x - 5.0);
}
void main() {
  vec3 c = ramp(vT);
  if (uLine > 0.5) {
    outColor = vec4(c * (0.55 + 1.3 * vS), 0.55 + 0.45 * vS);
  } else {
    // dark surface tinted by temperature, brighter where the members disagree
    outColor = vec4(c * (0.06 + 0.5 * vS * vS), 1.0);
  }
}`;

export async function initWeather() {
  const view = $("#wx-view"), canvas = $("canvas", view);
  const meta = await getJSON("data/gefs.json");
  const img = await loadImage("data/gefs.jpg");
  const [leads, rows, cols] = meta.shape;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x020306, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);

  const tex = new THREE.Texture(img);
  Object.assign(tex, { flipY: false, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, colorSpace: THREE.NoColorSpace });
  tex.needsUpdate = true;

  // map extent: lon span scaled by cos(lat) so the US looks right
  const lat0 = meta.lat[0], lat1 = meta.lat[1], lon0 = meta.lon[0], lon1 = meta.lon[1];
  const kx = Math.cos(THREE.MathUtils.degToRad(38));
  const MW = 10, MH = MW * (lat0 - lat1) / ((lon1 - lon0) * kx);
  const uniforms = {
    uData: { value: tex }, uLead: { value: 0 }, uSpreadMax: { value: Math.min(14, meta.spread_max) },
    uTmin: { value: meta.tmin }, uTmax: { value: meta.tmax }, uH: { value: 1.25 },
    uLeads: { value: leads }, uRows: { value: rows }, uCols: { value: cols }, uLine: { value: 0 },
  };
  const surfGeo = new THREE.PlaneGeometry(MW, MH, cols - 1, rows - 1);
  const surf = new THREE.Mesh(surfGeo, new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms, vertexShader: VERT, fragmentShader: FRAG,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  }));
  // ridgelines: every second latitude row
  const pts = [];
  for (let r = 0; r < rows; r += 2) {
    for (let c = 0; c < cols - 1; c++) {
      for (const cc of [c, c + 1]) pts.push((cc / (cols - 1) - 0.5) * MW, (0.5 - r / (rows - 1)) * MH, 0, cc / (cols - 1), 1 - r / (rows - 1));
    }
  }
  const lg = new THREE.BufferGeometry();
  const buf = new Float32Array(pts);
  const ib = new THREE.InterleavedBuffer(buf, 5);
  lg.setAttribute("position", new THREE.InterleavedBufferAttribute(ib, 3, 0));
  lg.setAttribute("uv", new THREE.InterleavedBufferAttribute(ib, 2, 3));
  const lineU = { ...uniforms, uLine: { value: 1 } };
  const lines = new THREE.LineSegments(lg, new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms: lineU, vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  const map = new THREE.Group();
  map.add(surf, lines);
  map.rotation.x = -Math.PI / 2;
  scene.add(map);

  // city markers projected to HTML labels
  const labelBox = $("#wx-labels", view);
  const cities = CITIES.map(([name, la, lo]) => {
    const d = document.createElement("div");
    d.className = "wx-label";
    d.textContent = name;
    labelBox.appendChild(d);
    return { name, d, x: ((lo - lon0) / (lon1 - lon0) - 0.5) * MW, y: (0.5 - (lat0 - la) / (lat0 - lat1)) * MH, la, lo };
  });

  let aspect = 1, W = 1, H = 1;
  autosize(renderer, canvas, (w, h) => { aspect = w / h; W = w; H = h; camera.aspect = aspect; camera.updateProjectionMatrix(); }, 2);

  // time controls
  const lead = $("#wx-lead"), out = $("#wx-lead-out"), play = $("#wx-play"), tag = $("#wx-tag");
  const fill = rangeFill(lead);
  let playing = !reduced, hours = 0, drag = false, yaw = 0, yawT = 0, lastX = 0;
  const setPlay = (p) => { playing = p; play.textContent = p ? "❚❚" : "▶"; play.setAttribute("aria-label", p ? "Pause" : "Play"); };
  setPlay(playing);
  play.addEventListener("click", () => setPlay(!playing));
  lead.addEventListener("input", () => { hours = +lead.value; setPlay(false); });
  canvas.addEventListener("pointerdown", (e) => { drag = true; lastX = e.clientX; });
  addEventListener("pointerup", () => (drag = false));
  canvas.addEventListener("pointermove", (e) => { if (drag) { yawT = clamp(yawT + (e.clientX - lastX) * 0.004, -0.6, 0.6); lastX = e.clientX; } });
  canvas.style.touchAction = "pan-y";

  const fan = fanChart($("#fan"), meta.denver);
  const v = new THREE.Vector3();
  let t0 = 0;
  loop(view, (dt) => {
    t0 += dt;
    if (playing) hours = (hours + dt * 24) % 241;
    lead.value = Math.round(hours);
    fill();
    const day = hours / 24;
    out.textContent = `day ${day.toFixed(1)}`;
    tag.textContent = `+${Math.round(hours)} h · p90 − p10`;
    const li = clamp(hours / 6, 0, leads - 1);
    uniforms.uLead.value = lineU.uLead.value = li;
    fan.mark(hours);

    // camera: low oblique view from the south, gentle sway
    yaw += (yawT + (reduced ? 0 : Math.sin(t0 * 0.15) * 0.1) - yaw) * 0.05;
    const portrait = aspect < 1.25;
    const dist = Math.max(12, 5.55 / (Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * aspect));
    const el_ = portrait ? 0.9 : 0.78;
    camera.position.set(Math.sin(yaw) * dist * Math.cos(el_), dist * Math.sin(el_), Math.cos(yaw) * dist * Math.cos(el_));
    camera.lookAt(0, 0, portrait ? 0.2 : 0.35);

    renderer.render(scene, camera);
    for (const c of cities) {
      v.set(c.x, 0.05, -c.y).project(camera);
      c.d.style.left = ((v.x + 1) / 2) * W + "px";
      c.d.style.top = ((1 - v.y) / 2) * H + "px";
      c.d.style.opacity = portrait && (c.name === "San Francisco" || c.name === "Atlanta") ? 0 : 1;
    }
  });

  miniRace($("#wx-race"), [
    { tool: "rankwise · AVX-512", s: 0.032, rw: true },
    { tool: "fastnanquantile", s: 2.3 },
    { tool: "NumPy quantile", s: 2.62 },
  ], "p10 · p50 · p90 of 31 × 81 × 121 × 257");
}

function fanChart(svg, d) {
  svg.removeAttribute("preserveAspectRatio");
  const H = 200;
  let W = 360;
  const g = el("g", {}, svg);
  let marker, lbl, xs, ys;
  const val = $("#fan-val");
  function draw() {
    W = Math.max(240, svg.clientWidth || 360);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    g.innerHTML = "";
    const n = d.lead_h.length, tmax = d.lead_h[n - 1];
    let lo = Infinity, hi = -Infinity;
    for (const m of d.members) for (const x of m) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
    lo = Math.floor(lo - 1); hi = Math.ceil(hi + 1);
    xs = (h) => 26 + (h / tmax) * (W - 34);
    ys = (t) => 8 + (1 - (t - lo) / (hi - lo)) * (H - 26);
    for (let t = Math.ceil(lo / 10) * 10; t <= hi; t += 10) {
      el("line", { x1: 26, x2: W - 8, y1: ys(t), y2: ys(t), stroke: "rgba(255,255,255,0.06)" }, g);
      el("text", { x: 0, y: ys(t) + 3.5, "font-size": 10, fill: "#5f697b", "font-family": "JetBrains Mono Variable, monospace" }, g).textContent = `${t}°`;
    }
    for (let day = 0; day <= 10; day += 2) el("text", { x: xs(day * 24), y: H - 2, "font-size": 10, fill: "#5f697b", "text-anchor": "middle", "font-family": "JetBrains Mono Variable, monospace" }, g).textContent = `d${day}`;
    const path = (arr) => arr.map((t, i) => `${i ? "L" : "M"}${xs(d.lead_h[i]).toFixed(1)},${ys(t).toFixed(1)}`).join("");
    const area = (a, b) => path(a) + b.map((t, i) => `L${xs(d.lead_h[b.length - 1 - i]).toFixed(1)},${ys(b[b.length - 1 - i]).toFixed(1)}`).join("") + "Z";
    const rev = (a) => a;
    el("path", { d: area(d.fan[0], rev(d.fan[4]).slice()), fill: "rgba(79,141,255,0.16)" }, g);
    el("path", { d: area(d.fan[1], d.fan[3].slice()), fill: "rgba(79,141,255,0.24)" }, g);
    for (const m of d.members) el("path", { d: path(m), fill: "none", stroke: "rgba(255,255,255,0.16)", "stroke-width": 0.8 }, g);
    el("path", { d: path(d.fan[2]), fill: "none", stroke: "#5ee6ff", "stroke-width": 2.2, filter: "drop-shadow(0 0 4px rgba(94,230,255,0.7))" }, g);
    marker = el("line", { y1: 4, y2: H - 16, stroke: "rgba(255,255,255,0.5)", "stroke-dasharray": "2 3" }, g);
    lbl = el("circle", { r: 4, fill: "#5ee6ff", stroke: "#020306", "stroke-width": 2 }, g);
  }
  draw();
  new ResizeObserver(draw).observe(svg);
  let lastH = -1;
  return {
    mark(h) {
      if (!marker) return;
      const x = xs(h);
      marker.setAttribute("x1", x); marker.setAttribute("x2", x);
      const i = Math.min(d.lead_h.length - 1, Math.round(h / 3));
      lbl.setAttribute("cx", xs(d.lead_h[i])); lbl.setAttribute("cy", ys(d.fan[2][i]));
      if (i !== lastH) {
        lastH = i;
        val.textContent = `p50 ${d.fan[2][i].toFixed(1)}° · p10–p90 ${(d.fan[4][i] - d.fan[0][i]).toFixed(1)}°`;
      }
    },
  };
}

/** A compact real-time race for a case study, started when it scrolls into view. */
export function miniRace(root, rows, caption) {
  root.innerHTML = `<div class="slice-head mono"><span>${caption}</span><span class="dim">1 core</span></div>` + rows.map((r) => `<div class="mr-row${r.rw ? " rw" : ""}"><span>${r.tool}</span><span class="t">—</span><div class="mr-bar"><i></i></div></div>`).join("");
  const els = [...root.querySelectorAll(".mr-row")].map((e) => ({ t: e.querySelector(".t"), i: e.querySelector("i") }));
  const max = Math.max(...rows.map((r) => r.s));
  let t = 0, started = false;
  whenNear(root, () => (started = true), "-15%");
  loop(root, (dt) => {
    if (!started || t > max + 0.5) return;
    t += dt;
    rows.forEach((r, k) => {
      const f = Math.min(1, t / r.s);
      els[k].i.style.width = (f * r.s / max) * 100 + "%";
      els[k].t.textContent = f >= 1 ? secs(r.s) : "…";
    });
  });
}
