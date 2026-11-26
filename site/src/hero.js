// Hero: the 40 synthetic subs as a 3D deck. A probe, a cyan needle of light, raster-scans the deck
// along a serpentine path (rows top first, joined by half-circle turns), skewering one pixel through
// every frame and sigma-clipping its 40 values live: one glint per frame, cyan if kept, orange and
// ringed if rejected. Behind the scan head each pixel stack counts as reduced to one clean value: a
// soft round footprint along the swept path clears the noisy frames there to faint violet glass,
// rankwise's stacked frame shows through mid-deck, and a violet thread of light (one bead per reduced
// stack) cools off behind the head. After the last row it holds, the deck re-forms in a wave from back
// to front and the scan restarts. Hovering (fine pointers) pauses the scan and hands the probe to the
// pointer; reduced motion freezes it mid-scan.
import * as THREE from "three";
import { $, autosize, coarse, getJSON, loadImage, loop, reduced, smoothstep } from "./util.js";
import { sigmaClip } from "./clip.js";

const W = 4.2, H = 2.8, GAP = 0.15;
const BANDS = 12;                                   // scan rows
const ROW_S = 1.0;                                  // seconds per row
const INTRO_S = 0.6, HOLD_S = 2.2, RESET_S = 1.6;

// The scan path in plane units (x right, y down from the top-left corner): serpentine rows joined by
// half-circle turns, walked at constant speed. s = path length walked.
const PH = H / BANDS, PR = PH / 2, XA = 0.035 * W, XB = W - XA;
const RL = XB - XA, SEG = RL + Math.PI * PR, TOTAL = BANDS * RL + (BANDS - 1) * Math.PI * PR;
const SPEED = RL / ROW_S;
function pathAt(s, out) {
  s = Math.min(TOTAL, Math.max(0, s));
  const r = Math.min(BANDS - 1, Math.floor(s / SEG)), t = s - r * SEG, yc = (r + 0.5) * PH;
  const even = r % 2 === 0;
  if (t <= RL) return out.set(even ? XA + t : XB - t, yc);
  const a = (t - RL) / PR - Math.PI / 2;            // -pi/2 (this row) .. pi/2 (the next)
  return out.set((even ? XB : XA) + (even ? 1 : -1) * PR * Math.cos(a), yc + PR + PR * Math.sin(a));
}

// Soft footprint of the swept path: fully reduced within R0 of it, feathered out to R1 (plane units).
const R0 = 0.13, R1 = 0.24;
const f = (x) => x.toFixed(5);
const PATH = /* glsl */ `
  const float B = ${f(BANDS)}, PH = ${f(PH)}, PR = ${f(PR)}, XA = ${f(XA)}, XB = ${f(XB)}, RL = ${f(RL)}, SEG = ${f(SEG)};
  const float R0 = ${f(R0)}, R1 = ${f(R1)};
  const vec2 WH = vec2(${f(W)}, ${f(H)});
  uniform float uS, uGrow, uReset;   // path swept so far; head footprint scale; re-forming the deck, 0..1
  uniform vec2 uHead;                // scan head, plane units
  // nearest point of row r to p: (distance, path length there); clampX measures only across the row
  vec2 rowNear(vec2 p, float r, float clampX) {
    float yc = (r + 0.5) * PH, x = clamp(p.x, XA, XB);
    float d = mix(length(vec2(p.x - x, p.y - yc)), abs(p.y - yc), clampX);
    return vec2(d, r * SEG + (mod(r, 2.0) < 0.5 ? x - XA : XB - x));
  }
  // how far p is from the swept part of the path (the deck's side margins count with their row),
  // and how long ago (in path length) the head passed there
  float sweptDist(vec2 p, out float age) {
    float d = length(p - uHead) / max(uGrow, 1e-3);
    float b0 = clamp(floor(p.y / PH), 0.0, B - 1.0);
    age = 0.0;
    for (int k = -1; k <= 1; k++) {
      float r = b0 + float(k);
      if (r < 0.0 || r > B - 1.0) continue;
      vec2 n = rowNear(p, r, 1.0);
      if (n.y <= uS && n.x < d) { d = n.x; age = uS - n.y; }
    }
    return d;
  }
  float swept(vec2 p, out float age) { return 1.0 - smoothstep(R0, R1, sweptDist(p, age)); }
  // the deck re-forms in a wave from back to front (depth 0 = back frame, 1 = front frame)
  float reformT(float depth) { return clamp((uReset - 0.82 * depth) / 0.18, 0.0, 1.0); }
  float reform(float depth) { float t = reformT(depth); return t * t * (3.0 - 2.0 * t); }
  vec3 toOut(vec3 c) { return linearToOutputTexel(vec4(c, 1.0)).rgb; }`;

export async function initHero() {
  const root = $("#hero");
  const canvas = $(".hero-canvas", root);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x04060a, 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 200);
  camera.position.set(0, 0, 15);

  const meta = await getJSON("data/astro.json");
  const [subsImg, stackImg] = await Promise.all([loadImage("data/hero_subs.jpg"), loadImage("data/hero_stack.jpg")]);
  const N = meta.frames, cols = meta.hero.cols, rows = Math.ceil(N / cols);

  const tex = new THREE.Texture(subsImg);
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  const stackTex = new THREE.Texture(stackImg);
  stackTex.colorSpace = THREE.SRGBColorSpace;
  stackTex.anisotropy = 4;
  stackTex.needsUpdate = true;

  // CPU copy at half size for the probe
  const cw = subsImg.width >> 1, ch = subsImg.height >> 1;
  const cv = new OffscreenCanvasOr(cw, ch);
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.drawImage(subsImg, 0, 0, cw, ch);
  const pix = cx.getImageData(0, 0, cw, ch).data;
  const tw = cw / cols, th = ch / rows;

  const deck = new THREE.Group();
  scene.add(deck);

  // shared by every layer of the deck; plane units, x right, y down
  const scanU = {
    uS: { value: -1 }, uGrow: { value: 0 }, uReset: { value: 0 }, uHead: { value: new THREE.Vector2(XA, PR) },
    uProbe: { value: new THREE.Vector2(XA, PR) }, uBeam: { value: 0 },
  };

  // ---- frames: two instanced draws (behind and in front of the stacked frame), z from a uniform
  const uniforms = {
    ...scanU,
    uAtlas: { value: tex }, uGrid: { value: new THREE.Vector2(cols, rows) }, uN: { value: N },
    uGap: { value: GAP }, uOpacity: { value: 0.92 }, uTime: { value: 0 }, uHot: { value: new Float32Array(64) },
  };
  const frameMat = new THREE.ShaderMaterial({
    uniforms, transparent: true, premultipliedAlpha: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      attribute float aIdx;
      uniform float uGap, uN, uTime;
      varying vec2 vUv; varying float vIdx; varying float vDepth;
      void main() {
        vUv = uv; vIdx = aIdx;
        vec3 p = position;
        p.z += (aIdx - (uN - 1.0) * 0.5) * uGap + 0.025 * sin(uTime * 1.3 + aIdx * 0.45);
        vDepth = aIdx / (uN - 1.0);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      ${PATH}
      uniform sampler2D uAtlas; uniform vec2 uGrid, uProbe; uniform float uOpacity, uBeam; uniform float uHot[64];
      varying vec2 vUv; varying float vIdx; varying float vDepth;
      void main() {
        vec2 tuv = vec2(vUv.x, 1.0 - vUv.y), p = tuv * WH;   // top-down, like the probe
        float col = mod(vIdx, uGrid.x), row = floor(vIdx / uGrid.x);
        vec3 c = texture2D(uAtlas, (vec2(col, row) + tuv) / uGrid).rgb;
        float shade = 0.55 + 0.45 * vDepth;
        vec2 e = min(p, WH - p);
        float edge = 1.0 - smoothstep(0.0, 0.016, min(e.x, e.y));
        float hot = uHot[int(vIdx)];

        // reduced stacks: their noisy values are spent, so the frames there clear to glass
        float age;
        float done = swept(p, age) * (1.0 - reform(vDepth));
        float a = uOpacity * mix(0.55, 1.0, vDepth) * (1.0 - done);
        // the cleared frames turn to violet glass, lit at their rims, with a wake just behind the head
        float rim = exp(-min(e.x, e.y) * 12.0);
        vec3 add = vec3(0.42, 0.36, 1.0) * (0.003 + 0.013 * rim + 0.017 * exp(-age * ${f(1 / (SPEED * 0.5))})) * done;
        add += edge * mix(vec3(0.3, 0.5, 0.9), vec3(0.45, 0.38, 1.0), done) * (0.4 - 0.2 * done) * shade;
        add += (edge + rim * 0.15) * vec3(0.5, 0.45, 1.0) * sin(3.14159 * reformT(vDepth)) * 0.6;   // each frame flashes as it returns

        // the probe pierces every frame: a faint cyan halo; a rejected value rings orange
        float d = length(p - uProbe), d2 = d * d;
        add += uBeam * vec3(0.37, 0.9, 1.0) * exp(-d2 * 60.0) * 0.02;
        add += uBeam * hot * vec3(1.0, 0.58, 0.22) * (exp(-pow((d - 0.1) * 60.0, 2.0)) * 0.7 + exp(-d2 * 25.0) * 0.08);
        gl_FragColor = vec4(toOut(c * shade) * a + add, a);   // light adds in display space
      }`,
  });
  const plane = new THREE.PlaneGeometry(W, H);
  const makeFrames = (i0, n) => {
    const g = new THREE.InstancedBufferGeometry().copy(plane);
    g.setAttribute("aIdx", new THREE.InstancedBufferAttribute(Float32Array.from({ length: n }, (_, i) => i0 + i), 1));
    g.instanceCount = n;
    const m = new THREE.Mesh(g, frameMat);
    m.frustumCulled = false;
    deck.add(m);
    return m;
  };
  const half = N >> 1;
  makeFrames(0, half).renderOrder = 0;
  makeFrames(half, N - half).renderOrder = 2;

  // ---- the stacked result, mid-deck: shown wherever the scan has been, with the cooling trail
  const DECAY = SPEED * 0.8, BEAD = 0.13;
  const result = new THREE.Mesh(plane, new THREE.ShaderMaterial({
    uniforms: { ...scanU, uMap: { value: stackTex }, uFlash: { value: 0 } },
    transparent: true, premultipliedAlpha: true, depthTest: false, depthWrite: false,
    vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      ${PATH}
      uniform sampler2D uMap; uniform vec2 uProbe; uniform float uBeam, uFlash;
      varying vec2 vUv;
      // one stretch of path: a thin thread with a soft afterglow, and a bead per reduced stack
      void acc(vec2 n, inout float thread, inout float bead) {
        float age = uS - n.y;
        if (age < -0.02) return;
        float k = exp(-max(age, 0.0) / ${f(DECAY)}) * smoothstep(-0.02, 0.0, age);
        thread = max(thread, k * exp(-n.x * n.x * 5000.0) + k * k * 0.25 * exp(-n.x * n.x * 120.0));
        float ds = (fract(n.y / ${f(BEAD)}) - 0.5) * ${f(BEAD)};
        bead = max(bead, k * exp(-(ds * ds + n.x * n.x) * 1500.0));
      }
      void main() {
        vec2 tuv = vec2(vUv.x, 1.0 - vUv.y), p = tuv * WH;
        vec3 c = texture2D(uMap, vUv).rgb * (1.0 + uFlash * 0.25);
        float fade = 1.0 - reform(0.5);
        float age;
        float a = swept(p, age) * fade;
        float thread = 0.0, bead = 0.0;
        float b0 = clamp(floor(p.y / PH), 0.0, B - 1.0);
        for (int k = -1; k <= 1; k++) {
          float r = b0 + float(k);
          if (r < 0.0 || r > B - 1.0) continue;
          acc(rowNear(p, r, 0.0), thread, bead);
        }
        // the half-circle turn beside p
        float kb = clamp(floor(p.y / PH + 0.5), 1.0, B - 1.0), r = kb - 1.0;
        float side = mod(r, 2.0) < 0.5 ? 1.0 : -1.0;
        vec2 v = p - vec2(side > 0.0 ? XB : XA, kb * PH);
        if (v.x * side > 0.0) acc(vec2(abs(length(v) - PR), r * SEG + RL + PR * (atan(v.y, abs(v.x)) + 1.5708)), thread, bead);
        vec3 add = (vec3(0.45, 0.36, 1.0) * thread * 0.4 + vec3(0.72, 0.66, 1.0) * bead * 0.7) * fade;
        // where the probe crosses the stacked frame, the clean value is written: a small white-cyan spark
        float d2 = dot(p - uProbe, p - uProbe);
        add += uBeam * vec3(0.6, 0.95, 1.0) * (exp(-d2 * 500.0) * 0.8 + exp(-d2 * 30.0) * 0.08);
        gl_FragColor = vec4(toOut(c) * a + add, a);
      }`,
  }));
  result.renderOrder = 1;
  deck.add(result);

  // ---- probe: a needle of light (a camera-facing ribbon with a soft profile) plus one glint per frame
  const probe = new THREE.Group();
  deck.add(probe);
  const rodGeo = new THREE.BufferGeometry();
  rodGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
  rodGeo.setAttribute("aT", new THREE.BufferAttribute(new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), 2));
  rodGeo.setIndex([0, 1, 2, 2, 1, 3]);
  const rodU = { uA: { value: new THREE.Vector3() }, uB: { value: new THREE.Vector3() }, uWidth: { value: 0.07 }, uBeam: scanU.uBeam, uPulse: { value: 1 } };
  const rod = new THREE.Mesh(rodGeo, new THREE.ShaderMaterial({
    uniforms: rodU, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute vec2 aT; uniform vec3 uA, uB; uniform float uWidth; varying vec2 vT;
      void main() {
        vec3 a = (modelViewMatrix * vec4(uA, 1.0)).xyz, b = (modelViewMatrix * vec4(uB, 1.0)).xyz;
        vec3 p = mix(a, b, aT.x);
        p += normalize(cross(b - a, p)) * aT.y * uWidth;
        vT = aT;
        gl_Position = projectionMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      uniform float uBeam, uPulse; varying vec2 vT;
      void main() {
        float s2 = vT.y * vT.y;
        float ends = smoothstep(0.0, 0.12, vT.x) * smoothstep(1.0, 0.88, vT.x);
        vec3 c = vec3(0.8, 0.98, 1.0) * exp(-s2 * 160.0) + vec3(0.2, 0.7, 1.0) * exp(-s2 * 7.0) * 0.24 * uPulse;
        gl_FragColor = vec4(c * ends * uBeam, 1.0);
      }`,
  }));
  rod.frustumCulled = false;
  const ppos = new Float32Array(N * 3), pcol = new Float32Array(N * 3), psize = new Float32Array(N), prej = new Float32Array(N);
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.BufferAttribute(ppos, 3));
  pgeo.setAttribute("color", new THREE.BufferAttribute(pcol, 3));
  pgeo.setAttribute("size", new THREE.BufferAttribute(psize, 1));
  pgeo.setAttribute("rej", new THREE.BufferAttribute(prej, 1));
  const pts = new THREE.Points(pgeo, new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1 }, uBeam: scanU.uBeam }, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute float size, rej; attribute vec3 color; varying vec3 vC; varying float vR; uniform float uScale;
      void main() { vC = color; vR = rej; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = size * uScale * (12.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: /* glsl */ `
      uniform float uBeam; varying vec3 vC; varying float vR;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0, d2 = d * d;
        float a = smoothstep(0.4, 0.16, d) + exp(-d2 * 4.0) * 0.45;
        a += vR * exp(-pow((d - 0.58) * 11.0, 2.0)) * 0.7;
        a *= 1.0 - smoothstep(0.8, 1.0, d);
        gl_FragColor = vec4(vC * a * uBeam, 1.0);
      }`,
  }));
  pts.frustumCulled = false;
  probe.add(rod, pts);
  rod.renderOrder = 3;
  pts.renderOrder = 4;

  // ---- distant star dust for depth: soft round motes that breathe slowly
  const dn = 700, dp = new Float32Array(dn * 3), dsz = new Float32Array(dn), dph = new Float32Array(dn);
  for (let i = 0; i < dn; i++) {
    dp[i * 3] = (Math.random() - 0.5) * 60;
    dp[i * 3 + 1] = (Math.random() - 0.5) * 40;
    dp[i * 3 + 2] = -10 - Math.random() * 40;
    dsz[i] = 1 + Math.random() * Math.random() * 3;
    dph[i] = Math.random();
  }
  const dust = new THREE.BufferGeometry();
  dust.setAttribute("position", new THREE.BufferAttribute(dp, 3));
  dust.setAttribute("aSize", new THREE.BufferAttribute(dsz, 1));
  dust.setAttribute("aPhase", new THREE.BufferAttribute(dph, 1));
  const dustU = { uTime: { value: 0 }, uScale: { value: 1 } };
  const dustPts = new THREE.Points(dust, new THREE.ShaderMaterial({
    uniforms: dustU, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      attribute float aSize, aPhase; uniform float uTime, uScale; varying float vA;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * uScale * (36.0 / -mv.z) + 1.0;
        vA = 0.7 + 0.3 * sin(uTime * (0.4 + aPhase) + aPhase * 50.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      varying float vA;
      void main() { float d = length(gl_PointCoord - 0.5) * 2.0; gl_FragColor = vec4(vec3(0.42, 0.52, 0.78) * exp(-d * d * 5.0) * vA * 0.75, 1.0); }`,
  }));
  dustPts.renderOrder = -1;
  scene.add(dustPts);

  // ---- probe values: sigma clip the 40 values of one pixel
  const vals = new Float64Array(N), hotT = new Float32Array(N), hot = uniforms.uHot.value;
  let lastKey = "";
  function sample(u, v) {
    const x = Math.min(tw - 1, Math.max(0, Math.floor(u * tw))), y = Math.min(th - 1, Math.max(0, Math.floor(v * th)));
    const key = x + "," + y;
    if (key === lastKey) return;
    lastKey = key;
    for (let i = 0; i < N; i++) {
      const ox = (i % cols) * tw + x, oy = Math.floor(i / cols) * th + y;
      const o = (oy * cw + ox) * 4;
      vals[i] = 0.3 * pix[o] + 0.55 * pix[o + 1] + 0.15 * pix[o + 2];
    }
    const r = sigmaClip(vals, N, 3, 5);
    for (let i = 0; i < N; i++) hotT[i] = r.kept[i] === 0 ? 1 : 0;
  }
  // glints ease between kept and rejected so the clip reads as a change, not a flicker
  function glints(k) {
    for (let i = 0; i < N; i++) {
      const h = (hot[i] += (hotT[i] - hot[i]) * k);
      const b = 0.6 + 0.9 * vals[i] / 255;
      pcol[i * 3] = 0.25 * b + (1.0 - 0.25 * b) * h;
      pcol[i * 3 + 1] = 0.75 * b + (0.58 - 0.75 * b) * h;
      pcol[i * 3 + 2] = 1.0 * b + (0.22 - 1.0 * b) * h;
      psize[i] = 14 + 22 * h;
      prej[i] = h;
    }
    pgeo.attributes.color.needsUpdate = true;
    pgeo.attributes.size.needsUpdate = true;
    pgeo.attributes.rej.needsUpdate = true;
  }

  // pointer → plane coordinates (hovering pauses the scan and hands the probe to the pointer)
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const hit = new THREE.Mesh(plane, new THREE.MeshBasicMaterial({ visible: false }));
  deck.add(hit);
  const head = new THREE.Vector2(XA, PR), probeP = head.clone(), targetP = head.clone();
  let hovering = false;
  function onPointer(e) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const h = ray.intersectObject(hit)[0];
    if (h && h.uv) {
      targetP.set(h.uv.x * W, (1 - h.uv.y) * H);
      hovering = true;
    } else hovering = false;
  }
  if (!coarse) {
    canvas.addEventListener("pointermove", onPointer);
    canvas.addEventListener("pointerleave", () => (hovering = false));
  }

  // ---- layout
  let portrait = false;
  autosize(renderer, canvas, (w, h) => {
    camera.aspect = w / h;
    portrait = w / h < 1.05;
    camera.fov = portrait ? 42 : 30;
    camera.updateProjectionMatrix();
    const px = Math.min(2, renderer.getPixelRatio());
    pts.material.uniforms.uScale.value = px * (portrait ? 0.8 : 1);
    dustU.uScale.value = px;
    rodU.uWidth.value = portrait ? 0.085 : 0.07;
  }, coarse ? 1.75 : 2);

  // ---- the scan: intro (the probe ignites) → scan → hold on the clean frame → the deck re-forms → again
  let phase = reduced ? "scan" : "intro", phaseT = 0, S = reduced ? TOTAL * 0.45 : 0, t0 = 0, beam = reduced ? 1 : 0;
  const depth = (N + 4) * GAP + 0.4;
  glints(1);
  loop(root, (dt) => {
    const step = reduced ? 0 : dt;
    t0 += step;
    // deck pose: a 3/4 view with a slow sway
    const pose = portrait
      ? { x: -0.35, y: 2.35, s: 0.7, ry: -0.72, rx: 0.2 }
      : { x: 2.5, y: 0.9, s: 0.84, ry: -0.7, rx: 0.16 };
    deck.position.set(pose.x, pose.y, 0);
    deck.scale.setScalar(pose.s);
    const sway = reduced ? 0 : Math.sin(t0 * 0.25) * 0.06;
    deck.rotation.set(pose.rx + sway * 0.4, pose.ry + sway, 0);
    uniforms.uTime.value = t0;
    dustU.uTime.value = t0;

    // scan clock: paused while the pointer drives the probe
    if (!hovering) {
      phaseT += step;
      if (phase === "intro" && phaseT >= INTRO_S) { phase = "scan"; phaseT = 0; S = 0; }
      else if (phase === "scan") {
        S += SPEED * step;
        if (S >= TOTAL) { phase = "hold"; phaseT = 0; }
      } else if (phase === "hold") {
        S += SPEED * step;                           // the trail keeps cooling
        if (phaseT >= HOLD_S) { phase = "reset"; phaseT = 0; }
      } else if (phase === "reset" && phaseT >= RESET_S) {
        phase = "intro"; phaseT = 0; S = 0;
        pathAt(0, probeP);                           // relight at the start, not slide there
      }
    }
    pathAt(S, head);
    const intro = phase === "intro";
    scanU.uS.value = intro ? -1 : S;
    scanU.uGrow.value = intro ? smoothstep(INTRO_S * 0.35, INTRO_S, phaseT) : 1;
    scanU.uReset.value = phase === "reset" ? smoothstep(0, RESET_S, phaseT) : 0;
    scanU.uHead.value.copy(head);
    // a soft swell of light as the last row completes
    result.material.uniforms.uFlash.value = phase === "hold" ? Math.exp(-phaseT * 2.2) * smoothstep(0, 0.15, phaseT) : 0;

    // the probe: rides the scan head, or follows the pointer
    const beamT = hovering ? 1 : intro ? smoothstep(0, INTRO_S, phaseT) : phase === "scan" ? 1
      : phase === "hold" ? 1 - smoothstep(0.1, 0.8, phaseT) : 0;
    beam += (beamT - beam) * (reduced ? 1 : Math.min(1, dt * 10));
    scanU.uBeam.value = beam;
    if (!hovering) targetP.copy(head);
    const gap = probeP.distanceTo(targetP);
    probeP.lerp(targetP, reduced || gap < 0.08 ? 1 : hovering ? 0.35 : 0.14);
    scanU.uProbe.value.copy(probeP);
    sample(probeP.x / W, probeP.y / H);
    glints(reduced ? 1 : Math.min(1, dt * 14));

    const px = probeP.x - W / 2, py = H / 2 - probeP.y;
    rodU.uA.value.set(px, py, -depth / 2);
    rodU.uB.value.set(px, py, depth / 2);
    for (let i = 0; i < N; i++) {
      ppos[i * 3] = px; ppos[i * 3 + 1] = py;
      ppos[i * 3 + 2] = (i - (N - 1) / 2) * GAP + 0.025 * Math.sin(t0 * 1.3 + i * 0.45);
    }
    pgeo.attributes.position.needsUpdate = true;
    rodU.uPulse.value = reduced ? 1 : 0.85 + 0.15 * Math.sin(t0 * 9);
    renderer.render(scene, camera);
  });
  root.classList.add("ready");
}

function OffscreenCanvasOr(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}
