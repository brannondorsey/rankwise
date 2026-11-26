// Hero: the 40 synthetic subs as a 3D deck. A probe raster-scans the deck band by band, skewering
// one pixel through every frame and sigma-clipping its 40 values. Behind the scan head each pixel
// stack has been reduced to one clean value: the noisy frames there fade to ghosts and rankwise's
// stacked frame shows through.
import * as THREE from "three";
import { $, autosize, coarse, ease, getJSON, loadImage, loop, reduced, smoothstep } from "./util.js";
import { sigmaClip } from "./clip.js";

const W = 4.2, H = 2.8, GAP = 0.15;
const BANDS = 12;        // scan rows
const BAND_S = 1.0;      // seconds per row
const STEP = 0.14;       // a row change, in rows' time
const HOLD_S = 1.8, RESET_S = 1.1;

// GLSL shared by the frames and the stacked frame: where the scan has been, in top-down uv.
const SCAN = /* glsl */ `
  uniform float uScan, uBands, uReveal;
  // the scan position (in rows) at which pixel (x, y) is reached; serpentine, top row first
  float reachedAt(vec2 tuv) {
    float band = min(floor(tuv.y * uBands), uBands - 1.0);
    float x = mod(band, 2.0) < 0.5 ? tuv.x : 1.0 - tuv.x;
    return band + x;
  }`;

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

  const scanU = {
    uScan: { value: 0 }, uBands: { value: BANDS }, uReveal: { value: 1 },
    uRod: { value: new THREE.Vector2(0, 0.5 / BANDS) }, uHead: { value: 1 }, uBeam: { value: 1 },
  };

  // ---- frames: one instanced draw call, z from a uniform
  const geo = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(W, H));
  const idx = new Float32Array(N).map((_, i) => i);
  geo.setAttribute("aIdx", new THREE.InstancedBufferAttribute(idx, 1));
  geo.instanceCount = N;
  const uniforms = {
    ...scanU,
    uAtlas: { value: tex }, uGrid: { value: new THREE.Vector2(cols, rows) }, uN: { value: N },
    uGap: { value: GAP }, uOpacity: { value: 0.92 }, uTime: { value: 0 }, uWave: { value: 1 },
    uProbe: { value: new THREE.Vector2(0.5, 0.5) }, uHot: { value: new Float32Array(64) },
  };
  const frames = new THREE.Mesh(geo, new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      attribute float aIdx;
      uniform float uGap, uN, uTime, uWave;
      varying vec2 vUv; varying float vIdx; varying float vDepth;
      void main() {
        vUv = uv; vIdx = aIdx;
        vec3 p = position;
        float k = aIdx - (uN - 1.0) * 0.5;
        p.z += k * uGap + uWave * 0.025 * sin(uTime * 1.3 + aIdx * 0.45);
        vDepth = aIdx / (uN - 1.0);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      ${SCAN}
      uniform sampler2D uAtlas; uniform vec2 uGrid; uniform float uOpacity; uniform float uHot[64];
      uniform vec2 uProbe, uRod; uniform float uHead, uBeam;
      varying vec2 vUv; varying float vIdx; varying float vDepth;
      void main() {
        vec2 tuv = vec2(vUv.x, 1.0 - vUv.y);   // top-down, like the probe
        float col = mod(vIdx, uGrid.x), row = floor(vIdx / uGrid.x);
        vec2 uv = (vec2(col, row) + tuv) / uGrid;
        vec3 c = texture2D(uAtlas, uv).rgb;
        vec2 e = min(vUv, 1.0 - vUv) * vec2(${W.toFixed(1)}, ${H.toFixed(1)});
        float edge = 1.0 - smoothstep(0.0, 0.018, min(e.x, e.y));
        float hot = uHot[int(vIdx)];
        vec3 rim = mix(vec3(0.35, 0.6, 1.0), vec3(1.0, 0.62, 0.25), hot);
        c = c * (0.55 + 0.45 * vDepth) + edge * rim * (0.35 + 0.65 * hot);
        float a = uOpacity * mix(0.55, 1.0, vDepth);

        // scanned stacks: the noisy frames there are spent, so they fade to violet ghosts
        float age = uScan - reachedAt(tuv);
        float done = step(0.0, age) * uReveal;
        vec3 ghost = vec3(0.42, 0.36, 1.0) * (0.3 + 0.7 * edge);
        c = mix(c, ghost, done * 0.9);
        a *= 1.0 - done * (0.975 - 0.55 * edge);

        // the probe pierces every frame: a cyan glint, and an orange ring where the value was rejected
        float d = length((tuv - uProbe) * vec2(${W.toFixed(1)}, ${H.toFixed(1)}));
        c += uBeam * vec3(0.37, 0.9, 1.0) * smoothstep(0.09, 0.0, d) * 0.9;
        c += hot * vec3(1.0, 0.55, 0.15) * (smoothstep(0.11, 0.08, d) - smoothstep(0.08, 0.05, d)) * 1.5;
        // the scan head: a thin sheet of light across the current row, through the whole deck
        float band = floor(tuv.y * uBands), rodBand = floor(uRod.y * uBands);
        float head = (band == rodBand ? 1.0 : 0.0) * exp(-abs(tuv.x - uRod.x) * 140.0);
        c += uHead * uBeam * vec3(0.37, 0.9, 1.0) * head * 0.55;
        a = max(a, uHead * uBeam * head * 0.5 * uOpacity);
        gl_FragColor = vec4(c, a);
        #include <colorspace_fragment>
      }`,
  }));
  frames.frustumCulled = false;
  deck.add(frames);

  // ---- the stacked result: shown wherever the scan has been, and everywhere once collapsed
  const resultU = { ...scanU, uMap: { value: stackTex }, uOpacity: { value: 0 }, uFx: { value: 1 } };
  const result = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.ShaderMaterial({
    uniforms: resultU, transparent: true, depthTest: false, depthWrite: false,
    vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      ${SCAN}
      uniform sampler2D uMap; uniform float uOpacity, uFx;
      varying vec2 vUv;
      void main() {
        vec2 tuv = vec2(vUv.x, 1.0 - vUv.y);
        vec3 c = texture2D(uMap, vUv).rgb;
        float age = uScan - reachedAt(tuv);
        float done = step(0.0, age) * uReveal;
        float fx = done * uFx;
        // fresh rows glow and cool off: a comet trail behind the scan head
        c += vec3(0.33, 0.3, 1.0) * exp(-age * 2.2) * fx * 0.4;
        // faint scan lines between rows, and a bright seam along the scan front
        float fy = fract(tuv.y * uBands);
        float seam = 1.0 - smoothstep(0.0, 0.035, min(fy, 1.0 - fy));
        c += vec3(0.45, 0.5, 1.0) * seam * fx * 0.16;
        vec2 e = min(vUv, 1.0 - vUv) * vec2(${W.toFixed(1)}, ${H.toFixed(1)});
        c += vec3(0.5, 0.45, 1.0) * (1.0 - smoothstep(0.0, 0.02, min(e.x, e.y))) * fx * 0.8;
        gl_FragColor = vec4(c, max(uOpacity, done));
        #include <colorspace_fragment>
      }`,
  }));
  deck.add(result);

  // ---- probe: a glowing skewer plus one point per frame
  const probe = new THREE.Group();
  deck.add(probe);
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 1, 6, 1, true), new THREE.MeshBasicMaterial({ color: 0x9ff4ff, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthTest: false }));
  rod.rotation.x = Math.PI / 2;
  const glow = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1, 10, 1, true), new THREE.MeshBasicMaterial({ color: 0x3fb8ff, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false }));
  glow.rotation.x = Math.PI / 2;
  probe.add(glow, rod);
  const ppos = new Float32Array(N * 3), pcol = new Float32Array(N * 3), psize = new Float32Array(N);
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.BufferAttribute(ppos, 3));
  pgeo.setAttribute("color", new THREE.BufferAttribute(pcol, 3));
  pgeo.setAttribute("size", new THREE.BufferAttribute(psize, 1));
  const pts = new THREE.Points(pgeo, new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1 } }, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float size; attribute vec3 color; varying vec3 vC; uniform float uScale;
      void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position,1.); gl_PointSize = size * uScale * (12.0 / -mv.z); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; void main(){ float d = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.0, d); a = a*a + smoothstep(0.18,0.1,d); gl_FragColor = vec4(vC * a, 1.0); }`,
  }));
  pts.frustumCulled = false;
  probe.add(pts);
  // the scan head is light, not matter: draw it over the stacked frame
  result.renderOrder = 1;
  for (const o of [rod, glow, pts]) o.renderOrder = 2;

  // ---- distant star dust for depth
  const dust = new THREE.BufferGeometry();
  const dn = 700, dp = new Float32Array(dn * 3);
  for (let i = 0; i < dn; i++) {
    dp[i * 3] = (Math.random() - 0.5) * 60;
    dp[i * 3 + 1] = (Math.random() - 0.5) * 40;
    dp[i * 3 + 2] = -10 - Math.random() * 40;
  }
  dust.setAttribute("position", new THREE.BufferAttribute(dp, 3));
  scene.add(new THREE.Points(dust, new THREE.PointsMaterial({ color: 0x6f86b8, size: 0.06, transparent: true, opacity: 0.6, sizeAttenuation: true })));

  // ---- probe values: sigma clip the 40 values of one pixel
  const vals = new Float64Array(N);
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
    const hot = uniforms.uHot.value;
    for (let i = 0; i < N; i++) {
      const isRej = r.kept[i] === 0;
      hot[i] = isRej ? 1 : 0;
      const b = 0.35 + vals[i] / 255;
      if (isRej) { pcol[i * 3] = 1.0; pcol[i * 3 + 1] = 0.55; pcol[i * 3 + 2] = 0.15; psize[i] = 34; }
      else { pcol[i * 3] = 0.25 * b; pcol[i * 3 + 1] = 0.75 * b; pcol[i * 3 + 2] = 1.0 * b; psize[i] = 14; }
    }
    pgeo.attributes.color.needsUpdate = true;
    pgeo.attributes.size.needsUpdate = true;
  }

  // pointer → deck-local UV (hovering pauses the scan and hands the probe to the pointer)
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const hit = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ visible: false }));
  deck.add(hit);
  const probeUV = new THREE.Vector2(0.02, 0.5 / BANDS), targetUV = probeUV.clone();
  let hovering = false;
  function onPointer(e) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const h = ray.intersectObject(hit)[0];
    if (h && h.uv) {
      targetUV.set(h.uv.x, 1 - h.uv.y);
      hovering = true;
    } else hovering = false;
  }
  if (!coarse) {
    canvas.addEventListener("pointermove", onPointer);
    canvas.addEventListener("pointerleave", () => (hovering = false));
  }

  // ---- the scan: serpentine rows, top first; hold on the clean frame, then fade back and rescan
  const ROW = 1 + STEP, END = BANDS * ROW - STEP;
  let tau = reduced ? END * 0.45 : 0, phase = "scan", phaseT = 0;
  const scanHead = (tau) => {
    const b = Math.min(BANDS - 1, Math.floor(tau / ROW)), t = tau - b * ROW;
    const dir = b % 2 === 0 ? 1 : -1;
    const f = Math.min(1, t);
    const x = 0.015 + 0.97 * (dir > 0 ? f : 1 - f);
    const y = (b + 0.5 + (t > 1 ? ease((t - 1) / STEP) : 0)) / BANDS;
    return { cover: b + f, x, y };
  };

  // ---- layout
  let portrait = false;
  autosize(renderer, canvas, (w, h) => {
    camera.aspect = w / h;
    portrait = w / h < 1.05;
    camera.fov = portrait ? 42 : 30;
    camera.updateProjectionMatrix();
    pts.material.uniforms.uScale.value = Math.min(2, renderer.getPixelRatio()) * (portrait ? 0.8 : 1);
  }, coarse ? 1.75 : 2);

  let t0 = 0, head = 1, cover = 0;
  loop(root, (dt) => {
    const step = reduced ? 0 : dt;
    t0 += step;
    // deck pose: a 3/4 view with a slow sway
    const pose = portrait
      ? { x: -0.35, y: 1.55, s: 0.7, ry: -0.72, rx: 0.2 }
      : { x: 2.5, y: 0.9, s: 0.84, ry: -0.7, rx: 0.16 };
    deck.position.set(pose.x, pose.y, 0);
    deck.scale.setScalar(pose.s);
    const sway = reduced ? 0 : Math.sin(t0 * 0.25) * 0.06;
    deck.rotation.set(pose.rx + sway * 0.4, pose.ry + sway, 0);
    uniforms.uTime.value = t0;

    // scan clock: paused while the pointer drives the probe
    if (!hovering) {
      phaseT += step;
      if (phase === "scan") {
        tau = Math.min(END, tau + step / BAND_S);
        if (tau >= END) { phase = "hold"; phaseT = 0; }
      } else if (phase === "hold" && phaseT > HOLD_S) { phase = "reset"; phaseT = 0; }
      else if (phase === "reset" && phaseT > RESET_S) { phase = "scan"; phaseT = 0; tau = 0; }
    }
    const s = scanHead(tau);
    if (phase === "scan") cover = s.cover;
    else if (phase === "hold") cover = BANDS;
    scanU.uScan.value = cover;
    scanU.uReveal.value = phase === "reset" ? 1 - smoothstep(0, RESET_S, phaseT) : 1;
    head += ((!hovering && phase === "scan" ? 1 : 0) - head) * (reduced ? 1 : Math.min(1, dt * 8));
    scanU.uHead.value = head;

    // probe: rides the scan head, or follows the pointer
    if (!hovering) targetUV.set(phase === "reset" ? 0.015 : s.x, phase === "reset" ? 0.5 / BANDS : s.y);
    const gap = probeUV.distanceTo(targetUV);
    probeUV.lerp(targetUV, hovering ? 0.35 : gap < 0.02 ? 1 : 0.14);
    scanU.uRod.value.copy(probeUV);
    uniforms.uProbe.value.copy(probeUV);
    sample(probeUV.x, probeUV.y);
    const px = (probeUV.x - 0.5) * W, py = (0.5 - probeUV.y) * H;
    const depth = (N + 4) * GAP + 0.4;
    rod.position.set(px, py, 0); rod.scale.y = depth;
    glow.position.set(px, py, 0); glow.scale.y = depth;
    for (let i = 0; i < N; i++) {
      ppos[i * 3] = px; ppos[i * 3 + 1] = py;
      ppos[i * 3 + 2] = (i - (N - 1) / 2) * GAP + 0.025 * Math.sin(t0 * 1.3 + i * 0.45);
    }
    pgeo.attributes.position.needsUpdate = true;
    const pulse = reduced ? 1 : 0.85 + 0.15 * Math.sin(t0 * 9);
    rod.material.opacity = pulse;
    glow.material.opacity = 0.25 * pulse;
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
