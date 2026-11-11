// Hero: the 40 synthetic subs as a 3D deck. A probe skewers one pixel through every frame and
// sigma-clips its 40 values; scrolling collapses the deck into rankwise's stacked frame.
import * as THREE from "three";
import { $, autosize, clamp, coarse, ease, getJSON, lerp, loadImage, loop, reduced, smoothstep } from "./util.js";
import { sigmaClip } from "./clip.js";

const W = 4.2, H = 2.8, GAP = 0.15;

export async function initHero() {
  const root = $("#hero");
  const canvas = $(".hero-canvas", root);
  const copy = $(".hero-copy", root), out = $(".hero-out", root), hud = $(".hero-hud", root);
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

  // ---- frames: one instanced draw call, z from a uniform
  const geo = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(W, H));
  const idx = new Float32Array(N).map((_, i) => i);
  geo.setAttribute("aIdx", new THREE.InstancedBufferAttribute(idx, 1));
  geo.instanceCount = N;
  const uniforms = {
    uAtlas: { value: tex }, uGrid: { value: new THREE.Vector2(cols, rows) }, uN: { value: N },
    uGap: { value: GAP }, uOpacity: { value: 0.9 }, uTime: { value: 0 }, uWave: { value: 1 },
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
      uniform sampler2D uAtlas; uniform vec2 uGrid; uniform float uOpacity; uniform float uHot[64];
      uniform vec2 uProbe;
      varying vec2 vUv; varying float vIdx; varying float vDepth;
      void main() {
        float col = mod(vIdx, uGrid.x), row = floor(vIdx / uGrid.x);
        vec2 uv = (vec2(col, row) + vec2(vUv.x, 1.0 - vUv.y)) / uGrid;
        vec3 c = texture2D(uAtlas, uv).rgb;
        vec2 e = min(vUv, 1.0 - vUv) * vec2(${W.toFixed(1)}, ${H.toFixed(1)});
        float edge = 1.0 - smoothstep(0.0, 0.018, min(e.x, e.y));
        float hot = uHot[int(vIdx)];
        vec3 rim = mix(vec3(0.35, 0.6, 1.0), vec3(1.0, 0.62, 0.25), hot);
        c = c * (0.55 + 0.45 * vDepth) + edge * rim * (0.35 + 0.65 * hot);
        float a = uOpacity * mix(0.55, 1.0, vDepth);
        // a soft ring where the probe crosses a frame whose value was rejected
        float d = length((vUv - uProbe) * vec2(${W.toFixed(1)}, ${H.toFixed(1)}));
        c += hot * vec3(1.0, 0.55, 0.15) * (smoothstep(0.11, 0.08, d) - smoothstep(0.08, 0.05, d)) * 1.5;
        gl_FragColor = vec4(c, a);
        #include <colorspace_fragment>
      }`,
  }));
  frames.frustumCulled = false;
  deck.add(frames);

  // ---- the stacked result
  const result = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ map: stackTex, transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
  deck.add(result);

  // ---- probe: a glowing skewer plus one point per frame
  const probe = new THREE.Group();
  deck.add(probe);
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 6, 1, true), new THREE.MeshBasicMaterial({ color: 0x5ee6ff, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthTest: false }));
  rod.rotation.x = Math.PI / 2;
  const glow = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1, 8, 1, true), new THREE.MeshBasicMaterial({ color: 0x4f8dff, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false }));
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
  let probeUV = new THREE.Vector2(0.62, 0.42), targetUV = probeUV.clone(), hovering = false, lastKey = "";
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
    const rej = [];
    for (let i = 0; i < N; i++) {
      const isRej = r.kept[i] === 0;
      hot[i] = isRej ? 1 : 0;
      const b = 0.35 + vals[i] / 255;
      if (isRej) { pcol[i * 3] = 1.0; pcol[i * 3 + 1] = 0.55; pcol[i * 3 + 2] = 0.15; psize[i] = 34; rej.push(i + 1); }
      else { pcol[i * 3] = 0.25 * b; pcol[i * 3 + 1] = 0.75 * b; pcol[i * 3 + 2] = 1.0 * b; psize[i] = 14; }
    }
    pgeo.attributes.color.needsUpdate = true;
    pgeo.attributes.size.needsUpdate = true;
    $("#hud-probe").textContent = `${(u * meta.hero.w * 4) | 0}, ${(v * meta.hero.h * 4) | 0}`;
    $("#hud-rej").textContent = rej.length ? `${rej.length} · frame ${rej.slice(0, 3).join(", ")}${rej.length > 3 ? "…" : ""}` : "0";
  }

  // pointer → deck-local UV
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const hit = new THREE.Mesh(new THREE.PlaneGeometry(W, H), new THREE.MeshBasicMaterial({ visible: false }));
  deck.add(hit);
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

  // ---- layout & scroll
  let portrait = false;
  autosize(renderer, canvas, (w, h) => {
    camera.aspect = w / h;
    portrait = w / h < 1.05;
    camera.fov = portrait ? 42 : 30;
    camera.updateProjectionMatrix();
    pts.material.uniforms.uScale.value = Math.min(2, renderer.getPixelRatio()) * (portrait ? 0.8 : 1);
  }, coarse ? 1.75 : 2);

  let p = 0;
  const onScroll = () => {
    const span = root.offsetHeight - innerHeight;
    p = clamp(scrollY / Math.max(1, span));
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  let t0 = 0;
  const L = loop(root, (dt, t) => {
    t0 += reduced ? 0 : dt;
    const e = ease(clamp((p - 0.04) / 0.8));
    // deck pose: 3/4 view → face-on, centered
    const pose = portrait
      ? { x: -0.35, y: 1.55, s: 0.7, ry: -0.72, rx: 0.2 }
      : { x: 2.5, y: 0.9, s: 0.84, ry: -0.7, rx: 0.16 };
    const end = portrait ? { x: 0, y: 0.25, s: 1.05 } : { x: 0, y: 0, s: 1.35 };
    deck.position.set(lerp(pose.x, end.x, e), lerp(pose.y, end.y, e), 0);
    deck.scale.setScalar(lerp(pose.s, end.s, e));
    const sway = reduced ? 0 : Math.sin(t0 * 0.25) * 0.06;
    deck.rotation.set(lerp(pose.rx + sway * 0.4, 0, e), lerp(pose.ry + sway, 0, e), 0);
    uniforms.uGap.value = GAP * (1 - e);
    uniforms.uWave.value = 1 - e;
    uniforms.uTime.value = t0;
    uniforms.uOpacity.value = 0.92 * (1 - smoothstep(0.85, 1.0, e));
    result.material.opacity = smoothstep(0.55, 0.95, e);
    result.position.z = (N * 0.5 + 0.5) * GAP * (1 - e);

    // probe: follows the pointer, otherwise wanders across the trails
    if (!hovering) targetUV.set(0.5 + 0.36 * Math.sin(t0 * 0.21 + 0.6), 0.5 + 0.32 * Math.sin(t0 * 0.33));
    probeUV.lerp(targetUV, hovering ? 0.35 : 0.08);
    uniforms.uProbe.value.copy(probeUV);
    sample(probeUV.x, probeUV.y);
    const px = (probeUV.x - 0.5) * W, py = (0.5 - probeUV.y) * H;
    const depth = (N + 4) * GAP * (1 - e) + 0.4;
    rod.position.set(px, py, 0); rod.scale.y = depth;
    glow.position.set(px, py, 0); glow.scale.y = depth;
    for (let i = 0; i < N; i++) {
      ppos[i * 3] = px; ppos[i * 3 + 1] = py;
      ppos[i * 3 + 2] = (i - (N - 1) / 2) * GAP * (1 - e) + (1 - e) * 0.025 * Math.sin(t0 * 1.3 + i * 0.45);
    }
    pgeo.attributes.position.needsUpdate = true;
    probe.visible = e < 0.6;
    rod.material.opacity = glow.material.opacity = 1 - smoothstep(0.3, 0.6, e);

    // copy & HUD
    copy.style.opacity = String(1 - smoothstep(0.02, 0.3, p));
    copy.style.transform = `translateY(${-p * 80}px)`;
    hud.style.opacity = String(1 - smoothstep(0.2, 0.45, p));
    out.style.opacity = String(smoothstep(0.7, 0.92, p));
    out.style.transform = `translate(-50%, ${portrait ? "calc(-50% + 34vw)" : "calc(-50% + 30vh)"})`;
    $("#hud-n").textContent = e > 0.95 ? "1 frame" : `${N} subs`;
    renderer.render(scene, camera);
  });
  addEventListener("scroll", L.kick, { passive: true });
  root.classList.add("ready");
}

function OffscreenCanvasOr(w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}
