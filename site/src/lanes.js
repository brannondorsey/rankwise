// 02 · 16 neighbouring pixels × 9 frames, one register per frame. Every compare-exchange of two
// registers (a min and a max) advances all 16 sorts at once; outliers sink, the median row lights up.
// Values are real: 16 pixels of the synthetic subs where satellite trails cross.
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { $, autosize, clamp, easeOut, getJSON, loop, reduced, segmented, smoothstep } from "./util.js";
import { batcher, medianRanks, pruneMask } from "./netgen.js";

const LANES = 16;
const STEP = 0.36; // seconds per instruction pair

export async function initLanes() {
  const view = $("#lanes-view"), canvas = $("canvas", view);
  const meta = await getJSON("data/astro.json");
  const data0 = meta.lanes.codes; // [rows][16]
  const R = data0.length;
  const full = batcher(R);
  const keep = pruneMask(full, R, medianRanks(R));
  const net = full.filter((_, i) => keep[i]);
  const mid = medianRanks(R)[0];

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x020306, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x020306, 26, 48);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  const world = new THREE.Group();
  scene.add(world);

  scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x080a12, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(6, 10, 12);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x5ee6ff, 1.0);
  rim.position.set(-10, 4, -8);
  scene.add(rim);

  // colour: rank inside the clean range → deep blue … cyan … white; outliers → orange
  const flat = data0.flat().slice().sort((a, b) => a - b);
  const med = flat[flat.length >> 1], cut = med + 40;
  const clean = flat.filter((v) => v <= cut);
  const lo = clean[Math.floor(clean.length * 0.03)], hi = clean[clean.length - 1];
  const ramp = [new THREE.Color(0x14307a), new THREE.Color(0x2a6cff), new THREE.Color(0x5ee6ff), new THREE.Color(0xe8f6ff)];
  const hot = new THREE.Color(0xff8a2a);
  function colorOf(v, out) {
    if (v > cut) return out.copy(hot);
    const t = clamp((v - lo) / (hi - lo)) * (ramp.length - 1);
    const i = Math.min(ramp.length - 2, Math.floor(t));
    return out.copy(ramp[i]).lerp(ramp[i + 1], t - i);
  }

  const geo = new RoundedBoxGeometry(0.8, 0.8, 0.8, 3, 0.12);
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0.15 });
  mat.onBeforeCompile = (s) => {
    s.uniforms.uDim = dimU;
    s.fragmentShader = s.fragmentShader
      .replace("#include <common>", "#include <common>\nuniform float uDim;")
      .replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\ntotalEmissiveRadiance += vColor * 0.42;\n#endif")
      .replace("#include <dithering_fragment>", "#include <dithering_fragment>\ngl_FragColor.rgb *= uDim;");
  };
  const dimU = { value: 1 };
  const cubes = new THREE.InstancedMesh(geo, mat, R * LANES);
  world.add(cubes);
  // median row highlight drawn separately so the rest can dim
  const midMat = mat.clone();
  midMat.onBeforeCompile = (s) => {
    s.fragmentShader = s.fragmentShader.replace("#include <emissivemap_fragment>", "#include <emissivemap_fragment>\n#ifdef USE_INSTANCING_COLOR\ntotalEmissiveRadiance += vColor * 0.9;\n#endif");
  };
  const midCubes = new THREE.InstancedMesh(geo, midMat, LANES);
  midCubes.visible = false;
  world.add(midCubes);

  // rails: one per (row, register group)
  const railMat = new THREE.MeshBasicMaterial({ color: 0x4f8dff, transparent: true, opacity: 0.08, depthWrite: false });
  let rails = [];
  const railGroup = new THREE.Group();
  world.add(railGroup);

  // register labels
  const labels = [];
  for (let r = 0; r < R; r++) {
    const c = document.createElement("canvas");
    c.width = 160; c.height = 64;
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
    sp.scale.set(1.9, 0.76, 1);
    world.add(sp);
    labels.push({ c, t, sp });
  }
  const medLabel = (() => {
    const c = document.createElement("canvas");
    c.width = 256; c.height = 64;
    const x = c.getContext("2d");
    x.font = "600 34px 'JetBrains Mono Variable', monospace";
    x.fillStyle = "#5ee6ff";
    x.textBaseline = "middle";
    x.fillText("median ×16", 6, 34);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false, opacity: 0 }));
    sp.scale.set(3.6, 0.9, 1);
    world.add(sp);
    return { sp, c, x, t };
  })();

  const X = (lane) => (lane - (LANES - 1) / 2) * 1.0;
  const Y = (row) => ((R - 1) / 2 - row) * 1.0;

  // state
  let L = 16, regName = (r) => `zmm${r}`, isaName = "avx512";
  let val, slot, steps, si, phase, hold, ins;
  // slot[row][lane] = index of the cube currently in that cell; val per cube
  function reset() {
    val = data0.flat();
    slot = Array.from({ length: R }, (_, r) => Array.from({ length: LANES }, (_, l) => r * LANES + l));
    steps = [];
    for (const [a, b] of net) for (let g = 0; g < LANES / L; g++) steps.push({ a, b, g });
    si = 0; phase = 0; hold = 0; ins = 0;
    $("#lanes-ins").textContent = "0";
    dimU.value = 1;
    midCubes.visible = false;
    medLabel.sp.material.opacity = 0;
    asm("", "");
  }
  function buildRails() {
    railGroup.clear();
    rails = [];
    for (let r = 0; r < R; r++) {
      rails[r] = [];
      for (let g = 0; g < LANES / L; g++) {
        const m = new THREE.Mesh(new THREE.BoxGeometry(L - 0.12, 0.94, 0.94), railMat.clone());
        m.position.set(X(g * L + (L - 1) / 2), Y(r), 0);
        railGroup.add(m);
        rails[r][g] = m;
      }
      const { c, t, sp } = labels[r];
      const x = c.getContext("2d");
      x.clearRect(0, 0, c.width, c.height);
      x.font = "500 30px 'JetBrains Mono Variable', monospace";
      x.fillStyle = r === mid ? "#5ee6ff" : "#7d8799";
      x.textAlign = "right";
      x.textBaseline = "middle";
      x.fillText(regName(r), c.width - 8, 34);
      t.needsUpdate = true;
      sp.position.set(X(0) - 1.6, Y(r), 0);
    }
    medLabel.sp.position.set(X(LANES - 1) + 2.6, Y(mid), 0);
  }

  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3(), col = new THREE.Color();
  function place(cube, x, y, z, s) {
    m4.compose(p3.set(x, y, z), q, s3.setScalar(s));
    cubes.setMatrixAt(cube, m4);
  }
  for (let i = 0; i < R * LANES; i++) cubes.setColorAt(i, colorOf(data0.flat()[i], col));

  function asm(l1, l2) {
    const el = $("#lanes-asm");
    el.innerHTML = l1 ? `<div>${l1}</div><div>${l2}</div>` : `<div><span class="cm">${LANES} pixels · ${R} frames</span></div><div><span class="cm">median of 9: ${net.length} of ${full.length} comparators</span></div>`;
  }
  function asmFor(a, b) {
    const [mn, mx, sfx] = isaName === "neon" ? ["fmin", "fmax", ".4s"] : ["vminps", "vmaxps", ""];
    const r = (k) => `<span class="reg">${regName(k)}${sfx}</span>`;
    return [`<span class="op">${mn}</span> ${r(a)} ← ${r(a)}, ${r(b)}`, `<span class="op">${mx}</span> ${r(b)} ← ${r(a)}, ${r(b)}`];
  }

  segmented($("#lanes-isa"), (d) => {
    L = +d.l;
    isaName = L === 16 ? "avx512" : L === 8 ? "avx2" : "neon";
    regName = L === 16 ? (r) => `zmm${r}` : L === 8 ? (r) => `ymm${r}` : (r) => `v${r}`;
    $("#lanes-sorts").textContent = L;
    buildRails();
    reset();
  });
  $("#lanes-play").addEventListener("click", reset);
  buildRails();
  reset();

  // camera fit + gentle drag
  let aspect = 1, yaw = 0, yawT = 0, dragging = false, lastX = 0;
  autosize(renderer, canvas, (w, h) => {
    aspect = w / h;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
  }, 2);
  canvas.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; });
  addEventListener("pointerup", () => (dragging = false));
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    yawT = clamp(yawT + (e.clientX - lastX) * 0.005, -0.7, 0.7);
    lastX = e.clientX;
  });
  canvas.style.touchAction = "pan-y";

  let t0 = 0;
  loop(view, (dt) => {
    t0 += dt;
    // fit 16 lanes + labels to the width, 9 rows to the height
    const narrow = aspect < 1.4;
    const needW = LANES + (narrow ? 3.2 : 5.5), needH = R + (narrow ? 1.2 : 2.5);
    const tanH = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const dist = Math.max(needH / 2 / tanH, needW / 2 / (tanH * aspect)) * 1.02;
    yaw += (yawT + (reduced ? 0 : Math.sin(t0 * 0.3) * 0.12) - yaw) * 0.06;
    camera.position.set(Math.sin(-0.32 + yaw) * dist, 2.4 + Math.sin(t0 * 0.21) * 0.3, Math.cos(-0.32 + yaw) * dist);
    camera.lookAt(0.6, aspect < 1.4 ? 0.7 : -0.2, 0);

    // advance the network
    let active = null, u = 0;
    if (si < steps.length) {
      phase += dt / STEP * (reduced ? 4 : 1);
      active = steps[si];
      u = clamp(phase);
      if (phase >= 1) {
        // commit the swap
        const { a, b, g } = active;
        for (let l = g * L; l < (g + 1) * L; l++) {
          const ca = slot[a][l], cb = slot[b][l];
          if (val[ca] > val[cb]) { slot[a][l] = cb; slot[b][l] = ca; }
        }
        ins += 2;
        $("#lanes-ins").textContent = ins;
        si++;
        phase = 0;
        if (si < steps.length) { const [l1, l2] = asmFor(steps[si].a, steps[si].b); asm(l1, l2); }
      } else if (phase < dt / STEP * 1.5) {
        const [l1, l2] = asmFor(active.a, active.b);
        asm(l1, l2);
      }
    } else {
      hold += dt;
      if (hold > 4.5) reset();
    }

    const done = si >= steps.length;
    const fade = done ? smoothstep(0, 0.6, hold) : 0;
    dimU.value = 1 - 0.72 * fade;
    medLabel.sp.material.opacity = fade;
    midCubes.visible = done;

    for (let r = 0; r < R; r++) {
      for (let l = 0; l < LANES; l++) {
        const c = slot[r][l];
        let x = X(l), y = Y(r), z = 0, s = 1;
        if (active && (r === active.a || r === active.b) && l >= active.g * L && l < (active.g + 1) * L) {
          const other = r === active.a ? slot[active.b][l] : slot[active.a][l];
          const swap = r === active.a ? val[c] > val[other] : val[other] > val[c];
          const k = easeOut(clamp((u - 0.15) / 0.75));
          s = 1 + 0.1 * Math.sin(Math.PI * clamp(u / 0.3));
          if (swap) {
            const ty = r === active.a ? Y(active.b) : Y(active.a);
            y = y + (ty - y) * k;
            z = Math.sin(Math.PI * k) * (r === active.a ? 1.1 : -1.1);
          }
        }
        if (done && r === mid) s = 0;
        place(c, x, y, z, s);
      }
    }
    cubes.instanceMatrix.needsUpdate = true;
    if (done) {
      for (let l = 0; l < LANES; l++) {
        const c = slot[mid][l];
        m4.compose(p3.set(X(l), Y(mid), 0), q, s3.setScalar(1.06 + 0.05 * Math.sin(t0 * 3 + l * 0.4)));
        midCubes.setMatrixAt(l, m4);
        midCubes.setColorAt(l, colorOf(val[c], col));
      }
      midCubes.instanceMatrix.needsUpdate = true;
      midCubes.instanceColor.needsUpdate = true;
    }
    // rails
    for (let r = 0; r < R; r++) for (let g = 0; g < rails[r].length; g++) {
      const on = active && (r === active.a || r === active.b) && g === active.g;
      const target = on ? 0.32 : done && r === mid ? 0.28 * fade : 0.06;
      const m = rails[r][g].material;
      m.opacity += (target - m.opacity) * 0.25;
      m.color.setHex(on ? 0x5ee6ff : done && r === mid ? 0x5ee6ff : 0x4f8dff);
    }
    renderer.render(scene, camera);
  });
}
