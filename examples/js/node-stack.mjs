// Stack 40 synthetic astro subs with satellite trails in Node (or Bun): sigma-clipped mean
// plus the rejection map, written as PNGs with no dependencies.
//
//   node rankwise-wasm/scripts/build.mjs        # once: builds rankwise-wasm/dist/*.wasm
//   node examples/js/node-stack.mjs [outdir]    # or: bun examples/js/node-stack.mjs
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { init } from "../../rankwise-wasm/js/rankwise.js";

const W = 640;
const H = 400;
const N = 40;
const outdir = process.argv[2] ?? "stack-output";

// ---- synthetic subs: a faint nebula, stars, sky noise, hot pixels and one trail per frame
let seed = 7;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const sky = new Float32Array(W * H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const dx = (x - W * 0.55) / 170;
    const dy = (y - H * 0.5) / 95;
    sky[y * W + x] = 1000 + 260 * Math.exp(-(dx * dx + dy * dy)) * (0.7 + 0.3 * Math.sin(x / 23 + y / 31));
  }
}
for (let s = 0; s < 260; s++) {
  const cx = rnd() * W;
  const cy = rnd() * H;
  const peak = 300 + 6000 * rnd() ** 4;
  for (let y = Math.max(0, cy - 6 | 0); y < Math.min(H, cy + 7); y++) {
    for (let x = Math.max(0, cx - 6 | 0); x < Math.min(W, cx + 7); x++) {
      sky[y * W + x] += peak * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 3);
    }
  }
}
const frames = Array.from({ length: N }, () => {
  const f = Float32Array.from(sky, (v) => v + gauss() * Math.sqrt(v) * 0.9);
  // a satellite trail
  const a = rnd() * Math.PI;
  const x0 = rnd() * W;
  const y0 = rnd() * H;
  for (let t = -800; t < 800; t++) {
    const x = Math.round(x0 + t * Math.cos(a));
    const y = Math.round(y0 + t * Math.sin(a));
    for (let w = -1; w <= 1; w++) if (x >= 0 && x < W && y + w >= 0 && y + w < H) f[(y + w) * W + x] += 4000;
  }
  for (let h = 0; h < 30; h++) f[(rnd() * W * H) | 0] = 60000; // cosmic rays / hot pixels
  return f;
});

// ---- rankwise
const t0 = performance.now();
const rw = await init();
const tLoad = performance.now() - t0;

const t1 = performance.now();
const { value: stacked, mask, rejected } = rw.sigmaClipMean(frames, { sigma: 3, maxiters: 5, mask: true, returnCounts: true });
const tClip = performance.now() - t1;
const t2 = performance.now();
const med = rw.median(frames);
const tMed = performance.now() - t2;

let total = 0;
for (const r of rejected) total += r;
console.log(`rankwise ${rw.version} (${rw.build} build, ${rw.isa} kernels), loaded in ${tLoad.toFixed(0)} ms`);
console.log(`sigma-clipped mean of ${N} × ${W}×${H}: ${tClip.toFixed(1)} ms, ${total} values rejected (${((100 * total) / (N * W * H)).toFixed(2)}%)`);
console.log(`median: ${tMed.toFixed(1)} ms`);

// ---- PNGs (grayscale, asinh stretch)
function crc32(buf) {
  let c = -1;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}
function png(gray, w, h) {
  const raw = Buffer.alloc((w + 1) * h);
  for (let y = 0; y < h; y++) raw.set(gray.subarray(y * w, (y + 1) * w), y * (w + 1) + 1);
  const chunk = (type, data) => {
    const b = Buffer.alloc(12 + data.length);
    b.writeUInt32BE(data.length, 0);
    b.write(type, 4, "ascii");
    data.copy(b, 8);
    b.writeUInt32BE(crc32(b.subarray(4, 8 + data.length)), 8 + data.length);
    return b;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr.set([8, 0, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}
const stretch = (img) => Uint8Array.from(img, (v) => Math.max(0, Math.min(255, 255 * Math.asinh(Math.max(0, v - 900) / 40) / Math.asinh(4000 / 40))));
const rejMap = new Uint8Array(W * H);
for (let k = 0; k < N; k++) for (let i = 0; i < W * H; i++) if (mask[k][i]) rejMap[i] = 255;

await mkdir(outdir, { recursive: true });
await writeFile(join(outdir, "one-sub.png"), png(stretch(frames[0]), W, H));
await writeFile(join(outdir, "median.png"), png(stretch(med), W, H));
await writeFile(join(outdir, "sigma-clipped.png"), png(stretch(stacked), W, H));
await writeFile(join(outdir, "rejected.png"), png(rejMap, W, H));
console.log(`wrote ${outdir}/{one-sub,median,sigma-clipped,rejected}.png`);
