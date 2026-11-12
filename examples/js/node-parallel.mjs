// Use every core, two ways. The native addon splits one call itself (`threads`); WebAssembly
// has no threads, so there one instance per worker thread reduces a strip of pixels of frames
// held in SharedArrayBuffers (no copies between threads). Workers also keep the event loop
// free, with either backend.
//
//   node examples/js/node-parallel.mjs [workers]     # also runs under Bun
import { availableParallelism } from "node:os";
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { init } from "../../rankwise-js/js/node.js"; // "rankwise" once installed

if (isMainThread) {
  const workers = Number(process.argv[2] ?? availableParallelism());
  const n = 31;
  const npix = 2048 * 2048;
  // frames in shared memory, as an app would decode or receive them
  const frames = Array.from({ length: n }, (_, k) => {
    const f = new Float32Array(new SharedArrayBuffer(npix * 4));
    let s = 99991 * (k + 1);
    for (let i = 0; i < npix; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      f[i] = 1000 + (s / 2 ** 32 - 0.5) * 60 + (s % 1024 === 0 ? 5000 : 0);
    }
    return f;
  });
  const out = new Float32Array(new SharedArrayBuffer(npix * 4));

  const pool = Array.from({ length: workers }, () => new Worker(new URL(import.meta.url), { workerData: { frames, out } }));
  const ready = pool.map((w) => new Promise((r) => w.once("message", r)));
  await Promise.all(ready);
  const run = (op) =>
    Promise.all(
      pool.map((w, i) => {
        const p0 = Math.floor((npix * i) / workers / 64) * 64;
        const p1 = i === workers - 1 ? npix : Math.floor((npix * (i + 1)) / workers / 64) * 64;
        return new Promise((r) => {
          w.once("message", r);
          w.postMessage({ op, p0, p1 });
        });
      }),
    );

  const rw = await init();
  console.log(`main thread: ${rw.backend} (${rw.isa})`);
  for (const op of ["median", "sigmaClipMean"]) {
    await run(op); // warm-up (the engine optimizes the kernels after the first calls)
    let best = Infinity;
    for (let r = 0; r < 3; r++) {
      const t = performance.now();
      await run(op);
      best = Math.min(best, performance.now() - t);
    }
    let ref = rw[op](frames); // single-thread reference (warm-up first)
    let single = Infinity;
    for (let r = 0; r < 3; r++) {
      const t = performance.now();
      ref = rw[op](frames);
      single = Math.min(single, performance.now() - t);
    }
    const same = ref.every((v, i) => v === out[i]);
    console.log(
      `${op.padEnd(14)} 31 × 2048² (${rw.backend}): 1 thread ${single.toFixed(0)} ms, ${workers} workers ${best.toFixed(0)} ms (${(single / best).toFixed(1)}x), ${same ? "identical" : "MISMATCH"}`,
    );
    if (!same) process.exitCode = 1;
    if (rw.backend === "native") {
      let t = Infinity;
      let r;
      for (let i = 0; i < 3; i++) {
        const t0 = performance.now();
        r = rw[op](frames, { threads: workers });
        t = Math.min(t, performance.now() - t0);
      }
      const ok = r.every((v, i) => v === ref[i]);
      console.log(`${"".padEnd(14)} native { threads: ${workers} }: ${t.toFixed(0)} ms (${(single / t).toFixed(1)}x), ${ok ? "identical" : "MISMATCH"}`);
      if (!ok) process.exitCode = 1;
    }
  }
  for (const w of pool) await w.terminate();
} else {
  const rw = await init();
  const { frames, out } = workerData;
  parentPort.postMessage("ready");
  parentPort.on("message", ({ op, p0, p1 }) => {
    rw[op](frames.map((f) => f.subarray(p0, p1)), { out: out.subarray(p0, p1) });
    parentPort.postMessage("done");
  });
}
