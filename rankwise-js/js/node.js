// rankwise for Node, Bun and Deno: the native addon when this platform has one, else
// WebAssembly. Package managers install only the addon package that matches the machine
// (an optional dependency with `os`/`cpu`/`libc`); this module loads it, and falls back to the
// WebAssembly build, saying why, when it is missing or cannot load.
//
//   import { init } from "rankwise";      // resolves here under the "node" export condition
//   const rw = await init();
//   rw.backend        // "native" or "wasm"
//   rw.isa            // "avx512", "avx2", "neon", "simd128", ...
//   rw.fallbackReason // why native was not used, or null
//
// Choose explicitly with `init({ backend: "native" | "wasm" | "auto" })` or the
// RANKWISE_BACKEND environment variable; RANKWISE_NATIVE_PATH loads an addon from a file.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { Rankwise } from "./core.js";
import { NativeEngine } from "./native.js";
import { addonFile, packageName, platformKey } from "./platforms.js";
import { chooseSimd, wasmEngine, wasmEngineSync, wasmUrl } from "./wasm.js";

export { Rankwise, RankwiseError, ZeroCopyArray } from "./core.js";
export { simdSupported, WasmArray } from "./wasm.js";

const require = createRequire(import.meta.url);
const BACKENDS = ["auto", "native", "wasm"];

function env(name) {
  try {
    return globalThis.process?.env?.[name] || undefined; // Deno without --allow-env throws
  } catch {
    return undefined;
  }
}

function pickBackend(options) {
  const b = options.backend ?? env("RANKWISE_BACKEND") ?? "auto";
  if (!BACKENDS.includes(b)) throw new RangeError(`rankwise: backend must be "auto", "native" or "wasm", got ${JSON.stringify(b)}`);
  return b;
}

let native = null;

/**
 * Load the native addon once: `{ engine }`, or `{ reason }` saying why not. Tries
 * RANKWISE_NATIVE_PATH if set, else a local build (`npm/<platform>/`, in a checkout of the
 * repository) and then the platform's package.
 */
function loadNative() {
  if (native) return native;
  const key = platformKey(undefined, undefined, (p) => readFileSync(p, "utf8"));
  const explicit = env("RANKWISE_NATIVE_PATH");
  const candidates = explicit
    ? [explicit]
    : [fileURLToPath(new URL(`../npm/${key}/${addonFile(key)}`, import.meta.url)), packageName(key)];
  const tried = [];
  for (const c of candidates) {
    try {
      native = { engine: new NativeEngine(require(c), key) };
      return native;
    } catch (e) {
      tried.push(`${c}: ${String(e?.message ?? e).split("\n")[0]}`);
    }
  }
  native = { reason: `no native addon for ${key} could be loaded (${tried.join("; ")})` };
  return native;
}

function resolveSync(options) {
  const backend = pickBackend(options);
  if (backend !== "wasm" && options.wasm === undefined) {
    const n = loadNative();
    if (n.engine) return { rw: new Rankwise(n.engine) };
    if (backend === "native") throw new Error(`rankwise: ${n.reason}`);
    return { wasm: true, reason: n.reason };
  }
  return { wasm: true, reason: null };
}

const cache = new Map();

/**
 * Load rankwise: the native addon if it loads (`backend: "auto"`, the default), else
 * WebAssembly. Resolves to a {@link Rankwise}; `rw.backend` says which, and
 * `rw.fallbackReason` why native was not used.
 *
 * Options: `backend` ("auto" | "native" | "wasm"; default RANKWISE_BACKEND or "auto"),
 * and for WebAssembly `simd` and `wasm` as in the browser build.
 */
export function init(options = {}) {
  let key;
  try {
    key = `${pickBackend(options)}|${options.simd ?? "auto"}`;
  } catch (e) {
    return Promise.reject(e);
  }
  const make = async () => {
    const r = resolveSync(options);
    if (r.rw) return r.rw;
    const rw = new Rankwise(await wasmEngine(options));
    rw.fallbackReason = r.reason;
    return rw;
  };
  if (options.wasm !== undefined) return make();
  if (!cache.has(key)) {
    const p = make();
    p.catch(() => cache.delete(key));
    cache.set(key, p);
  }
  return cache.get(key);
}

/**
 * Synchronous {@link init}. With no argument (or options) it makes the same choice as
 * `init()`; with WebAssembly bytes or a compiled module it loads those.
 */
export function initSync(arg = {}) {
  if (arg instanceof WebAssembly.Module || arg instanceof ArrayBuffer || ArrayBuffer.isView(arg)) {
    return new Rankwise(wasmEngineSync(arg));
  }
  const r = resolveSync(arg);
  if (r.rw) return r.rw;
  const bytes = arg.wasm ?? readFileSync(wasmUrl(chooseSimd(arg)));
  const rw = new Rankwise(wasmEngineSync(bytes instanceof URL ? readFileSync(bytes) : bytes));
  rw.fallbackReason = r.reason;
  return rw;
}

export default init;
