// Cargo runner for wasm32-wasip1 test binaries: runs them under Node's WASI, so the core
// crate's oracle tests check the simd128 kernels against the portable ones bit for bit.
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node rankwise-js/scripts/wasi-run.mjs"
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";

const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [file, ...args], env: process.env, preopens: { ".": "." }, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(file));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
