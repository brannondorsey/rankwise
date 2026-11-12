// Build the two WebAssembly modules the package ships:
//   dist/rankwise-simd128.wasm   compiled with +simd128 (every current browser, Node >= 16.4, Bun, Deno)
//   dist/rankwise-portable.wasm  no SIMD, for older engines
// The loader picks one at runtime. Usage: node scripts/build.mjs [--dev] [--only simd128|portable]
import { spawn } from "node:child_process";
import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const root = join(pkg, "..");
const args = process.argv.slice(2);
const dev = args.includes("--dev");
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;

const builds = [
  { name: "simd128", rustflags: "-C target-feature=+simd128" },
  { name: "portable", rustflags: "" },
].filter((b) => !only || b.name === only);

function cargo(b) {
  const targetDir = join(root, "target", `wasm-${b.name}`);
  const env = { ...process.env, RUSTFLAGS: [process.env.RUSTFLAGS, b.rustflags].filter(Boolean).join(" ") };
  if (dev) {
    // fast iteration: no fat LTO (about 3x quicker to build, a few percent slower)
    env.CARGO_PROFILE_RELEASE_LTO = "false";
    env.CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "16";
  }
  const cmd = ["build", "--release", "-p", "rankwise-wasm", "--target", "wasm32-unknown-unknown", "--target-dir", targetDir];
  return new Promise((resolve, reject) => {
    const p = spawn("cargo", cmd, { cwd: root, env, stdio: "inherit" });
    p.on("exit", (code) => (code === 0 ? resolve(join(targetDir, "wasm32-unknown-unknown", "release", "rankwise_wasm.wasm")) : reject(new Error(`cargo exited with ${code}`))));
  });
}

await mkdir(join(pkg, "dist"), { recursive: true });
const outs = await Promise.all(builds.map(cargo));
for (const [i, b] of builds.entries()) {
  const dst = join(pkg, "dist", `rankwise-${b.name}.wasm`);
  await copyFile(outs[i], dst);
  console.log(`${dst}  ${((await stat(dst)).size / 1024).toFixed(0)} KiB`);
}
