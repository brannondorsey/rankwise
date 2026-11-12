// Build the package's binaries.
//
//   node scripts/build.mjs                 both WebAssembly modules into dist/
//   node scripts/build.mjs --native        the Node-API addon for this machine into npm/<platform>/
//   node scripts/build.mjs --all           both
//
// Options: --dev (no fat LTO: about 3x quicker to build, a few percent slower), --only
// simd128|portable, --target <rust triple> (cross-compile the addon; its platform is looked up
// in js/platforms.js).
//
// dist/rankwise-simd128.wasm needs WebAssembly SIMD (every current browser, Node >= 16.4, Bun,
// Deno); dist/rankwise-portable.wasm runs anywhere. npm/<platform>/ is laid out as the
// platform's npm package (see scripts/pack.mjs).
import { spawn } from "node:child_process";
import { copyFile, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PLATFORMS, addonFile, packageName, platformKey } from "../js/platforms.js";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const root = join(pkg, "..");
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const value = (f) => (flag(f) ? args[args.indexOf(f) + 1] : null);
const dev = flag("--dev");
const wantNative = flag("--native") || flag("--all");
const wantWasm = !flag("--native") || flag("--all");

function cargo(cmd, { rustflags = "", targetDir }) {
  const env = { ...process.env, RUSTFLAGS: [process.env.RUSTFLAGS, rustflags].filter(Boolean).join(" ") };
  if (dev) {
    env.CARGO_PROFILE_RELEASE_LTO = "false";
    env.CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "16";
  }
  return new Promise((resolve, reject) => {
    const p = spawn("cargo", [...cmd, "--target-dir", targetDir], { cwd: root, env, stdio: "inherit" });
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`cargo exited with ${code}`))));
  });
}

async function wasm() {
  const builds = [
    { name: "simd128", rustflags: "-C target-feature=+simd128" },
    { name: "portable", rustflags: "" },
  ].filter((b) => !value("--only") || b.name === value("--only"));
  await mkdir(join(pkg, "dist"), { recursive: true });
  await Promise.all(
    builds.map(async (b) => {
      const targetDir = join(root, "target", `wasm-${b.name}`);
      await cargo(["build", "--release", "-p", "rankwise-js", "--target", "wasm32-unknown-unknown"], { ...b, targetDir });
      const dst = join(pkg, "dist", `rankwise-${b.name}.wasm`);
      await copyFile(join(targetDir, "wasm32-unknown-unknown", "release", "rankwise_js.wasm"), dst);
      console.log(`${dst}  ${((await stat(dst)).size / 1024).toFixed(0)} KiB`);
    }),
  );
}

/** The npm package of one platform's addon. */
export function platformManifest(key, version) {
  const p = PLATFORMS[key];
  const manifest = {
    name: packageName(key),
    version,
    description: `rankwise's native addon for ${key} (installed automatically with rankwise)`,
    main: addonFile(key),
    files: [addonFile(key)],
    os: [p.os],
    cpu: [p.cpu],
    ...(p.libc ? { libc: [p.libc] } : {}),
    license: "MIT OR Apache-2.0",
    repository: { type: "git", url: "https://github.com/brannondorsey/rankwise", directory: "rankwise-js" },
    engines: { node: ">=18" },
    // remove once the scope is registered (see HANDOFF.md)
    private: true,
  };
  return manifest;
}

async function native() {
  const triple = value("--target");
  const key = triple ? Object.keys(PLATFORMS).find((k) => PLATFORMS[k].rust === triple) : platformKey();
  if (!PLATFORMS[key]) throw new Error(`no native platform entry for ${triple ?? key} in js/platforms.js`);
  const targetDir = join(root, "target", "node");
  const cmd = ["build", "--release", "-p", "rankwise-js", "--features", "node"];
  if (triple) cmd.push("--target", triple);
  await cargo(cmd, { targetDir });
  const os = PLATFORMS[key].os;
  const lib = os === "win32" ? "rankwise_js.dll" : os === "darwin" ? "librankwise_js.dylib" : "librankwise_js.so";
  const out = join(pkg, "npm", key);
  await mkdir(out, { recursive: true });
  const dst = join(out, addonFile(key));
  await copyFile(join(targetDir, ...(triple ? [triple] : []), "release", lib), dst);
  const { version } = JSON.parse(await import("node:fs/promises").then((fs) => fs.readFile(join(pkg, "package.json"), "utf8")));
  await writeFile(join(out, "package.json"), JSON.stringify(platformManifest(key, version), null, 2) + "\n");
  console.log(`${dst}  ${((await stat(dst)).size / 1024).toFixed(0)} KiB`);
}

if (fileURLToPath(import.meta.url) === process.argv[1]) {
  await Promise.all([wantWasm && wasm(), wantNative && native()].filter(Boolean));
}
