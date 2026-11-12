// Assemble the npm release: one package per platform addon plus `rankwise` itself, whose
// staged package.json lists every platform package as an optional dependency (npm, pnpm,
// yarn and bun then install only the one whose os/cpu/libc match). The repository's own
// package.json never lists them: the scope is unregistered, and an install in a checkout
// must not fetch packages someone else could publish under those names.
//
//   node scripts/build.mjs --all                 # dist/*.wasm and npm/<this platform>/
//   node scripts/pack.mjs [--out release] [--only-built]
//
// In CI, each platform's job uploads npm/<platform>/ and the release job runs this with all
// of them present. --only-built lists only the platforms present (local install tests).
import { execFileSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PLATFORMS, addonFile, packageName } from "../js/platforms.js";
import { platformManifest } from "./build.mjs";

const pkg = join(dirname(fileURLToPath(import.meta.url)), "..");

const exists = (p) => stat(p).then(() => true, () => false);
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const pack = (dir, out) => execFileSync(npm, ["pack", "--pack-destination", out, "--silent"], { cwd: dir, encoding: "utf8", shell: process.platform === "win32" }).trim();

/** Pack every built platform package and the main package into `out`. */
export async function packAll({ out = join(pkg, "release"), onlyBuilt = false, quiet = false } = {}) {
  out = resolve(out);
  const manifest = JSON.parse(await readFile(join(pkg, "package.json"), "utf8"));
  for (const f of ["rankwise-simd128.wasm", "rankwise-portable.wasm"]) {
    if (!(await exists(join(pkg, "dist", f)))) throw new Error(`dist/${f} is missing: run node scripts/build.mjs first`);
  }
  await mkdir(out, { recursive: true });
  const built = [];
  for (const key of Object.keys(PLATFORMS)) {
    const dir = join(pkg, "npm", key);
    if (!(await exists(join(dir, addonFile(key))))) continue;
    await writeFile(join(dir, "package.json"), JSON.stringify(platformManifest(key, manifest.version), null, 2) + "\n");
    built.push({ key, tarball: join(out, pack(dir, out)) });
  }
  const listed = onlyBuilt ? built.map((b) => b.key) : Object.keys(PLATFORMS);
  const missing = listed.filter((k) => !built.some((b) => b.key === k));
  if (missing.length && !quiet) console.warn(`warning: no addon built for ${missing.join(", ")} (they will fall back to WebAssembly)`);

  // stage the main package with its optional dependencies
  const stage = await mkdtemp(join(tmpdir(), "rankwise-pack-"));
  try {
    for (const f of ["js", "dist", "README.md"]) await cp(join(pkg, f), join(stage, f), { recursive: true });
    const staged = { ...manifest };
    delete staged.devDependencies;
    staged.scripts = {};
    staged.optionalDependencies = Object.fromEntries(listed.map((k) => [packageName(k), manifest.version]));
    await writeFile(join(stage, "package.json"), JSON.stringify(staged, null, 2) + "\n");
    for (const f of await readdir(join(stage, "dist"))) if (!f.endsWith(".wasm")) await rm(join(stage, "dist", f));
    const main = join(out, pack(stage, out));
    return { main, platforms: built, optional: listed };
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
}

if (fileURLToPath(import.meta.url) === process.argv[1]) {
  const args = process.argv.slice(2);
  const r = await packAll({ out: args.includes("--out") ? args[args.indexOf("--out") + 1] : undefined, onlyBuilt: args.includes("--only-built") });
  console.log(r.main);
  for (const p of r.platforms) console.log(p.tarball);
}
