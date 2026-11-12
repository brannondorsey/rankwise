// The install story, end to end: pack the release (scripts/pack.mjs), serve it from a
// throwaway local npm registry, install `rankwise` the way a user would, and check what
// lands on disk and what runs.
//
//   node scripts/build.mjs --all && node test/package.test.mjs
//
// Proves that
//   * npm and bun install exactly one addon package, the one matching os/cpu/libc (other
//     platforms are stand-ins here, also checked with --os/--cpu/--libc),
//   * `import "rankwise"` then runs the native addon in Node and Bun,
//   * without optional dependencies it runs WebAssembly and says why,
//   * a browser bundle resolves the WebAssembly entry and contains no native loader.
import assert from "node:assert/strict";
import { execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";
import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PLATFORMS, addonFile, packageName, platformKey } from "../js/platforms.js";
import { platformManifest } from "../scripts/build.mjs";

const work = mkdtempSync(join(tmpdir(), "rankwise-install-"));
const release = join(work, "release");
const host = platformKey();
const bun = (() => {
  try {
    execFileSync("bun", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
})();

// ---- the release: the real main package and host addon, stand-ins for other platforms
const { packAll } = await import("../scripts/pack.mjs");
await packAll({ out: release, quiet: true }); // every platform listed; only this one's addon is real
const version = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")).version;
for (const key of Object.keys(PLATFORMS)) {
  if (existsSync(join(release, `rankwise-${key}-${version}.tgz`))) continue;
  const dir = join(work, "stub", key);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, addonFile(key)), "a stand-in: this platform's addon is built in CI");
  writeFileSync(join(dir, "package.json"), JSON.stringify(platformManifest(key, version)));
  execFileSync("npm", ["pack", "--pack-destination", release, "--silent"], { cwd: dir });
}
assert.ok(existsSync(join(release, `rankwise-${host}-${version}.tgz`)), `build the ${host} addon first (scripts/build.mjs --native)`);

// ---- a registry serving those tarballs (packuments carry os/cpu/libc, as npmjs.org's do)
const packages = new Map();
for (const f of readdirSync(release)) {
  const file = join(release, f);
  const pj = JSON.parse(execFileSync("tar", ["-xzOf", file, "package/package.json"], { encoding: "utf8" }));
  const data = readFileSync(file);
  packages.set(pj.name, { pj, file: f, data, integrity: `sha512-${createHash("sha512").update(data).digest("base64")}` });
}
const requested = new Set();
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url);
  if (url.startsWith("/-/files/")) {
    const p = [...packages.values()].find((x) => x.file === url.slice(9));
    requested.add(p?.pj.name);
    return p ? res.writeHead(200, { "Content-Type": "application/octet-stream" }).end(p.data) : res.writeHead(404).end();
  }
  const p = packages.get(url.slice(1));
  if (!p) return res.writeHead(404, { "Content-Type": "application/json" }).end('{"error":"not found"}');
  const base = `http://127.0.0.1:${server.address().port}`;
  const { private: _, ...meta } = p.pj;
  const doc = {
    name: p.pj.name,
    "dist-tags": { latest: p.pj.version },
    versions: { [p.pj.version]: { ...meta, dist: { tarball: `${base}/-/files/${p.file}`, integrity: p.integrity } } },
  };
  res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(doc));
}).listen(0, "127.0.0.1");
await new Promise((r) => server.once("listening", r));
const registry = `http://127.0.0.1:${server.address().port}/`;

// ---- install into fresh projects
const PROBE = `import { init } from "rankwise";
const rw = await init();
console.log(JSON.stringify({ backend: rw.backend, build: rw.build, isa: rw.isa, fallbackReason: rw.fallbackReason,
  median: Array.from(rw.median([Float32Array.of(1, 5), Float32Array.of(2, 6), Float32Array.of(3, 4)])) }));
`;
// installs run async: the registry is served from this process's event loop
const sh = promisify(execFile);
async function project(name, installer, flags = []) {
  const dir = join(work, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "package.json"), JSON.stringify({ name, private: true, type: "module" }));
  writeFileSync(join(dir, "probe.mjs"), PROBE);
  const quiet = { cwd: dir, encoding: "utf8", timeout: 300_000 };
  if (installer === "npm") {
    await sh("npm", ["install", `rankwise@${version}`, `--registry=${registry}`, `--cache=${join(work, "npm-cache")}`, "--no-audit", "--no-fund", ...flags], quiet);
  } else {
    await sh("bun", ["add", `rankwise@${version}`, `--registry=${registry}`, `--cache-dir=${join(work, "bun-cache")}`, ...flags], quiet);
  }
  const scope = join(dir, "node_modules", "@rankwise");
  return { dir, addons: existsSync(scope) ? readdirSync(scope).sort() : [] };
}
const run = (dir, cmd = process.execPath) => JSON.parse(execFileSync(cmd, ["probe.mjs"], { cwd: dir, encoding: "utf8", env: { ...process.env, RANKWISE_BACKEND: "", RANKWISE_NATIVE_PATH: "" } }));

const results = [];
const check = (name, f) => {
  f();
  results.push(name);
  console.log(`ok   ${name}`);
};

try {
  const npmProject = await project("npm-app", "npm");
  check(`npm installs only ${packageName(host)} and Node runs it natively`, () => {
    assert.deepEqual(npmProject.addons, [host]);
    const r = run(npmProject.dir);
    assert.equal(r.backend, "native");
    assert.equal(r.build, host);
    assert.equal(r.fallbackReason, null);
    assert.deepEqual(r.median, [2, 5]);
    console.log(`     Node: ${r.backend} ${r.build}, isa ${r.isa}`);
  });
  if (bun) {
    check("Bun runs the same install natively", () => {
      const r = run(npmProject.dir, "bun");
      assert.equal(r.backend, "native");
    });
    const bunProject = await project("bun-app", "bun");
    check(`bun install picks the same single addon`, () => {
      assert.deepEqual(bunProject.addons, [host]);
      assert.equal(run(bunProject.dir, "bun").backend, "native");
    });
  }

  const lean = await project("no-optional", "npm", ["--omit=optional"]);
  check("--omit=optional installs no addon and runs WebAssembly, saying why", () => {
    assert.deepEqual(lean.addons, []);
    const r = run(lean.dir);
    assert.equal(r.backend, "wasm");
    assert.match(r.fallbackReason, new RegExp(`${packageName(host)}: Cannot find`));
    assert.deepEqual(r.median, [2, 5]);
  });

  // what other machines would get from the same `npm install`
  for (const [key, p] of Object.entries(PLATFORMS)) {
    if (key === host) continue;
    const flags = [`--os=${p.os}`, `--cpu=${p.cpu}`, ...(p.libc ? [`--libc=${p.libc}`] : [])];
    const other = await project(`as-${key}`, "npm", flags);
    check(`npm install ${flags.join(" ")} installs only ${packageName(key)}`, () => assert.deepEqual(other.addons, [key]));
  }
  const musl = await project("as-linux-x64-musl", "npm", ["--os=linux", "--cpu=x64", "--libc=musl"]);
  check("a platform without a prebuilt addon (linux x64 musl) installs none: WebAssembly at runtime", () => {
    assert.deepEqual(musl.addons, []);
  });

  check("bundling for the browser resolves the WebAssembly entry, with no native loader", () => {
    writeFileSync(join(npmProject.dir, "app.mjs"), 'import { init } from "rankwise";\nconsole.log(await init());\n');
    const esbuild = (platform) =>
      execFileSync("npx", ["-y", "esbuild@0.25.10", "app.mjs", "--bundle", "--format=esm", `--platform=${platform}`, "--log-level=error",
        "--external:node:*"], { cwd: npmProject.dir, encoding: "utf8", env: { ...process.env, npm_config_registry: "https://registry.npmjs.org/" } });
    const web = esbuild("browser");
    assert.ok(!/createRequire|loadNative|RANKWISE_NATIVE_PATH/.test(web), "the browser bundle contains the native loader");
    assert.match(web, /rankwise-simd128\.wasm/);
    const node = esbuild("node");
    assert.match(node, /RANKWISE_NATIVE_PATH/);
  });
} finally {
  server.close();
  rmSync(work, { recursive: true, force: true });
}
console.log(`${results.length} install checks passed; tarballs fetched: ${[...requested].filter(Boolean).sort().join(", ")}`);
