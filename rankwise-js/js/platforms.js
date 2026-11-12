// Platforms with a prebuilt native addon. Each ships as its own npm package, an optional
// dependency of `rankwise` whose `os`, `cpu` and `libc` fields let npm, pnpm, yarn and bun
// install only the one that matches; names follow napi-rs ("linux-x64-gnu").
// The build and release scripts read this table too.

export const SCOPE = "@rankwise";

export const PLATFORMS = {
  "linux-x64-gnu": { os: "linux", cpu: "x64", libc: "glibc", rust: "x86_64-unknown-linux-gnu" },
  "linux-arm64-gnu": { os: "linux", cpu: "arm64", libc: "glibc", rust: "aarch64-unknown-linux-gnu" },
  "darwin-arm64": { os: "darwin", cpu: "arm64", rust: "aarch64-apple-darwin" },
  "darwin-x64": { os: "darwin", cpu: "x64", rust: "x86_64-apple-darwin" },
  "win32-x64-msvc": { os: "win32", cpu: "x64", rust: "x86_64-pc-windows-msvc" },
};

function musl(readText) {
  try {
    const r = globalThis.process?.report?.getReport?.();
    const header = typeof r === "string" ? JSON.parse(r).header : r?.header;
    if (header && "glibcVersionRuntime" in header) return !header.glibcVersionRuntime;
  } catch {}
  try {
    return readText?.("/usr/bin/ldd").includes("musl") ?? false;
  } catch {
    return false;
  }
}

/**
 * This process's platform key (e.g. "linux-x64-gnu"), whether or not a binding is built for
 * it. `readText` reads a file (used to tell musl from glibc where the process report can't).
 */
export function platformKey(platform = globalThis.process?.platform, arch = globalThis.process?.arch, readText) {
  if (platform === "linux") return `linux-${arch}-${musl(readText) ? "musl" : "gnu"}`;
  if (platform === "win32") return `win32-${arch}-msvc`;
  return `${platform}-${arch}`;
}

/** The npm package that holds the addon for `key`. */
export const packageName = (key) => `${SCOPE}/${key}`;

/** The addon's file name inside that package. */
export const addonFile = (key) => `rankwise.${key}.node`;
