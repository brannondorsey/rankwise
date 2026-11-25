// Run the conformance suite in headless Chromium: serve the repository root, open
// test/browser.html, wait for its summary. Needs Playwright (`npm i --no-save playwright` and
// `npx playwright install chromium`, or a global install on NODE_PATH).
//
//   node test/browser.mjs
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };
const server = createServer(async (req, res) => {
  const path = normalize(join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)));
  if (!path.startsWith(root)) return res.writeHead(403).end();
  try {
    if (!(await stat(path)).isFile()) throw new Error();
    res.writeHead(200, { "Content-Type": types[extname(path)] ?? "application/octet-stream" });
    createReadStream(path).pipe(res);
  } catch {
    res.writeHead(404).end();
  }
}).listen(0, "127.0.0.1");
await new Promise((r) => server.once("listening", r));

const { chromium } = createRequire(import.meta.url)("playwright");
const browser = await chromium.launch();
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/rankwise-js/test/browser.html`);
await page.waitForFunction(() => globalThis.conformance, null, { timeout: 600_000 });
const { summary, failed } = await page.evaluate(() => globalThis.conformance);
console.log(summary);
if (failed) console.log(await page.textContent("#log"));
await browser.close();
server.close();
process.exitCode = failed ? 1 : 0;
