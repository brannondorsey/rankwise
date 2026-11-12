// An ensemble-statistics HTTP service on Bun: POST a stack of float32 members, get back
// quantiles (or a median / sigma-clipped mean) as raw float32 or JSON.
//
//   node rankwise-wasm/scripts/build.mjs              # once
//   bun examples/js/bun-server.ts                     # serves on :8787
//   bun examples/js/bun-server.ts --selftest          # starts, posts a 31-member grid, checks, exits
//
//   curl -s --data-binary @members.f32 \
//     "http://localhost:8787/quantile?members=31&q=0.1,0.5,0.9" -o quantiles.f32
//
// The body is `members` float32 frames back to back (little-endian); the response is one
// float32 array per statistic, back to back, with the layout in the X-Rankwise-* headers.
import { init, RankwiseError } from "../../rankwise-wasm/js/rankwise.js";

const rw = await init();
const port = Number(process.env.PORT ?? 8787);

function frames(body: ArrayBuffer, members: number): Float32Array[] {
  if (!Number.isInteger(members) || members < 1) throw new RangeError("members must be a positive integer");
  const all = new Float32Array(body);
  if (all.length % members) throw new RangeError(`${all.length} values do not split into ${members} members`);
  const cells = all.length / members;
  return Array.from({ length: members }, (_, k) => all.subarray(k * cells, (k + 1) * cells));
}

function reply(arrays: Float32Array[], names: string[], ms: number, json: boolean): Response {
  const headers = {
    "X-Rankwise-Outputs": names.join(","),
    "X-Rankwise-Cells": String(arrays[0].length),
    "X-Rankwise-Ms": ms.toFixed(2),
    "X-Rankwise-Build": `${rw.build}/${rw.isa}`,
  };
  if (json) return Response.json(Object.fromEntries(names.map((n, j) => [n, Array.from(arrays[j])])), { headers });
  const out = new Float32Array(arrays.length * arrays[0].length);
  arrays.forEach((a, j) => out.set(a, j * a.length));
  return new Response(out, { headers: { ...headers, "Content-Type": "application/octet-stream" } });
}

const server = Bun.serve({
  port,
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === "GET" && url.pathname === "/") {
      return Response.json({ rankwise: rw.version, build: rw.build, isa: rw.isa, endpoints: ["/quantile", "/median", "/sigma-clip"] });
    }
    if (req.method !== "POST") return new Response("POST a float32 stack", { status: 405 });
    try {
      const x = frames(await req.arrayBuffer(), Number(url.searchParams.get("members")));
      const json = url.searchParams.get("format") === "json";
      const t = performance.now();
      if (url.pathname === "/quantile") {
        const q = (url.searchParams.get("q") ?? "0.1,0.5,0.9").split(",").map(Number);
        // float32 out: interpolate as np.quantile(x, np.float32(q)) does
        const res = rw.quantile(x, Float32Array.from(q), { method: url.searchParams.get("method") ?? "linear" }) as Float32Array[];
        return reply(res, q.map((v) => `q${v}`), performance.now() - t, json);
      }
      if (url.pathname === "/median") return reply([rw.median(x) as Float32Array], ["median"], performance.now() - t, json);
      if (url.pathname === "/sigma-clip") {
        const sigma = Number(url.searchParams.get("sigma") ?? 3);
        return reply([rw.sigmaClipMean(x, { sigma }) as Float32Array], ["mean"], performance.now() - t, json);
      }
      return new Response("not found", { status: 404 });
    } catch (e) {
      const status = e instanceof RankwiseError || e instanceof RangeError || e instanceof TypeError ? 400 : 500;
      return new Response(String((e as Error).message), { status });
    }
  },
});
console.log(`rankwise ${rw.version} (${rw.build}) serving on http://localhost:${server.port}`);

if (process.argv.includes("--selftest")) {
  const members = 31;
  const cells = 721 * 1440; // a 0.25° global grid
  const body = new Float32Array(members * cells);
  for (let i = 0; i < body.length; i++) body[i] = 280 + 15 * Math.sin(i % cells) + ((i * 2654435761) % 1000) / 100;
  const t = performance.now();
  const res = await fetch(`http://localhost:${server.port}/quantile?members=${members}&q=0.1,0.5,0.9`, { method: "POST", body });
  const out = new Float32Array(await res.arrayBuffer());
  const total = performance.now() - t;
  const kernel = res.headers.get("X-Rankwise-Ms");
  // check one cell against a sort
  const col = Array.from({ length: members }, (_, k) => body[k * cells + 12345]).sort((a, b) => a - b);
  console.log(`p10/p50/p90 of ${members} × ${cells} cells: ${kernel} ms in rankwise, ${total.toFixed(0)} ms round trip`);
  console.log(`cell 12345: p50 ${out[cells + 12345]} (sorted column's middle: ${col[15]})`);
  server.stop();
  if (out[cells + 12345] !== col[15]) process.exit(1);
}
