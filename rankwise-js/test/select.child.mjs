// One backend choice per process, for api.test.mjs: prints what init() (or initSync()) chose.
import { init, initSync } from "../js/node.js";

const { sync, ...options } = JSON.parse(process.argv[2] ?? "{}");
const rw = sync ? initSync(options) : await init(options);
const f = [Float32Array.of(1, 5, 9), Float32Array.of(2, 6, 7), Float32Array.of(3, 4, 8)];
console.log(
  JSON.stringify({
    backend: rw.backend,
    build: rw.build,
    isa: rw.isa,
    isas: rw.availableIsas(),
    fallbackReason: rw.fallbackReason,
    median: Array.from(rw.median(f)),
  }),
);
