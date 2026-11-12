//! The kernels behind the `rankwise` npm package, for two front ends that share one request
//! format ([`request`]):
//!
//! * **WebAssembly** (`wasm32-unknown-unknown`, with or without `+simd128`): plain `rw_*`
//!   exports over linear memory, no wasm-bindgen, so one file runs in every engine.
//! * **Node-API** (`--features node`): a native addon for Node, Bun and Deno that reads and
//!   writes the callers' TypedArrays in place, with every region bounds-checked, and can
//!   use AVX-512 and threads.
//!
//! The JS layer (`js/`) owns NumPy's semantics and builds the requests; which front end it
//! talks to is chosen at load time (see `js/node.js`).

#![allow(clippy::missing_safety_doc)]
// A host build without `--features node` has no front end; the request module is still
// compiled (and linted) there.
#![cfg_attr(not(any(target_family = "wasm", feature = "node")), allow(dead_code))]

pub mod request;

#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(feature = "node")]
mod node;
