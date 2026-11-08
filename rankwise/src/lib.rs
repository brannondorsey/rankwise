//! # rankwise
//!
//! Lane-parallel order statistics. rankwise reduces one axis of an array (a stack of frames,
//! ensemble members, scenes, ...) by giving every SIMD lane its own slice: register `k` holds
//! frame `k` for 16 pixels at once (AVX-512, f32), and a Batcher sorting network, pruned to
//! the ranks you ask for, sorts all 16 stacks with branch-free min/max instructions.
//!
//! Results are the same as NumPy's (`median`, `quantile` with every `method`, `nan*`
//! variants), SciPy's (`trim_mean`, `iqr`, `median_abs_deviation`) and astropy's
//! (`sigma_clip` decisions), on every ISA: AVX-512, AVX2, NEON and a portable fallback,
//! chosen at runtime.
//!
//! ```
//! let frames: Vec<Vec<f32>> = (0..31).map(|k| vec![k as f32; 1000]).collect();
//! let views: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();
//! let mut master = vec![0f32; 1000];
//! rankwise::median_frames(&views, &mut master).unwrap();
//! assert_eq!(master[0], 15.0);
//! ```

#![allow(
    clippy::missing_safety_doc,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity,
    clippy::manual_range_contains
)]

mod arch;
mod clip;
mod engine;
mod error;
mod frames;
mod layout;
mod net;
#[doc(hidden)]
pub mod netgen;
mod plan;
pub mod quantile;
pub mod simd;
mod spec;

#[cfg(feature = "ndarray")]
mod nd;

pub use error::{Error, Result};
pub use frames::*;
pub use layout::{c_strides, RawMask, RawOut, RawStack};
pub use plan::{describe, natural_dtype, network_size, reduce_raw, Normalize, Options, Staging};
pub use simd::{resolve_isa, set_isa_override, Isa};
pub use spec::*;

#[cfg(feature = "ndarray")]
pub use nd::{Reduce, Reduction};

/// Deepest stack v0.1 reduces.
pub const MAX_FRAMES: usize = 512;

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
