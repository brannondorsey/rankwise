use core::fmt;

/// Everything that can go wrong. Validation happens before any kernel runs, so a call either
/// fails with one of these or writes every output.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// No frames, or more than [`crate::MAX_FRAMES`].
    BadDepth(usize),
    /// Frames, outputs or parameters disagree on shape.
    Shape(String),
    /// An argument is outside its domain (a quantile outside [0, 1], a negative sigma, ...).
    Param(String),
    /// The dtype is not supported for this operation.
    DType(String),
    /// The requested ISA cannot run on this CPU.
    UnsupportedIsa(&'static str),
    /// `RANKWISE_ISA` names no known ISA.
    UnknownIsa(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadDepth(n) => {
                write!(f, "stack depth {n} is out of range: rankwise v0.1 reduces 1..={} frames", crate::MAX_FRAMES)
            }
            Error::Shape(s) => write!(f, "shape mismatch: {s}"),
            Error::Param(s) => write!(f, "invalid parameter: {s}"),
            Error::DType(s) => write!(f, "unsupported dtype: {s}"),
            Error::UnsupportedIsa(s) => write!(f, "this CPU cannot run the {s} kernels"),
            Error::UnknownIsa(s) => write!(f, "unknown ISA {s:?} (expected scalar, avx2, avx512, neon or auto)"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
