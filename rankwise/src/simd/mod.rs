//! The small vector abstraction every kernel is written against.
//!
//! One generic kernel is compiled once per instruction set: the per-ISA modules implement
//! [`Vector`] for f32, f64 and u16 lanes, and `#[target_feature]` entry points instantiate the
//! generic code with those types. All methods are `#[inline(always)]`, so the intrinsics end up
//! inside the entry point that enables their target feature.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// A per-lane boolean.
pub trait Mask: Copy {
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    /// `self & !o`
    fn andnot(self, o: Self) -> Self;
    fn not(self) -> Self;
    fn any(self) -> bool;
    /// Lane `i` is bit `i`.
    fn bits(self) -> u64;
}

/// Element types the kernels sort.
pub trait Element: Copy + PartialOrd + Send + Sync + 'static + core::fmt::Debug {
    const MAX: Self;
    const ZERO: Self;
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
    fn from_usize(x: usize) -> Self;
    fn to_usize(self) -> usize;
    fn is_nan(self) -> bool;
}

impl Element for f32 {
    const MAX: Self = f32::INFINITY;
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        x as f32
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        // Only ever called on small non-negative integers (counts, ranks, frame indices).
        debug_assert!(self >= 0.0 && self <= 65535.0);
        unsafe { self.to_int_unchecked::<u32>() as usize }
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}
impl Element for f64 {
    const MAX: Self = f64::INFINITY;
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        x as f64
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        debug_assert!(self >= 0.0 && self <= 65535.0);
        unsafe { self.to_int_unchecked::<u32>() as usize }
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}
impl Element for u16 {
    const MAX: Self = u16::MAX;
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(x: f64) -> Self {
        x as u16
    }
    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        x as u16
    }
    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        false
    }
}

/// A SIMD register of `L` lanes.
///
/// `min`/`max` only need to be exact for non-NaN inputs: NaNs are replaced before any
/// network runs. Comparisons are ordered (false when either side is NaN).
pub trait Vector: Copy + 'static {
    type E: Element;
    type M: Mask;
    const L: usize;
    /// Largest stack depth to run as an unrolled, register-resident network.
    const UNROLL: usize;
    unsafe fn splat(x: Self::E) -> Self;
    unsafe fn load(p: *const Self::E) -> Self;
    unsafe fn store(self, p: *mut Self::E);
    unsafe fn min(self, o: Self) -> Self;
    unsafe fn max(self, o: Self) -> Self;
    unsafe fn add(self, o: Self) -> Self;
    unsafe fn sub(self, o: Self) -> Self;
    /// Lane-wise product (wrapping for integers).
    unsafe fn mul(self, o: Self) -> Self;
    unsafe fn eq(self, o: Self) -> Self::M;
    unsafe fn lt(self, o: Self) -> Self::M;
    unsafe fn gt(self, o: Self) -> Self::M;
    /// `m ? a : b`, lane by lane.
    unsafe fn select(m: Self::M, a: Self, b: Self) -> Self;
    /// Lanes holding NaN (always false for integers).
    unsafe fn nan_mask(self) -> Self::M;
    /// Lanes holding NaN or ±inf (always false for integers).
    unsafe fn nonfinite_mask(self) -> Self::M;
    unsafe fn mask_none() -> Self::M;
}

/// Floating-point lanes.
pub trait FVector: Vector {
    unsafe fn div(self, o: Self) -> Self;
    unsafe fn sqrt(self) -> Self;
    unsafe fn abs(self) -> Self;
}

/// The instruction sets rankwise can dispatch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Isa {
    /// Portable Rust: fixed-size arrays the compiler vectorizes for the baseline target.
    Scalar = 0,
    /// x86-64 AVX2: 8 f32 lanes.
    Avx2 = 1,
    /// x86-64 AVX-512 F/BW/DQ/VL: 16 f32 lanes.
    Avx512 = 2,
    /// aarch64 Advanced SIMD: 4 f32 lanes.
    Neon = 3,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Scalar => "scalar",
            Isa::Avx2 => "avx2",
            Isa::Avx512 => "avx512",
            Isa::Neon => "neon",
        }
    }

    pub fn from_name(s: &str) -> Option<Isa> {
        match s.trim().to_ascii_lowercase().as_str() {
            "scalar" => Some(Isa::Scalar),
            "avx2" => Some(Isa::Avx2),
            "avx512" | "avx-512" => Some(Isa::Avx512),
            "neon" => Some(Isa::Neon),
            _ => None,
        }
    }

    /// f32 lanes per register.
    pub fn f32_lanes(self) -> usize {
        match self {
            Isa::Scalar => 8,
            Isa::Avx2 => 8,
            Isa::Avx512 => 16,
            Isa::Neon => 4,
        }
    }

    /// Whether this CPU can run `self`.
    pub fn is_supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512dq")
                    && std::arch::is_x86_feature_detected!("avx512vl")
            }
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// Every ISA this CPU supports, fastest first.
    pub fn available() -> Vec<Isa> {
        [Isa::Avx512, Isa::Neon, Isa::Avx2, Isa::Scalar].into_iter().filter(|i| i.is_supported()).collect()
    }

    /// The fastest ISA this CPU supports, ignoring overrides.
    pub fn best() -> Isa {
        Isa::available()[0]
    }
}

impl core::fmt::Display for Isa {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

use std::sync::atomic::{AtomicU8, Ordering};

/// 255 = no programmatic override.
static OVERRIDE: AtomicU8 = AtomicU8::new(255);

/// Force an ISA for subsequent calls in this process (`None` restores detection).
///
/// Errors if the CPU cannot run the requested ISA.
pub fn set_isa_override(isa: Option<Isa>) -> crate::Result<()> {
    match isa {
        None => OVERRIDE.store(255, Ordering::Relaxed),
        Some(i) => {
            if !i.is_supported() {
                return Err(crate::Error::UnsupportedIsa(i.name()));
            }
            OVERRIDE.store(i as u8, Ordering::Relaxed)
        }
    }
    Ok(())
}

/// The ISA forced with [`set_isa_override`], if any (not `RANKWISE_ISA`).
pub fn isa_override() -> Option<Isa> {
    match OVERRIDE.load(Ordering::Relaxed) {
        255 => None,
        0 => Some(Isa::Scalar),
        1 => Some(Isa::Avx2),
        2 => Some(Isa::Avx512),
        _ => Some(Isa::Neon),
    }
}

/// The ISA a call will use: an explicit request, else [`set_isa_override`], else the
/// `RANKWISE_ISA` environment variable, else the fastest supported ISA.
pub fn resolve_isa(requested: Option<Isa>) -> crate::Result<Isa> {
    let pick = |i: Isa| -> crate::Result<Isa> {
        if i.is_supported() {
            Ok(i)
        } else {
            Err(crate::Error::UnsupportedIsa(i.name()))
        }
    };
    if let Some(i) = requested {
        return pick(i);
    }
    if let Some(i) = isa_override() {
        return pick(i);
    }
    if let Ok(s) = std::env::var("RANKWISE_ISA") {
        if !s.is_empty() && s != "auto" {
            let i = Isa::from_name(&s).ok_or(crate::Error::UnknownIsa(s))?;
            return pick(i);
        }
    }
    Ok(Isa::best())
}
