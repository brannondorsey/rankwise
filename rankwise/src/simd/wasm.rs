//! WebAssembly SIMD (`simd128`): 4 f32, 2 f64 or 8 u16 lanes per 128-bit register.
//!
//! WebAssembly has no runtime feature detection: a module either uses `simd128` or it does
//! not, so this backend exists only when the crate is compiled with
//! `-C target-feature=+simd128` (the JS loader picks that build or the portable one).
//!
//! `min`/`max` use `pmin`/`pmax` with the operands swapped, which is exactly x86's
//! `minps`/`maxps` (`a < b ? a : b`, second operand on ties and ±0), so results match x86 on
//! every host, and V8 lowers them to those instructions on x86-64. The IEEE `f32x4.min`
//! orders -0 below +0 and lowers to a longer sequence.

use super::{FVector, Mask, Vector};
use core::arch::wasm32::*;

#[derive(Clone, Copy)]
pub struct M32(pub v128);
#[derive(Clone, Copy)]
pub struct M64(pub v128);
#[derive(Clone, Copy)]
pub struct M16(pub v128);

macro_rules! mask {
    ($m:ident, $bitmask:ident) => {
        impl Mask for $m {
            #[inline(always)]
            fn and(self, o: Self) -> Self {
                $m(v128_and(self.0, o.0))
            }
            #[inline(always)]
            fn or(self, o: Self) -> Self {
                $m(v128_or(self.0, o.0))
            }
            #[inline(always)]
            fn andnot(self, o: Self) -> Self {
                $m(v128_andnot(self.0, o.0))
            }
            #[inline(always)]
            fn not(self) -> Self {
                $m(v128_not(self.0))
            }
            #[inline(always)]
            fn any(self) -> bool {
                v128_any_true(self.0)
            }
            #[inline(always)]
            fn bits(self) -> u64 {
                $bitmask(self.0) as u64
            }
        }
    };
}
mask!(M32, i32x4_bitmask);
mask!(M64, i64x2_bitmask);
mask!(M16, i16x8_bitmask);

#[derive(Clone, Copy)]
pub struct F32x4(pub v128);
#[derive(Clone, Copy)]
pub struct F64x2(pub v128);
#[derive(Clone, Copy)]
pub struct U16x8(pub v128);

/// Unrolled networks keep every frame in a local, which engines map to 16 (x86-64) or 32
/// (arm64) vector registers and spill beyond. 16, 24 and 32 measured within a few percent of
/// each other in V8 on x86-64, interpreted networks 1.4x slower (see docs/javascript.md).
const UNROLL: usize = 24;

macro_rules! fvec {
    ($v:ident, $m:ident, $e:ty, $l:expr, $splat:ident, $pmin:ident, $pmax:ident, $add:ident, $sub:ident,
     $mul:ident, $div:ident, $sqrt:ident, $abs:ident, $eq:ident, $ne:ident, $lt:ident, $gt:ident, $inf:expr) => {
        impl Vector for $v {
            type E = $e;
            type M = $m;
            const L: usize = $l;
            const UNROLL: usize = UNROLL;
            #[inline(always)]
            unsafe fn splat(x: $e) -> Self {
                $v($splat(x))
            }
            #[inline(always)]
            unsafe fn load(p: *const $e) -> Self {
                $v(v128_load(p as *const v128))
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut $e) {
                v128_store(p as *mut v128, self.0)
            }
            #[inline(always)]
            unsafe fn min(self, o: Self) -> Self {
                $v($pmin(o.0, self.0))
            }
            #[inline(always)]
            unsafe fn max(self, o: Self) -> Self {
                $v($pmax(o.0, self.0))
            }
            #[inline(always)]
            unsafe fn add(self, o: Self) -> Self {
                $v($add(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn sub(self, o: Self) -> Self {
                $v($sub(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn mul(self, o: Self) -> Self {
                $v($mul(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn eq(self, o: Self) -> $m {
                $m($eq(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn lt(self, o: Self) -> $m {
                $m($lt(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn gt(self, o: Self) -> $m {
                $m($gt(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn select(m: $m, a: Self, b: Self) -> Self {
                $v(v128_bitselect(a.0, b.0, m.0))
            }
            #[inline(always)]
            unsafe fn nan_mask(self) -> $m {
                $m($ne(self.0, self.0))
            }
            #[inline(always)]
            unsafe fn nonfinite_mask(self) -> $m {
                $m(v128_not($lt($abs(self.0), $splat($inf))))
            }
            #[inline(always)]
            unsafe fn mask_none() -> $m {
                $m(i64x2_splat(0))
            }
        }
        impl FVector for $v {
            #[inline(always)]
            unsafe fn div(self, o: Self) -> Self {
                $v($div(self.0, o.0))
            }
            #[inline(always)]
            unsafe fn sqrt(self) -> Self {
                $v($sqrt(self.0))
            }
            #[inline(always)]
            unsafe fn abs(self) -> Self {
                $v($abs(self.0))
            }
        }
    };
}

fvec!(
    F32x4,
    M32,
    f32,
    4,
    f32x4_splat,
    f32x4_pmin,
    f32x4_pmax,
    f32x4_add,
    f32x4_sub,
    f32x4_mul,
    f32x4_div,
    f32x4_sqrt,
    f32x4_abs,
    f32x4_eq,
    f32x4_ne,
    f32x4_lt,
    f32x4_gt,
    f32::INFINITY
);
fvec!(
    F64x2,
    M64,
    f64,
    2,
    f64x2_splat,
    f64x2_pmin,
    f64x2_pmax,
    f64x2_add,
    f64x2_sub,
    f64x2_mul,
    f64x2_div,
    f64x2_sqrt,
    f64x2_abs,
    f64x2_eq,
    f64x2_ne,
    f64x2_lt,
    f64x2_gt,
    f64::INFINITY
);

impl Vector for U16x8 {
    type E = u16;
    type M = M16;
    const L: usize = 8;
    const UNROLL: usize = UNROLL;
    #[inline(always)]
    unsafe fn splat(x: u16) -> Self {
        U16x8(u16x8_splat(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> Self {
        U16x8(v128_load(p as *const v128))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u16) {
        v128_store(p as *mut v128, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        U16x8(u16x8_min(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        U16x8(u16x8_max(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        U16x8(i16x8_add(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        U16x8(i16x8_sub(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        U16x8(i16x8_mul(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M16 {
        M16(i16x8_eq(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M16 {
        M16(u16x8_lt(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M16 {
        M16(u16x8_gt(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M16, a: Self, b: Self) -> Self {
        U16x8(v128_bitselect(a.0, b.0, m.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M16 {
        M16(i64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M16 {
        M16(i64x2_splat(0))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M16 {
        M16(i64x2_splat(0))
    }
}
