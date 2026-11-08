//! aarch64 Advanced SIMD (NEON): 4 f32, 2 f64 or 8 u16 lanes per 128-bit register.
//! NEON is part of the aarch64 baseline, so no runtime detection is needed.

use super::{FVector, Mask, Vector};
use core::arch::aarch64::*;

#[derive(Clone, Copy)]
pub struct M32(pub uint32x4_t);
#[derive(Clone, Copy)]
pub struct M64(pub uint64x2_t);
#[derive(Clone, Copy)]
pub struct M16(pub uint16x8_t);

impl Mask for M32 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M32(vandq_u32(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M32(vorrq_u32(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M32(vbicq_u32(self.0, o.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M32(vmvnq_u32(self.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u32(self.0) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        let mut a = [0u32; 4];
        unsafe { vst1q_u32(a.as_mut_ptr(), self.0) };
        (0..4).fold(0u64, |b, i| b | (((a[i] != 0) as u64) << i))
    }
}
impl Mask for M64 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M64(vandq_u64(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M64(vorrq_u64(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M64(vbicq_u64(self.0, o.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M64(vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(self.0)))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u32(vreinterpretq_u32_u64(self.0)) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        let mut a = [0u64; 2];
        unsafe { vst1q_u64(a.as_mut_ptr(), self.0) };
        ((a[0] != 0) as u64) | (((a[1] != 0) as u64) << 1)
    }
}
impl Mask for M16 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M16(vandq_u16(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M16(vorrq_u16(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M16(vbicq_u16(self.0, o.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M16(vmvnq_u16(self.0)) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { vmaxvq_u16(self.0) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        let mut a = [0u16; 8];
        unsafe { vst1q_u16(a.as_mut_ptr(), self.0) };
        (0..8).fold(0u64, |b, i| b | (((a[i] != 0) as u64) << i))
    }
}

#[derive(Clone, Copy)]
pub struct F32x4(pub float32x4_t);
#[derive(Clone, Copy)]
pub struct F64x2(pub float64x2_t);
#[derive(Clone, Copy)]
pub struct U16x8(pub uint16x8_t);

impl Vector for F32x4 {
    type E = f32;
    type M = M32;
    const L: usize = 4;
    const UNROLL: usize = 24;
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        F32x4(vdupq_n_f32(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        F32x4(vld1q_f32(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        vst1q_f32(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F32x4(vminq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F32x4(vmaxq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F32x4(vaddq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F32x4(vsubq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F32x4(vmulq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M32 {
        M32(vceqq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M32 {
        M32(vcltq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M32 {
        M32(vcgtq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M32, a: Self, b: Self) -> Self {
        F32x4(vbslq_f32(m.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M32 {
        M32(vmvnq_u32(vceqq_f32(self.0, self.0)))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M32 {
        M32(vmvnq_u32(vcltq_f32(vabsq_f32(self.0), vdupq_n_f32(f32::INFINITY))))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M32 {
        M32(vdupq_n_u32(0))
    }
}
impl FVector for F32x4 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F32x4(vdivq_f32(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F32x4(vsqrtq_f32(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F32x4(vabsq_f32(self.0))
    }
}

impl Vector for F64x2 {
    type E = f64;
    type M = M64;
    const L: usize = 2;
    const UNROLL: usize = 24;
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        F64x2(vdupq_n_f64(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        F64x2(vld1q_f64(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        vst1q_f64(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F64x2(vminq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F64x2(vmaxq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F64x2(vaddq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F64x2(vsubq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F64x2(vmulq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M64 {
        M64(vceqq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M64 {
        M64(vcltq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M64 {
        M64(vcgtq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M64, a: Self, b: Self) -> Self {
        F64x2(vbslq_f64(m.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M64 {
        let eq = vreinterpretq_u32_u64(vceqq_f64(self.0, self.0));
        M64(vreinterpretq_u64_u32(vmvnq_u32(eq)))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M64 {
        let fin = vreinterpretq_u32_u64(vcltq_f64(vabsq_f64(self.0), vdupq_n_f64(f64::INFINITY)));
        M64(vreinterpretq_u64_u32(vmvnq_u32(fin)))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M64 {
        M64(vdupq_n_u64(0))
    }
}
impl FVector for F64x2 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F64x2(vdivq_f64(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F64x2(vsqrtq_f64(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F64x2(vabsq_f64(self.0))
    }
}

impl Vector for U16x8 {
    type E = u16;
    type M = M16;
    const L: usize = 8;
    const UNROLL: usize = 24;
    #[inline(always)]
    unsafe fn splat(x: u16) -> Self {
        U16x8(vdupq_n_u16(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> Self {
        U16x8(vld1q_u16(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u16) {
        vst1q_u16(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        U16x8(vminq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        U16x8(vmaxq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        U16x8(vaddq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        U16x8(vsubq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        U16x8(vmulq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M16 {
        M16(vceqq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M16 {
        M16(vcltq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M16 {
        M16(vcgtq_u16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M16, a: Self, b: Self) -> Self {
        U16x8(vbslq_u16(m.0, a.0, b.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M16 {
        M16(vdupq_n_u16(0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M16 {
        M16(vdupq_n_u16(0))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M16 {
        M16(vdupq_n_u16(0))
    }
}
