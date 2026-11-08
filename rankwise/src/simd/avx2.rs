//! x86-64 AVX2: 8 f32, 4 f64 or 16 u16 lanes per 256-bit register.

use super::{FVector, Mask, Vector};
use core::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct M32(pub __m256);
#[derive(Clone, Copy)]
pub struct M64(pub __m256d);
#[derive(Clone, Copy)]
pub struct M16(pub __m256i);

impl Mask for M32 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M32(_mm256_and_ps(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M32(_mm256_or_ps(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M32(_mm256_andnot_ps(o.0, self.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M32(_mm256_xor_ps(self.0, _mm256_castsi256_ps(_mm256_set1_epi32(-1)))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_ps(self.0) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        unsafe { _mm256_movemask_ps(self.0) as u32 as u64 }
    }
}
impl Mask for M64 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M64(_mm256_and_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M64(_mm256_or_pd(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M64(_mm256_andnot_pd(o.0, self.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M64(_mm256_xor_pd(self.0, _mm256_castsi256_pd(_mm256_set1_epi32(-1)))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_pd(self.0) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        unsafe { _mm256_movemask_pd(self.0) as u32 as u64 }
    }
}
impl Mask for M16 {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        unsafe { M16(_mm256_and_si256(self.0, o.0)) }
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        unsafe { M16(_mm256_or_si256(self.0, o.0)) }
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        unsafe { M16(_mm256_andnot_si256(o.0, self.0)) }
    }
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { M16(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
    }
    #[inline(always)]
    fn any(self) -> bool {
        unsafe { _mm256_movemask_epi8(self.0) != 0 }
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        let m = unsafe { _mm256_movemask_epi8(self.0) } as u32;
        let mut b = 0u64;
        for i in 0..16 {
            b |= (((m >> (2 * i)) & 1) as u64) << i;
        }
        b
    }
}

#[derive(Clone, Copy)]
pub struct F32x8(pub __m256);
#[derive(Clone, Copy)]
pub struct F64x4(pub __m256d);
#[derive(Clone, Copy)]
pub struct U16x16(pub __m256i);

impl Vector for F32x8 {
    type E = f32;
    type M = M32;
    const L: usize = 8;
    const UNROLL: usize = 16;
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        F32x8(_mm256_set1_ps(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        F32x8(_mm256_loadu_ps(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        _mm256_storeu_ps(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F32x8(_mm256_min_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F32x8(_mm256_max_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F32x8(_mm256_add_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F32x8(_mm256_sub_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F32x8(_mm256_mul_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M32 {
        M32(_mm256_cmp_ps::<_CMP_EQ_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M32 {
        M32(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M32 {
        M32(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M32, a: Self, b: Self) -> Self {
        F32x8(_mm256_blendv_ps(b.0, a.0, m.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M32 {
        M32(_mm256_cmp_ps::<_CMP_UNORD_Q>(self.0, self.0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M32 {
        let a = self.abs();
        M32(_mm256_cmp_ps::<_CMP_NLT_UQ>(a.0, _mm256_set1_ps(f32::INFINITY)))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M32 {
        M32(_mm256_setzero_ps())
    }
}
impl FVector for F32x8 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F32x8(_mm256_div_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F32x8(_mm256_sqrt_ps(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F32x8(_mm256_andnot_ps(_mm256_set1_ps(-0.0), self.0))
    }
}

impl Vector for F64x4 {
    type E = f64;
    type M = M64;
    const L: usize = 4;
    const UNROLL: usize = 16;
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        F64x4(_mm256_set1_pd(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        F64x4(_mm256_loadu_pd(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        _mm256_storeu_pd(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F64x4(_mm256_min_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F64x4(_mm256_max_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F64x4(_mm256_add_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F64x4(_mm256_sub_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F64x4(_mm256_mul_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M64 {
        M64(_mm256_cmp_pd::<_CMP_EQ_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M64 {
        M64(_mm256_cmp_pd::<_CMP_LT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M64 {
        M64(_mm256_cmp_pd::<_CMP_GT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: M64, a: Self, b: Self) -> Self {
        F64x4(_mm256_blendv_pd(b.0, a.0, m.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M64 {
        M64(_mm256_cmp_pd::<_CMP_UNORD_Q>(self.0, self.0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M64 {
        let a = self.abs();
        M64(_mm256_cmp_pd::<_CMP_NLT_UQ>(a.0, _mm256_set1_pd(f64::INFINITY)))
    }
    #[inline(always)]
    unsafe fn mask_none() -> M64 {
        M64(_mm256_setzero_pd())
    }
}
impl FVector for F64x4 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F64x4(_mm256_div_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F64x4(_mm256_sqrt_pd(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F64x4(_mm256_andnot_pd(_mm256_set1_pd(-0.0), self.0))
    }
}

impl Vector for U16x16 {
    type E = u16;
    type M = M16;
    const L: usize = 16;
    const UNROLL: usize = 16;
    #[inline(always)]
    unsafe fn splat(x: u16) -> Self {
        U16x16(_mm256_set1_epi16(x as i16))
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> Self {
        U16x16(_mm256_loadu_si256(p as *const __m256i))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u16) {
        _mm256_storeu_si256(p as *mut __m256i, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        U16x16(_mm256_min_epu16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        U16x16(_mm256_max_epu16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        U16x16(_mm256_add_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        U16x16(_mm256_sub_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        U16x16(_mm256_mullo_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> M16 {
        M16(_mm256_cmpeq_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> M16 {
        // a < b  <=>  max(a, b) != a
        let m = _mm256_cmpeq_epi16(_mm256_max_epu16(self.0, o.0), self.0);
        M16(_mm256_xor_si256(m, _mm256_set1_epi16(-1)))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> M16 {
        o.lt(self)
    }
    #[inline(always)]
    unsafe fn select(m: M16, a: Self, b: Self) -> Self {
        U16x16(_mm256_blendv_epi8(b.0, a.0, m.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> M16 {
        M16(_mm256_setzero_si256())
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> M16 {
        M16(_mm256_setzero_si256())
    }
    #[inline(always)]
    unsafe fn mask_none() -> M16 {
        M16(_mm256_setzero_si256())
    }
}
