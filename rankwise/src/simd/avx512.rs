//! x86-64 AVX-512 (F, BW, DQ, VL): 16 f32, 8 f64 or 32 u16 lanes per 512-bit register.
//! Comparisons write opmask registers, so masked passes cost one instruction.

use super::{FVector, Mask, Vector};
use core::arch::x86_64::*;

macro_rules! kmask {
    ($name:ident, $t:ty) => {
        #[derive(Clone, Copy)]
        pub struct $name(pub $t);
        impl Mask for $name {
            #[inline(always)]
            fn and(self, o: Self) -> Self {
                $name(self.0 & o.0)
            }
            #[inline(always)]
            fn or(self, o: Self) -> Self {
                $name(self.0 | o.0)
            }
            #[inline(always)]
            fn andnot(self, o: Self) -> Self {
                $name(self.0 & !o.0)
            }
            #[inline(always)]
            fn not(self) -> Self {
                $name(!self.0)
            }
            #[inline(always)]
            fn any(self) -> bool {
                self.0 != 0
            }
            #[inline(always)]
            fn bits(self) -> u64 {
                self.0 as u64
            }
        }
    };
}
kmask!(K8, u8);
kmask!(K16, u16);
kmask!(K32, u32);

#[derive(Clone, Copy)]
pub struct F32x16(pub __m512);
#[derive(Clone, Copy)]
pub struct F64x8(pub __m512d);
#[derive(Clone, Copy)]
pub struct U16x32(pub __m512i);

impl Vector for F32x16 {
    type E = f32;
    type M = K16;
    const L: usize = 16;
    const UNROLL: usize = 32;
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        F32x16(_mm512_set1_ps(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        F32x16(_mm512_loadu_ps(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        _mm512_storeu_ps(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F32x16(_mm512_min_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F32x16(_mm512_max_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F32x16(_mm512_add_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F32x16(_mm512_sub_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F32x16(_mm512_mul_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> K16 {
        K16(_mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> K16 {
        K16(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> K16 {
        K16(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: K16, a: Self, b: Self) -> Self {
        F32x16(_mm512_mask_blend_ps(m.0, b.0, a.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> K16 {
        K16(_mm512_cmp_ps_mask::<_CMP_UNORD_Q>(self.0, self.0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> K16 {
        K16(_mm512_cmp_ps_mask::<_CMP_NLT_UQ>(
            _mm512_abs_ps(self.0),
            _mm512_set1_ps(f32::INFINITY),
        ))
    }
    #[inline(always)]
    unsafe fn mask_none() -> K16 {
        K16(0)
    }
}
impl FVector for F32x16 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F32x16(_mm512_div_ps(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F32x16(_mm512_sqrt_ps(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F32x16(_mm512_abs_ps(self.0))
    }
}

impl Vector for F64x8 {
    type E = f64;
    type M = K8;
    const L: usize = 8;
    const UNROLL: usize = 32;
    #[inline(always)]
    unsafe fn splat(x: f64) -> Self {
        F64x8(_mm512_set1_pd(x))
    }
    #[inline(always)]
    unsafe fn load(p: *const f64) -> Self {
        F64x8(_mm512_loadu_pd(p))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f64) {
        _mm512_storeu_pd(p, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        F64x8(_mm512_min_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        F64x8(_mm512_max_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        F64x8(_mm512_add_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        F64x8(_mm512_sub_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        F64x8(_mm512_mul_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> K8 {
        K8(_mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> K8 {
        K8(_mm512_cmp_pd_mask::<_CMP_LT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> K8 {
        K8(_mm512_cmp_pd_mask::<_CMP_GT_OQ>(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: K8, a: Self, b: Self) -> Self {
        F64x8(_mm512_mask_blend_pd(m.0, b.0, a.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> K8 {
        K8(_mm512_cmp_pd_mask::<_CMP_UNORD_Q>(self.0, self.0))
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> K8 {
        K8(_mm512_cmp_pd_mask::<_CMP_NLT_UQ>(
            _mm512_abs_pd(self.0),
            _mm512_set1_pd(f64::INFINITY),
        ))
    }
    #[inline(always)]
    unsafe fn mask_none() -> K8 {
        K8(0)
    }
}
impl FVector for F64x8 {
    #[inline(always)]
    unsafe fn div(self, o: Self) -> Self {
        F64x8(_mm512_div_pd(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sqrt(self) -> Self {
        F64x8(_mm512_sqrt_pd(self.0))
    }
    #[inline(always)]
    unsafe fn abs(self) -> Self {
        F64x8(_mm512_abs_pd(self.0))
    }
}

impl Vector for U16x32 {
    type E = u16;
    type M = K32;
    const L: usize = 32;
    const UNROLL: usize = 32;
    #[inline(always)]
    unsafe fn splat(x: u16) -> Self {
        U16x32(_mm512_set1_epi16(x as i16))
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> Self {
        U16x32(_mm512_loadu_si512(p as *const __m512i))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut u16) {
        _mm512_storeu_si512(p as *mut __m512i, self.0)
    }
    #[inline(always)]
    unsafe fn min(self, o: Self) -> Self {
        U16x32(_mm512_min_epu16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn max(self, o: Self) -> Self {
        U16x32(_mm512_max_epu16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        U16x32(_mm512_add_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn sub(self, o: Self) -> Self {
        U16x32(_mm512_sub_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        U16x32(_mm512_mullo_epi16(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn eq(self, o: Self) -> K32 {
        K32(_mm512_cmpeq_epu16_mask(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn lt(self, o: Self) -> K32 {
        K32(_mm512_cmplt_epu16_mask(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn gt(self, o: Self) -> K32 {
        K32(_mm512_cmpgt_epu16_mask(self.0, o.0))
    }
    #[inline(always)]
    unsafe fn select(m: K32, a: Self, b: Self) -> Self {
        U16x32(_mm512_mask_blend_epi16(m.0, b.0, a.0))
    }
    #[inline(always)]
    unsafe fn nan_mask(self) -> K32 {
        K32(0)
    }
    #[inline(always)]
    unsafe fn nonfinite_mask(self) -> K32 {
        K32(0)
    }
    #[inline(always)]
    unsafe fn mask_none() -> K32 {
        K32(0)
    }
}
