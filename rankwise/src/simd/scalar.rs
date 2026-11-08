//! Portable fallback: fixed-size arrays. Lane counts mirror AVX2 (8 f32 / 4 f64 / 16 u16) so
//! the blocking logic is the same everywhere; the compiler vectorizes the loops for whatever
//! baseline the target has (SSE2 on x86-64).

use super::{Element, FVector, Mask, Vector};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitMask<const L: usize>(pub u64);

impl<const L: usize> Mask for BitMask<L> {
    #[inline(always)]
    fn and(self, o: Self) -> Self {
        BitMask(self.0 & o.0)
    }
    #[inline(always)]
    fn or(self, o: Self) -> Self {
        BitMask(self.0 | o.0)
    }
    #[inline(always)]
    fn andnot(self, o: Self) -> Self {
        BitMask(self.0 & !o.0)
    }
    #[inline(always)]
    fn not(self) -> Self {
        BitMask(!self.0 & if L >= 64 { u64::MAX } else { (1u64 << L) - 1 })
    }
    #[inline(always)]
    fn any(self) -> bool {
        self.0 != 0
    }
    #[inline(always)]
    fn bits(self) -> u64 {
        self.0
    }
}

macro_rules! scalar_vec {
    ($name:ident, $t:ty, $l:expr, float = $float:tt) => {
        #[derive(Clone, Copy, Debug)]
        #[repr(C)]
        pub struct $name(pub [$t; $l]);

        impl Vector for $name {
            type E = $t;
            type M = BitMask<$l>;
            const L: usize = $l;
            const UNROLL: usize = 0;
            #[inline(always)]
            unsafe fn splat(x: $t) -> Self {
                $name([x; $l])
            }
            #[inline(always)]
            unsafe fn load(p: *const $t) -> Self {
                $name(core::ptr::read_unaligned(p as *const [$t; $l]))
            }
            #[inline(always)]
            unsafe fn store(self, p: *mut $t) {
                core::ptr::write_unaligned(p as *mut [$t; $l], self.0)
            }
            #[inline(always)]
            unsafe fn min(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = if self.0[i] < o.0[i] { self.0[i] } else { o.0[i] };
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn max(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = if self.0[i] > o.0[i] { self.0[i] } else { o.0[i] };
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn add(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = scalar_vec!(@add $float, self.0[i], o.0[i]);
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn sub(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = scalar_vec!(@sub $float, self.0[i], o.0[i]);
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn mul(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = scalar_vec!(@mul $float, self.0[i], o.0[i]);
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn eq(self, o: Self) -> BitMask<$l> {
                let mut m = 0u64;
                for i in 0..$l {
                    m |= ((self.0[i] == o.0[i]) as u64) << i;
                }
                BitMask(m)
            }
            #[inline(always)]
            unsafe fn lt(self, o: Self) -> BitMask<$l> {
                let mut m = 0u64;
                for i in 0..$l {
                    m |= ((self.0[i] < o.0[i]) as u64) << i;
                }
                BitMask(m)
            }
            #[inline(always)]
            unsafe fn gt(self, o: Self) -> BitMask<$l> {
                let mut m = 0u64;
                for i in 0..$l {
                    m |= ((self.0[i] > o.0[i]) as u64) << i;
                }
                BitMask(m)
            }
            #[inline(always)]
            unsafe fn select(m: BitMask<$l>, a: Self, b: Self) -> Self {
                let mut r = b.0;
                for i in 0..$l {
                    if (m.0 >> i) & 1 == 1 {
                        r[i] = a.0[i];
                    }
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn nan_mask(self) -> BitMask<$l> {
                let mut m = 0u64;
                for i in 0..$l {
                    m |= (Element::is_nan(self.0[i]) as u64) << i;
                }
                BitMask(m)
            }
            #[inline(always)]
            unsafe fn nonfinite_mask(self) -> BitMask<$l> {
                scalar_vec!(@nonfinite $float, self, $l)
            }
            #[inline(always)]
            unsafe fn mask_none() -> BitMask<$l> {
                BitMask(0)
            }
        }
        scalar_vec!(@fvec $float, $name, $t, $l);
    };
    (@add true, $a:expr, $b:expr) => { $a + $b };
    (@add false, $a:expr, $b:expr) => { $a.wrapping_add($b) };
    (@sub true, $a:expr, $b:expr) => { $a - $b };
    (@mul true, $a:expr, $b:expr) => { $a * $b };
    (@mul false, $a:expr, $b:expr) => { $a.wrapping_mul($b) };
    (@sub false, $a:expr, $b:expr) => { $a.wrapping_sub($b) };
    (@nonfinite true, $s:expr, $l:expr) => {{
        let mut m = 0u64;
        for i in 0..$l {
            m |= ((!$s.0[i].is_finite()) as u64) << i;
        }
        BitMask(m)
    }};
    (@nonfinite false, $s:expr, $l:expr) => { BitMask(0) };
    (@fvec true, $name:ident, $t:ty, $l:expr) => {
        impl FVector for $name {
            #[inline(always)]
            unsafe fn div(self, o: Self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = self.0[i] / o.0[i];
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn sqrt(self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = self.0[i].sqrt();
                }
                $name(r)
            }
            #[inline(always)]
            unsafe fn abs(self) -> Self {
                let mut r = self.0;
                for i in 0..$l {
                    r[i] = self.0[i].abs();
                }
                $name(r)
            }
        }
    };
    (@fvec false, $name:ident, $t:ty, $l:expr) => {};
}

scalar_vec!(F32s, f32, 8, float = true);
scalar_vec!(F64s, f64, 4, float = true);
scalar_vec!(U16s, u16, 16, float = false);
