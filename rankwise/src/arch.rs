//! Per-ISA instantiation of the generic engine.
//!
//! Every `run_*` and `net_*` function below is the one place where a target feature is
//! enabled; the generic kernels and the vector methods they call are `#[inline(always)]`,
//! so they compile into these functions with the right instructions.

use crate::engine::{self, Job, Unit};
use crate::net::{self, NetPlan};
use crate::simd::{FVector, Isa, Vector};
use crate::spec::DType;

/// A vector type that can run comparator networks.
pub(crate) trait SortVec: Vector {
    unsafe fn apply_net(b: *mut Self, plan: &NetPlan);
}

/// The three vector types of one ISA.
pub(crate) trait Arch {
    type F32: SortVec + FVector<E = f32>;
    type F64: SortVec + FVector<E = f64>;
    type U16: SortVec<E = u16>;
}

pub(crate) struct ScalarArch;
mod scalar_impl {
    use super::*;
    use crate::simd::scalar::{F32s, F64s, U16s};
    impl Arch for ScalarArch {
        type F32 = F32s;
        type F64 = F64s;
        type U16 = U16s;
    }
    macro_rules! sv {
        ($t:ty) => {
            impl SortVec for $t {
                #[inline(never)]
                unsafe fn apply_net(b: *mut Self, plan: &NetPlan) {
                    net::apply::<Self>(b, plan)
                }
            }
        };
    }
    sv!(F32s);
    sv!(F64s);
    sv!(U16s);
    pub(crate) unsafe fn run(job: &Job, u: Unit) {
        match job.sdt {
            DType::F32 => engine::run_unit::<ScalarArch, F32s>(job, u),
            DType::F64 => engine::run_unit::<ScalarArch, F64s>(job, u),
            _ => engine::run_unit::<ScalarArch, U16s>(job, u),
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use crate::simd::avx2::{F32x8, F64x4, U16x16};
    use crate::simd::avx512::{F32x16, F64x8, U16x32};

    pub(crate) struct Avx2Arch;
    pub(crate) struct Avx512Arch;
    impl Arch for Avx2Arch {
        type F32 = F32x8;
        type F64 = F64x4;
        type U16 = U16x16;
    }
    impl Arch for Avx512Arch {
        type F32 = F32x16;
        type F64 = F64x8;
        type U16 = U16x32;
    }

    macro_rules! entries {
        ($feat:literal, $arch:ty, $net:ident, $run:ident, $( $v:ty => $dt:path ),* ) => {
            $(
                impl SortVec for $v {
                    #[inline(always)]
                    unsafe fn apply_net(b: *mut Self, plan: &NetPlan) {
                        $net::<$v>(b, plan)
                    }
                }
            )*
            #[target_feature(enable = $feat)]
            unsafe fn $net<V: Vector>(b: *mut V, plan: &NetPlan) {
                net::apply::<V>(b, plan)
            }
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn $run(job: &Job, u: Unit) {
                match job.sdt {
                    $( $dt => engine::run_unit::<$arch, $v>(job, u), )*
                    _ => unreachable!(),
                }
            }
        };
    }
    entries!("avx2", Avx2Arch, net_avx2, run_avx2,
        F32x8 => DType::F32, F64x4 => DType::F64, U16x16 => DType::U16);
    entries!("avx512f,avx512bw,avx512dq,avx512vl", Avx512Arch, net_avx512, run_avx512,
        F32x16 => DType::F32, F64x8 => DType::F64, U16x32 => DType::U16);
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use crate::simd::neon::{F32x4, F64x2, U16x8};
    pub(crate) struct NeonArch;
    impl Arch for NeonArch {
        type F32 = F32x4;
        type F64 = F64x2;
        type U16 = U16x8;
    }
    macro_rules! sv {
        ($t:ty) => {
            impl SortVec for $t {
                #[inline(never)]
                unsafe fn apply_net(b: *mut Self, plan: &NetPlan) {
                    net::apply::<Self>(b, plan)
                }
            }
        };
    }
    sv!(F32x4);
    sv!(F64x2);
    sv!(U16x8);
    pub(crate) unsafe fn run(job: &Job, u: Unit) {
        match job.sdt {
            DType::F32 => engine::run_unit::<NeonArch, F32x4>(job, u),
            DType::F64 => engine::run_unit::<NeonArch, F64x2>(job, u),
            _ => engine::run_unit::<NeonArch, U16x8>(job, u),
        }
    }
}

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use crate::simd::wasm::{F32x4, F64x2, U16x8};
    pub(crate) struct Simd128Arch;
    impl Arch for Simd128Arch {
        type F32 = F32x4;
        type F64 = F64x2;
        type U16 = U16x8;
    }
    macro_rules! sv {
        ($t:ty) => {
            impl SortVec for $t {
                #[inline(never)]
                unsafe fn apply_net(b: *mut Self, plan: &NetPlan) {
                    net::apply::<Self>(b, plan)
                }
            }
        };
    }
    sv!(F32x4);
    sv!(F64x2);
    sv!(U16x8);
    pub(crate) unsafe fn run(job: &Job, u: Unit) {
        match job.sdt {
            DType::F32 => engine::run_unit::<Simd128Arch, F32x4>(job, u),
            DType::F64 => engine::run_unit::<Simd128Arch, F64x2>(job, u),
            _ => engine::run_unit::<Simd128Arch, U16x8>(job, u),
        }
    }
}

/// Run one unit of work on the job's ISA.
pub(crate) unsafe fn run(job: &Job, u: Unit) {
    match job.isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => x86::run_avx512(job, u),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => x86::run_avx2(job, u),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => arm::run(job, u),
        #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
        Isa::Simd128 => wasm::run(job, u),
        _ => scalar_impl::run(job, u),
    }
}

/// Widest stack the unrolled networks handle for this ISA and sort dtype.
pub(crate) fn unroll_for(isa: Isa, dt: DType) -> usize {
    use crate::simd::scalar::{F32s, F64s, U16s};
    let s = match dt {
        DType::F32 => F32s::UNROLL,
        DType::F64 => F64s::UNROLL,
        _ => U16s::UNROLL,
    };
    match isa {
        Isa::Scalar => s,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => {
            use crate::simd::avx2::*;
            match dt {
                DType::F32 => F32x8::UNROLL,
                DType::F64 => F64x4::UNROLL,
                _ => U16x16::UNROLL,
            }
        }
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => {
            use crate::simd::avx512::*;
            match dt {
                DType::F32 => F32x16::UNROLL,
                DType::F64 => F64x8::UNROLL,
                _ => U16x32::UNROLL,
            }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            use crate::simd::neon::*;
            match dt {
                DType::F32 => F32x4::UNROLL,
                DType::F64 => F64x2::UNROLL,
                _ => U16x8::UNROLL,
            }
        }
        #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
        Isa::Simd128 => {
            use crate::simd::wasm::*;
            match dt {
                DType::F32 => F32x4::UNROLL,
                DType::F64 => F64x2::UNROLL,
                _ => U16x8::UNROLL,
            }
        }
        #[allow(unreachable_patterns)]
        _ => s,
    }
}

/// Lanes per vector for this ISA and sort dtype.
pub(crate) fn lanes_for(isa: Isa, dt: DType) -> usize {
    let f = isa.f32_lanes();
    match dt {
        DType::F64 => f / 2,
        DType::U16 => f * 2,
        _ => f,
    }
}
