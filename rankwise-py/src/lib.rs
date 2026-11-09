//! Python extension: a thin, GIL-free bridge from NumPy buffers to the rankwise core.
//!
//! The Python package (`python/rankwise/__init__.py`) owns NumPy's semantics (argument
//! parsing, dtype rules, output allocation, warnings) and passes raw addresses and byte
//! strides here, keeping every array alive for the duration of the call.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rankwise::{
    presets, reduce_raw, ArgRank, Center, Clip, Combine, DType, Error, Isa, NanPolicy, Normalize, Op, Options,
    OutRounding, Precision, QForm, QuantileMethod, QuantileSpec, RawMask, RawOut, RawStack, Rejection, Spread, Staging,
};

fn err(e: Error) -> PyErr {
    match e {
        Error::DType(_) => PyTypeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn dtype(s: &str) -> PyResult<DType> {
    DType::from_name(s).ok_or_else(|| PyTypeError::new_err(format!("unsupported dtype {s}")))
}

fn get<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Option<T>> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract::<T>().map_err(Into::into)?)),
        _ => Ok(None),
    }
}

fn req<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<T> {
    get(d, k)?.ok_or_else(|| PyValueError::new_err(format!("missing op field {k:?}")))
}

fn parse_clip(d: &Bound<'_, PyDict>) -> PyResult<Clip> {
    let method: String = req(d, "method")?;
    let low: f64 = get(d, "low")?.unwrap_or(3.0);
    let high: f64 = get(d, "high")?.unwrap_or(3.0);
    let maxiters: Option<u32> = get(d, "maxiters")?;
    let preset: Option<String> = get(d, "preset")?;
    let mut c = match (preset.as_deref(), method.as_str()) {
        (Some("pixinsight"), "winsorized") => presets::pixinsight_winsorized(low, high).max_iters(maxiters),
        (Some("pixinsight"), m) => {
            return Err(PyValueError::new_err(format!("the pixinsight preset covers 'winsorized', not {m:?}")))
        }
        (Some("siril"), "sigma") => presets::siril_sigma(low, high),
        (Some("siril"), "mad") => {
            return Err(PyValueError::new_err(
                "Siril's k-MAD clipping is not reproduced yet (see docs/presets.md); use method='mad' for astropy's mad_std",
            ))
        }
        (_, "sigma") => Clip::sigma(low, high).max_iters(maxiters),
        (_, "mad") => Clip::mad(low, high).max_iters(maxiters),
        (_, "median_sigma") => Clip::median_sigma(low, high),
        (_, "winsorized") => Clip::winsorized(low, high).max_iters(maxiters),
        (_, "percentile") => Clip::percentile(low, high),
        (_, "none") => Clip::none(),
        (_, m) => return Err(PyValueError::new_err(format!("unknown rejection method {m:?}"))),
    };
    if let Some(s) = get::<String>(d, "cenfunc")? {
        c = c.center(match s.as_str() {
            "median" => Center::Median,
            "mean" => Center::Mean,
            _ => return Err(PyValueError::new_err(format!("{s} is an invalid cenfunc."))),
        });
    }
    if let Some(s) = get::<String>(d, "stdfunc")? {
        // method="mad" fixes the spread; letting a default stdfunc through silently turned it
        // back into std clipping.
        if method == "mad" && s != "mad_std" {
            return Err(PyValueError::new_err(format!(
                "method='mad' clips with mad_std; use method='sigma', stdfunc={s:?} for that spread"
            )));
        }
        if method != "sigma" && method != "mad" {
            return Err(PyValueError::new_err(format!("stdfunc applies to method='sigma', not {method:?}")));
        }
        c = c.spread(match s.as_str() {
            "std" => Spread::Std,
            "mad_std" => Spread::MadStd,
            _ => return Err(PyValueError::new_err(format!("{s} is an invalid stdfunc."))),
        });
    }
    if let Some(s) = get::<String>(d, "combine")? {
        c = c.combine(match s.as_str() {
            "mean" => Combine::Mean,
            "median" => Combine::Median,
            "sum" => Combine::Sum,
            _ => return Err(PyValueError::new_err(format!("unknown combine {s:?}"))),
        });
    }
    if let Some(s) = get::<String>(d, "precision")? {
        c = c.precision(match s.as_str() {
            "f64" | "float64" => Precision::F64,
            "f32" | "float32" => Precision::F32,
            _ => return Err(PyValueError::new_err(format!("unknown precision {s:?}"))),
        });
    }
    if let Some(b) = get::<bool>(d, "mask_infinite")? {
        c.mask_infinite = b;
    }
    if let (Some(cut), Rejection::Winsorized { cutoff, .. }) = (get::<f64>(d, "cutoff")?, &mut c.rejection) {
        *cutoff = cut;
    }
    Ok(c)
}

fn parse_op(d: &Bound<'_, PyDict>) -> PyResult<Op> {
    let kind: String = req(d, "kind")?;
    Ok(match kind.as_str() {
        "select" => Op::SelectRanks(req(d, "ranks")?),
        "median" => Op::Median,
        "quantile" => {
            let form = match req::<String>(d, "form")?.as_str() {
                "weak" => QForm::Weak,
                "weak_int" => QForm::WeakInt,
                "f32" => QForm::F32,
                _ => QForm::F64,
            };
            let out_dtype = match get::<String>(d, "out_dtype")? {
                Some(s) => Some(dtype(&s)?),
                None => None,
            };
            let out_rounding = match get::<String>(d, "out_rounding")?.as_deref() {
                Some("cast_once") => OutRounding::CastOnce,
                _ => OutRounding::NumpyQuantile,
            };
            Op::Quantile(QuantileSpec {
                q: req(d, "q")?,
                method: QuantileMethod::from_name(&req::<String>(d, "method")?).map_err(err)?,
                form,
                out_dtype,
                out_rounding,
            })
        }
        "min" => Op::Min,
        "max" => Op::Max,
        "range" => Op::Range,
        "argmin" => Op::ArgMin,
        "argmax" => Op::ArgMax,
        "argmedian" => Op::ArgRank(ArgRank::Median),
        "argrank" => Op::ArgRank(ArgRank::Fixed(req(d, "rank")?)),
        "trim_mean" => Op::TrimMean(req(d, "proportion")?),
        "iqr" => Op::Iqr { lo: req(d, "lo")?, hi: req(d, "hi")?, scale: req(d, "scale")? },
        "mad" => Op::Mad { scale: req(d, "scale")? },
        "clip" => Op::Clip(parse_clip(d)?),
        k => return Err(PyValueError::new_err(format!("unknown op {k:?}"))),
    })
}

fn parse_opts(d: &Bound<'_, PyDict>) -> PyResult<Options> {
    let isa = match get::<String>(d, "isa")? {
        Some(s) if s != "auto" => {
            Some(Isa::from_name(&s).ok_or_else(|| PyValueError::new_err(format!("unknown ISA {s:?}")))?)
        }
        _ => None,
    };
    let scale: Option<Vec<f64>> = get(d, "scale")?;
    let offset: Option<Vec<f64>> = get(d, "offset")?;
    Ok(Options {
        nan: if get::<bool>(d, "omit")?.unwrap_or(false) { NanPolicy::Omit } else { NanPolicy::Propagate },
        nodata: get(d, "nodata")?,
        normalize: if scale.is_some() || offset.is_some() { Some(Normalize { scale, offset }) } else { None },
        weights: get(d, "weights")?,
        isa,
        threads: get::<usize>(d, "threads")?.unwrap_or(1),
        staging: match get::<String>(d, "staging")?.as_deref() {
            Some("always") => Staging::Always,
            Some("never") => Staging::Never,
            Some("skew") => Staging::Skew,
            _ => Staging::Auto,
        },
    })
}

type OutSpec = (usize, String, Vec<isize>);

/// Run one reduction. Addresses must stay valid for the call (the Python layer holds the
/// arrays); the GIL is released while the kernels run.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (op, in_dtype, frames, shape, strides, outs, mask, opts))]
fn reduce(
    py: Python<'_>,
    op: &Bound<'_, PyDict>,
    in_dtype: &str,
    frames: Vec<usize>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    outs: Vec<OutSpec>,
    mask: Option<(Vec<usize>, String, Vec<isize>)>,
    opts: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let op = parse_op(op)?;
    let opts = parse_opts(opts)?;
    let dt = dtype(in_dtype)?;
    let out_dts: Vec<DType> = outs.iter().map(|o| dtype(&o.1)).collect::<PyResult<_>>()?;
    let mask_dt = match &mask {
        Some(m) => Some(dtype(&m.1)?),
        None => None,
    };
    py.detach(move || {
        let fptrs: Vec<*const u8> = frames.iter().map(|&a| a as *const u8).collect();
        let stack = RawStack { dtype: dt, frames: &fptrs, shape: &shape, strides: &strides };
        let raw_outs: Vec<RawOut> = outs
            .iter()
            .zip(out_dts.iter())
            .map(|(o, &d)| RawOut { ptr: o.0 as *mut u8, dtype: d, strides: &o.2 })
            .collect();
        let mptrs: Vec<*mut u8> =
            mask.as_ref().map(|m| m.0.iter().map(|&a| a as *mut u8).collect()).unwrap_or_default();
        let rm = mask.as_ref().map(|m| RawMask { frames: &mptrs, dtype: mask_dt.unwrap(), strides: &m.2 });
        // SAFETY: the Python layer validated shapes and keeps every buffer alive.
        unsafe { reduce_raw(&stack, &op, &raw_outs, rm.as_ref(), &opts) }
    })
    .map_err(err)
}

#[pyfunction]
fn available_isas() -> Vec<&'static str> {
    Isa::available().into_iter().map(|i| i.name()).collect()
}

#[pyfunction]
#[pyo3(signature = (name=None))]
fn set_isa(name: Option<&str>) -> PyResult<()> {
    let isa = match name {
        None | Some("auto") => None,
        Some(s) => Some(Isa::from_name(s).ok_or_else(|| PyValueError::new_err(format!("unknown ISA {s:?}")))?),
    };
    rankwise::set_isa_override(isa).map_err(err)
}

#[pyfunction]
fn current_isa() -> PyResult<&'static str> {
    rankwise::resolve_isa(None).map(|i| i.name()).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (n, ranks=None))]
fn network_size(n: usize, ranks: Option<Vec<usize>>) -> usize {
    rankwise::network_size(n, ranks.as_deref())
}

#[pyfunction]
fn describe(
    op: &Bound<'_, PyDict>,
    in_dtype: &str,
    frames: Vec<usize>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    opts: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let op = parse_op(op)?;
    let opts = parse_opts(opts)?;
    let fptrs: Vec<*const u8> = frames.iter().map(|&a| a as *const u8).collect();
    let stack = RawStack { dtype: dtype(in_dtype)?, frames: &fptrs, shape: &shape, strides: &strides };
    rankwise::describe(&stack, &op, &opts).map_err(err)
}

#[pymodule]
fn _rankwise(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(reduce, m)?)?;
    m.add_function(wrap_pyfunction!(available_isas, m)?)?;
    m.add_function(wrap_pyfunction!(set_isa, m)?)?;
    m.add_function(wrap_pyfunction!(current_isa, m)?)?;
    m.add_function(wrap_pyfunction!(network_size, m)?)?;
    m.add_function(wrap_pyfunction!(describe, m)?)?;
    m.add("__version__", rankwise::VERSION)?;
    m.add("MAX_FRAMES", rankwise::MAX_FRAMES)?;
    let _ = PyList::empty(m.py());
    Ok(())
}
