//! Array arguments that accept any NumPy dtype castable to the kernel's element type.
//!
//! The kernels run on `f32` (and `Complex32`), but NumPy produces `float64` by default.
//! An array of the right dtype is borrowed as is; anything else goes through one
//! `numpy.asarray(x, dtype=...)` call. `numpy::PyArrayLike` is not used because it
//! converts a 1-D array element by element through a `Vec`.

use std::ops::Deref;

use numpy::ndarray::{Dimension, Ix1, Ix2};
use numpy::{
    get_array_module, Element, PyArray, PyArrayMethods, PyReadonlyArray, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{intern, Borrowed};

/// A read-only array argument cast to element type `T` when its dtype differs.
pub struct ArrayIn<'py, T: Element, D: Dimension>(PyReadonlyArray<'py, T, D>);

pub type ArrayIn1<'py, T> = ArrayIn<'py, T, Ix1>;
pub type ArrayIn2<'py, T> = ArrayIn<'py, T, Ix2>;

impl<'py, T: Element, D: Dimension> Deref for ArrayIn<'py, T, D> {
    type Target = PyReadonlyArray<'py, T, D>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'py, T, D> FromPyObject<'a, 'py> for ArrayIn<'py, T, D>
where
    T: Element + 'py,
    D: Dimension + 'py,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = ob.cast::<PyArray<T, D>>() {
            return Ok(Self(array.try_readonly()?));
        }

        let py = ob.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "dtype"), T::get_dtype(py))?;
        let converted = get_array_module(py)?
            .getattr(intern!(py, "asarray"))?
            .call((ob,), Some(&kwargs))?;
        if let Some(ndim) = D::NDIM {
            let got = converted.cast::<PyUntypedArray>()?.ndim();
            if got != ndim {
                return Err(PyTypeError::new_err(format!(
                    "expected a {ndim}-D array, got {got}-D"
                )));
            }
        }
        Ok(Self(
            converted.cast_into::<PyArray<T, D>>()?.try_readonly()?,
        ))
    }
}

/// A mono audio argument: an [`ArrayIn1<f32>`] that accepts floating-point
/// samples only. Cast to `f32` as they are, 16-bit PCM values would be read as
/// samples up to ±32768 instead of ±1, about 90 dB too loud, with no error.
pub struct AudioIn1<'py>(ArrayIn1<'py, f32>);

impl<'py> Deref for AudioIn1<'py> {
    type Target = PyReadonlyArray<'py, f32, Ix1>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// `numpy.asarray(ob)`, refused unless its samples are floating-point.
fn float_samples<'py>(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ob.py();
    let array = get_array_module(py)?
        .getattr(intern!(py, "asarray"))?
        .call1((ob,))?;
    let dtype = array.getattr(intern!(py, "dtype"))?;
    let kind: String = dtype.getattr(intern!(py, "kind"))?.extract()?;
    if kind != "f" {
        return Err(PyTypeError::new_err(format!(
            "audio must be floating-point samples in [-1, 1], got dtype {dtype}; \
             scale integer PCM first, e.g. y / 32768.0 for 16-bit"
        )));
    }
    Ok(array)
}

impl<'a, 'py> FromPyObject<'a, 'py> for AudioIn1<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.cast::<PyArray<f32, Ix1>>().is_err() {
            float_samples(ob)?;
        }
        ArrayIn1::extract(ob).map(Self)
    }
}

/// An audio argument of the analyzer: mono samples (1-D) or channels (2-D,
/// shaped (channels, samples)). Floating-point only, as [`AudioIn1`].
pub enum AudioIn<'py> {
    Mono(ArrayIn1<'py, f32>),
    Channels(ArrayIn2<'py, f32>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for AudioIn<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // A float32 array is borrowed as it is, without asking NumPy.
        let ndim = if ob.cast::<PyArray<f32, Ix1>>().is_ok() {
            1
        } else if ob.cast::<PyArray<f32, Ix2>>().is_ok() {
            2
        } else {
            float_samples(ob)?
                .getattr(intern!(ob.py(), "ndim"))?
                .extract()?
        };
        match ndim {
            1 => ArrayIn1::extract(ob).map(Self::Mono),
            2 => ArrayIn2::extract(ob).map(Self::Channels),
            n => Err(PyTypeError::new_err(format!(
                "audio must be a 1-D array (mono) or a 2-D array shaped \
                 (channels, samples), got {n}-D"
            ))),
        }
    }
}
