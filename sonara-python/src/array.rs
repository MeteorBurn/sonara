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

impl<'a, 'py> FromPyObject<'a, 'py> for AudioIn1<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if ob.cast::<PyArray<f32, Ix1>>().is_err() {
            let py = ob.py();
            let dtype = get_array_module(py)?
                .getattr(intern!(py, "asarray"))?
                .call1((ob,))?
                .getattr(intern!(py, "dtype"))?;
            let kind: String = dtype.getattr(intern!(py, "kind"))?.extract()?;
            if kind != "f" {
                return Err(PyTypeError::new_err(format!(
                    "audio must be floating-point samples in [-1, 1], got dtype {dtype}; \
                     scale integer PCM first, e.g. y / 32768.0 for 16-bit"
                )));
            }
        }
        ArrayIn1::extract(ob).map(Self)
    }
}
