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
