//! Python bindings for the hand-crafted similarity / embedding vector.

use pyo3::prelude::*;

use sonara::similarity as rs;
use sonara::types::Float;

/// Weighted cosine-free similarity between two embedding vectors, in `[0, 1]`
/// (higher = more similar). See `sonara::similarity` for the metric definition.
#[pyfunction]
#[pyo3(name = "similarity")]
pub fn py_similarity(a: Vec<Float>, b: Vec<Float>) -> Float {
    rs::similarity(&a, &b)
}

/// Distance between two embedding vectors, in `[0, 1]` (0 = identical).
#[pyfunction]
#[pyo3(name = "embedding_distance")]
pub fn py_embedding_distance(a: Vec<Float>, b: Vec<Float>) -> Float {
    rs::distance(&a, &b)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose the current embedding layout version + dimensionality so callers can
    // validate stored vectors before comparing.
    m.add("SIMILARITY_VERSION", rs::SIMILARITY_VERSION)?;
    m.add("EMBEDDING_DIM", rs::EMBEDDING_DIM)?;
    m.add_function(wrap_pyfunction!(py_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_embedding_distance, m)?)?;
    Ok(())
}
