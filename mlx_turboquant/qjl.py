"""Quantized Johnson-Lindenstrauss (QJL) transform.

1-bit QJL provides unbiased inner product estimation on quantization residuals.
Used in TurboQuant_prod (Algorithm 2) as a residual correction stage.

The projection matrix S has i.i.d. N(0,1) entries (per the paper).
Dequantization: x̃_qjl = (√(π/2) / d) · γ · S^T · z

Reference: QJL (Zandieh et al., 2024a)
"""

import math

import mlx.core as mx
import numpy as np

# Cache projection matrices
_projection_cache: dict[tuple[int, int, int], mx.array] = {}

DEFAULT_SEED = 12345


def generate_projection_matrix(d: int, m: int, seed: int = DEFAULT_SEED) -> mx.array:
    """Generate random Gaussian projection matrix for QJL.

    Args:
        d: Input dimension
        m: Projection dimension (number of random projections)
        seed: Random seed

    Returns:
        Projection matrix of shape (m, d) with i.i.d. N(0, 1) entries
    """
    rng = np.random.RandomState(seed)
    S = rng.randn(m, d).astype(np.float32)
    return mx.array(S)


def get_projection_matrix(d: int, m: int, seed: int = DEFAULT_SEED) -> mx.array:
    """Get cached projection matrix."""
    key = (d, m, seed)
    if key not in _projection_cache:
        _projection_cache[key] = generate_projection_matrix(d, m, seed)
    return _projection_cache[key]


def qjl_quantize(residual: mx.array, projection: mx.array) -> tuple[mx.array, mx.array]:
    """Apply 1-bit QJL quantization to residual vectors.

    Args:
        residual: Residual vectors (..., d)
        projection: Projection matrix (m, d) with N(0,1) entries

    Returns:
        (signs, norms) where:
            signs: Sign bits as boolean, shape (..., m)
            norms: L2 norms of residuals, shape (...)
    """
    norms = mx.linalg.norm(residual, axis=-1)  # (...)

    # Project: (..., d) @ (d, m) -> (..., m)
    projected = residual @ projection.T  # (..., m)

    # Store signs as boolean (True = positive)
    signs = projected >= 0  # (..., m)

    return signs, norms


def qjl_dequantize(signs: mx.array, norms: mx.array, projection: mx.array, d: int) -> mx.array:
    """Reconstruct residual estimate from QJL quantization.

    Formula: x̃_qjl = (√(π/2) / d) · γ · S^T · z
    where z = signs converted to ±1, S has N(0,1) entries, γ = ||r||.

    Args:
        signs: Sign bits (..., m) as boolean
        norms: Residual L2 norms (...)
        projection: Projection matrix (m, d) with N(0,1) entries
        d: Original dimension

    Returns:
        Estimated residual (..., d)
    """
    # Convert bools to ±1
    z = mx.where(signs, mx.array(1.0), mx.array(-1.0))  # (..., m)

    # S^T @ z: (..., m) @ (m, d) -> (..., d)
    reconstructed = z @ projection  # (..., d)

    # Scale: (√(π/2) / d) · γ
    scale = (math.sqrt(math.pi / 2) / d) * norms[..., None]  # (..., 1)

    return scale * reconstructed


def qjl_inner_product(
    query: mx.array,
    signs: mx.array,
    norms: mx.array,
    projection: mx.array,
    d: int,
) -> mx.array:
    """Estimate inner product ⟨q, r⟩ using QJL without full dequantization.

    ⟨q, r⟩ ≈ (√(π/2) / d) · γ · ⟨S·q, z⟩

    Args:
        query: Query vectors (..., d)
        signs: QJL sign bits (..., m) as boolean
        norms: Residual norms (...)
        projection: Projection matrix (m, d) with N(0,1) entries
        d: Original dimension

    Returns:
        Estimated inner products (...)
    """
    # Project query: (..., d) @ (d, m) -> (..., m)
    q_proj = query @ projection.T  # (..., m)

    # Convert signs to ±1
    z = mx.where(signs, mx.array(1.0), mx.array(-1.0))  # (..., m)

    # Dot product in projection space
    dot = mx.sum(q_proj * z, axis=-1)  # (...)

    # Scale: (√(π/2) / d) · γ
    scale = (math.sqrt(math.pi / 2) / d) * norms

    return scale * dot
