"""Lloyd-Max optimal scalar quantizer for Beta-distributed coordinates.

After rotating a unit vector by a random orthogonal matrix, each coordinate
follows Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1]. In high dimensions this
converges to N(0, 1/d). The Lloyd-Max algorithm finds optimal centroids that
minimize E[(X - Q(X))^2] for this distribution.
"""

import math
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

CODEBOOK_DIR = Path(__file__).parent / "codebooks"


def _user_cache_dir() -> Path:
    """User-writable cache directory for codebooks computed at runtime.

    Honors XDG_CACHE_HOME; falls back to ~/.cache. The package-shipped
    codebooks under CODEBOOK_DIR are still preferred when present.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("MLX_TURBOQUANT_CACHE")
    if base:
        return Path(base) / "mlx_turboquant"
    return Path.home() / ".cache" / "mlx_turboquant"


# Cache computed codebooks in memory
_codebook_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a coordinate of a rotated unit vector in d dimensions.

    This is Beta((d-1)/2, (d-1)/2) rescaled to [-1, 1].
    """
    from scipy import special

    alpha = (d - 1) / 2.0
    # Normalization constant for Beta(alpha, alpha) on [-1, 1]
    log_norm = special.gammaln(2 * alpha) - 2 * special.gammaln(alpha) - (2 * alpha - 1) * math.log(2)
    log_pdf = log_norm + (alpha - 1) * np.log(np.maximum(1 - x * x, 1e-300))
    return np.exp(log_pdf)


def gaussian_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Gaussian approximation N(0, 1/d) for high dimensions."""
    var = 1.0 / d
    return np.exp(-0.5 * x * x / var) / math.sqrt(2 * math.pi * var)


def lloyd_max(d: int, bits: int, num_iter: int = 500, grid_size: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal codebook for rotated unit-sphere coordinates.

    Args:
        d: Dimension of vectors (head_dim)
        bits: Number of quantization bits
        num_iter: Number of Lloyd-Max iterations
        grid_size: Number of grid points for numerical integration

    Returns:
        (centroids, boundaries) - centroids shape (2^bits,), boundaries shape (2^bits - 1,)
    """
    n_levels = 1 << bits

    # Use Gaussian approximation for d >= 32 (accurate and faster)
    sigma = 1.0 / math.sqrt(d)
    # Support: ±6 sigma covers essentially all probability mass
    x_min, x_max = -6 * sigma, 6 * sigma

    x_grid = np.linspace(x_min, x_max, grid_size)
    dx = x_grid[1] - x_grid[0]

    if d >= 32:
        pdf_vals = gaussian_pdf(x_grid, d)
    else:
        pdf_vals = beta_pdf(x_grid, d)

    # Normalize PDF on grid
    pdf_vals = pdf_vals / (np.sum(pdf_vals) * dx)

    # Initialize centroids using quantiles of the distribution
    cdf = np.cumsum(pdf_vals) * dx
    cdf = cdf / cdf[-1]  # Ensure CDF ends at 1
    quantile_positions = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    centroids = np.interp(quantile_positions, cdf, x_grid)

    for _ in range(num_iter):
        # Compute boundaries (midpoints between centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Compute new centroids as conditional expectations
        all_bounds = np.concatenate([[x_min], boundaries, [x_max]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            mask = (x_grid >= all_bounds[i]) & (x_grid < all_bounds[i + 1])
            if i == n_levels - 1:
                mask = (x_grid >= all_bounds[i]) & (x_grid <= all_bounds[i + 1])

            weighted = pdf_vals[mask]
            if weighted.sum() > 0:
                new_centroids[i] = np.sum(x_grid[mask] * weighted) / weighted.sum()
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < 1e-12:
            break
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids.astype(np.float32), boundaries.astype(np.float32)


def compute_theoretical_mse(d: int, bits: int, centroids: np.ndarray, boundaries: np.ndarray,
                            grid_size: int = 10000) -> float:
    """Compute the expected MSE per coordinate for a given codebook."""
    sigma = 1.0 / math.sqrt(d)
    x_min, x_max = -6 * sigma, 6 * sigma
    x_grid = np.linspace(x_min, x_max, grid_size)
    dx = x_grid[1] - x_grid[0]

    if d >= 32:
        pdf_vals = gaussian_pdf(x_grid, d)
    else:
        pdf_vals = beta_pdf(x_grid, d)
    pdf_vals = pdf_vals / (np.sum(pdf_vals) * dx)

    n_levels = len(centroids)
    all_bounds = np.concatenate([[x_min], boundaries, [x_max]])
    mse = 0.0

    for i in range(n_levels):
        mask = (x_grid >= all_bounds[i]) & (x_grid < all_bounds[i + 1])
        if i == n_levels - 1:
            mask = (x_grid >= all_bounds[i]) & (x_grid <= all_bounds[i + 1])
        mse += np.sum((x_grid[mask] - centroids[i]) ** 2 * pdf_vals[mask]) * dx

    return float(mse)


def get_codebook(d: int, bits: int) -> tuple[mx.array, mx.array]:
    """Get precomputed codebook for given dimension and bit-width.

    Returns MLX arrays of (centroids, boundaries).
    """
    key = (d, bits)
    if key not in _codebook_cache:
        filename = f"codebook_d{d}_b{bits}.npz"
        shipped_path = CODEBOOK_DIR / filename
        user_path = _user_cache_dir() / filename

        if shipped_path.exists():
            data = np.load(shipped_path)
            _codebook_cache[key] = (data["centroids"], data["boundaries"])
        elif user_path.exists():
            data = np.load(user_path)
            _codebook_cache[key] = (data["centroids"], data["boundaries"])
        else:
            centroids, boundaries = lloyd_max(d, bits)
            _codebook_cache[key] = (centroids, boundaries)
            # Persist to user-writable cache; package dir may be read-only when pip-installed.
            try:
                user_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(user_path, centroids=centroids, boundaries=boundaries)
            except OSError:
                pass

    centroids, boundaries = _codebook_cache[key]
    return mx.array(centroids, dtype=mx.float32), mx.array(boundaries, dtype=mx.float32)


def precompute_codebooks(
    dims: tuple[int, ...] = (64, 96, 128, 256),
    bits_range: tuple[int, ...] = (2, 3, 4),
) -> None:
    """Precompute and save codebooks for common configurations."""
    CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for d in dims:
        for b in bits_range:
            npz_path = CODEBOOK_DIR / f"codebook_d{d}_b{b}.npz"
            if not npz_path.exists():
                centroids, boundaries = lloyd_max(d, b)
                np.savez(npz_path, centroids=centroids, boundaries=boundaries)


def quantize_scalar(x: mx.array, centroids: mx.array, boundaries: mx.array) -> mx.array:
    """Quantize values to nearest centroid index using vectorized boundary comparison.

    Args:
        x: Values to quantize, any shape
        centroids: Codebook centroids (2^bits,)
        boundaries: Decision boundaries (2^bits - 1,)

    Returns:
        Indices as uint8, same shape as x
    """
    original_shape = x.shape
    flat = x.reshape(-1)

    # Vectorized: compare each value against all boundaries at once
    # flat: (N,) -> (N, 1), boundaries: (B,) -> (1, B)
    # comparisons: (N, B) boolean matrix
    comparisons = flat[:, None] > boundaries[None, :]  # (N, num_boundaries)

    # Sum across boundaries = index (number of boundaries exceeded)
    indices = mx.sum(comparisons.astype(mx.uint8), axis=-1).astype(mx.uint8)  # (N,)

    return indices.reshape(original_shape)


def dequantize_scalar(indices: mx.array, centroids: mx.array) -> mx.array:
    """Map indices back to centroid values.

    Args:
        indices: Quantized indices (uint8), any shape
        centroids: Codebook centroids (2^bits,)

    Returns:
        Dequantized values (float32), same shape as indices
    """
    return centroids[indices.astype(mx.int32)]
