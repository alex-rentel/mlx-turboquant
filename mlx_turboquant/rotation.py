"""Random orthogonal rotation matrix generation.

The rotation is the key insight of TurboQuant: it makes coordinates of unit
vectors approximately independent and identically distributed (Beta distribution),
allowing per-coordinate scalar quantization to be near-optimal.
"""

import mlx.core as mx
import numpy as np

# Cache rotation matrices in memory
_rotation_cache: dict[tuple[int, int], mx.array] = {}

DEFAULT_SEED = 42


def generate_rotation_matrix(d: int, seed: int = DEFAULT_SEED) -> mx.array:
    """Generate a deterministic random orthogonal matrix via QR decomposition.

    Args:
        d: Dimension (head_dim)
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix of shape (d, d)
    """
    rng = np.random.RandomState(seed)
    # Generate random Gaussian matrix
    G = rng.randn(d, d).astype(np.float32)
    # QR decomposition
    Q, R = np.linalg.qr(G)
    # Ensure deterministic sign convention (positive diagonal of R)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[np.newaxis, :]
    return mx.array(Q)


def get_rotation_matrix(d: int, seed: int = DEFAULT_SEED) -> mx.array:
    """Get cached rotation matrix for given dimension.

    Args:
        d: Dimension (head_dim)
        seed: Random seed for reproducibility

    Returns:
        Orthogonal matrix of shape (d, d)
    """
    key = (d, seed)
    if key not in _rotation_cache:
        _rotation_cache[key] = generate_rotation_matrix(d, seed)
    return _rotation_cache[key]


def rotate(x: mx.array, rotation: mx.array) -> mx.array:
    """Apply rotation to vectors.

    Args:
        x: Input vectors (..., d)
        rotation: Orthogonal matrix (d, d)

    Returns:
        Rotated vectors (..., d)
    """
    return x @ rotation.T


def inverse_rotate(y: mx.array, rotation: mx.array) -> mx.array:
    """Apply inverse rotation (transpose of orthogonal matrix).

    Args:
        y: Rotated vectors (..., d)
        rotation: Orthogonal matrix (d, d)

    Returns:
        Original-space vectors (..., d)
    """
    return y @ rotation


def pre_rotate_query(query: mx.array, rotation: mx.array) -> mx.array:
    """Pre-rotate query vectors for fused attention-from-compressed.

    Computes ``Q_rot = Q @ R.T`` — the same forward rotation that is
    applied to K during quantization. This is the key transformation
    that lets us compute dot products against packed codebook indices
    without materializing dequantized K vectors.

    Math:
        K_hat[k] = norms[k] * (centroids[idx[k,:]] @ R)
        Q[q] . K_hat[k] = norms[k] * Q[q] @ R.T . centroids[idx[k,:]]
                        = norms[k] * Q_rot[q] . centroids[idx[k,:]]

    So if we pre-rotate Q once per decode step, the fused QK kernel
    only needs centroid lookups — no inverse rotation per token.

    Args:
        query: Query vectors, shape (..., d). Typically
               (B, H, T_q, D) in the attention hot path.
        rotation: Rotation matrix (d, d) float32.

    Returns:
        Rotated query vectors, same shape as input.

    Notes:
        This is mathematically identical to ``rotate(query, rotation)`` —
        both are just ``query @ rotation.T``. It exists as a named
        function so the fused-attention code path is self-documenting.
    """
    return query @ rotation.T


def hadamard_matrix(d: int) -> mx.array:
    """Generate a normalized Walsh-Hadamard matrix of size d.

    d must be a power of 2. The matrix is orthogonal: H @ H.T = I.

    Args:
        d: Dimension (must be power of 2)

    Returns:
        Normalized Hadamard matrix (d, d)
    """
    if d & (d - 1) != 0:
        raise ValueError(f"Dimension {d} must be a power of 2 for Hadamard transform")

    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    H = H / np.sqrt(d)
    return mx.array(H)


def randomized_hadamard(d: int, seed: int = DEFAULT_SEED) -> mx.array:
    """Generate randomized Hadamard matrix: D @ H where D is random sign diagonal.

    This gives O(d log d) rotation with similar distributional properties.
    Requires d to be a power of 2.

    Args:
        d: Dimension (must be power of 2)
        seed: Random seed

    Returns:
        Orthogonal matrix (d, d)
    """
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    H = hadamard_matrix(d)
    D = mx.diag(mx.array(signs))
    return D @ H
