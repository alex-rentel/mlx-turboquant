"""Core TurboQuant quantization algorithms.

Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
from the paper. These operate on raw MLX arrays — no model awareness.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .codebook import get_codebook, quantize_scalar, dequantize_scalar
from .rotation import get_rotation_matrix, rotate, inverse_rotate
from .packing import pack_indices, unpack_indices
from .qjl import (
    get_projection_matrix,
    qjl_quantize,
    qjl_dequantize,
)


@dataclass
class QuantizedTensor:
    """Compressed representation of a vector."""
    packed_indices: mx.array   # Bit-packed quantization indices
    norms: mx.array            # L2 norms (float16)
    bits: int                  # Bit-width used
    d: int                     # Original dimension
    # For TurboQuant_prod only:
    qjl_signs: Optional[mx.array] = None
    qjl_norms: Optional[mx.array] = None


class TurboQuantMSE:
    """Algorithm 1: TurboQuant_mse — MSE-optimized quantization.

    rotate → normalize → quantize each coordinate with Lloyd-Max → store indices + norm
    """

    def __init__(self, d: int, bits: int = 4, seed: int = 42):
        """
        Args:
            d: Vector dimension (head_dim)
            bits: Quantization bit-width (2, 3, or 4)
            seed: Random seed for rotation matrix
        """
        self.d = d
        self.bits = bits
        self.seed = seed

        # Get precomputed components
        self.rotation = get_rotation_matrix(d, seed)
        self.centroids, self.boundaries = get_codebook(d, bits)

    def quantize(self, x: mx.array) -> QuantizedTensor:
        """Quantize vectors.

        Args:
            x: Input vectors of shape (..., d)

        Returns:
            QuantizedTensor with packed indices and norms
        """
        # 1. Compute and store norms (float32 to avoid overflow)
        x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
        norms = mx.linalg.norm(x_f32, axis=-1)  # (...)

        # 2. Normalize to unit sphere (avoid division by zero)
        safe_norms = mx.maximum(norms, mx.array(1e-10))
        x_norm = x_f32 / safe_norms[..., None]

        # 3. Rotate
        y = rotate(x_norm, self.rotation)  # (..., d)

        # 4. Quantize each coordinate
        indices = quantize_scalar(y, self.centroids, self.boundaries)  # (..., d) uint8

        # 5. Pack indices
        packed = pack_indices(indices, self.bits)

        return QuantizedTensor(
            packed_indices=packed,
            norms=norms,
            bits=self.bits,
            d=self.d,
        )

    def dequantize(self, qt: QuantizedTensor) -> mx.array:
        """Dequantize back to approximate vectors.

        Args:
            qt: QuantizedTensor from quantize()

        Returns:
            Reconstructed vectors (..., d)
        """
        # 1. Unpack indices
        indices = unpack_indices(qt.packed_indices, qt.bits, qt.d)

        # 2. Look up centroids
        y_hat = dequantize_scalar(indices, self.centroids)  # (..., d)

        # 3. Inverse rotate
        x_hat = inverse_rotate(y_hat, self.rotation)  # (..., d)

        # 4. Rescale by norms
        norms = qt.norms.astype(mx.float32) if qt.norms.dtype != mx.float32 else qt.norms
        x_hat = x_hat * norms[..., None]

        return x_hat


class TurboQuantProd:
    """Algorithm 2: TurboQuant_prod — Inner-product optimized quantization.

    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residual for unbiased
    inner product estimation.

    WARNING: Community consensus is that this hurts quality as a drop-in
    KV cache replacement. Use TurboQuantMSE unless you have fused attention kernels.
    """

    def __init__(self, d: int, bits: int = 4, qjl_dim: Optional[int] = None,
                 seed: int = 42, qjl_seed: int = 12345):
        """
        Args:
            d: Vector dimension
            bits: Total bit budget (MSE gets bits-1, QJL gets 1 bit)
            qjl_dim: Projection dimension for QJL (default: d)
            seed: Seed for rotation matrix
            qjl_seed: Seed for QJL projection
        """
        self.d = d
        self.bits = bits
        self.qjl_dim = qjl_dim or d

        # MSE stage uses (bits - 1) bits
        self.mse = TurboQuantMSE(d, bits=bits - 1, seed=seed)

        # QJL projection matrix
        self.projection = get_projection_matrix(d, self.qjl_dim, qjl_seed)

    def quantize(self, x: mx.array) -> QuantizedTensor:
        """Quantize with MSE + QJL residual correction.

        Args:
            x: Input vectors (..., d)

        Returns:
            QuantizedTensor with MSE indices, QJL signs, and norms
        """
        # 1. MSE quantize with (bits-1) bits
        qt_mse = self.mse.quantize(x)

        # 2. Compute residual
        x_hat_mse = self.mse.dequantize(qt_mse)
        residual = x - x_hat_mse  # (..., d)

        # 3. QJL quantize residual
        qjl_signs, qjl_norms = qjl_quantize(residual, self.projection)

        return QuantizedTensor(
            packed_indices=qt_mse.packed_indices,
            norms=qt_mse.norms,
            bits=self.bits,
            d=self.d,
            qjl_signs=qjl_signs,
            qjl_norms=qjl_norms,
        )

    def dequantize(self, qt: QuantizedTensor) -> mx.array:
        """Dequantize MSE + QJL correction.

        Args:
            qt: QuantizedTensor from quantize()

        Returns:
            Reconstructed vectors (..., d)
        """
        # 1. MSE dequantize (using bits - 1)
        mse_qt = QuantizedTensor(
            packed_indices=qt.packed_indices,
            norms=qt.norms,
            bits=self.bits - 1,
            d=self.d,
        )
        x_hat_mse = self.mse.dequantize(mse_qt)

        # 2. QJL dequantize residual
        if qt.qjl_signs is not None and qt.qjl_norms is not None:
            x_hat_qjl = qjl_dequantize(
                qt.qjl_signs, qt.qjl_norms, self.projection, self.d
            )
            return x_hat_mse + x_hat_qjl

        return x_hat_mse
