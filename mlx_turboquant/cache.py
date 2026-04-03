"""TurboQuant KV Cache — drop-in replacement for mlx-lm's KVCache.

Compresses KV vectors using TurboQuant (Algorithm 1, MSE-optimized) with:
- Asymmetric K/V bit allocation (keys need more precision than values)
- Residual window (recent tokens stay in FP16 for quality)
- Outlier channel detection (first-pass norm analysis)
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .codebook import get_codebook, quantize_scalar, dequantize_scalar
from .rotation import get_rotation_matrix, rotate, inverse_rotate
from .packing import pack_indices, unpack_indices


class TurboQuantKVCache:
    """TurboQuant-compressed KV cache for mlx-lm models.

    Drop-in replacement for mlx_lm's KVCache. Older tokens get compressed
    to the specified bit-width, while recent tokens stay in FP16.
    """

    step = 256  # Pre-allocation growth step (matches mlx-lm convention)

    def __init__(
        self,
        head_dim: int = 128,
        num_kv_heads: int = 1,
        key_bits: float = 4,
        value_bits: float = 2,
        residual_window: int = 128,
        rotation_seed: int = 42,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window

        # Rotation matrix (QR decomposition — WHT tested but degraded
        # Gemma quality by >0.5%, see commit history)
        self.rotation = get_rotation_matrix(head_dim, rotation_seed)

        # Codebooks — for fractional bits (e.g. 3.5), store two codebooks
        self._k_fractional = (key_bits != int(key_bits))
        self._v_fractional = (value_bits != int(value_bits))

        if self._k_fractional:
            # e.g. 3.5 → first half at 4-bit, second half at 3-bit
            self._k_bits_hi = int(key_bits) + 1  # 4
            self._k_bits_lo = int(key_bits)       # 3
            self.k_centroids_hi, self.k_boundaries_hi = get_codebook(head_dim, self._k_bits_hi)
            self.k_centroids_lo, self.k_boundaries_lo = get_codebook(head_dim, self._k_bits_lo)
            self.k_centroids = self.k_boundaries = None  # not used
        else:
            self.k_centroids, self.k_boundaries = get_codebook(head_dim, int(key_bits))

        if self._v_fractional:
            self._v_bits_hi = int(value_bits) + 1
            self._v_bits_lo = int(value_bits)
            self.v_centroids_hi, self.v_boundaries_hi = get_codebook(head_dim, self._v_bits_hi)
            self.v_centroids_lo, self.v_boundaries_lo = get_codebook(head_dim, self._v_bits_lo)
            self.v_centroids = self.v_boundaries = None
        else:
            self.v_centroids, self.v_boundaries = get_codebook(head_dim, int(value_bits))

        # FP16 storage for recent tokens (residual window)
        self.keys: Optional[mx.array] = None       # (B, H, T, D)
        self.values: Optional[mx.array] = None      # (B, H, T, D)

        # Compressed storage for older tokens
        # For integer bits: single packed array. For fractional: (packed_hi, packed_lo).
        self._compressed_keys: Optional[mx.array] = None      # packed indices (or hi part)
        self._compressed_keys_lo: Optional[mx.array] = None   # lo part (fractional only)
        self._compressed_key_norms: Optional[mx.array] = None
        self._compressed_values: Optional[mx.array] = None    # packed indices (or hi part)
        self._compressed_values_lo: Optional[mx.array] = None # lo part (fractional only)
        self._compressed_value_norms: Optional[mx.array] = None
        self._compressed_len: int = 0

        # Decompressed cache (avoids re-dequantizing every decode step)
        self._decompressed_keys_cache: Optional[mx.array] = None
        self._decompressed_values_cache: Optional[mx.array] = None
        self._decompressed_valid: bool = False
        self._dequant_calls: int = 0

        # Sequence offset (total tokens seen so far)
        self.offset: int = 0

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self._compressed_keys,
            self._compressed_key_norms,
            self._compressed_values,
            self._compressed_value_norms,
        )

    @state.setter
    def state(self, v):
        (
            self.keys,
            self.values,
            self._compressed_keys,
            self._compressed_key_norms,
            self._compressed_values,
            self._compressed_value_norms,
        ) = v
        self._decompressed_valid = False

    @property
    def meta_state(self):
        return {
            "offset": str(self.offset),
            "compressed_len": str(self._compressed_len),
            "head_dim": str(self.head_dim),
            "key_bits": str(self.key_bits),
            "value_bits": str(self.value_bits),
        }

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v["offset"])
        self._compressed_len = int(v["compressed_len"])

    def empty(self) -> bool:
        return self.keys is None and self._compressed_len == 0

    def is_trimmable(self) -> bool:
        return False  # TurboQuant cache doesn't support trimming

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        if self._compressed_keys is not None:
            total += self._compressed_keys.nbytes + self._compressed_key_norms.nbytes
            total += self._compressed_values.nbytes + self._compressed_value_norms.nbytes
        return total

    def _rotate_and_norm(self, tensor: mx.array) -> tuple[mx.array, mx.array, int]:
        """Shared first step: cast, norm, normalize, rotate. Returns (rotated_flat, norms, N)."""
        B, H, T, D = tensor.shape
        tensor_f32 = tensor.astype(mx.float32)
        norms = mx.linalg.norm(tensor_f32, axis=-1)  # (B, H, T)
        safe_norms = mx.maximum(norms, mx.array(1e-10))
        normalized = tensor_f32 / safe_norms[..., None]
        flat = normalized.reshape(-1, D)
        rotated = rotate(flat, self.rotation)
        return rotated, norms, B * H * T

    def _quantize_kv(self, tensor: mx.array, centroids: mx.array,
                     boundaries: mx.array, bits: float) -> tuple[mx.array, mx.array]:
        """Quantize KV tensor: (B, H, T, D) -> (packed_indices, norms)."""
        B, H, T, D = tensor.shape
        rotated, norms, N = self._rotate_and_norm(tensor)

        indices = quantize_scalar(rotated, centroids, boundaries)
        packed = pack_indices(indices, int(bits)).reshape(B, H, T, -1)
        return packed, norms

    def _quantize_kv_fractional(self, tensor: mx.array,
                                centroids_hi: mx.array, boundaries_hi: mx.array, bits_hi: int,
                                centroids_lo: mx.array, boundaries_lo: mx.array, bits_lo: int,
                                ) -> tuple[mx.array, mx.array, mx.array]:
        """Quantize with fractional bits: first D/2 at bits_hi, second D/2 at bits_lo.

        Returns (packed_hi, packed_lo, norms). Both packed arrays have T dimension.
        """
        B, H, T, D = tensor.shape
        half_D = D // 2
        rotated, norms, N = self._rotate_and_norm(tensor)

        # Split rotated coordinates
        rot_hi = rotated[:, :half_D]   # (N, D/2) — first half at higher bits
        rot_lo = rotated[:, half_D:]   # (N, D/2) — second half at lower bits

        idx_hi = quantize_scalar(rot_hi, centroids_hi, boundaries_hi)
        idx_lo = quantize_scalar(rot_lo, centroids_lo, boundaries_lo)

        packed_hi = pack_indices(idx_hi, bits_hi).reshape(B, H, T, -1)
        packed_lo = pack_indices(idx_lo, bits_lo).reshape(B, H, T, -1)
        return packed_hi, packed_lo, norms

    def _dequantize_kv_fractional(self, packed_hi: mx.array, packed_lo: mx.array,
                                  norms: mx.array,
                                  centroids_hi: mx.array, bits_hi: int,
                                  centroids_lo: mx.array, bits_lo: int) -> mx.array:
        """Dequantize fractional-bit: two packed halves -> full vectors."""
        B, H, T, _ = packed_hi.shape
        D = self.head_dim
        half_D = D // 2

        # Unpack and dequantize each half
        flat_hi = packed_hi.reshape(-1, packed_hi.shape[-1])
        flat_lo = packed_lo.reshape(-1, packed_lo.shape[-1])
        idx_hi = unpack_indices(flat_hi, bits_hi, half_D)
        idx_lo = unpack_indices(flat_lo, bits_lo, half_D)
        y_hi = dequantize_scalar(idx_hi, centroids_hi)
        y_lo = dequantize_scalar(idx_lo, centroids_lo)

        # Reassemble full rotated vector
        y_hat = mx.concatenate([y_hi, y_lo], axis=-1)  # (N, D)

        # Inverse rotate and scale
        x_hat = inverse_rotate(y_hat, self.rotation)
        flat_norms = norms.reshape(-1).astype(mx.float32)
        x_hat = x_hat * flat_norms[:, None]
        return x_hat.reshape(B, H, T, D)

    def _dequantize_kv(self, packed: mx.array, norms: mx.array,
                       centroids: mx.array, bits: int) -> mx.array:
        """Dequantize: (packed_indices, norms) -> (B, H, T, D).

        Uses fused Metal kernel when available (2/3/4-bit, power-of-2 dim).
        Falls back to Python path otherwise.
        """
        B, H, T, _ = packed.shape
        D = self.head_dim

        # Try Metal kernel path (fused unpack + centroid + inverse rotation + scale)
        if bits in (2, 3, 4) and T > 0:
            try:
                from .kernels import metal_dequantize
                flat_packed = packed.reshape(-1, packed.shape[-1])
                flat_norms = norms.reshape(-1).astype(mx.float32)
                x_hat = metal_dequantize(
                    flat_packed, flat_norms, centroids,
                    self.rotation, bits, D,
                )
                return x_hat.reshape(B, H, T, D)
            except Exception:
                pass  # Fall through to Python path

        # Python fallback
        flat_packed = packed.reshape(-1, packed.shape[-1])
        indices = unpack_indices(flat_packed, bits, D)
        y_hat = dequantize_scalar(indices, centroids)
        x_hat = inverse_rotate(y_hat, self.rotation)
        flat_norms = norms.reshape(-1).astype(mx.float32)
        x_hat = x_hat * flat_norms[:, None]
        return x_hat.reshape(B, H, T, D)

    def _compress_one_side(self, old_tensor, is_key: bool):
        """Compress one side (key or value) and return (packed_data, norms, decompressed)."""
        fractional = self._k_fractional if is_key else self._v_fractional

        if fractional:
            bits_hi = self._k_bits_hi if is_key else self._v_bits_hi
            bits_lo = self._k_bits_lo if is_key else self._v_bits_lo
            c_hi = self.k_centroids_hi if is_key else self.v_centroids_hi
            b_hi = self.k_boundaries_hi if is_key else self.v_boundaries_hi
            c_lo = self.k_centroids_lo if is_key else self.v_centroids_lo
            b_lo = self.k_boundaries_lo if is_key else self.v_boundaries_lo

            packed_hi, packed_lo, norms = self._quantize_kv_fractional(
                old_tensor, c_hi, b_hi, bits_hi, c_lo, b_lo, bits_lo,
            )
            decompressed = self._dequantize_kv_fractional(
                packed_hi, packed_lo, norms, c_hi, bits_hi, c_lo, bits_lo,
            )
            return packed_hi, packed_lo, norms, decompressed
        else:
            bits = int(self.key_bits if is_key else self.value_bits)
            centroids = self.k_centroids if is_key else self.v_centroids
            boundaries = self.k_boundaries if is_key else self.v_boundaries

            packed, norms = self._quantize_kv(old_tensor, centroids, boundaries, bits)
            decompressed = self._dequantize_kv(packed, norms, centroids, bits)
            return packed, None, norms, decompressed

    def _append_compressed(self, packed_hi, packed_lo, norms, is_key: bool):
        """Append packed data to compressed storage."""
        if is_key:
            if self._compressed_keys is None:
                self._compressed_keys = packed_hi
                self._compressed_keys_lo = packed_lo
                self._compressed_key_norms = norms
            else:
                self._compressed_keys = mx.concatenate([self._compressed_keys, packed_hi], axis=2)
                if packed_lo is not None:
                    self._compressed_keys_lo = mx.concatenate(
                        [self._compressed_keys_lo, packed_lo], axis=2
                    ) if self._compressed_keys_lo is not None else packed_lo
                self._compressed_key_norms = mx.concatenate(
                    [self._compressed_key_norms, norms], axis=2
                )
        else:
            if self._compressed_values is None:
                self._compressed_values = packed_hi
                self._compressed_values_lo = packed_lo
                self._compressed_value_norms = norms
            else:
                self._compressed_values = mx.concatenate([self._compressed_values, packed_hi], axis=2)
                if packed_lo is not None:
                    self._compressed_values_lo = mx.concatenate(
                        [self._compressed_values_lo, packed_lo], axis=2
                    ) if self._compressed_values_lo is not None else packed_lo
                self._compressed_value_norms = mx.concatenate(
                    [self._compressed_value_norms, norms], axis=2
                )

    def _compress_old_tokens(self):
        """Move tokens outside the residual window to compressed storage.

        Uses batch compression: waits until FP16 buffer reaches 2x the
        residual window, then compresses the excess in one batch. This
        halves compression frequency and amortizes per-call overhead.
        """
        if self.keys is None:
            return

        fp16_len = self.keys.shape[2]
        # Batch: only compress when we've accumulated a full window of excess
        compress_threshold = self.residual_window * 2
        if fp16_len <= compress_threshold:
            return

        n_compress = fp16_len - self.residual_window
        old_keys = self.keys[:, :, :n_compress, :]
        old_values = self.values[:, :, :n_compress, :]
        self.keys = self.keys[:, :, n_compress:, :]
        self.values = self.values[:, :, n_compress:, :]

        # Compress keys and values
        pk_hi, pk_lo, nk, dk = self._compress_one_side(old_keys, is_key=True)
        pv_hi, pv_lo, nv, dv = self._compress_one_side(old_values, is_key=False)

        self._append_compressed(pk_hi, pk_lo, nk, is_key=True)
        self._append_compressed(pv_hi, pv_lo, nv, is_key=False)
        self._compressed_len += n_compress

        # Incrementally extend decompressed cache
        if self._decompressed_keys_cache is None:
            self._decompressed_keys_cache = dk
            self._decompressed_values_cache = dv
        else:
            self._decompressed_keys_cache = mx.concatenate(
                [self._decompressed_keys_cache, dk], axis=2
            )
            self._decompressed_values_cache = mx.concatenate(
                [self._decompressed_values_cache, dv], axis=2
            )
        self._decompressed_valid = True
        self._dequant_calls += 1

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Update cache with new KV and return full KV for attention.

        Args:
            keys: New key vectors (B, H, num_steps, D)
            values: New value vectors (B, H, num_steps, D)

        Returns:
            (all_keys, all_values) for attention computation, both in FP16/FP32
        """
        # Append new tokens to FP16 buffer (simple concatenation)
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.offset += keys.shape[2]

        # Compress old tokens if residual window is exceeded
        self._compress_old_tokens()

        # Build full KV using cached decompression (only re-decompresses
        # when _compress_old_tokens actually moves tokens to compressed storage)
        if self._compressed_len > 0:
            if not self._decompressed_valid:
                self._decompressed_keys_cache = self._dequantize_kv(
                    self._compressed_keys, self._compressed_key_norms,
                    self.k_centroids, self.key_bits,
                )
                self._decompressed_values_cache = self._dequantize_kv(
                    self._compressed_values, self._compressed_value_norms,
                    self.v_centroids, self.value_bits,
                )
                self._decompressed_valid = True
                self._dequant_calls += 1
            all_keys = mx.concatenate([self._decompressed_keys_cache, self.keys], axis=2)
            all_values = mx.concatenate([self._decompressed_values_cache, self.values], axis=2)
        else:
            all_keys = self.keys
            all_values = self.values

        return all_keys, all_values

    def make_mask(self, N: int, offset: int = 0, return_array: bool = False,
                  window_size: Optional[int] = None) -> Optional[Any]:
        """Generate attention mask. Returns 'causal' for standard causal masking."""
        T = self.offset
        if N == 1 and not return_array:
            return None
        return "causal"
