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
        key_bits: int = 4,
        value_bits: int = 2,
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

        # Codebooks
        self.k_centroids, self.k_boundaries = get_codebook(head_dim, key_bits)
        self.v_centroids, self.v_boundaries = get_codebook(head_dim, value_bits)

        # FP16 storage for recent tokens (residual window)
        self.keys: Optional[mx.array] = None       # (B, H, T, D)
        self.values: Optional[mx.array] = None      # (B, H, T, D)

        # Compressed storage for older tokens
        self._compressed_keys: Optional[mx.array] = None      # packed indices
        self._compressed_key_norms: Optional[mx.array] = None  # float16 norms
        self._compressed_values: Optional[mx.array] = None
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

    def _quantize_kv(self, tensor: mx.array, centroids: mx.array,
                     boundaries: mx.array, bits: int) -> tuple[mx.array, mx.array]:
        """Quantize KV tensor: (B, H, T, D) -> (packed_indices, norms)."""
        B, H, T, D = tensor.shape

        # Cast to float32 for norm computation (float16 overflows on large vectors)
        tensor_f32 = tensor.astype(mx.float32)

        # Compute norms in float32
        norms = mx.linalg.norm(tensor_f32, axis=-1)  # (B, H, T)
        safe_norms = mx.maximum(norms, mx.array(1e-10))

        # Normalize
        normalized = tensor_f32 / safe_norms[..., None]  # (B, H, T, D)

        # Rotate: need to reshape for matmul
        flat = normalized.reshape(-1, D)  # (B*H*T, D)
        rotated = rotate(flat, self.rotation)  # (B*H*T, D)

        # Quantize each coordinate
        indices = quantize_scalar(rotated, centroids, boundaries)  # (B*H*T, D)

        # Pack
        packed = pack_indices(indices, bits)  # (B*H*T, packed_dim)
        packed = packed.reshape(B, H, T, -1)

        # Store norms as float32 (float16 overflows for models like Qwen
        # with key norms > 65504; cost: 4 bytes vs 2 bytes per token per head)
        return packed, norms

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

    def _compress_old_tokens(self):
        """Move tokens outside the residual window to compressed storage."""
        if self.keys is None:
            return

        fp16_len = self.keys.shape[2]

        # Only compress if we have more FP16 tokens than the window allows
        if fp16_len <= self.residual_window:
            return

        # How many tokens to compress
        n_compress = fp16_len - self.residual_window

        # Split: old tokens to compress | recent tokens to keep in FP16
        old_keys = self.keys[:, :, :n_compress, :]
        old_values = self.values[:, :, :n_compress, :]
        self.keys = self.keys[:, :, n_compress:, :]
        self.values = self.values[:, :, n_compress:, :]

        # Quantize old tokens
        packed_k, norms_k = self._quantize_kv(
            old_keys, self.k_centroids, self.k_boundaries, self.key_bits
        )
        packed_v, norms_v = self._quantize_kv(
            old_values, self.v_centroids, self.v_boundaries, self.value_bits
        )

        # Append to compressed storage
        if self._compressed_keys is None:
            self._compressed_keys = packed_k
            self._compressed_key_norms = norms_k
            self._compressed_values = packed_v
            self._compressed_value_norms = norms_v
        else:
            self._compressed_keys = mx.concatenate(
                [self._compressed_keys, packed_k], axis=2
            )
            self._compressed_key_norms = mx.concatenate(
                [self._compressed_key_norms, norms_k], axis=2
            )
            self._compressed_values = mx.concatenate(
                [self._compressed_values, packed_v], axis=2
            )
            self._compressed_value_norms = mx.concatenate(
                [self._compressed_value_norms, norms_v], axis=2
            )
        self._compressed_len += n_compress

        # Incrementally extend the decompressed cache instead of invalidating.
        # Dequantize ONLY the newly compressed tokens and append.
        new_decompressed_k = self._dequantize_kv(
            packed_k, norms_k, self.k_centroids, self.key_bits,
        )
        new_decompressed_v = self._dequantize_kv(
            packed_v, norms_v, self.v_centroids, self.value_bits,
        )
        if self._decompressed_keys_cache is None:
            self._decompressed_keys_cache = new_decompressed_k
            self._decompressed_values_cache = new_decompressed_v
        else:
            self._decompressed_keys_cache = mx.concatenate(
                [self._decompressed_keys_cache, new_decompressed_k], axis=2
            )
            self._decompressed_values_cache = mx.concatenate(
                [self._decompressed_values_cache, new_decompressed_v], axis=2
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
