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
from .qjl import get_projection_matrix, qjl_quantize, qjl_dequantize


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
        fp16_sink_size: int = 0,
        chunk_size: int = 0,
        qjl_correction: bool = False,
        qjl_n_proj: int = 32,
    ):
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.fp16_sink_size = fp16_sink_size
        self.chunk_size = chunk_size
        self.qjl_correction = qjl_correction
        self.qjl_n_proj = qjl_n_proj

        # Lazy QJL projection matrix — only allocated when correction is on.
        # Uses the global get_projection_matrix cache so all caches with the
        # same (head_dim, n_proj) share the same projection.
        if qjl_correction:
            self._qjl_projection = get_projection_matrix(head_dim, qjl_n_proj)
        else:
            self._qjl_projection = None

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

        # FP16 attention sink: first N tokens, NEVER compressed.
        # Independent of the sliding residual window. Used to permanently
        # preserve system prompt tokens that define tool schemas, instruction
        # format, etc. Disabled when fp16_sink_size == 0.
        self.sink_keys: Optional[mx.array] = None   # (B, H, fp16_sink_size, D)
        self.sink_values: Optional[mx.array] = None # (B, H, fp16_sink_size, D)
        self._sink_len: int = 0                     # valid tokens in sink (0..fp16_sink_size)

        # FP16 storage for recent tokens (residual window)
        self.keys: Optional[mx.array] = None       # (B, H, capacity, D) pre-allocated
        self.values: Optional[mx.array] = None      # (B, H, capacity, D)
        self._fp16_len: int = 0                     # valid tokens in FP16 buffer
        self._fp16_capacity: int = 0                # allocated capacity

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

        # v0.8.0: fused attention mode. When True, update_and_fetch
        # returns sink + residual keys only (no decompressed middle)
        # and the patched SDPA uses the packed state directly via
        # get_fused_state(). V is still fully dequantized — we do not
        # fuse V in v0.8.0.
        self._use_fused_attention: bool = False

        # Sequence offset (total tokens seen so far)
        self.offset: int = 0

    # ------------------------------------------------------------------
    # v0.8.0 fused attention accessors
    # ------------------------------------------------------------------

    @property
    def has_compressed(self) -> bool:
        """True if there are any compressed tokens available for fused QK."""
        return self._compressed_len > 0

    def get_fused_state(self) -> dict:
        """Return the packed state needed by the fused SDPA path.

        The patched scaled_dot_product_attention reads this dict to
        run fused_qk_scores_*_batched against the compressed region
        without ever materializing dequantized K.

        Returns:
            dict with:
              packed_keys: (B, H_kv, T_compressed, packed_dim) uint8
              key_norms:   (B, H_kv, T_compressed) float32
              key_centroids: (2^bits,) float32  — None if fractional
              rotation:    (D, D) float32
              key_bits:    int
              sink_len:    int — number of tokens pinned in sink (positions 0..sink_len)
              compressed_len: int — number of compressed tokens
              fp16_len:    int — number of residual FP16 tokens
              head_dim:    int
        """
        return {
            # Packed K state (used by fused_qk_scores_4bit_batched in v0.8.0)
            "packed_keys": self._compressed_keys,
            "key_norms": self._compressed_key_norms,
            "key_centroids": self.k_centroids,
            "rotation": self.rotation,
            "key_bits": int(self.key_bits),
            # Packed V state (added in v0.9.0 for full fused attention kernel).
            # V is already compressed alongside K in _compress_one_side and
            # the packed indices + norms live in _compressed_values /
            # _compressed_value_norms. The decompressed cache exists for
            # the standard SDPA path but is not required when the fused
            # kernel reads packed V directly.
            "packed_values": self._compressed_values,
            "value_norms": self._compressed_value_norms,
            "value_centroids": self.v_centroids,
            "value_bits": int(self.value_bits),
            # Position metadata
            "sink_len": self._sink_len,
            "compressed_len": self._compressed_len,
            "fp16_len": self._fp16_len,
            "head_dim": self.head_dim,
        }

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self._compressed_keys,
            self._compressed_key_norms,
            self._compressed_values,
            self._compressed_value_norms,
            self.sink_keys,
            self.sink_values,
            self._compressed_keys_lo,
            self._compressed_values_lo,
        )

    @state.setter
    def state(self, v):
        # Backward compatible state loading by tuple length:
        #   6 elements — pre-v0.6.0 (no sink, no fractional lo parts)
        #   8 elements — v0.6.0 with sink, no fractional lo parts
        #  10 elements — v0.6.0 with sink and fractional lo parts (current)
        n = len(v)
        if n == 6:
            (
                self.keys,
                self.values,
                self._compressed_keys,
                self._compressed_key_norms,
                self._compressed_values,
                self._compressed_value_norms,
            ) = v
        elif n == 8:
            (
                self.keys,
                self.values,
                self._compressed_keys,
                self._compressed_key_norms,
                self._compressed_values,
                self._compressed_value_norms,
                self.sink_keys,
                self.sink_values,
            ) = v
        else:
            (
                self.keys,
                self.values,
                self._compressed_keys,
                self._compressed_key_norms,
                self._compressed_values,
                self._compressed_value_norms,
                self.sink_keys,
                self.sink_values,
                self._compressed_keys_lo,
                self._compressed_values_lo,
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
            "fp16_sink_size": str(self.fp16_sink_size),
            "sink_len": str(self._sink_len),
            "fp16_len": str(self._fp16_len),
            "fp16_capacity": str(self._fp16_capacity),
        }

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v["offset"])
        self._compressed_len = int(v["compressed_len"])
        if "fp16_sink_size" in v:
            self.fp16_sink_size = int(v["fp16_sink_size"])
        if "sink_len" in v:
            self._sink_len = int(v["sink_len"])
        if "fp16_len" in v:
            self._fp16_len = int(v["fp16_len"])
        if "fp16_capacity" in v:
            self._fp16_capacity = int(v["fp16_capacity"])

    def empty(self) -> bool:
        return self._fp16_len == 0 and self._compressed_len == 0 and self._sink_len == 0

    def is_trimmable(self) -> bool:
        return False  # TurboQuant cache doesn't support trimming

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        total = 0
        elem_size = 4  # float32
        D = self.head_dim
        if self.keys is not None and self._fp16_len > 0:
            # Only count valid FP16 tokens, not pre-allocated capacity
            B, H = self.keys.shape[0], self.keys.shape[1]
            total += 2 * B * H * self._fp16_len * D * elem_size  # keys + values
        if self.sink_keys is not None and self._sink_len > 0:
            B, H = self.sink_keys.shape[0], self.sink_keys.shape[1]
            total += 2 * B * H * self._sink_len * D * elem_size  # sink keys + values
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

    def _rebuild_decompressed_cache(self) -> None:
        """Re-dequantize the entire compressed cache from packed storage.

        Used by the lazy state-reload path: when `cache.state` is set from
        a saved tuple, the in-memory `_decompressed_*_cache` is None but the
        compressed arrays are populated. This rebuilds the dense cache.

        Handles both fractional and non-fractional bit widths. Note that
        QJL correction CANNOT be re-applied here because the original
        pre-quantization tensors are no longer available — the corrected
        cache would have been baked in at compression time during the
        original session.
        """
        if self._k_fractional:
            self._decompressed_keys_cache = self._dequantize_kv_fractional(
                self._compressed_keys, self._compressed_keys_lo,
                self._compressed_key_norms,
                self.k_centroids_hi, self._k_bits_hi,
                self.k_centroids_lo, self._k_bits_lo,
            )
        else:
            self._decompressed_keys_cache = self._dequantize_kv(
                self._compressed_keys, self._compressed_key_norms,
                self.k_centroids, int(self.key_bits),
            )

        if self._v_fractional:
            self._decompressed_values_cache = self._dequantize_kv_fractional(
                self._compressed_values, self._compressed_values_lo,
                self._compressed_value_norms,
                self.v_centroids_hi, self._v_bits_hi,
                self.v_centroids_lo, self._v_bits_lo,
            )
        else:
            self._decompressed_values_cache = self._dequantize_kv(
                self._compressed_values, self._compressed_value_norms,
                self.v_centroids, int(self.value_bits),
            )
        self._decompressed_valid = True

    def _apply_qjl_correction(self, original: mx.array, decompressed: mx.array) -> mx.array:
        """Apply 1-bit QJL sign-sketch correction to a dequantized tensor.

        Computes the residual r = original - decompressed, takes a 1-bit
        sign sketch via random Gaussian projection, then reconstructs an
        unbiased estimate of r and adds it back to the decompressed tensor.

        Unlike sharpner/cache_v2.py which stores the sketch but never reads
        it back, we apply the correction immediately at compression time.
        The corrected tensor is what gets stored in _decompressed_keys_cache,
        so QJL sign bits never need to be persisted — they are transient
        within this call. Zero memory overhead, ~5% extra compute per chunk.

        Args:
            original: Pre-quantization tensor, shape (B, H, T, D)
            decompressed: Dequantized tensor from MSE quant, same shape

        Returns:
            Corrected tensor (B, H, T, D) with reduced quantization error
        """
        residual = (original.astype(mx.float32)
                    - decompressed.astype(mx.float32))
        # Flatten leading axes for projection
        B, H, T, D = residual.shape
        flat_r = residual.reshape(-1, D)
        signs, r_norms = qjl_quantize(flat_r, self._qjl_projection)
        correction_flat = qjl_dequantize(signs, r_norms, self._qjl_projection, D)
        correction = correction_flat.reshape(B, H, T, D)
        return decompressed + correction

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
        else:
            bits = int(self.key_bits if is_key else self.value_bits)
            centroids = self.k_centroids if is_key else self.v_centroids
            boundaries = self.k_boundaries if is_key else self.v_boundaries

            packed, norms = self._quantize_kv(old_tensor, centroids, boundaries, bits)
            decompressed = self._dequantize_kv(packed, norms, centroids, bits)
            packed_hi, packed_lo = packed, None

        if self.qjl_correction:
            decompressed = self._apply_qjl_correction(old_tensor, decompressed)

        return packed_hi, packed_lo, norms, decompressed

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

        Two compression strategies, selected by `self.chunk_size`:

        - chunk_size == 0 (default, v0.5.0 batch behavior):
            Wait until the FP16 buffer reaches 2x residual_window, then
            compress everything beyond residual_window in one call.
            Single Metal dispatch per drain. Variable shape per call.
            Phase 3 v0.6.0 benchmarks showed this is the fastest mode.

        - chunk_size > 0 (opt-in fixed-size chunks):
            Drain whole chunks of `chunk_size` tokens whenever the buffer
            exceeds `residual_window + chunk_size`. Multiple Metal dispatches
            per drain (one per chunk). Stable shapes per call. Architecturally
            friendlier for future kernels that template on chunk dimension.

        Sink storage is independent and never touched here.
        """
        if self.keys is None:
            return

        if self.chunk_size <= 0:
            # v0.5.0 batch path — single drain, variable size, fastest in
            # current Metal kernels which do not template on chunk dimension.
            compress_threshold = self.residual_window * 2
            if self._fp16_len <= compress_threshold:
                return
            n_compress = self._fp16_len - self.residual_window
            self._drain_chunk(n_compress)
            return

        # Chunked path — fixed-size drains in a loop.
        chunk_size = self.chunk_size
        threshold = self.residual_window + chunk_size
        while self._fp16_len >= threshold:
            self._drain_chunk(chunk_size)

    def _drain_chunk(self, n_compress: int) -> None:
        """Compress and evict the first `n_compress` tokens from the FP16 buffer.

        Shared between the v0.5.0 batch path and the chunked path.
        """
        old_keys = self.keys[:, :, :n_compress, :]
        old_values = self.values[:, :, :n_compress, :]
        # Shift remaining to front of pre-allocated buffer
        remaining = self._fp16_len - n_compress
        self.keys[:, :, :remaining, :] = self.keys[:, :, n_compress:self._fp16_len, :]
        self.values[:, :, :remaining, :] = self.values[:, :, n_compress:self._fp16_len, :]
        self._fp16_len = remaining

        # Compress keys and values for this chunk
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

    def _route_sink(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Peel off sink-bound tokens from the head of the incoming batch.

        Returns the (keys, values) that should flow into the residual buffer
        after sink tokens have been written to sink storage. If the sink is
        disabled or already full, returns the inputs unchanged.
        """
        if self.fp16_sink_size <= 0:
            return keys, values

        prev_offset = self.offset
        if prev_offset >= self.fp16_sink_size:
            return keys, values  # sink already full

        T_new = keys.shape[2]
        n_sink_new = min(self.fp16_sink_size - prev_offset, T_new)
        if n_sink_new <= 0:
            return keys, values

        sink_k = keys[:, :, :n_sink_new, :]
        sink_v = values[:, :, :n_sink_new, :]

        if self.sink_keys is None:
            B, H, _, D = keys.shape
            self.sink_keys = mx.zeros(
                (B, H, self.fp16_sink_size, D), dtype=keys.dtype
            )
            self.sink_values = mx.zeros(
                (B, H, self.fp16_sink_size, D), dtype=values.dtype
            )

        # Write sink tokens at their absolute position [prev_offset, prev_offset + n_sink_new)
        self.sink_keys[:, :, prev_offset:prev_offset + n_sink_new, :] = sink_k
        self.sink_values[:, :, prev_offset:prev_offset + n_sink_new, :] = sink_v
        self._sink_len = prev_offset + n_sink_new

        # Return remaining tokens (may be empty)
        if n_sink_new == T_new:
            empty_k = keys[:, :, :0, :]
            empty_v = values[:, :, :0, :]
            return empty_k, empty_v
        return keys[:, :, n_sink_new:, :], values[:, :, n_sink_new:, :]

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
        T_total = keys.shape[2]

        # Route sink-bound tokens out of the residual flow first.
        keys_rest, values_rest = self._route_sink(keys, values)
        T_new = keys_rest.shape[2]

        # Append remaining tokens to FP16 residual buffer using pre-allocation
        if T_new > 0:
            if self.keys is None:
                # First residual write: pre-allocate with headroom
                B, H, _, D = keys_rest.shape
                cap = max(T_new, self.residual_window * 2) + 256
                self._fp16_capacity = cap
                self._fp16_len = T_new
                self.keys = mx.zeros((B, H, cap, D), dtype=keys_rest.dtype)
                self.values = mx.zeros((B, H, cap, D), dtype=values_rest.dtype)
                self.keys[:, :, :T_new, :] = keys_rest
                self.values[:, :, :T_new, :] = values_rest
            else:
                new_len = self._fp16_len + T_new
                if new_len > self._fp16_capacity:
                    # Grow buffer (rare after initial prefill)
                    B, H, _, D = self.keys.shape
                    new_cap = new_len + 256
                    new_k = mx.zeros((B, H, new_cap, D), dtype=keys_rest.dtype)
                    new_v = mx.zeros((B, H, new_cap, D), dtype=values_rest.dtype)
                    new_k[:, :, :self._fp16_len, :] = self.keys[:, :, :self._fp16_len, :]
                    new_v[:, :, :self._fp16_len, :] = self.values[:, :, :self._fp16_len, :]
                    self.keys = new_k
                    self.values = new_v
                    self._fp16_capacity = new_cap
                # Write new tokens in-place (no concat)
                self.keys[:, :, self._fp16_len:self._fp16_len + T_new, :] = keys_rest
                self.values[:, :, self._fp16_len:self._fp16_len + T_new, :] = values_rest
                self._fp16_len = new_len

        self.offset += T_total

        # Compress old tokens if residual window is exceeded.
        # Note: sink storage is separate and never touched by compression.
        self._compress_old_tokens()

        # Build full KV: [sink (FP16, permanent) | decompressed_middle | residual_fp16]
        # The decompressed_*_cache is built incrementally inside
        # _compress_old_tokens(). The branch below handles state-reload paths
        # (cache.state setter from a saved tuple), where the in-memory cache
        # is empty but compressed_* arrays are populated from disk.
        if self._compressed_len > 0 and not self._decompressed_valid:
            self._rebuild_decompressed_cache()

        # In fused attention mode (v0.8.0), we skip materializing the
        # decompressed middle for KEYS — the patched SDPA will read the
        # packed state directly via get_fused_state(). Values are still
        # fully dequantized because v0.8.0 does not fuse V.
        #
        # But only skip the middle when this is a decode step (T_total == 1):
        # the fused SDPA path only handles T_q=1 in v0.8.0 and falls through
        # to the original wrapper for prefill. Returning sparse keys during
        # prefill would break the fallback because keys and values would have
        # mismatched sequence lengths.
        skip_compressed_middle = (
            self._use_fused_attention and T_total == 1
        )
        parts_k = []
        parts_v = []
        if self._sink_len > 0:
            parts_k.append(self.sink_keys[:, :, :self._sink_len, :])
            parts_v.append(self.sink_values[:, :, :self._sink_len, :])
        if self._compressed_len > 0:
            if not skip_compressed_middle:
                parts_k.append(self._decompressed_keys_cache)
            parts_v.append(self._decompressed_values_cache)
        if self._fp16_len > 0:
            parts_k.append(self.keys[:, :, :self._fp16_len, :])
            parts_v.append(self.values[:, :, :self._fp16_len, :])

        if len(parts_k) == 0:
            # Cache fully empty (e.g., zero-token update before any state).
            # Return an empty (B, H, 0, D) array shaped from the input.
            all_keys = keys[:, :, :0, :]
            all_values = values[:, :, :0, :]
        elif len(parts_k) == 1:
            all_keys = parts_k[0]
            all_values = parts_v[0]
        else:
            all_keys = mx.concatenate(parts_k, axis=2)
            all_values = mx.concatenate(parts_v, axis=2)

        return all_keys, all_values

    def make_mask(self, N: int, offset: int = 0, return_array: bool = False,
                  window_size: Optional[int] = None) -> Optional[Any]:
        """Generate attention mask. Returns 'causal' for standard causal masking."""
        T = self.offset
        if N == 1 and not return_array:
            return None
        return "causal"
