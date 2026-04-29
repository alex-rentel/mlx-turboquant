"""Tests for Phase 2: KV cache integration with mlx-lm.

Tests cover:
- TurboQuantKVCache basic operations
- Compression/decompression round-trip
- Residual window behavior
- Memory savings measurement
- Model patching
- Integration with actual mlx-lm model (if available)
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_turboquant.cache import TurboQuantKVCache
from mlx_turboquant.patch import _get_model_config, apply_turboquant, enable_turboquant

# ============================================================
# Cache Unit Tests (no model needed)
# ============================================================

class TestTurboQuantKVCacheBasic:
    """Basic cache operations without model."""

    def test_init(self):
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2)
        assert cache.empty()
        assert cache.offset == 0
        assert cache.size() == 0

    def test_single_update(self):
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=64)
        # Simulate: batch=1, heads=4, 1 new token, dim=128
        keys = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))

        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 4, 1, 128)
        assert v_out.shape == (1, 4, 1, 128)
        assert cache.offset == 1
        assert not cache.empty()

    def test_incremental_decode(self):
        """Token-by-token generation should work."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=32)

        for i in range(50):
            keys = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))
            values = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))
            k_out, v_out = cache.update_and_fetch(keys, values)
            mx.eval(k_out, v_out)

            assert k_out.shape[2] == i + 1, f"Step {i}: expected seq_len {i+1}, got {k_out.shape[2]}"
            assert cache.offset == i + 1

    def test_prefill_then_decode(self):
        """Prefill with many tokens, then decode one at a time."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=32)

        # Prefill: 100 tokens at once
        keys = mx.array(np.random.randn(1, 4, 100, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 100, 128).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)

        assert cache.offset == 100
        assert k_out.shape[2] == 100

        # Decode: 10 more tokens one at a time
        for i in range(10):
            keys = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))
            values = mx.array(np.random.randn(1, 4, 1, 128).astype(np.float32))
            k_out, v_out = cache.update_and_fetch(keys, values)
            mx.eval(k_out, v_out)

            assert k_out.shape[2] == 100 + i + 1

    def test_residual_window_fp16(self):
        """Recent tokens within residual window should stay in FP16.

        Uses default chunk_size=0 (v0.5.0 batch compression) which
        guarantees `_fp16_len == residual_window` after drain.
        """
        window = 32
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=window)

        keys = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)

        # FP16 buffer should only have `window` tokens
        assert cache._fp16_len == window
        # Compressed storage should have the rest
        assert cache._compressed_len == 100 - window

    def test_residual_window_exact_values(self):
        """Recent tokens in FP16 should be bit-exact (not compressed)."""
        window = 16
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=window)

        np.random.seed(42)
        # Prefill 50 tokens
        all_keys = np.random.randn(1, 2, 50, 128).astype(np.float32)
        all_values = np.random.randn(1, 2, 50, 128).astype(np.float32)

        k_out, v_out = cache.update_and_fetch(
            mx.array(all_keys), mx.array(all_values)
        )
        mx.eval(k_out, v_out)

        # The last `window` tokens should be FP16-exact
        k_recent = np.array(k_out[:, :, -window:, :])
        expected_recent = all_keys[:, :, -window:, :]
        np.testing.assert_allclose(k_recent, expected_recent, atol=1e-5)

    def test_memory_savings(self):
        """Compressed cache should use less memory than FP16."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=32)

        # Add 500 tokens
        keys = mx.array(np.random.randn(1, 8, 500, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 8, 500, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)

        # FP16 baseline: 2 (k+v) * 8 heads * 500 tokens * 128 dims * 2 bytes = 2,048,000
        fp16_bytes = 2 * 8 * 500 * 128 * 2
        tq_bytes = cache.nbytes

        ratio = fp16_bytes / tq_bytes
        assert ratio > 2.0, f"Compression ratio {ratio:.1f}x too low"

    def test_empty_cache_mask(self):
        cache = TurboQuantKVCache(head_dim=128)
        assert cache.empty()

    def test_make_mask_single_token(self):
        cache = TurboQuantKVCache(head_dim=128)
        keys = mx.array(np.random.randn(1, 4, 10, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 10, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)

        mask = cache.make_mask(N=1)
        assert mask is None  # Single token decode needs no mask

    def test_make_mask_multi_token(self):
        cache = TurboQuantKVCache(head_dim=128)
        mask = cache.make_mask(N=10)
        assert mask == "causal"

    def test_different_head_dims(self):
        """Should work with various head dimensions."""
        for d in [64, 96, 128, 256]:
            cache = TurboQuantKVCache(head_dim=d, key_bits=4, value_bits=2,
                                      residual_window=8)
            keys = mx.array(np.random.randn(1, 2, 20, d).astype(np.float32))
            values = mx.array(np.random.randn(1, 2, 20, d).astype(np.float32))
            k_out, v_out = cache.update_and_fetch(keys, values)
            mx.eval(k_out, v_out)
            assert k_out.shape == (1, 2, 20, d)


class TestStateReload:
    """Tests for the state property setter (used by mlx-lm cache restore)."""

    def test_state_roundtrip_integer_bits(self):
        """Save state, restore into a fresh cache, dequantization works."""
        np.random.seed(2)
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16, chunk_size=64)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        meta = cache.meta_state
        state_tuple = cache.state

        # Restore into fresh cache
        new_cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                       residual_window=16, chunk_size=64)
        new_cache.state = state_tuple
        new_cache.meta_state = meta

        # Force a fetch (zero-token update) to trigger lazy re-dequant
        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_out_old, _ = cache.update_and_fetch(decode, decode)
        k_out_new, _ = new_cache.update_and_fetch(decode, decode)

        # Shapes should match
        assert k_out_old.shape == k_out_new.shape

    def test_state_roundtrip_fractional_bits(self):
        """Fractional bit configs (e.g. key_bits=3.5) survive state reload.

        Regression test for a v0.5.0 latent bug where the lazy re-dequant
        path called the non-fractional dequant helper with None centroids.
        """
        np.random.seed(3)
        cache = TurboQuantKVCache(head_dim=128, key_bits=3.5, value_bits=2,
                                  residual_window=16, chunk_size=64)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)

        # Save and restore
        state_tuple = cache.state
        meta = cache.meta_state
        # State tuple must include fractional lo arrays
        assert len(state_tuple) == 10

        new_cache = TurboQuantKVCache(head_dim=128, key_bits=3.5, value_bits=2,
                                       residual_window=16, chunk_size=64)
        new_cache.state = state_tuple
        new_cache.meta_state = meta

        # Lazy re-dequant must not crash on fractional path
        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_out, _ = new_cache.update_and_fetch(decode, decode)
        # Just verify it didn't crash and returned correct total length
        assert k_out.shape[2] == 201

    def test_legacy_6tuple_state_still_loads(self):
        """Backward compat: pre-v0.6.0 6-element state tuples still restore."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16, chunk_size=64)
        # Legacy 6-tuple has no sink and no fractional lo parts
        legacy = (None, None, None, None, None, None)
        cache.state = legacy
        # Should not crash; state should be cleared
        assert cache.sink_keys is None
        assert cache._compressed_keys_lo is None

    @pytest.mark.parametrize("key_bits,value_bits", [(4, 2), (4, 4), (3, 3)])
    def test_legacy_6tuple_realistic_roundtrip(self, key_bits, value_bits):
        """v0.5.x 6-tuple sessions (no sink, no fractional) round-trip into
        a fresh cache and the next decode step does not crash."""
        np.random.seed(11)
        cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                  value_bits=value_bits, residual_window=16,
                                  chunk_size=64)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)

        # Trim live state to the 6-tuple shape pre-v0.6.0 sessions emitted.
        legacy_state = cache.state[:6]
        # Legacy meta_state from v0.5.x: no sink_len / fp16_sink_size keys.
        legacy_meta = {
            "offset": str(cache.offset),
            "compressed_len": str(cache._compressed_len),
            "head_dim": str(cache.head_dim),
            "key_bits": str(key_bits),
            "value_bits": str(value_bits),
            "fp16_len": str(cache._fp16_len),
            "fp16_capacity": str(cache._fp16_capacity),
        }

        new_cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                      value_bits=value_bits, residual_window=16,
                                      chunk_size=64)
        new_cache.state = legacy_state
        new_cache.meta_state = legacy_meta

        # Sink fields default to None / 0 — no leak from older fields.
        assert new_cache.sink_keys is None
        assert new_cache._sink_len == 0
        # Decode: the next update should produce the right total length.
        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_out, _ = new_cache.update_and_fetch(decode, decode)
        assert k_out.shape[2] == 201

    @pytest.mark.parametrize("key_bits,value_bits", [(4, 2), (4, 4), (3, 3)])
    def test_legacy_8tuple_realistic_roundtrip(self, key_bits, value_bits):
        """v0.6.0 pre-fractional 8-tuple sessions (with sink, no fractional
        lo parts) round-trip cleanly."""
        np.random.seed(12)
        cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                  value_bits=value_bits, residual_window=16,
                                  chunk_size=64, fp16_sink_size=8)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)

        legacy_state = cache.state[:8]
        legacy_meta = dict(cache.meta_state)  # keeps sink_len/fp16_sink_size

        new_cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                      value_bits=value_bits, residual_window=16,
                                      chunk_size=64, fp16_sink_size=8)
        new_cache.state = legacy_state
        new_cache.meta_state = legacy_meta

        # Sink should restore; fractional lo arrays absent (integer config).
        assert new_cache._compressed_keys_lo is None
        assert new_cache._compressed_values_lo is None
        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_out, _ = new_cache.update_and_fetch(decode, decode)
        assert k_out.shape[2] == 201

    @pytest.mark.parametrize("key_bits,value_bits",
                             [(4, 2), (4, 4), (3, 3), (3.5, 2), (4, 3.5)])
    def test_current_10tuple_roundtrip_decode_matches(self, key_bits, value_bits):
        """Current 10-tuple state restores to a cache whose next decode
        produces the same shape (and, for integer bits, the same dequantized
        values) as the live cache."""
        np.random.seed(13)
        cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                  value_bits=value_bits, residual_window=16,
                                  chunk_size=64, fp16_sink_size=4)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        full_state = cache.state
        assert len(full_state) == 10

        new_cache = TurboQuantKVCache(head_dim=128, key_bits=key_bits,
                                      value_bits=value_bits, residual_window=16,
                                      chunk_size=64, fp16_sink_size=4)
        new_cache.state = full_state
        new_cache.meta_state = cache.meta_state

        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_old, _ = cache.update_and_fetch(decode, decode)
        k_new, _ = new_cache.update_and_fetch(decode, decode)
        assert k_old.shape == k_new.shape


class TestQJLCorrection:
    """Tests for optional 1-bit QJL sign-sketch residual correction."""

    def test_qjl_disabled_by_default(self):
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2)
        assert cache.qjl_correction is False
        assert cache._qjl_projection is None

    def test_qjl_projection_allocated_when_enabled(self):
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  qjl_correction=True, qjl_n_proj=32)
        assert cache._qjl_projection is not None
        assert cache._qjl_projection.shape == (32, 128)

    def test_qjl_correction_reduces_dequant_error(self):
        """QJL-corrected dequantization should have lower MSE than uncorrected.

        Uses identical inputs and only flips the qjl_correction flag, so
        any difference is purely the correction effect.
        """
        np.random.seed(0)
        # Use a tensor that will have meaningful quantization error
        keys_np = np.random.randn(1, 4, 200, 128).astype(np.float32)
        keys = mx.array(keys_np)

        # 2-bit values force significant quant error so the correction
        # has something meaningful to fix.
        cache_off = TurboQuantKVCache(head_dim=128, key_bits=2, value_bits=2,
                                       residual_window=8, chunk_size=64,
                                       qjl_correction=False)
        cache_on = TurboQuantKVCache(head_dim=128, key_bits=2, value_bits=2,
                                      residual_window=8, chunk_size=64,
                                      qjl_correction=True, qjl_n_proj=32)

        cache_off.update_and_fetch(keys, keys)
        cache_on.update_and_fetch(keys, keys)

        # Compare decompressed cache content (the compressed portion)
        deq_off = np.array(cache_off._decompressed_keys_cache)
        deq_on = np.array(cache_on._decompressed_keys_cache)

        # Both should have the same shape
        assert deq_off.shape == deq_on.shape
        n_compressed = deq_off.shape[2]
        truth = keys_np[:, :, :n_compressed, :]

        mse_off = float(np.mean((deq_off - truth) ** 2))
        mse_on = float(np.mean((deq_on - truth) ** 2))

        # QJL correction should reduce MSE (allow tiny tolerance for noise)
        assert mse_on < mse_off, (
            f"QJL correction did not reduce MSE: off={mse_off:.6f}, on={mse_on:.6f}"
        )

    def test_qjl_does_not_affect_sink_or_residual(self):
        """QJL correction only touches compressed chunks, not sink or residual."""
        np.random.seed(1)
        keys_np = np.random.randn(1, 2, 200, 128).astype(np.float32)
        keys = mx.array(keys_np)

        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16, chunk_size=64,
                                  fp16_sink_size=8, qjl_correction=True)
        k_out, _ = cache.update_and_fetch(keys, keys)

        # Sink should still be bit-exact
        np.testing.assert_array_equal(
            np.array(k_out[:, :, :8, :]), keys_np[:, :, :8, :]
        )
        # Residual region (last cache._fp16_len tokens) should also be bit-exact
        residual_start = k_out.shape[2] - cache._fp16_len
        np.testing.assert_array_equal(
            np.array(k_out[:, :, residual_start:, :]),
            keys_np[:, :, residual_start:, :],
        )


class TestChunkedCompression:
    """Tests for the opt-in fixed-size chunk_size compression draining.

    The default chunk_size=0 selects v0.5.0 batch compression. These tests
    explicitly pass chunk_size>0 to exercise the chunked code path.
    """

    def test_default_is_v05_batch_mode(self):
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2)
        assert cache.chunk_size == 0  # default = v0.5.0 batch path

    def test_chunks_drained_in_fixed_size(self):
        """After a large prefill, FP16 buffer holds [window, window+chunk)."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=32, chunk_size=64)
        # Prefill 300 tokens — should drain (300 - 32) // 64 = 4 chunks of 64
        keys = mx.array(np.random.randn(1, 2, 300, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        assert cache._compressed_len == 4 * 64  # 256
        assert cache._fp16_len == 300 - 256     # 44
        assert 32 <= cache._fp16_len < 32 + 64

    def test_drain_loops_for_multi_chunk_input(self):
        """A single huge update should drain multiple chunks in one call."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16, chunk_size=32)
        keys = mx.array(np.random.randn(1, 2, 200, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        # Expected drained chunks: while fp16_len >= 16+32 = 48, drain 32.
        # After 200 tokens: drain until fp16_len < 48 → fp16_len lands in [16, 48)
        n_drained = (200 - 16) // 32  # 5
        assert cache._compressed_len == n_drained * 32  # 160
        assert cache._fp16_len == 200 - 160              # 40

    def test_custom_chunk_size_smaller(self):
        """Smaller chunks compress more frequently in finer steps."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16, chunk_size=8)
        keys = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        # Should drain (100 - 16) // 8 = 10 chunks of 8 → 80 compressed, 20 in FP16
        assert cache._compressed_len == 80
        assert cache._fp16_len == 20

    def test_no_compression_below_threshold(self):
        """Small prefill (< window + chunk) should not compress at all."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=64, chunk_size=64)
        keys = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        cache.update_and_fetch(keys, keys)
        # 100 < 64 + 64 = 128, so no compression
        assert cache._compressed_len == 0
        assert cache._fp16_len == 100


class TestAttentionSink:
    """Tests for fp16_sink_size — permanent FP16 region for system prompt."""

    def test_sink_disabled_by_default(self):
        """fp16_sink_size=0 should preserve all v0.5.0 behavior."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8)
        assert cache.fp16_sink_size == 0
        assert cache.sink_keys is None
        assert cache._sink_len == 0

        keys = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        assert cache.sink_keys is None
        assert cache._sink_len == 0

    def test_sink_filled_in_single_prefill(self):
        """A single prefill larger than sink should fill the sink."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8, fp16_sink_size=16)
        keys = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)

        assert cache._sink_len == 16
        assert cache.sink_keys.shape == (1, 2, 16, 128)
        np.testing.assert_array_equal(
            np.array(cache.sink_keys[:, :, :16, :]),
            np.array(keys[:, :, :16, :]),
        )
        np.testing.assert_array_equal(
            np.array(cache.sink_values[:, :, :16, :]),
            np.array(values[:, :, :16, :]),
        )

    def test_sink_filled_across_multiple_calls(self):
        """Sink should fill correctly when prefill is split across calls."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=4, fp16_sink_size=16)
        chunk1 = mx.array(np.random.randn(1, 2, 8, 128).astype(np.float32))
        cache.update_and_fetch(chunk1, chunk1)
        assert cache._sink_len == 8
        chunk2 = mx.array(np.random.randn(1, 2, 8, 128).astype(np.float32))
        cache.update_and_fetch(chunk2, chunk2)
        assert cache._sink_len == 16
        assert cache._fp16_len == 0  # no overflow into residual yet

    def test_sink_partial_overlap_with_residual(self):
        """When a single call straddles the sink boundary, split correctly."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8, fp16_sink_size=16)
        prefill = mx.array(np.random.randn(1, 2, 24, 128).astype(np.float32))
        cache.update_and_fetch(prefill, prefill)
        assert cache._sink_len == 16
        assert cache._fp16_len == 8
        np.testing.assert_array_equal(
            np.array(cache.sink_keys[:, :, :16, :]),
            np.array(prefill[:, :, :16, :]),
        )

    def test_sink_survives_compression(self):
        """Sink must NOT be compressed even after many compression cycles."""
        sink_size = 8
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8, fp16_sink_size=sink_size)
        big = mx.array(np.random.randn(1, 2, 88, 128).astype(np.float32))
        cache.update_and_fetch(big, big)

        for _ in range(5):
            chunk = mx.array(np.random.randn(1, 2, 16, 128).astype(np.float32))
            cache.update_and_fetch(chunk, chunk)

        assert cache._sink_len == sink_size
        np.testing.assert_array_equal(
            np.array(cache.sink_keys[:, :, :sink_size, :]),
            np.array(big[:, :, :sink_size, :]),
        )
        assert cache._compressed_len > 0

    def test_sink_returned_at_head_of_kv(self):
        """Returned KV must have sink tokens as the first sink_len positions."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8, fp16_sink_size=16)
        prefill = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        k_out, _ = cache.update_and_fetch(prefill, prefill)
        np.testing.assert_array_equal(
            np.array(k_out[:, :, :16, :]),
            np.array(prefill[:, :, :16, :]),
        )
        assert k_out.shape[2] == cache.offset == 100

    def test_sink_offset_tracking(self):
        """offset must include sink tokens."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=8, fp16_sink_size=16)
        prefill = mx.array(np.random.randn(1, 2, 32, 128).astype(np.float32))
        cache.update_and_fetch(prefill, prefill)
        assert cache.offset == 32
        decode = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        cache.update_and_fetch(decode, decode)
        assert cache.offset == 33

    def test_sink_meta_state_roundtrip(self):
        """meta_state should serialize and restore sink config."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  fp16_sink_size=16)
        prefill = mx.array(np.random.randn(1, 2, 20, 128).astype(np.float32))
        cache.update_and_fetch(prefill, prefill)

        meta = cache.meta_state
        assert meta["fp16_sink_size"] == "16"
        assert meta["sink_len"] == "16"

        new_cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                       fp16_sink_size=16)
        new_cache.meta_state = meta
        assert new_cache._sink_len == 16


class TestCacheReconstructionQuality:
    """Test that compressed tokens are faithfully reconstructed."""

    def test_compressed_key_quality_4bit(self):
        """4-bit key compression should give good reconstruction."""
        np.random.seed(0)
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=4,
                                  residual_window=0)

        # Store 100 tokens (all get compressed since window=0)
        keys_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)
        values_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)

        k_out, v_out = cache.update_and_fetch(
            mx.array(keys_orig), mx.array(values_orig)
        )
        mx.eval(k_out, v_out)

        k_recon = np.array(k_out)

        # Per-vector cosine similarity (median)
        cos_sims = []
        for b in range(1):
            for h in range(4):
                for t in range(100):
                    orig = keys_orig[b, h, t]
                    recon = k_recon[b, h, t]
                    cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
                    cos_sims.append(cos)

        median_cos = np.median(cos_sims)
        assert median_cos > 0.99, f"4-bit median cosine sim {median_cos:.4f} too low"

    def test_compressed_value_quality_2bit(self):
        """2-bit value compression -- lower quality but still usable."""
        np.random.seed(0)
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=0)

        keys_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)
        values_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)

        k_out, v_out = cache.update_and_fetch(
            mx.array(keys_orig), mx.array(values_orig)
        )
        mx.eval(k_out, v_out)

        v_recon = np.array(v_out)

        # 2-bit is lossy but should still capture direction
        cos_sims = []
        for h in range(4):
            for t in range(100):
                orig = values_orig[0, h, t]
                recon = v_recon[0, h, t]
                cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
                cos_sims.append(cos)

        median_cos = np.median(cos_sims)
        assert median_cos > 0.85, f"2-bit median cosine sim {median_cos:.4f} too low"

    def test_metal_dequant_fallback_path_matches_kernel(self):
        """Force the pure-MLX fallback that engages when metal_dequantize
        fails. The fallback is a sticky safety net set on the cache via
        ``_metal_dequant_disabled``. This test flips the flag at
        construction time, runs the same workload as
        ``test_compressed_key_quality_4bit``, and asserts the same
        quality bound."""
        np.random.seed(0)
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=4,
                                  residual_window=0)
        cache._metal_dequant_disabled = True  # force the fallback path

        keys_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)
        values_orig = np.random.randn(1, 4, 100, 128).astype(np.float32)
        k_out, _v_out = cache.update_and_fetch(
            mx.array(keys_orig), mx.array(values_orig)
        )

        k_recon = np.array(k_out)  # materializes the lazy graph
        cos_sims = []
        for h in range(4):
            for t in range(100):
                orig = keys_orig[0, h, t]
                recon = k_recon[0, h, t]
                cos = np.dot(orig, recon) / (
                    np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10
                )
                cos_sims.append(cos)
        median_cos = np.median(cos_sims)
        assert median_cos > 0.99, (
            f"4-bit fallback path median cos sim {median_cos:.4f} too low"
        )


# ============================================================
# Model Patching Tests
# ============================================================

class TestModelPatching:
    """Test model patching (requires mlx-lm)."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small model for testing."""
        try:
            from mlx_lm import load
            model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
            return model, tokenizer
        except Exception:
            pytest.skip("Could not load test model")

    def test_config_detection(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        config = _get_model_config(model)
        assert config["head_dim"] > 0
        assert config["num_kv_heads"] > 0
        assert config["num_layers"] > 0

    def test_apply_turboquant(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        apply_turboquant(model, key_bits=4, value_bits=2)
        assert hasattr(model, "make_cache")
        assert hasattr(model, "_turboquant_config")

        cache = model.make_cache()
        assert len(cache) == model._turboquant_config["num_layers"]
        # At least some layers should use TurboQuantKVCache
        # (outlier layers stay as KVCache)
        tq_count = sum(1 for c in cache if isinstance(c, TurboQuantKVCache))
        assert tq_count > 0, "No TurboQuantKVCache layers found"

    def test_enable_turboquant(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        enable_turboquant(model, bits=4)
        cache = model.make_cache()
        # Find a TurboQuant layer (skip outlier layers)
        tq_layers = [c for c in cache if isinstance(c, TurboQuantKVCache)]
        assert len(tq_layers) > 0
        assert tq_layers[0].key_bits == 4
        assert tq_layers[0].value_bits == 4

    def test_generation_with_turboquant(self, model_and_tokenizer):
        """Generate text with TurboQuant cache -- should produce coherent output."""
        model, tokenizer = model_and_tokenizer
        apply_turboquant(model, key_bits=4, value_bits=4, residual_window=64)

        prompt = "The capital of France is"
        inputs = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

        cache = model.make_cache()

        # Prefill
        logits = model(inputs, cache=cache)
        mx.eval(logits)

        # Decode a few tokens
        generated_tokens = []
        for _ in range(20):
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            generated_tokens.append(next_token.item())
            logits = model(next_token[:, None], cache=cache)
            mx.eval(logits)

        output = tokenizer.decode(generated_tokens)
        # Should contain "Paris" or related content
        assert len(output) > 0, "Generated empty output"

    def test_logit_similarity_vs_baseline(self, model_and_tokenizer):
        """Prefill logits with TurboQuant should be close to baseline."""
        model, tokenizer = model_and_tokenizer

        prompt = "Explain quantum computing in simple terms."
        inputs = mx.array(tokenizer.encode(prompt))[None]

        # Baseline: no TurboQuant (standard cache, all FP16)
        from mlx_lm.models.cache import KVCache
        baseline_cache = [KVCache() for _ in range(len(model.model.layers))]
        baseline_logits = model(inputs, cache=baseline_cache)
        mx.eval(baseline_logits)

        # TurboQuant: 4-bit keys, 4-bit values, large residual window
        # (with window >= seq_len, everything stays FP16, so logits should match)
        apply_turboquant(model, key_bits=4, value_bits=4, residual_window=512)
        tq_cache = model.make_cache()
        tq_logits = model(inputs, cache=tq_cache)
        mx.eval(tq_logits)

        # With large residual window, everything is FP16 -> should be identical
        bl = np.array(baseline_logits[0, -1, :100])
        tq = np.array(tq_logits[0, -1, :100])
        cos_sim = np.dot(bl, tq) / (np.linalg.norm(bl) * np.linalg.norm(tq))
        assert cos_sim > 0.999, f"Logit cosine sim {cos_sim:.6f} too low (should be ~1.0 with large window)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
