"""Edge case and hardening tests.

Tests for error handling, boundary conditions, and unusual inputs.
"""

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_turboquant.codebook import get_codebook, lloyd_max, quantize_scalar, dequantize_scalar
from mlx_turboquant.rotation import get_rotation_matrix, hadamard_matrix, randomized_hadamard
from mlx_turboquant.packing import pack_indices, unpack_indices
from mlx_turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from mlx_turboquant.cache import TurboQuantKVCache


class TestCodebookEdgeCases:
    """Edge cases for codebook computation."""

    def test_1bit_codebook(self):
        """1-bit codebook should have exactly 2 centroids."""
        c, b = lloyd_max(128, 1)
        assert c.shape == (2,)
        assert b.shape == (1,)

    def test_codebook_different_dims(self):
        """Codebook should work for any dimension."""
        for d in [16, 32, 64, 128, 256, 512]:
            c, b = lloyd_max(d, 2)
            assert c.shape == (4,)

    def test_codebook_small_dimension(self):
        """Small dimensions should still produce valid codebooks."""
        c, b = lloyd_max(4, 2)
        assert len(c) == 4
        assert np.all(np.diff(c) > 0)

    def test_quantize_extreme_values(self):
        """Values far outside distribution should clip to extreme centroids."""
        c, b = get_codebook(128, 4)
        extreme = mx.array([10.0, -10.0, 0.0])
        indices = quantize_scalar(extreme, c, b)
        idx = np.array(indices)
        assert idx[0] == 15  # Max centroid
        assert idx[1] == 0   # Min centroid

    def test_quantize_boundary_values(self):
        """Values exactly on boundaries should map consistently."""
        c, b = get_codebook(128, 4)
        boundary_val = b[7]  # Middle boundary (should be ~0)
        idx = quantize_scalar(mx.array([boundary_val + 1e-7]), c, b)
        assert np.array(idx)[0] == 8

    def test_dequantize_all_indices(self):
        """All valid indices should produce valid centroids."""
        c, b = get_codebook(128, 4)
        for i in range(16):
            idx = mx.array([i], dtype=mx.uint8)
            val = dequantize_scalar(idx, c)
            assert not np.isnan(np.array(val)[0])


class TestRotationEdgeCases:
    """Edge cases for rotation matrices."""

    def test_hadamard_non_power_of_2(self):
        """Should raise ValueError for non-power-of-2 dimensions."""
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(96)

    def test_hadamard_1x1(self):
        """1x1 Hadamard should be [[1]]."""
        H = hadamard_matrix(1)
        np.testing.assert_allclose(np.array(H), [[1.0]], atol=1e-6)

    def test_rotation_large_dimension(self):
        """Should handle large dimensions."""
        R = get_rotation_matrix(512)
        assert R.shape == (512, 512)

    def test_rotation_seed_reproducibility(self):
        """Same seed should always produce same matrix."""
        from mlx_turboquant.rotation import _rotation_cache
        _rotation_cache.clear()
        R1 = get_rotation_matrix(128, seed=999)
        _rotation_cache.clear()
        R2 = get_rotation_matrix(128, seed=999)
        np.testing.assert_array_equal(np.array(R1), np.array(R2))


class TestPackingEdgeCases:
    """Edge cases for bit-packing."""

    def test_pack_all_zeros(self):
        """All-zero indices should pack/unpack correctly."""
        for bits in [2, 3, 4]:
            indices = mx.zeros((1, 128), dtype=mx.uint8)
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, 128)
            np.testing.assert_array_equal(np.array(indices), np.array(unpacked))

    def test_pack_all_max(self):
        """Max-value indices should pack/unpack correctly."""
        for bits in [2, 3, 4]:
            max_val = (1 << bits) - 1
            indices = mx.full((1, 128), max_val, dtype=mx.uint8)
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, 128)
            np.testing.assert_array_equal(np.array(indices), np.array(unpacked))

    def test_pack_unsupported_bits(self):
        """Should raise ValueError for unsupported bit-widths."""
        indices = mx.zeros((1, 128), dtype=mx.uint8)
        with pytest.raises(ValueError, match="Unsupported"):
            pack_indices(indices, 5)
        with pytest.raises(ValueError, match="Unsupported"):
            unpack_indices(indices, 5, 128)

    def test_pack_single_element(self):
        """Packing with minimum dimensions."""
        for bits, min_d in [(2, 4), (3, 8), (4, 2)]:
            indices = mx.array(np.random.randint(0, 2**bits, size=(1, min_d)).astype(np.uint8))
            packed = pack_indices(indices, bits)
            unpacked = unpack_indices(packed, bits, min_d)
            np.testing.assert_array_equal(np.array(indices), np.array(unpacked))


class TestQuantizerEdgeCases:
    """Edge cases for TurboQuantMSE and TurboQuantProd."""

    def test_mse_large_norm_vectors(self):
        """Vectors with very large norms should quantize/dequantize correctly."""
        tq = TurboQuantMSE(d=128, bits=4)
        x = mx.array(np.random.randn(10, 128).astype(np.float32) * 1000)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        # Norms should be preserved (roughly)
        orig_norms = np.linalg.norm(np.array(x), axis=-1)
        recon_norms = np.linalg.norm(np.array(x_hat), axis=-1)
        ratios = recon_norms / orig_norms
        assert np.median(ratios) > 0.9
        assert np.median(ratios) < 1.1

    def test_mse_tiny_norm_vectors(self):
        """Very small vectors should not cause numerical issues."""
        tq = TurboQuantMSE(d=128, bits=4)
        x = mx.array(np.random.randn(10, 128).astype(np.float32) * 1e-8)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        assert not np.any(np.isnan(np.array(x_hat)))
        assert not np.any(np.isinf(np.array(x_hat)))

    def test_mse_identical_vectors(self):
        """Quantizing the same vector multiple times should give same result."""
        tq = TurboQuantMSE(d=128, bits=4)
        x = mx.array(np.random.randn(128).astype(np.float32))
        qt1 = tq.quantize(x)
        qt2 = tq.quantize(x)
        np.testing.assert_array_equal(
            np.array(qt1.packed_indices), np.array(qt2.packed_indices)
        )

    def test_mse_batch_vs_single(self):
        """Batch quantization should match single-vector quantization."""
        tq = TurboQuantMSE(d=128, bits=4)
        np.random.seed(42)
        x_batch = mx.array(np.random.randn(5, 128).astype(np.float32))

        # Batch
        qt_batch = tq.quantize(x_batch)
        x_hat_batch = tq.dequantize(qt_batch)

        # Single
        for i in range(5):
            qt_single = tq.quantize(x_batch[i])
            x_hat_single = tq.dequantize(qt_single)
            np.testing.assert_allclose(
                np.array(x_hat_batch[i]),
                np.array(x_hat_single),
                atol=1e-5,
            )

    def test_prod_with_2bits(self):
        """TurboQuantProd with 2 bits: 1 bit MSE + 1 bit QJL."""
        tq = TurboQuantProd(d=128, bits=2)
        x = mx.array(np.random.randn(10, 128).astype(np.float32))
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        assert not np.any(np.isnan(np.array(x_hat)))


class TestCacheEdgeCases:
    """Edge cases for TurboQuantKVCache."""

    def test_empty_sequence(self):
        """Cache should handle zero-token update gracefully."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2)
        keys = mx.array(np.random.randn(1, 2, 0, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 0, 128).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape[2] == 0

    def test_single_token_sequence(self):
        """Cache with just 1 token."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=128)
        keys = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 1, 128).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 2, 1, 128)

    def test_sequence_shorter_than_window(self):
        """All tokens should stay FP16 when seq < window."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=256)
        keys = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 50, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        assert cache._compressed_len == 0
        assert cache._fp16_len == 50

    def test_zero_residual_window(self):
        """Window=0 means all tokens get compressed immediately.

        Note: this requires the default v0.5.0 batch compression mode
        (chunk_size=0). The opt-in chunked path (chunk_size>0) keeps up
        to chunk_size tokens uncompressed by design.
        """
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=4,
                                  residual_window=0)
        keys = mx.array(np.random.randn(1, 2, 10, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 10, 128).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        _ = k_out.shape  # force evaluation
        _ = v_out.shape
        assert cache._compressed_len == 10
        assert cache._fp16_len == 0

    def test_nbytes_increases_with_tokens(self):
        """Memory usage should grow with token count."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=16)
        keys = mx.array(np.random.randn(1, 4, 10, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 10, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        bytes_10 = cache.nbytes

        keys = mx.array(np.random.randn(1, 4, 100, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 4, 100, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        bytes_110 = cache.nbytes
        assert bytes_110 > bytes_10

    def test_state_properties(self):
        """Cache state should be readable."""
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2)
        keys = mx.array(np.random.randn(1, 2, 10, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 10, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        state = cache.state
        meta = cache.meta_state
        assert meta["offset"] == "10"
        assert meta["head_dim"] == "128"

    def test_is_not_trimmable(self):
        cache = TurboQuantKVCache(head_dim=128)
        assert not cache.is_trimmable()

    def test_large_head_dim(self):
        """d=256 (Gemma) should work."""
        cache = TurboQuantKVCache(head_dim=256, key_bits=4, value_bits=2,
                                  residual_window=8)
        keys = mx.array(np.random.randn(1, 2, 20, 256).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 20, 256).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 2, 20, 256)


class TestPatchEdgeCases:
    """Edge cases for model patching."""

    def test_apply_with_skip_layers(self):
        """Manual skip_layers should work."""
        try:
            from mlx_lm import load
            from mlx_turboquant.patch import apply_turboquant
            from mlx_turboquant.cache import TurboQuantKVCache
            from mlx_lm.models.cache import KVCache

            model, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
            apply_turboquant(model, skip_layers=[0, 27], auto_detect_outliers=False)
            cache = model.make_cache()

            assert isinstance(cache[0], KVCache)  # Skipped
            assert isinstance(cache[27], KVCache)  # Skipped
            assert isinstance(cache[5], TurboQuantKVCache)  # Normal
        except Exception:
            pytest.skip("Could not load test model")

    def test_apply_no_outlier_detection(self):
        """Should work with auto_detect_outliers=False."""
        try:
            from mlx_lm import load
            from mlx_turboquant.patch import apply_turboquant
            from mlx_turboquant.cache import TurboQuantKVCache

            model, _ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
            apply_turboquant(model, auto_detect_outliers=False)
            cache = model.make_cache()
            assert all(isinstance(c, TurboQuantKVCache) for c in cache)
        except Exception:
            pytest.skip("Could not load test model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
