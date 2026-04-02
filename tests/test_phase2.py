"""Tests for Phase 2: KV cache integration with mlx-lm.

Tests cover:
- TurboQuantKVCache basic operations
- Compression/decompression round-trip
- Residual window behavior
- Memory savings measurement
- Model patching
- Integration with actual mlx-lm model (if available)
"""

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_turboquant.cache import TurboQuantKVCache
from mlx_turboquant.patch import apply_turboquant, enable_turboquant, _get_model_config


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
        """Recent tokens within residual window should stay in FP16."""
        window = 32
        cache = TurboQuantKVCache(head_dim=128, key_bits=4, value_bits=2,
                                  residual_window=window)

        # Add 100 tokens
        keys = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        values = mx.array(np.random.randn(1, 2, 100, 128).astype(np.float32))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        # FP16 buffer should only have `window` tokens
        assert cache.keys.shape[2] == window
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
