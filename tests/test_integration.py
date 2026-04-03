"""Integration test -- loads smallest available model and verifies cache works."""

import pytest


@pytest.mark.slow
def test_cache_interface_with_real_model():
    """Verify TurboQuantKVCache works with mlx-lm's actual calling convention."""
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_turboquant import apply_turboquant
    except ImportError:
        pytest.skip("mlx-lm not installed")

    try:
        model, tokenizer = load("mlx-community/Qwen3-0.6B-4bit")
    except Exception:
        try:
            model, tokenizer = load("mlx-community/Qwen3-1.7B-4bit")
        except Exception:
            pytest.skip("No small model available")

    apply_turboquant(model, key_bits=4, value_bits=4, residual_window=32,
                     auto_detect_outliers=False)
    cache = model.make_cache()

    inputs = mx.array([[1, 2, 3, 4, 5]])
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    assert logits.shape[0] == 1
    assert logits.shape[1] == 5

    for _ in range(3):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        logits = model(next_token[:, None], cache=cache)
        mx.eval(logits)

    assert logits.shape[1] == 1
