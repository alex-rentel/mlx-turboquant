"""Speed benchmarks: tokens/second for prefill and decode.

Compares baseline FP16, mlx-lm quantized cache, and TurboQuant.
"""

import time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from mlx_turboquant import apply_turboquant, TurboQuantKVCache


def benchmark_speed(model_path: str, num_decode_tokens: int = 50,
                    prompt_length: int = 256):
    """Measure prefill and decode speed."""
    print(f"\n{'='*60}")
    print(f"Speed Benchmark")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt_length} tokens, Decode: {num_decode_tokens} tokens")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)

    # Generate a prompt of the right length
    base_prompt = "The quick brown fox jumps over the lazy dog. "
    prompt = base_prompt * (prompt_length // 10 + 1)
    tokens = tokenizer.encode(prompt)[:prompt_length]
    inputs = mx.array(tokens)[None]

    configs = [
        ("FP16 baseline", None),
        ("TQ K4/V4 w128", (4, 4, 128)),
        ("TQ K4/V2 w128", (4, 2, 128)),
        ("TQ K4/V2 w32", (4, 2, 32)),
    ]

    print(f"{'Config':<25} {'Prefill (tok/s)':>15} {'Decode (tok/s)':>15} {'Overhead':>10}")
    print("-" * 67)

    baseline_decode_speed = None

    for name, tq_config in configs:
        if tq_config is None:
            # FP16 baseline
            cache = [KVCache() for _ in range(len(model.model.layers))]
        else:
            k_bits, v_bits, window = tq_config
            apply_turboquant(model, key_bits=k_bits, value_bits=v_bits,
                           residual_window=window)
            cache = model.make_cache()

        # Warmup
        warmup_cache = [KVCache() for _ in range(len(model.model.layers))]
        _ = model(inputs[:, :4], cache=warmup_cache)
        mx.eval(_)

        # Fresh cache
        if tq_config is None:
            cache = [KVCache() for _ in range(len(model.model.layers))]
        else:
            cache = model.make_cache()

        # Prefill
        t0 = time.perf_counter()
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        t_prefill = time.perf_counter() - t0
        prefill_speed = prompt_length / t_prefill

        # Decode
        t0 = time.perf_counter()
        for _ in range(num_decode_tokens):
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            logits = model(next_token[:, None], cache=cache)
            mx.eval(logits)
        t_decode = time.perf_counter() - t0
        decode_speed = num_decode_tokens / t_decode

        if baseline_decode_speed is None:
            baseline_decode_speed = decode_speed
            overhead = ""
        else:
            pct = (1 - decode_speed / baseline_decode_speed) * 100
            overhead = f"{pct:>+8.1f}%"

        print(f"{name:<25} {prefill_speed:>13.1f} {decode_speed:>13.1f} {overhead:>10}")


if __name__ == "__main__":
    benchmark_speed("mlx-community/Qwen2.5-7B-Instruct-4bit")
