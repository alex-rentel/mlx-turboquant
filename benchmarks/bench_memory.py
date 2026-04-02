"""Memory benchmarks: measure actual KV cache memory usage.

Compares FP16, mlx-lm quantized cache, and TurboQuant at various context lengths.
"""

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from mlx_turboquant import apply_turboquant, TurboQuantKVCache


def measure_cache_memory(model_path: str,
                         context_lengths: list[int] = [512, 1024, 2048, 4096]):
    """Measure actual cache memory at various context lengths."""
    print(f"\n{'='*60}")
    print(f"Memory Benchmark")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)

    # Detect head_dim and num_kv_heads
    from mlx_turboquant.patch import _get_model_config
    config = _get_model_config(model)
    hd = config["head_dim"]
    nkv = config["num_kv_heads"]
    print(f"Config: {num_layers} layers, {nkv} KV heads, head_dim={hd}\n")

    header = f"{'Context':>8}"
    configs = [
        ("FP16", None),
        ("TQ K4/V4", (4, 4)),
        ("TQ K4/V2", (4, 2)),
        ("TQ K3/V2", (3, 2)),
    ]
    for name, _ in configs:
        header += f" {name:>12}"
    header += f" {'Best Ratio':>12}"
    print(header)
    print("-" * len(header))

    for ctx_len in context_lengths:
        row = f"{ctx_len:>8}"

        # Compute theoretical FP16 size
        # 2 (K+V) * num_layers * num_kv_heads * ctx_len * head_dim * 2 bytes
        fp16_bytes = 2 * num_layers * nkv * ctx_len * hd * 2
        row += f" {fp16_bytes/1e6:>10.1f}MB"

        best_ratio = 1.0
        for name, kv_bits in configs[1:]:
            k_bits, v_bits = kv_bits
            # Create TQ caches and simulate filling
            caches = [
                TurboQuantKVCache(
                    head_dim=hd, num_kv_heads=nkv,
                    key_bits=k_bits, value_bits=v_bits,
                    residual_window=128,
                )
                for _ in range(num_layers)
            ]

            # Fill with random data
            for cache in caches:
                keys = mx.array(np.random.randn(1, nkv, ctx_len, hd).astype(np.float32))
                values = mx.array(np.random.randn(1, nkv, ctx_len, hd).astype(np.float32))
                cache.update_and_fetch(keys, values)

            total_bytes = sum(c.nbytes for c in caches)
            ratio = fp16_bytes / total_bytes
            best_ratio = max(best_ratio, ratio)
            row += f" {total_bytes/1e6:>10.1f}MB"

        row += f" {best_ratio:>11.1f}x"
        print(row)


if __name__ == "__main__":
    measure_cache_memory("mlx-community/Qwen2.5-7B-Instruct-4bit")
