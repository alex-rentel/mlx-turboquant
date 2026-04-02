"""Quality benchmarks: cosine similarity, top-K accuracy, perplexity.

Compares FP16 baseline KV cache with TurboQuant at various bit-widths.
"""

import time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from mlx_turboquant import apply_turboquant, TurboQuantKVCache


def cosine_similarity_benchmark(model_path: str, bits_list: list[int] = [2, 3, 4],
                                 prompt: str = "Explain the theory of relativity in detail. " * 20):
    """Compare prefill logits between baseline and TurboQuant."""
    print(f"\n{'='*60}")
    print(f"Quality Benchmark: Cosine Similarity")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    inputs = mx.array(tokenizer.encode(prompt))[None]
    seq_len = inputs.shape[1]
    print(f"Prompt: {prompt[:60]}... ({seq_len} tokens)\n")

    # Baseline
    baseline_cache = [KVCache() for _ in range(len(model.model.layers))]
    baseline_logits = model(inputs, cache=baseline_cache)
    mx.eval(baseline_logits)
    bl = np.array(baseline_logits[0, -1, :]).astype(np.float64)

    print(f"{'Config':<30} {'Cosine Sim':>12} {'Top-1 Match':>12}")
    print("-" * 56)
    print(f"{'FP16 baseline':<30} {'1.000000':>12} {'Yes':>12}")

    for bits in bits_list:
        for kv_config in [(bits, bits), (bits, max(2, bits - 2))]:
            k_bits, v_bits = kv_config
            label = f"K{k_bits}/V{v_bits}"

            # Window small enough to force compression but large enough for quality
            window = max(4, seq_len // 4)
            apply_turboquant(model, key_bits=k_bits, value_bits=v_bits,
                           residual_window=window)
            tq_cache = model.make_cache()
            tq_logits = model(inputs, cache=tq_cache)
            mx.eval(tq_logits)
            tq = np.array(tq_logits[0, -1, :]).astype(np.float64)

            # Cosine similarity (float64 to avoid overflow)
            cos_sim = np.dot(bl, tq) / (np.linalg.norm(bl) * np.linalg.norm(tq) + 1e-30)

            # Top-1 match
            top1_match = "Yes" if np.argmax(bl) == np.argmax(tq) else "No"

            print(f"{label:<30} {cos_sim:>12.6f} {top1_match:>12}")

    return True


def top_k_retrieval_benchmark(model_path: str, k: int = 10):
    """Check if TurboQuant preserves which tokens the model attends to most."""
    print(f"\n{'='*60}")
    print(f"Quality Benchmark: Top-{k} Token Prediction Overlap")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    prompt = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = mx.array(tokenizer.encode(prompt))[None]

    # Baseline
    baseline_cache = [KVCache() for _ in range(len(model.model.layers))]
    baseline_logits = model(inputs, cache=baseline_cache)
    mx.eval(baseline_logits)
    bl_topk = set(np.argsort(np.array(baseline_logits[0, -1, :]))[-k:])

    print(f"{'Config':<20} {'Top-' + str(k) + ' Overlap':>15}")
    print("-" * 37)

    for bits in [2, 3, 4]:
        apply_turboquant(model, key_bits=bits, value_bits=bits, residual_window=4)
        tq_cache = model.make_cache()
        tq_logits = model(inputs, cache=tq_cache)
        mx.eval(tq_logits)
        tq_topk = set(np.argsort(np.array(tq_logits[0, -1, :]))[-k:])

        overlap = len(bl_topk & tq_topk) / k * 100
        print(f"{'K'+str(bits)+'/V'+str(bits):<20} {overlap:>14.1f}%")


if __name__ == "__main__":
    model_path = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    cosine_similarity_benchmark(model_path)
    top_k_retrieval_benchmark(model_path)
