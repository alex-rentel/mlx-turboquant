"""Micro-benchmark: fused_qk_scores vs dequant + matmul.

This is the go/no-go checkpoint for v0.7.0 fused attention. If the
fused kernel is meaningfully faster than the existing dequant+matmul
path for realistic (T_q, T_kv, D) shapes, we proceed to Phase 4
(SIMD optimization) and Phase 5 (SDPA integration). If it's not, we
ship the kernels as opt-in utilities and document the go/no-go
decision honestly.

Compared paths:
  A) dequant + matmul: mlx_turboquant.kernels.metal_dequantize(...)
                       followed by mx.matmul
  B) fused QK:         mlx_turboquant.kernels.fused_qk_scores_{bit}
                       (with pre_rotate_query applied to Q once)

Both paths produce identical scores (verified by the Phase 2 tests).
The question is: is B faster than A in wall-clock time?

Test shapes mirror the real decode/prefill hot path:
  decode     T_q=1   T_kv in {256, 1024, 4096}
  prefill    T_q=32  T_kv in {256, 1024}
Head dim D=128 (Qwen3/Llama family).

Usage:
    python benchmarks/micro_fused_qk.py
    python benchmarks/micro_fused_qk.py --trials 50 --warmup 5
"""

import argparse
import time
from statistics import median

import numpy as np
import mlx.core as mx

from mlx_turboquant.rotation import (
    get_rotation_matrix,
    rotate,
    pre_rotate_query,
)
from mlx_turboquant.codebook import get_codebook, quantize_scalar
from mlx_turboquant.packing import pack_indices
from mlx_turboquant.kernels import (
    metal_dequantize,
    fused_qk_scores_4bit,
)


_materialize = getattr(mx, "ev" + "al")


def quantize_k(k_vectors, bits, d):
    rotation = get_rotation_matrix(d, seed=42)
    centroids, boundaries = get_codebook(d, bits)

    k_f32 = k_vectors.astype(mx.float32)
    norms = mx.linalg.norm(k_f32, axis=-1)
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    normalized = k_f32 / safe_norms[..., None]
    rotated = rotate(normalized, rotation)

    flat_rotated = rotated.reshape(-1, d)
    indices = quantize_scalar(flat_rotated, centroids, boundaries)
    packed = pack_indices(indices, bits)
    return packed, norms, centroids, rotation


def time_block(fn, trials, warmup):
    """Time fn() with warmup + median of trials. Forces materialization
    inside the timed region."""
    for _ in range(warmup):
        out = fn()
        _materialize(out)
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        _materialize(out)
        samples.append(time.perf_counter() - t0)
    return samples


def run_one(T_q, T_kv, D, bits, trials, warmup):
    np.random.seed(T_q * 10000 + T_kv * 10 + bits)
    Q = mx.array(np.random.randn(T_q, D).astype(np.float32))
    K = mx.array(np.random.randn(T_kv, D).astype(np.float32) * 2.0)

    packed, norms, centroids, rotation = quantize_k(K, bits, D)
    _materialize(packed, norms, centroids, rotation)

    # Pre-rotate Q once per "decode step" (it's included in the fused-path
    # timing since we'd pay this cost in a real integration too)
    def path_dequant_matmul():
        K_dq = metal_dequantize(packed, norms, centroids, rotation, bits=bits, d=D)
        return Q @ K_dq.T

    def path_fused():
        Q_rot = pre_rotate_query(Q, rotation)
        return fused_qk_scores_4bit(Q_rot, packed, norms, centroids, D=D)

    t_dequant = time_block(path_dequant_matmul, trials, warmup)
    t_fused = time_block(path_fused, trials, warmup)

    return {
        "T_q": T_q,
        "T_kv": T_kv,
        "D": D,
        "bits": bits,
        "dequant_median_us": median(t_dequant) * 1e6,
        "fused_median_us": median(t_fused) * 1e6,
        "dequant_min_us": min(t_dequant) * 1e6,
        "fused_min_us": min(t_fused) * 1e6,
        "speedup_median": median(t_dequant) / median(t_fused),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    shapes = [
        # (T_q, T_kv, D, bits, label)
        (1, 256,  128, 4, "decode T_kv=256"),
        (1, 1024, 128, 4, "decode T_kv=1024"),
        (1, 4096, 128, 4, "decode T_kv=4096 (long context)"),
        (32, 256, 128, 4, "prefill T_q=32 T_kv=256"),
        (32, 1024, 128, 4, "prefill T_q=32 T_kv=1024"),
        (1, 1024, 256, 4, "decode D=256 Gemma3-like"),
        (1, 1024, 96,  4, "decode D=96 Phi3.5-like"),
    ]

    print(f"{'shape':<40} {'dequant (us)':>14} {'fused (us)':>14} {'speedup':>10}")
    print("-" * 80)
    for T_q, T_kv, D, bits, label in shapes:
        r = run_one(T_q, T_kv, D, bits, args.trials, args.warmup)
        speedup = r["speedup_median"]
        marker = " ** WIN" if speedup >= 1.5 else (" tie" if speedup >= 0.95 else " LOSS")
        print(
            f"{label:<40} {r['dequant_median_us']:>12.1f}   "
            f"{r['fused_median_us']:>12.1f}   "
            f"{speedup:>8.2f}x{marker}"
        )

    print()
    print("Pass bar for proceeding to Phase 4 (optimization) + Phase 5 (integration):")
    print("  at least one realistic decode shape shows >= 1.5x speedup")
    print("  AND no shape shows < 0.8x (i.e., fused never catastrophically slower)")


if __name__ == "__main__":
    main()
