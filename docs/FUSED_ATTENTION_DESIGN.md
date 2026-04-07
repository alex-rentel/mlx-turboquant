# Fused Attention-from-Compressed — Design Doc (v0.7.0)

**Status:** design, Phase 0 of v0.7.0. No code yet.
**Date:** 2026-04-07

## Problem

At v0.6.0, the decode hot path for a TurboQuant-patched model is:

1. Attention layer calls `cache.update_and_fetch(k_new, v_new)`
2. Cache appends new tokens to FP16 residual buffer
3. If residual exceeds threshold, old tokens get compressed (quantized to packed
   indices + per-token norms)
4. `update_and_fetch` rebuilds a **full dense FP16 K/V** by dequantizing every
   compressed token and concatenating `[sink | decompressed_middle | residual_fp16]`
5. Returns those dense tensors to the attention layer
6. Attention layer calls `mx.fast.scaled_dot_product_attention(q, k_dense, v_dense)`

Step 4 is the bottleneck. Every decode step materializes a
`(B, H, T_compressed, D)` float32/float16 tensor just so SDPA can multiply
`Q @ K^T`. At 4K context on Qwen3-8B this is `1 * 8 * 4000 * 128 * 2 = 8 MB` of
memory traffic per step for K alone, plus the same for V. Benchmarks show 22-36%
decode overhead vs FP16 baseline at 2K-4K context, driven almost entirely by this
rebuild-and-concatenate cost.

## The Observation

We never actually need dense K. We only need the scalar dot product
`Q[q] . K[k]` for each `(query_token, cache_token)` pair. And K[k] is stored in
a form that lets us compute this dot product without a full reconstruction, if
we rearrange the math correctly.

## The Math

### What we store

For each compressed token `k`:
- `norms[k]`: the original L2 norm of `K[k]` (float32)
- `packed[k, :]`: uint8-packed codebook indices, one per dimension
  (4-bit packing = `D/2` bytes per token; 2-bit = `D/4`; 3-bit = `3D/8`)
- Shared across all tokens:
  - `centroids`: the codebook (`2^bits` float32 values)
  - `R`: the rotation matrix `(D, D)` float32

The decompression pipeline (existing code in `kernels.py` /
`cache._dequantize_kv`) reconstructs `K[k]` as:

```
idx[k, j]      = unpack(packed[k], j)               # integer in [0, 2^bits)
y_hat[k, j]    = centroids[idx[k, j]]               # float32
K_hat[k]       = norms[k] * (y_hat[k] @ R)          # apply inverse rotation, scale
```

Note `y_hat[k] @ R` is the inverse rotation, because our forward quant path does
`y = (x / ||x||) @ R.T` (see `rotation.rotate`). So `R.T` is forward, `R` is inverse.

### What attention computes

The only thing attention needs from K is:

```
score[q, k] = Q[q] . K_hat[k]
            = Q[q] . (norms[k] * y_hat[k] @ R)
            = norms[k] * Q[q] . (y_hat[k] @ R)
            = norms[k] * (Q[q] @ R.T) . y_hat[k]
            = norms[k] * Q_rot[q] . y_hat[k]
            = norms[k] * sum_j( Q_rot[q, j] * centroids[idx[k, j]] )
```

Where `Q_rot = Q @ R.T = rotate(Q, R)` (same function we already use on K
during quantization; it's cached and cheap: one D×D matmul per decode step).

### Why this is a win

In the current decompression path, every decode step pays:

- For each of `T_compressed` tokens:
  - Read `D/2` packed bytes (4-bit case)
  - Unpack `D` indices
  - `D` centroid lookups → `y_hat[k]` (D float32 values)
  - `D × D` multiply-adds for `y_hat[k] @ R` (inverse rotation)
  - `D` multiplies by `norms[k]`
- Then the attention layer does another `D × T_compressed` multiply-adds for
  `Q @ K^T`.

**Total per step: `T_compressed × (D² + D)` FLOPs** for the K reconstruction
plus `T_compressed × D` FLOPs for the score computation, i.e.
`T_compressed × (D² + 2D)` FLOPs dominated by the `D²` inverse rotation.

In the fused path, Q_rot is computed once per step (one `(T_q, D) × (D, D)` matmul,
cheap: `T_q × D²`), then each `(q, k)` score is:

- Read `D/2` packed bytes (same)
- `D` centroid lookups (same)
- `D` multiply-adds for `sum_j( Q_rot[q, j] * centroids[idx[k, j]] )`
- 1 multiply by `norms[k]`

**Total per step: `T_q × D² + T_q × T_compressed × D` FLOPs.**

At `T_q = 1, T_compressed = 4000, D = 128`:
- Current: `4000 × (128² + 2×128) = ~66 MFLOP`
- Fused: `1 × 128² + 1 × 4000 × 128 = ~530 KFLOP`

**~125× reduction in FLOPs for the K-side** of attention. The realized speedup
will be less — this workload is already memory-bandwidth bound at the `read
packed bytes` step, not compute-bound — but the elimination of the `D × D`
inverse rotation per token is the key win.

## Kernel Design

### API

```python
def fused_qk_scores_4bit(
    q_rot: mx.array,      # (T_q, D) float32    — pre-rotated query for ONE head/batch
    packed_k: mx.array,   # (T_kv, D/2) uint8  — packed 4-bit indices
    norms_k: mx.array,    # (T_kv,) float32    — per-token norms
    centroids: mx.array,  # (16,) float32      — codebook
    D: int,               # head dimension
) -> mx.array:            # (T_q, T_kv) float32 — raw scores, pre-softmax, pre-scale
```

Variants: `fused_qk_scores_2bit`, `fused_qk_scores_3bit`.

The returned scores are **raw dot products**. Softmax, scaling, and masking are
the caller's problem — they happen in the SDPA dispatch where the compressed
scores are combined with the scores from the FP16 regions (sink and residual).

### Thread / threadgroup layout (first version — correctness-focused)

- **Grid:** `(T_q, T_kv, 1)` — one thread per output score.
- **Threadgroup size:** `(1, 32, 1)` — groups of 32 tokens along the `T_kv` axis,
  aligned to SIMD width. Each thread in a group is independent; no shared memory
  yet (we add it in Phase 4).
- **Inner loop:** serial over `j = 0 .. D-1`:
  - `byte_idx = j / 2; is_hi = (j & 1)`
  - `byte = packed_k[kv_idx, byte_idx]`
  - `idx = is_hi ? (byte >> 4) & 0x0F : byte & 0x0F`
  - `cent = centroids[idx]`
  - `acc += q_rot[q_idx, j] * cent`
- **Output:** `scores[q_idx, kv_idx] = norms_k[kv_idx] * acc`

This is single-element-per-thread, no shared memory, no SIMD reductions.
Functionally identical to the current `_dequant_4bit` kernel's structure — just
with a different output.

### Correctness-first performance (Phase 2)

Before optimizing:
1. Get the raw scores matching `dequant + matmul` to within `atol=1e-4`.
2. Handle 2-bit, 3-bit variants.
3. Handle edge cases: `T_kv = 0`, `T_q > 1` (prefill), non-power-of-2 D (Phi-3.5).

### Optimization (Phase 4, only if Phase 3 micro-benchmark justifies it)

1. **`simd_sum` across D.** Instead of one thread doing `D=128` serial multiplies,
   use 4 threads × 32 lanes each with `simd_sum` to combine. 32× arithmetic
   parallelism per thread.
2. **Threadgroup `shared_q_rot[D]`.** Load one query row into threadgroup memory
   once per group, then all threads in the group read from it. Saves `T_kv ×
   D` global reads per query row.
3. **Threadgroup `shared_centroids[16]`.** 16 float32 = 64 bytes, trivially fits.
   Loaded once per group. Saves `D × T_kv` centroid lookups per group (they all
   hit the same table).
4. **Unroll inner loop by 4.** For D = 128, unroll by 4 → 32 iterations.

## Integration Problem

This is where v0.7.0 has to make an honest trade-off.

### The blocker

mlx-lm's attention layers call:
```python
output = mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask=...)
```

where `k` and `v` are **dense tensors** returned from `cache.update_and_fetch(k, v)`.
There is no hook point in mlx-lm to substitute a custom SDPA implementation.

### Rejected approaches

1. **Return a wrapper object from `update_and_fetch`** — SDPA rejects anything
   that isn't an `mx.array`.
2. **Monkey-patch `mx.fast.scaled_dot_product_attention` globally** — affects
   every SDPA call in the process, fragile, prevents running baselines in the
   same session.
3. **Subclass `mx.array`** — MLX doesn't support user subclasses of arrays.

### Viable approaches

**Approach A: Per-model-family attention patch.** In `patch.py`, after
`apply_turboquant`, walk the model's attention modules and replace their
`__call__` method with a custom version that checks if the cache is a
`TurboQuantKVCache` and routes to the fused path. This requires per-family
patches because `Llama.Attention.__call__` is not the same as
`Qwen2.Attention.__call__` (different RoPE application, different scale factor,
etc.).

**Approach B: Custom TurboQuantKVCache method called from a dispatcher.**
Give `TurboQuantKVCache` a `fused_attention(q, scale, mask, ...)` method that
performs the full attention operation (scores + softmax + weighted sum).
Install a thin attention wrapper per model that calls
`cache.fused_attention(...)` instead of `SDPA(cache.update_and_fetch(...))`.
Essentially the same as A but with a cleaner method boundary.

**Approach C: Ship the kernel as a utility only.** Export
`pre_rotate_query`, `fused_qk_scores_{2,3,4}bit`, and document how to wire
them into a custom attention loop. Don't touch `patch.py` at all. Users who
want the speedup can integrate it themselves.

### Recommendation for v0.7.0

**Start with Approach C.** Ship the kernels and the utility. Prove they're
faster than `dequant + matmul` in isolation via micro-benchmark. Write a
reference attention loop that uses them. Document the per-model-family patch
problem honestly.

**Move Approach A/B to v0.8.0** if and only if the Phase 3 micro-benchmark
shows a meaningful speedup (>= 1.5× on fused vs dequant+matmul for
`T_compressed >= 1024`). The per-family attention patches are a significant
surface area — we should only commit to maintaining them if the perf win
justifies it.

This is the **kernel-first, integration-second** strategy. Phase 3's
micro-benchmark is the go/no-go checkpoint.

## Numerical Considerations

1. **Everything in float32 inside the kernel.** The quantization errors are
   already ~3%; float16 accumulation would add its own ~0.1% error, and at
   `D=128` with 128 additions the float16 error could grow to visible levels.
   Convert to the query dtype only on the final store.

2. **Rotation orientation.** Our stored representation is `y = x_norm @ R.T`,
   so the inverse is `x = y @ R`. To eliminate the inverse inside the kernel
   we pre-compute `Q_rot = Q @ R.T` (same direction as forward). Any confusion
   between R and R.T here will produce garbage scores — **the correctness
   test is the source of truth**.

3. **Packed layout.** Our 4-bit packing stores `(idx[2k], idx[2k+1])` as
   `(low_nibble, high_nibble)` of byte `k` (see `packing.pack_4bit:149`).
   The existing `_DEQUANT_4BIT_SOURCE` already has this unpack code; we can
   lift it directly.

## Fallback and Compatibility

- **All kernels are additive.** Nothing in v0.6.0's correct-and-slow path
  changes. `TurboQuantKVCache.update_and_fetch` still returns dense FP16
  tensors by default.
- **`apply_turboquant(..., use_fused_attention=True)` is opt-in**, defaulting
  to `False` in v0.7.0. Shipping as opt-in avoids any chance of regressing
  the 12-model benchmark baseline.
- **The correctness test runs on every PR.** Any kernel change that breaks
  `allclose(fused_scores, dequant_scores, atol=1e-4)` fails CI.

## Planned tests

1. `test_pre_rotate_query_math` — verify `pre_rotate_query(Q, R) @ y = Q @ (y @ R.T)`
   for random inputs.
2. `test_fused_qk_scores_4bit_matches_reference` — run the full pipeline on a
   small synthetic tensor: quantize some K vectors, run both `fused_qk_scores_4bit`
   and `dequantize + matmul`, assert `np.allclose` to `1e-4`.
3. `test_fused_qk_scores_2bit`, `test_fused_qk_scores_3bit` — same shape.
4. `test_fused_qk_scores_edge_cases`:
   - `T_q = 1` (decode case)
   - `T_q = 4` (small prefill)
   - `T_kv = 0` (no compressed tokens yet — must not crash)
   - `D = 96` (non-power-of-2, for Phi-3.5-mini)
5. `test_fused_qk_scores_no_nans` — sanity: no NaN/Inf in output for random input.

## What this doc does NOT cover (deferred to v0.8.0 or later)

- **Fusing V**. `softmax(scores) @ V_compressed` requires either a two-pass
  kernel (compute all scores, then normalize, then weighted sum) or online
  softmax. Both are possible but significantly more code.
- **Per-model attention patches.** See Approach A/B above. Conditional on
  Phase 3 going well.
- **Using SIMD group matmul primitives.** Apple Silicon GPUs have
  `simdgroup_multiply_accumulate` for matrix tiles. Out of scope for v0.7.0.
- **Partial compression for prefill.** The current design works for decode
  (T_q = 1) and for the compressed region during prefill. Prefill doesn't
  exercise this path because the FP16 residual holds all recent tokens.
