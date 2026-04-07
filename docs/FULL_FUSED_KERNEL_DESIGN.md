# Full Fused Attention Kernel — Design (branch `feat/full-fused-attention`)

**Status:** design, Phase 1 of v0.9.0 attempt.
**Goal:** A single Metal kernel that computes the complete decode-step attention output from packed K and V, matching `mx.fast.scaled_dot_product_attention` numerically and beating it in wall time for long contexts.
**Ancestor:** builds on `feat/fused-sdpa-qwen3` (v0.8.0 SDPA dispatch infrastructure).

## Why v0.8.0 failed, and why this might work

v0.8.0 built a correct SDPA replacement that decomposed attention into 6-7 separate Metal dispatches (fused_qk for K, matmul for sink K, matmul for residual K, concat scores, softmax, broadcast V, matmul for V). The sum of those dispatches was slower than `mx.fast.scaled_dot_product_attention` — Apple's single fused kernel. See `docs/FUSED_SDPA_RESULTS.md` for benchmarks.

**The correct baseline to beat is `mx.fast.sdpa` operating on dense FP16 K/V, not `dequant+matmul`.** We need to write our OWN single fused kernel that does Q@K^T + online softmax + weighted sum over V, all in one Metal dispatch, reading packed indices directly.

Sharpner's `_FUSED_ATTN_NOROT_SOURCE` is the reference implementation. This doc documents the adaptation for our cache layout and packing format.

## Math recap

Our stored KV representation (for compressed tokens):
- `K[k] = norms_k[k] * (centroids[k_idx[k, :]] @ R)`
- `V[k] = norms_v[k] * (centroids[v_idx[k, :]] @ R)`

Where R is the rotation matrix (orthogonal, `R @ R.T = I`), centroids is the Lloyd-Max 2-bit codebook (4 entries), and `k_idx`/`v_idx` are packed uint8 indices.

### QK dot product
```
Q[q] . K[k]^T = Q[q] . (centroids[k_idx] @ R)^T * norms_k[k]
              = (Q[q] @ R.T) . centroids[k_idx] * norms_k[k]
              = q_rot[q] . centroids[k_idx] * norms_k[k]
```

### V weighted sum
```
output[q] = sum_k( softmax(QK^T)[q, k] * V[k] )
          = sum_k( w[k] * norms_v[k] * (centroids[v_idx] @ R) )
          = (sum_k( w[k] * norms_v[k] * centroids[v_idx] )) @ R
          = acc_rotated @ R
```

Where `w[k]` is the softmax weight. So:
- Accumulate V contributions **in rotated space** inside the kernel: `acc_rotated[j] += w[k] * norms_v[k] * centroids[v_idx[k, j]]`.
- Single inverse rotation at the end: `output = acc_rotated @ R`.

The V inverse rotation is paid ONCE per query, not per KV token. O(D²) once, not O(D² × T_kv).

### Scale

Standard attention: `scale = 1/sqrt(head_dim)`. We absorb this into Q before the kernel runs: `q_scaled_rot = (Q / sqrt(D)) @ R.T`. One scalar multiply during the pre-rotation. Inside the kernel, no scale factor appears.

### Online softmax (Flash-style)

Per kv-token:
```
new_max = max(running_max, score)
alpha   = exp(running_max - new_max)          # rescales the old state
beta    = exp(score - new_max)                # weight for the new term
running_sum  = running_sum  * alpha + beta
running_acc  = running_acc  * alpha + beta * v_contribution
running_max  = new_max
```

Final: `output[j] = running_acc[j] / running_sum`.

## Kernel architecture (adapted from sharpner)

### Thread / threadgroup layout

- **Grid:** `(num_query_heads, 1, 1)` — one threadgroup per query head. Single batch (B=1) in v0.9.0 minimum viable version.
- **Threadgroup:** `(1024, 1, 1)` = 32 simdgroups × 32 lanes each.
- **Work distribution:**
  - Each simdgroup processes a **stride-32 slice** of the T_kv axis: simdgroup `s` handles tokens `s, s+32, s+64, ...`.
  - Within a simdgroup, each of the 32 lanes permanently owns **4 of the 128 output dimensions** (lane L owns dims `L*4, L*4+1, L*4+2, L*4+3`).
  - The QK dot product for one kv-token is computed via `simd_sum` across the 32 lanes.
  - The V accumulator is **not reduced** — each lane keeps its own 4-element slice of `acc_rotated` in registers throughout the T_kv loop.

### Register state per thread (during T_kv loop)

- `q[4]` — this lane's 4 elements of pre-rotated, pre-scaled query. Loaded once before the loop.
- `local_max` — running softmax max, scalar float.
- `local_sum` — running softmax sum, scalar float.
- `local_acc[4]` — this lane's 4 elements of running V accumulator, in rotated space.

Per kv-token:
1. Each lane reads its uint8 packed-K byte (holds 4 2-bit indices covering this lane's 4 dims).
2. Computes `partial = sum_i(q[i] * centroids[k_idx[i]])` — 4 multiply-adds.
3. `simd_sum(partial)` → full QK dot across 32 lanes, broadcast to every lane.
4. `score = simd_sum_result * norms_k[kv_head, k]` — each lane has the same score value.
5. Online softmax update — all 4 scalars updated identically on every lane (`local_max`, `local_sum`, `alpha`, `beta` are all scalar broadcasts, so the update is lockstep).
6. Each lane reads its uint8 packed-V byte (holds 4 2-bit indices for its 4 output dims).
7. `local_acc[i] = local_acc[i] * alpha + beta * centroids[v_idx[i]] * norms_v[kv_head, k]` — 4 updates per lane.

### Cross-simdgroup combine (after the T_kv loop)

The 32 simdgroups each have their own `local_max`, `local_sum`, `local_acc[4]` for their stride-32 slice of kv-tokens. Must combine into global results.

- `threadgroup float tg_max[32]` — one per simdgroup
- `threadgroup float tg_sum[32]` — one per simdgroup
- `threadgroup float tg_acc[32 * 128]` — one full 128-dim accumulator per simdgroup

Lane 0 of each simdgroup writes `local_max` and `local_sum`. All 32 lanes of each simdgroup write their 4 `local_acc` dims. Then one `threadgroup_barrier`.

Finally, every lane redundantly computes:
```
global_max = max over s of tg_max[s]
global_sum = 0
result[4] = 0
for s in 0..32:
    factor = exp(tg_max[s] - global_max)
    global_sum += tg_sum[s] * factor
    for i in 0..4:
        result[i] += tg_acc[s*128 + lane*4 + i] * factor
output_rotated[head * D + lane*4 + i] = result[i] / global_sum
```

This costs `32 * 4 = 128` float multiply-adds per lane for the final combine — trivially cheap next to the T_kv scan.

### Post-kernel work (Python side)

1. Pre-rotate and pre-scale Q: `q_rot = (Q / sqrt(D)) @ R.T`. Cheap: one `(H, D) × (D, D)` matmul.
2. Dispatch the kernel. Output is `output_rotated (H, D)`.
3. Inverse-rotate the output: `output = output_rotated @ R`. Another `(H, D) × (D, D)` matmul. Cheap.

Both pre/post rotations use MLX's built-in `@` (delegates to Accelerate BLAS). No custom code needed.

## Adaptation from sharpner to our layout

### Packing format

Sharpner: 2-bit packed into `uint32` words, 16 values per word, `D/16` words per token. At D=128, 8 words per token.

Us: 2-bit packed into `uint8` bytes, 4 values per byte, `D/4` bytes per token. At D=128, 32 bytes per token.

**Key observation**: our uint8 layout maps to the 32-lane pattern even more cleanly than sharpner's uint32. Lane L reads byte L, which contains exactly the 4 indices for lane L's 4 output dimensions. No bit-base shuffling needed.

```
Lane 0 → byte 0 → indices for dims 0, 1, 2, 3
Lane 1 → byte 1 → indices for dims 4, 5, 6, 7
...
Lane 31 → byte 31 → indices for dims 124, 125, 126, 127
```

Inside the lane, unpack with simple shift-and-mask:
```metal
uint byte = packed_k[offset + kv_idx * 32 + lane_id];
uint idx_0 = (byte >> 0) & 0x03;  // for dim lane_id*4 + 0
uint idx_1 = (byte >> 2) & 0x03;  // for dim lane_id*4 + 1
uint idx_2 = (byte >> 4) & 0x03;  // for dim lane_id*4 + 2
uint idx_3 = (byte >> 6) & 0x03;  // for dim lane_id*4 + 3
```

### V rotation state

Sharpner stores V pre-rotated in their cache. Our cache **already** stores V in rotated space — the packed indices encode `v_norm @ R.T` after the normalize+rotate pipeline in `_compress_one_side`. So no cache modification needed for V. Verified in Phase 2 (below).

### K centroids vs V centroids

Sharpner uses **one shared codebook** for K and V (both at 2-bit). For a single (d, bits) pair, `get_codebook` is deterministic, so at 2-bit K and 2-bit V we have identical centroids. We'll pass both to the kernel but they'll be the same tensor.

## Scope of the minimum viable kernel (Phase 3)

Match sharpner's narrow gate to de-risk implementation:

- **B = 1** (single batch)
- **T_q = 1** (decode only)
- **D = 128** (Qwen3, Llama, Mistral — the main target models)
- **2-bit K and 2-bit V** (same 4-entry codebook for both)
- **No sink, no residual** — all tokens are compressed
- **GQA supported** — kv_head derived from `head / (H_q / H_kv)`

Everything else falls back to the v0.8.0 decomposed path (which at least preserves correctness even if slower).

## Correctness test (Phase 3, written before the kernel)

```python
def test_full_fused_attention_matches_reference():
    """Build random Q and packed K/V. Compare fused kernel output to
    the reference path: dequant K, dequant V, standard attention."""
    # Setup
    D, H_q, H_kv = 128, 32, 8
    T_kv = 256
    Q = random (H_q, D)
    K_dense = random (H_kv, T_kv, D); quantize to 2-bit packed
    V_dense = random (H_kv, T_kv, D); quantize to 2-bit packed
    
    # Reference
    K_dq = dequant(packed_K, norms_K, centroids, rotation, bits=2)  # (H_kv, T_kv, D)
    V_dq = dequant(packed_V, norms_V, centroids, rotation, bits=2)  # (H_kv, T_kv, D)
    K_gqa = repeat(K_dq, H_q/H_kv, axis=0)  # broadcast to H_q
    V_gqa = repeat(V_dq, H_q/H_kv, axis=0)
    scores = (Q @ K_gqa.transpose(...)) / sqrt(D)   # (H_q, T_kv)
    weights = softmax(scores, axis=-1)
    reference = weights @ V_gqa                     # (H_q, D)
    
    # Fused kernel
    Q_rot = pre_rotate_query(Q / sqrt(D), rotation)
    output_rot = fused_attention_2bit_kernel(
        Q_rot, packed_K, packed_V, norms_K, norms_V,
        centroids, H_q, H_kv, D, T_kv
    )
    fused = inverse_rotate(output_rot, rotation)
    
    assert allclose(reference, fused, atol=1e-3)
```

## Phase sequence

1. **Phase 2 (pack V access)** — ✅ DONE on branch. `get_fused_state()` now returns `packed_values`, `value_norms`, `value_centroids`, `value_bits`. Verified packed V round-trips to match `_decompressed_values_cache`.
2. **Phase 3 — minimum kernel.** Write the kernel, correctness test, iterate until it passes.
3. **Phase 4 — benchmark decision gate.** A/B against `mx.fast.sdpa` on dense KV at T_kv = 256, 1K, 2K, 4K. If we beat it at any non-trivial context, proceed. If not, document and stop.
4. **Phase 5 — integration** (only if Phase 4 gate passes). Wire into `fused_sdpa.py` replacing the v0.8.0 decomposed path.
5. **Phase 6 — results doc** (always). Honest report, positive or negative.

## What's deferred

- K bits > 2 (4-bit K is our default default — will need a separate kernel variant if 2-bit K doesn't meet quality)
- T_q > 1 (prefill)
- B > 1 (batching)
- Sink + residual inside the kernel (can be handled by the two-kernel merge approach: standard SDPA on small FP16 regions, fused on packed region, merge softmax states)
- D ≠ 128 (Gemma3 uses D=256; Phi-3.5 uses D=96)
- simd_shuffle optimization for the combine step
- Q_head > K_head × 4 (some GQA ratios exceed the 4-per-KV assumption we inherit)

## Expected outcome (back-of-envelope)

Qwen3-8B decode at 2K context:
- Measured baseline (mx.fast.sdpa on dense): ~33 tok/s = 30 ms/step
- Theoretical memory-bandwidth limit (400 GB/s, 288 MB KV cache traffic per step): 0.72 ms/step
- Current overhead: 40× the mem-bound limit — compute/dispatch dominated

Packed KV reduces memory traffic ~5× (K: 128 bits/vec vs 2048 bits/vec; V: 64 bits/vec vs 2048 bits/vec). If the kernel is well-optimized and dispatch overhead is near-zero (single kernel instead of 6-7), realistic upside:
- 256 ctx: **loss or wash** (dispatch dominates)
- 1K ctx: **wash** (memory bandwidth and compute roughly equal)
- 2K ctx: **1.1-1.3× win** (memory bandwidth starts mattering)
- 4K ctx: **1.5-2× win** (memory bandwidth dominant, packed KV pays off)

**Honest caveat**: if `mx.fast.sdpa` is already near memory-bound at 4K, we can't beat it without packing (which we do). If it's compute-bound at 4K, the packed KV's extra compute cost (centroid indirection, per-token norm multiply) might offset the memory savings.

We'll find out.
