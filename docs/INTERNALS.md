# mlx-turboquant — internals & engineering history

This document collects the engineering-facing material that used to live
in the top-level README: version history, fused-kernel micro-benchmarks,
post-mortems on attempts that didn't ship, and the deeper architecture
notes. For user-facing install/usage docs, see [README.md](../README.md).

---

## Fused Metal kernels — what's on the supported path

The supported decode path uses two fused Metal kernels, both wired into
`TurboQuantKVCache`:

- **`metal_dequantize`** — single dispatch: unpack indices → centroid lookup
  → inverse rotation → norm scaling. Replaces the per-token Python path.
- **`metal_quantize_4bit`** — single dispatch for the quantize side. Used
  when tokens roll out of the FP16 residual window into compressed storage.

These brought the decode overhead from 57% (pure Python, v0.2.0) down to
11% (batch compression + pre-allocated FP16 window, v0.5.0) to where it is
today. At short contexts (~256 tokens) the quantized path is within 2-5%
of the FP16 baseline on most models.

## Fused QK scores kernels — research-only primitives

`mlx_turboquant.kernels` ships `fused_qk_scores_{2,3,4}bit` and
`mlx_turboquant.rotation.pre_rotate_query` as research-only primitives.
They are NOT wired into `apply_turboquant` or the decode path. They exist
because two integration attempts produced clean negative results that are
worth preserving.

### Micro-benchmark (isolated): v0.7.0 fused QK vs dequant+matmul

On M1 Max 64GB, median of 30 trials:

| Shape | dequant + matmul | fused kernel | Speedup |
|---|---|---|---|
| T_kv=256, D=128 | 301 μs | 266 μs | 1.13× |
| T_kv=1024, D=128 | 327 μs | 231 μs | 1.42× |
| T_kv=4096, D=128 | 487 μs | 230 μs | **2.12×** |
| T_kv=1024, D=256 (Gemma) | 491 μs | 242 μs | **2.03×** |

Full methodology in [BENCHMARKS_v07.md](../BENCHMARKS_v07.md).

### Integration attempt #1 — decomposed SDPA (v0.8.0)

Branch: [`feat/fused-sdpa-qwen3`](https://github.com/alex-rentel/mlx-turboquant/tree/feat/fused-sdpa-qwen3)

Replaced the `mx.fast.scaled_dot_product_attention` call with fused QK
for compressed K + standard matmul for sink/residual K + separate softmax
+ V matmul.

**Result:** 0.61×–0.99× vs the standard path. Correctness was perfect
(cos_sim = 1.000000). Decomposing one hyper-optimized dispatch into 6-7
separate dispatches added more overhead than the per-op speedup saved.

Full post-mortem: [FUSED_SDPA_RESULTS.md](FUSED_SDPA_RESULTS.md).

### Integration attempt #2 — full fused attention kernel (v0.9.0)

Branch: [`feat/full-fused-attention`](https://github.com/alex-rentel/mlx-turboquant/tree/feat/full-fused-attention)

Wrote a complete single-dispatch Metal kernel (online softmax + simd_sum
+ GQA + packed K and V) modeled on sharpner's `_FUSED_ATTN_NOROT_SOURCE`.

**Result:** 0.64×–0.86× vs `mx.fast.sdpa` across T_kv = 256, 512, 1024,
2048, 4096, 8192. Correctness was perfect (5/5 tests, atol < 2e-3). The
structural reason: at realistic decode shapes both paths are
dispatch/latency/compute bound, not memory-bandwidth bound. `mx.fast.sdpa`
runs at ~4.4× the dense memory-bound limit; our fused kernel runs at ~57×
its (smaller) packed-data limit. The centroid indirection (data-dependent
gather per lane per kv-token) kills the bandwidth advantage of smaller
packed data.

Full post-mortem: [FULL_FUSED_ATTENTION_RESULTS.md](FULL_FUSED_ATTENTION_RESULTS.md).

### The structural conclusion

`mx.fast.scaled_dot_product_attention` is not a wall you can break by
writing a better kernel around it. It is already optimized to within ~4×
of the memory-bandwidth ceiling on dense data. To actually beat it you'd
need one of:

1. **An upstream MLX PR** that makes `mx.fast.sdpa` itself aware of packed
   KV layouts. The correct path, but requires a contribution to Apple's
   MLX core.
2. **Much longer contexts (32K+)** — the regime where both paths become
   memory-bandwidth-dominated. Untested at the time of v0.9.0.
3. **Indirection-free compression** (NF4, bitplane, affine) that
   eliminates the per-token centroid gather. This is a different
   algorithm, not TurboQuant.
4. **Precomputed Q·centroid table** — precompute `q · c[j]` for all
   centroids once per query, then per-token work is a table gather with
   zero multiplications. Plausible for 2-bit (only 4 centroids), but the
   dominators in our kernel are compute/latency overhead (online softmax
   rescaling, scalar norm reads, cross-simdgroup combine), not the
   multiply itself — so the upside is uncertain and this is *speculative
   future work*, not a near-term commitment.

The tripwire test at `tests/test_fused_kernel_integration_tripwire.py`
enforces that these primitives stay out of the supported decode path. If
you are wiring them in, delete the tripwire and replace it with a real
end-to-end integration test.

---

## Version history

| Version | What Changed | Key Result |
|---|---|---|
| v0.2.0 | Real model testing, vectorized quantization | 6 model families validated |
| v0.3.0 | Fused Metal dequantize kernels | 57% → 33% decode overhead |
| v0.4.0 | 3.5-bit fractional, needle-in-haystack, PyPI packaging | 12/12 retrieval at 1K-8K |
| v0.5.0 | Batch compression, pre-allocated FP16 window | 33% → **11%** decode overhead |
| v0.6.0 | Attention sink, hybrid attention, QJL correction, 12-model sweep | +0.098 cos_sim on Qwen3-8B, 12 models × 8 families validated |
| v0.7.0 | Fused QK scores Metal kernel (4/3/2-bit) + `pre_rotate_query` utility | 2.12× speedup on long-context decode in isolation. Research primitive only. |
| v0.8.0 *(branch)* | Decomposed SDPA integration — fused QK + separate softmax + V matmul | Correct but 0.61×–0.99× vs `mx.fast.sdpa`. Branch only. |
| v0.9.0 *(branch)* | Full single-dispatch fused attention kernel | Correct but 0.64×–0.86× vs `mx.fast.sdpa`. Branch only. |
| v0.8.1 | Bug-fix pass on main: `nbytes` undercount on fractional configs, aliased in-place buffer shift in `_drain_chunk`, sticky Metal-kernel fallback, NaN guard in `detect_outlier_layers` | Tier-1 sweep re-run (7 models × 5 configs, 0 errors) confirmed no regression. |
| **v1.0.0** | Public API freeze, user-facing README rewrite, CI, v1.0.0 PyPI release | Stable library surface. Fused QK kernel officially demoted to research primitive. |

## Community implementations

- [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) — V2 + V3 dual-path, best QJL analysis in the community
- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) — Claims 98% FP16 speed via fused attention kernel (unverified)
- [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) — Clean drop-in KVCache, no Metal kernels
- [helgklaizar/turboquant_mlx](https://github.com/helgklaizar/turboquant_mlx) — Attention sink + dynamic chunking, affine quantization
- [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) — Qwen 3.5 35B benchmarks, "TurboQuant-inspired"

This repo differentiates with: fused Metal kernels on the Lloyd-Max
codebook path, asymmetric K/V bits, attention sink, outlier detection,
hybrid attention support, fractional 3.5-bit, and a 12-model benchmark
sweep with needle-in-haystack validation.
