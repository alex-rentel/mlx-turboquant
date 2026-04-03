# Benchmarks

All benchmarks on Apple M1 Max 64GB, 2026-04-02.

## Quality (cosine similarity vs FP16 baseline, ~500-token context)

| Model | head_dim | K4/V4 | K4/V2 | K3.5/V2 | K3/V2 |
|-------|----------|-------|-------|---------|-------|
| Qwen3-1.7B | 128 | 0.9914 | 0.9853 | — | 0.9687 |
| Qwen3-8B | 128 | **0.9994** | 0.9976 | 0.9963 | 0.9872 |
| Gemma3-1B | 256 | 0.9953 | 0.9802 | — | 0.9619 |
| Gemma3-4B | 256 | 0.9925 | 0.9848 | — | 0.9753 |
| Llama-3.2-3B | 128 | 0.9871 | 0.9478 | 0.9342 | — |
| Mistral-7B | 128 | 0.9980 | 0.9921 | 0.9852 | — |

## Needle-in-a-Haystack (Qwen3-8B, K4/V2, w128)

| Context | 25% | 50% | 75% |
|---------|-----|-----|-----|
| 1K | FOUND | FOUND | FOUND |
| 2K | FOUND | FOUND | FOUND |
| 4K | FOUND | FOUND | FOUND |
| 8K | FOUND | FOUND | FOUND |

**12/12 — matches FP16 baseline perfectly.**

## Memory (2K context, K4/V2, residual_window=128)

| Model | Baseline (FP16) | TurboQuant | Ratio |
|-------|-----------------|------------|-------|
| Qwen3-8B | 302.0 MB | 100.5 MB | **3.0x** |
| Gemma3-4B | 285.2 MB | 101.5 MB | **2.8x** |

## Speed (K4/V2, decode 50 tokens)

| Model | Context | Baseline | TurboQuant | Overhead |
|-------|---------|----------|------------|----------|
| Qwen3-1.7B | 512 | 35.5 tok/s | 29.8 tok/s | 16% |
| Qwen3-8B | 2K | 41.3 tok/s | 36.7 tok/s | **11%** |

### Speed History

| Version | Qwen3-8B 2K Overhead | Key Change |
|---------|---------------------|------------|
| v0.2.0 | 57% | Python-only dequantization |
| v0.3.0 | 33% | Fused Metal dequantize kernel |
| v0.4.0 | 31% | 3.5-bit, 6 models, needle-in-haystack |
| **v0.5.0** | **11%** | Batch compression + pre-allocated FP16 window |

### Optimization Breakdown (v0.4.0 → v0.5.0)

| Optimization | Impact | Why It Helped |
|-------------|--------|---------------|
| Batch compression (2x window threshold) | -7pt | Eliminates ALL quantize/dequant ops during short decode runs |
| Pre-allocated FP16 window (slice write) | -13pt | Eliminates per-token array allocation for window growth |

Optimizations tested and reverted:
- Argmin distance lookup: 4.5x slower than boundary search
- Float16 rotation in Metal: no speedup (kernel promotes to float32)
- Rotation matrix sharing: already shared via cache
- Prefill optimization: already same speed as baseline

Remaining overhead is MLX per-operation dispatch cost (~0.25ms × concatenate ops per step). Requires fused attention-from-compressed kernel to eliminate.

## Pure Quantizer Quality

The mathematical core achieves excellent per-vector reconstruction (no model involvement):

| dim | bits | Mean cos_sim | Median cos_sim |
|-----|------|-------------|----------------|
| 128 | 4 | 0.9954 | 0.9956 |
| 128 | 3.5 | 0.9943 | 0.9945 |
| 128 | 3 | 0.9831 | 0.9837 |
| 128 | 2 | 0.9406 | 0.9413 |
| 256 | 4 | 0.9953 | 0.9955 |
| 256 | 3 | 0.9828 | 0.9832 |
| 256 | 2 | 0.9401 | 0.9404 |

The logit-level quality gap comes from error compounding through transformer layers, not quantizer imprecision.
