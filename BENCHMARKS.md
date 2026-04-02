# Benchmarks

All benchmarks run on M1 Max 64GB with `mlx-community/Qwen2.5-7B-Instruct-4bit`.

## Quality (Cosine Similarity vs FP16 Baseline)

182-token prompt, outlier layers (0, 1, 3, 27) kept in FP16.

| Config | Cosine Sim | Top-1 Match | Top-10 Overlap |
|--------|-----------|-------------|----------------|
| FP16 baseline | 1.000 | Yes | 100% |
| K4/V4 | 0.951 | No | 70% |
| K4/V2 | 0.941 | No | 70% |
| K3/V3 | 0.876 | No | 60% |
| K3/V2 | 0.887 | No | 70% |
| K2/V2 | 0.768 | No | 70% |

**Notes:**
- Cosine similarity measures logit vector similarity at the last token position
- Outlier layer auto-detection is critical for Qwen models (layers with key norms >3x median are kept in FP16)
- 4-bit provides reasonable quality; 2-bit is lossy but usable for long-context scenarios

## Memory Compression

Qwen2.5-7B: 28 layers, 4 KV heads, head_dim=128. Residual window: 128 tokens.

| Context | FP16 | TQ K4/V4 | TQ K4/V2 | TQ K3/V2 | Best Ratio |
|---------|------|----------|----------|----------|------------|
| 512 | 29.4 MB | 20.5 MB | 19.2 MB | 18.5 MB | 1.6x |
| 1024 | 58.7 MB | 28.3 MB | 25.1 MB | 23.5 MB | 2.5x |
| 2048 | 117.4 MB | 43.9 MB | 37.0 MB | 33.6 MB | 3.5x |
| 4096 | 234.9 MB | 75.1 MB | 60.9 MB | 53.8 MB | 4.4x |

**Notes:**
- Compression ratio improves with context length (fixed overhead from residual window + outlier layers)
- At 4K+ context, K3/V2 achieves 4.4x compression
- Theoretical maximum at infinite context: ~5x for K4/V2, ~8x for K3/V2

## Speed (Tokens/Second)

256-token prefill, 50-token decode.

| Config | Prefill (tok/s) | Decode (tok/s) | Decode Overhead |
|--------|----------------|----------------|-----------------|
| FP16 baseline | 404 | 57.2 | -- |
| TQ K4/V4 w128 | 312 | 30.3 | +47% |
| TQ K4/V2 w128 | 338 | 30.9 | +46% |
| TQ K4/V2 w32 | 332 | 30.6 | +47% |

**Notes:**
- Decode overhead is ~47% due to dequantization (rotation matrix multiply + centroid lookup)
- This is a CPU-side Python implementation; a fused Metal kernel would reduce overhead significantly
- The value proposition is memory savings (longer contexts), not speed
- At very long contexts where FP16 would OOM, TurboQuant enables inference that wouldn't otherwise be possible

## Key Findings

1. **Outlier layer detection is essential** for Qwen models — without it, cosine sim drops from 0.95 to 0.63
2. **Asymmetric K/V allocation works**: K4/V2 is nearly as good as K4/V4 with better compression
3. **Residual window size** has minimal impact on quality for short prompts (tested w32 vs w128)
4. **Compression ratio scales** with context length — the main use case is fitting longer contexts in memory
