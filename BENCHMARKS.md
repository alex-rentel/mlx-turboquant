# Benchmarks

All benchmarks on Apple M1 Max 64GB, 2026-04-02.

## Quality (cosine similarity vs FP16 baseline)

~500-token prompt, residual_window=64. Compression active on ~430+ tokens.

| Model | head_dim | K4/V4 | K4/V2 | K3/V2 |
|-------|----------|-------|-------|-------|
| Qwen3-1.7B-4bit | 128 | 0.9914 | 0.9853 | 0.9687 |
| Qwen3-8B-4bit | 128 | 0.9994 | 0.9976 | 0.9872 |
| Gemma3-1B-it-4bit | 256 | 0.9953 | 0.9802 | 0.9619 |
| Gemma3-4B-it-4bit | 256 | 0.9925 | 0.9848 | 0.9753 |

**Observations:**
- Qwen3-8B at K4/V4 achieves 0.9994 cosine sim with 5/5 top-5 overlap
- Larger models (8B > 1.7B) are more robust to compression
- K4/V2 (asymmetric) provides good quality with better compression than K4/V4

## Memory (2K context, K4/V2, residual_window=128)

| Model | Baseline (FP16) | TurboQuant | Ratio |
|-------|-----------------|------------|-------|
| Qwen3-8B-4bit | 302.0 MB | 100.5 MB | 3.0x |
| Gemma3-4B-it-4bit | 285.2 MB | 101.5 MB | 2.8x |

## Speed (2K context, K4/V2, decode 50 tokens)

| Model | Baseline | TurboQuant | Overhead |
|-------|----------|------------|----------|
| Qwen3-8B-4bit | 38.1 tok/s | 16.5 tok/s | 57% |
| Gemma3-4B-it-4bit | 58.1 tok/s | 18.1 tok/s | 69% |

The overhead comes from software dequantization (rotation matrix multiply + centroid lookup per decode step). A fused Metal kernel would reduce this significantly.

## Generation Samples (2K context, diverse prompt)

**Qwen3-8B K4/V2 vs baseline:** Output identical for first 50 tokens.

**Gemma3-4B:** Coherent up to ~1K context. Degrades at 2K due to error compounding through 34 layers with only 1 KV head group. For Gemma3, use residual_window >= context_length/2 or K4/V4.

## Known Limitations

1. **Decode overhead (57-69%):** Software dequantization dominates. This is a Python/MLX implementation without custom Metal kernels. The value proposition is memory savings for long context, not speed.

2. **Gemma3 sensitivity at long context:** Gemma3's architecture (1-4 KV heads, head_dim=256) is more sensitive to KV compression error compounding than Qwen3 (8 KV heads, head_dim=128). With fewer KV heads, each head carries more information per position.

3. **Outlier layers:** Auto-detected and kept in FP16 (Qwen3-1.7B: 4 layers, Qwen3-8B: varies, Gemma3: 1 layer).

## Pure Quantizer Quality

The mathematical core achieves excellent per-vector reconstruction (no model involvement):

| dim | bits | Mean cos_sim | Median cos_sim |
|-----|------|-------------|----------------|
| 128 | 4 | 0.9954 | 0.9956 |
| 128 | 3 | 0.9831 | 0.9837 |
| 128 | 2 | 0.9406 | 0.9413 |
| 256 | 4 | 0.9953 | 0.9955 |
| 256 | 3 | 0.9828 | 0.9832 |
| 256 | 2 | 0.9401 | 0.9404 |

The logit-level quality gap comes from error compounding through transformer layers, not quantizer imprecision.
