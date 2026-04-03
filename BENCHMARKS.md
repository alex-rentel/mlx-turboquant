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

Quality is identical between Python (v0.2.0) and Metal (v0.3.0) paths. The Metal kernels are bit-accurate (max diff < 2e-6).

## Memory (2K context, K4/V2, residual_window=128)

| Model | Baseline (FP16) | TurboQuant | Ratio |
|-------|-----------------|------------|-------|
| Qwen3-8B-4bit | 302.0 MB | 100.5 MB | 3.0x |
| Gemma3-4B-it-4bit | 285.2 MB | 101.5 MB | 2.8x |

## Speed (K4/V2, decode 50 tokens)

### v0.3.0 (Fused Metal kernels)

| Model | Context | Baseline | TurboQuant | Overhead |
|-------|---------|----------|------------|----------|
| Qwen3-1.7B | 512 | 35.5 tok/s | 21.6 tok/s | 39% |
| Qwen3-8B | 512 | 19.1 tok/s | 13.8 tok/s | 28% |
| Qwen3-8B | 2K | 35.2 tok/s | 23.6 tok/s | 33% |

### Comparison: v0.2.0 (Python) vs v0.3.0 (Metal)

| Model | Context | v0.2.0 | v0.3.0 | Improvement |
|-------|---------|--------|--------|-------------|
| Qwen3-1.7B | 512 | 45% overhead | 39% overhead | -6pt |
| Qwen3-8B | 512 | ~45% overhead | 28% overhead | -17pt |
| Qwen3-8B | 2K | 57% overhead | 33% overhead | -24pt |

The fused Metal dequantize kernel eliminates 3 intermediate array allocations per call (unpack, centroid lookup, inverse rotation), reducing overhead by up to 24 percentage points.

## Known Limitations

1. **Remaining overhead (28-39%):** Dominated by `mx.concatenate` on every decode step (assembling decompressed + FP16 window). A fused attention-from-compressed kernel would eliminate this.

2. **Gemma3 sensitivity at long context:** Gemma3 (1-4 KV heads, head_dim=256) is more sensitive to compression error compounding than Qwen3 (8 KV heads, head_dim=128). For Gemma3, use larger residual window or K4/V4.

3. **Outlier layers:** Auto-detected and kept in FP16.

## Pure Quantizer Quality

| dim | bits | Mean cos_sim | Median cos_sim |
|-----|------|-------------|----------------|
| 128 | 4 | 0.9954 | 0.9956 |
| 128 | 3 | 0.9831 | 0.9837 |
| 128 | 2 | 0.9406 | 0.9413 |
| 256 | 4 | 0.9953 | 0.9955 |
| 256 | 3 | 0.9828 | 0.9832 |
| 256 | 2 | 0.9401 | 0.9404 |
