# mlx-turboquant

Near-optimal KV cache quantization for Apple Silicon. Faithful implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (arXiv:2504.19874, ICLR 2026).

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start

```bash
pip install mlx-turboquant
```

```python
from mlx_lm import load
from mlx_turboquant import apply_turboquant

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
apply_turboquant(model, key_bits=4, value_bits=2)
cache = model.make_cache()
# Use cache normally with model(inputs, cache=cache)
```

## Benchmarks (M1 Max 64GB, April 2026)

### Quality (cosine similarity vs FP16 baseline, ~500-token context)

| Model | head_dim | K4/V4 | K4/V2 | K3/V2 |
|-------|----------|-------|-------|-------|
| Qwen3-1.7B | 128 | 0.9914 | 0.9853 | 0.9687 |
| Qwen3-8B | 128 | **0.9994** | 0.9976 | 0.9872 |
| Gemma3-1B | 256 | 0.9953 | 0.9802 | 0.9619 |
| Gemma3-4B | 256 | 0.9925 | 0.9848 | 0.9753 |

### Memory (2K context, K4/V2)

| Model | Baseline | TurboQuant | Compression |
|-------|----------|------------|-------------|
| Qwen3-8B | 302 MB | 101 MB | **3.0x** |
| Gemma3-4B | 285 MB | 102 MB | **2.8x** |

### Speed (2K context, decode)

| Model | Baseline | TurboQuant | Overhead |
|-------|----------|------------|----------|
| Qwen3-8B | 38.1 tok/s | 16.5 tok/s | 57% |
| Gemma3-4B | 58.1 tok/s | 18.1 tok/s | 69% |

Speed overhead is from software dequantization (dense rotation matrix multiply per token). A fused Metal kernel would reduce this.

## How It Works

TurboQuant compresses each KV vector independently in three steps. First, a fixed random orthogonal matrix (QR decomposition) rotates the vector so that its coordinates become approximately independent and identically distributed. This rotation is the theoretical foundation: it transforms the vector quantization problem into a simpler per-coordinate scalar quantization problem.

Second, each rotated coordinate is quantized using a precomputed Lloyd-Max codebook optimized for the post-rotation distribution (Beta((d-1)/2, (d-1)/2), which converges to Gaussian in high dimensions). The codebook is computed once per (dimension, bit-width) pair and reused for all tokens. Only the quantization indices and the original vector's L2 norm are stored.

To decode, the indices are mapped back to centroids, the inverse rotation is applied, and the result is rescaled by the stored norm. The per-vector MSE is within 2.7x of the Shannon information-theoretic lower bound. At 4-bit, the pure quantizer achieves 0.9955 median cosine similarity per vector.

## Configuration

```python
apply_turboquant(
    model,
    key_bits=4,          # 2, 3, or 4 bits for keys
    value_bits=2,        # 2, 3, or 4 bits for values
    residual_window=128, # recent tokens stay in FP16
    auto_detect_outliers=True,  # skip layers with extreme key norms
    skip_layers=[0, 27], # manually specify layers to keep in FP16
)
```

- **Asymmetric K/V:** Keys need more precision than values. K4/V2 is a good default.
- **Residual window:** The last N tokens stay uncompressed. Larger = better quality, less compression.
- **Outlier detection:** Models like Qwen3 have layers with extreme key norms (>3x median). These are auto-detected and kept in FP16.
- **Few-KV-head safety:** Models with 1-2 KV heads (Gemma3-1B) auto-upgrade to K4/V3 minimum.

## Supported Models

Any model loaded via `mlx_lm.load()`:

- **Qwen3** (1.7B, 8B) -- head_dim=128, tested extensively
- **Gemma3** (1B, 4B) -- head_dim=256, works well up to ~1K context
- **Llama, Mistral, Phi** -- standard head_dim=128, expected to work (GQA supported)

## Limitations

1. **Decode overhead (45-69%):** No custom Metal kernels yet. The dense rotation matrix multiply dominates decode time. Memory savings scale with context but speed does not improve.
2. **Error compounding:** Small per-vector errors (~0.5%) compound through transformer layers. Larger models and more KV heads are more robust. Gemma3 with 1 KV head degrades at >1K context.
3. **Memory savings scale with context:** At short contexts (<residual_window), no compression occurs. The benefit grows with context length.

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Ali and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT -- crediting original authors (Zandieh, Daliri, Hadian, Mirrokni).
