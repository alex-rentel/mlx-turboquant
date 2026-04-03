# mlx-turboquant

Near-optimal KV cache quantization for Apple Silicon. Faithful implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026).

Compresses the KV cache during LLM inference by 3x with <0.3% quality loss. No training, no calibration data, fully data-oblivious. Drop-in for any [mlx-lm](https://github.com/ml-explore/mlx-lm) model.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Quick Start

```bash
git clone https://github.com/alex-rentel/mlx-turboquant.git
cd mlx-turboquant
pip install -e .
```

```python
from mlx_lm import load
from mlx_turboquant import apply_turboquant

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")
apply_turboquant(model, key_bits=4, value_bits=2)

cache = model.make_cache()
# Use cache normally — older tokens are compressed automatically
logits = model(inputs, cache=cache)
```

## Benchmarks (M1 Max 64GB, April 2026)

Tested on 4 real models across 2 architecture families (Qwen3: head_dim=128, Gemma3: head_dim=256).

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

### Speed (decode, K4/V2)

| Model | Context | Baseline | TurboQuant | Overhead |
|-------|---------|----------|------------|----------|
| Qwen3-1.7B | 512 | 35.5 tok/s | 29.8 tok/s | 16% |
| Qwen3-8B | 2K | 41.3 tok/s | 36.7 tok/s | **11%** |

Overhead reduced from 57% (v0.2.0, Python) → 33% (v0.3.0, Metal kernels) → **11%** (v0.5.0, batch compression + pre-allocated window).

### Needle-in-a-Haystack

Qwen3-8B K4/V2: **12/12 perfect** retrieval at 1K, 2K, 4K, 8K context — matches FP16 baseline.

## How It Works

TurboQuant implements both algorithms from the paper:

**Algorithm 1 — TurboQuant_mse (default).** A fixed random orthogonal matrix (QR decomposition) rotates each KV vector so that its coordinates become approximately independent and Beta-distributed. Each coordinate is then quantized using a precomputed Lloyd-Max codebook optimized for that distribution. Only the quantization indices and the vector's L2 norm are stored. To dequantize: look up centroids, inverse-rotate, rescale by norm. The per-vector MSE is within 2.7x of the Shannon information-theoretic lower bound.

**Algorithm 2 — TurboQuant_prod (opt-in).** Applies (b-1)-bit MSE quantization, computes the residual, then applies a 1-bit Quantized Johnson-Lindenstrauss (QJL) transform on the residual. This produces an unbiased inner product estimator. However, 6+ independent community implementations confirmed that QJL hurts quality as a drop-in KV cache replacement — it's only beneficial with fused attention kernels that consume the two-part representation directly. Algorithm 1 (MSE-only) is the default for this reason.

**Key design decisions informed by community findings:**
- Asymmetric K/V allocation — keys need more bits than values (K4/V2 recommended)
- Residual window — recent tokens stay in FP16 for quality (default: 128 tokens)
- Outlier layer detection — some models (Qwen3) have layers with extreme key norms that must stay uncompressed
- Few-KV-head safety — models with ≤2 KV heads (Gemma3-1B) auto-upgrade to K4/V3 minimum

## Configuration

```python
apply_turboquant(
    model,
    key_bits=4,                # 2, 3, 3.5, or 4 bits for keys
    value_bits=2,              # 2, 3, 3.5, or 4 bits for values
    residual_window=128,       # recent tokens stay in FP16
    auto_detect_outliers=True, # skip layers with extreme key norms
    skip_layers=[0, 27],       # manually specify FP16 layers
)
```

You can also use the low-level quantizer directly:

```python
from mlx_turboquant import TurboQuantMSE, TurboQuantProd

# Algorithm 1: MSE-optimized (recommended for KV cache)
tq = TurboQuantMSE(d=128, bits=4)
qt = tq.quantize(vectors)
vectors_hat = tq.dequantize(qt)

# Algorithm 2: Inner-product optimized (for custom attention kernels)
tq_prod = TurboQuantProd(d=128, bits=4)
qt = tq_prod.quantize(vectors)
```

## CLI

```bash
# Generate text with TurboQuant compression
mlx-turboquant generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Explain quantum computing" \
  --key-bits 4 --value-bits 2

# Run benchmarks
mlx-turboquant benchmark \
  --model mlx-community/Qwen3-8B-4bit \
  --benchmarks quality memory speed
```

## Supported Models

Any model loaded via `mlx_lm.load()`. Tested on:

| Family | Models Tested | head_dim | Notes |
|--------|--------------|----------|-------|
| Qwen3 | 1.7B, 8B | 128 | Excellent quality, has outlier layers |
| Gemma3 | 1B, 4B | 256 | Works well up to ~1K context |
| Llama 3.2 | 3B | 128 | Tested, coherent generation |
| Mistral | 7B v0.3 | 128 | Tested, excellent quality (0.998 cos_sim) |

## Project Structure

```
mlx_turboquant/
  codebook.py     # Lloyd-Max optimal scalar quantizer for Beta distributions
  rotation.py     # Random orthogonal rotation (QR) + Walsh-Hadamard
  quantizer.py    # TurboQuantMSE (Algorithm 1) + TurboQuantProd (Algorithm 2)
  qjl.py          # Quantized Johnson-Lindenstrauss transform
  packing.py      # 1/2/3/4-bit index packing into uint8
  kernels.py      # Fused Metal compute kernels (dequantize, quantize)
  cache.py        # TurboQuantKVCache — drop-in for mlx-lm's KVCache
  patch.py        # Model patching — apply_turboquant() monkey-patch
  cli.py          # Command-line interface
  codebooks/      # Precomputed Lloyd-Max codebooks (.npz)
benchmarks/       # Quality, memory, speed, needle-in-haystack benchmarks
tests/            # 137 unit tests + 1 integration test with real model
```

## Running Tests

```bash
# Unit tests (no model download, fast)
python -m pytest tests/ -q --ignore=tests/test_integration.py

# Integration test (downloads ~1GB model)
python -m pytest tests/test_integration.py -v -m slow

# Benchmarks (downloads models, takes several minutes)
python benchmarks/bench_quality.py
python benchmarks/bench_memory.py
python benchmarks/bench_speed.py
```

## Limitations

1. **Decode speed overhead (~11%):** Fused Metal kernels + batch compression + pre-allocated windows reduced overhead from 57% to 11%. The remaining overhead is MLX per-operation dispatch cost (~0.25ms per concatenate). A fused attention-from-compressed kernel would eliminate this entirely.
2. **Error compounding:** Per-vector reconstruction error (~0.5%) compounds through transformer layers. Models with more KV heads (Qwen3: 8 heads) are more robust than those with fewer (Gemma3-1B: 1 head).
3. **Context-dependent compression:** At contexts shorter than `residual_window`, no compression occurs. Memory savings grow with context length.

## Roadmap

### Completed

| Version | Feature | Result |
|---------|---------|--------|
| v0.2.0 | Real model testing, vectorized quantization | 6 model families validated |
| v0.3.0 | Fused Metal dequantize kernels | 57% → 33% overhead |
| v0.4.0 | 3.5-bit fractional, needle-in-haystack, PyPI packaging | 12/12 retrieval at 1K-8K |
| v0.5.0 | Batch compression + pre-allocated FP16 window | 33% → **11%** overhead |

### Next: Fused Attention-from-Compressed Kernel

The remaining 11% overhead is MLX dispatch cost from concatenating decompressed + FP16 tensors. The path to sub-5%:

1. **Fused attention-from-compressed kernel:** Compute Q @ K^T directly from packed indices without materializing the full dequantized K tensor. Another MLX TurboQuant implementation demonstrated 0.98x native speed with this approach.
2. **Walsh-Hadamard fast path:** WHT is O(d log d) vs O(d²) for QR. Works for Qwen3 but degrades Gemma3 by 1.6%. A dimension-adaptive strategy could use WHT where safe.

### Future

- Needle-in-a-haystack validation at 16K-32K context
- Integration with [eden-fleet](https://github.com/alex-rentel/eden-fleet) for distributed inference with compressed KV cache transfer between nodes
- Upstream contribution to mlx-lm as an optional KV cache backend

## Part of the Eden Ecosystem

mlx-turboquant is part of a local-first AI development ecosystem for Apple Silicon:

- [eden](https://github.com/alex-rentel/eden) — Local AI agent framework
- [eden-flywheel](https://github.com/alex-rentel/eden-flywheel) — Capture Claude Code sessions → training data → fine-tune → deploy
- [eden-fleet](https://github.com/alex-rentel/eden-fleet) — Distribute AI workloads across a Mac homelab via SSH
- [eden-models](https://github.com/alex-rentel/eden-models) — Training pipeline for 1-bit tool-calling LLMs on HPC
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Train ChatGPT-class models on Mac (port of Karpathy's nanochat)
- **mlx-turboquant** — KV cache compression for longer contexts on Apple Silicon

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT — crediting original authors (Zandieh, Daliri, Hadian, Mirrokni). See [LICENSE](LICENSE).
