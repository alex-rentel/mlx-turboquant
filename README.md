# mlx-turboquant

**Near-optimal KV cache quantization for Apple Silicon. Up to 4.4x memory savings.**

MLX-native implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026). Compress KV cache to 2-4 bits during inference -- run longer contexts with less RAM on your Mac.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## How it works

```
Input KV vector (FP16, 128 dims)
    |
    v
1. Random rotation (QR decomposition)
   -> Spreads energy evenly across dimensions
   -> Coordinates follow Beta distribution
    |
    v
2. Optimal scalar quantization (Lloyd-Max)
   -> Each coordinate quantized to 2-4 bits
   -> Codebook precomputed from Beta distribution
    |
    v
3. Bit-packed storage + float32 norm
   -> 4-bit: 66 bytes vs 256 bytes FP16 (3.9x)
   -> 2-bit: 36 bytes vs 256 bytes FP16 (7.1x)
```

**Data-oblivious:** No training data, calibration, or fine-tuning needed. Works on any model instantly.

## Quick Start

```bash
pip install mlx-turboquant
```

### Library API

```python
from mlx_turboquant import apply_turboquant
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Enable TurboQuant: 4-bit keys, 2-bit values
apply_turboquant(model, key_bits=4, value_bits=2)

# Generate with compressed KV cache -- same API, less memory
import mlx.core as mx
cache = model.make_cache()
inputs = mx.array(tokenizer.encode("Hello!"))[None]
logits = model(inputs, cache=cache)
```

### CLI

```bash
# Generate text
mlx-turboquant generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --key-bits 4 --value-bits 2 \
  --prompt "Explain quantum computing"

# Run benchmarks
mlx-turboquant benchmark \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --benchmarks quality memory speed
```

## Benchmarks

All benchmarks on M1 Max 64GB with `Qwen2.5-7B-Instruct-4bit`.

### Memory Compression

| Context | FP16 | TQ K4/V2 | TQ K3/V2 | Best Ratio |
|---------|------|----------|----------|------------|
| 512 | 29.4 MB | 19.2 MB | 18.5 MB | 1.6x |
| 1024 | 58.7 MB | 25.1 MB | 23.5 MB | 2.5x |
| 2048 | 117.4 MB | 37.0 MB | 33.6 MB | 3.5x |
| 4096 | 234.9 MB | 60.9 MB | 53.8 MB | 4.4x |

### Quality (vs FP16 baseline)

| Config | Cosine Sim | Top-10 Overlap |
|--------|-----------|----------------|
| K4/V4 | 0.951 | 70% |
| K4/V2 | 0.941 | 70% |
| K3/V3 | 0.876 | 60% |
| K2/V2 | 0.768 | 70% |

### Speed

| Config | Decode (tok/s) | Overhead |
|--------|----------------|----------|
| FP16 baseline | 57.2 | -- |
| TQ K4/V2 | 30.9 | +46% |

The overhead comes from software dequantization (rotation matrix multiply). A fused Metal kernel would reduce this significantly.

## Features

- **Asymmetric K/V bits:** Keys need more precision than values (K4/V2 is the sweet spot)
- **Residual window:** Recent tokens stay in FP16 for maximum quality
- **Outlier layer detection:** Auto-detects layers with extreme key norms (Qwen layer 0) and keeps them in FP16
- **Model agnostic:** Works with Llama, Qwen, Mistral, Gemma via mlx-lm
- **Both algorithms:** TurboQuant_mse (default, recommended) and TurboQuant_prod (QJL, opt-in)

## Architecture

```
mlx_turboquant/
  __init__.py         # apply_turboquant(), enable_turboquant()
  codebook.py         # Lloyd-Max optimal scalar quantizer
  rotation.py         # Random orthogonal rotation (QR decomposition)
  quantizer.py        # TurboQuantMSE, TurboQuantProd
  qjl.py              # Quantized Johnson-Lindenstrauss transform
  packing.py          # 1/2/3/4-bit packing into uint8
  cache.py            # TurboQuantKVCache (drop-in for mlx-lm)
  patch.py            # Model patching + outlier layer detection
  cli.py              # CLI entry points
  codebooks/          # Precomputed Lloyd-Max codebooks (.npz)
benchmarks/
  bench_quality.py    # Cosine similarity, top-K accuracy
  bench_memory.py     # Cache memory measurement
  bench_speed.py      # Tokens/second comparison
  needle_haystack.py  # Needle-in-a-haystack retrieval test
tests/                # 137 tests
```

## Credits

**Original paper:** Zandieh, Daliri, Hadian, Mirrokni -- *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"* (Google Research / Google DeepMind / NYU, ICLR 2026). [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

**Reference implementations:**
- [vivekvar-dl/turboquant](https://github.com/vivekvar-dl/turboquant) -- PyTorch reference
- [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) -- MLX port
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) -- V3 with community findings
- [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) -- Implementation insights

**MLX framework:** [Apple ML Explore](https://github.com/ml-explore/mlx)

## License

MIT
