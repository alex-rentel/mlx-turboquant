# mlx-turboquant

**Near-optimal KV cache compression for Apple Silicon. Drop-in for any
[mlx-lm](https://github.com/ml-explore/mlx-lm) model — 3-4× memory
savings with no training, no calibration data, and no architecture
changes.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/mlx-0.31+-orange.svg)](https://github.com/ml-explore/mlx)

Faithful implementation of [TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh,
Daliri, Hadian, Mirrokni — Google Research, ICLR 2026).

---

## Install

```bash
pip install mlx-turboquant
```

Or from source:

```bash
git clone https://github.com/alex-rentel/mlx-turboquant.git
cd mlx-turboquant
pip install -e .
```

Requires Python 3.10+, macOS 13+, Apple Silicon (M1/M2/M3/M4), and
`mlx >= 0.31`.

## Quick start

```python
from mlx_lm import load, generate
from mlx_turboquant import apply_turboquant

model, tokenizer = load("mlx-community/Qwen3-8B-4bit")

# Enable 4-bit keys, 2-bit values, pin the first 128 tokens in FP16.
apply_turboquant(model, key_bits=4, value_bits=2, fp16_sink_size=128)

# Everything else is mlx-lm as usual.
response = generate(model, tokenizer, prompt="Explain quantum entanglement.",
                    max_tokens=512)
print(response)
```

That's it. The KV cache is now ~3.6× smaller. Older tokens compress to
3-4 bits automatically. The first 128 tokens (system prompt) stay in FP16
permanently. The last 128 tokens (residual window) stay in FP16 until
they age out.

## Why

Every token an LLM generates stores key and value vectors for every
layer. At FP16, an 8B model's KV cache eats ~576 MB at 4K context and
scales linearly — 16K context means 2.3 GB just for the cache, on top of
the model weights. On a 64GB Mac running a 4-bit 8B model (~4.5 GB
weights), the KV cache becomes the bottleneck for long conversations,
multi-turn agents, and RAG pipelines.

`mlx-turboquant` compresses that cache to 2-4 bits per dimension with
near-zero quality loss on most models, using the TurboQuant algorithm
(rotation → Lloyd-Max scalar quantization → inverse rotation). Per-vector
reconstruction error is within 2.7× of the Shannon information-theoretic
lower bound.

## Which config should I use?

| Use case | Config | Compression | Quality (cos sim) |
|---|---|---|---|
| **Default / balanced** | `key_bits=4, value_bits=2, fp16_sink_size=128` | 3.6× | ≥ 0.98 on most models |
| **Tool calling / chat** | same as default | 3.6× | Sink preserves system prompt tokens |
| **Memory constrained** | `key_bits=3, value_bits=2, fp16_sink_size=128` | 4.0× | Model-dependent (see below) |
| **Conservative** | `key_bits=4, value_bits=4, fp16_sink_size=128` | 2.5-3× | Near-lossless |

Full argument list:

```python
apply_turboquant(
    model,
    key_bits=4,                # 2, 3, 3.5, or 4 bits for keys
    value_bits=2,              # 2, 3, 3.5, or 4 bits for values
    residual_window=128,       # recent tokens stay in FP16 (sliding)
    fp16_sink_size=128,        # first N tokens stay FP16 permanently
    auto_detect_outliers=True, # skip layers with extreme key norms
    skip_layers=None,          # manually pin specific layers in FP16
    chunk_size=0,              # 0 = batch compression, >0 = fixed chunks
    qjl_correction=False,      # experimental: 1-bit QJL residual correction
)
```

## Validated models

Seven primary models, benchmarked end-to-end on M1 Max 64GB. All numbers
use `K4/V2 + fp16_sink_size=128` (the recommended default). Cos sim is
measured against the FP16 baseline at 500 tokens of context.

| Model | Cos sim vs FP16 | Decode @ 2k (tok/s) | KV @ 4k (MB) |
|---|---|---|---|
| Qwen3-8B | **0.981** | 33.0 | 192 |
| DeepSeek-R1-Qwen3-8B | **0.991** | 32.0 | 192 |
| Llama-3.1-8B | **0.997** | 35.2 | 162 |
| Mistral-7B v0.3 | **0.998** | 36.0 | 162 |
| Qwen2.5-7B | **0.947** | 38.7 | 109 |
| Phi-3.5-mini | **0.997** | 29.9 | 492 |
| Qwen3.5-9B *(hybrid attn)* | 0.790 | 36.2 | 87 |

**Also works on:** Qwen3-4B, Qwen3-1.7B, Llama-3.2-3B, Gemma-3-4B,
Gemma-3-1B. Full tier-2 numbers in [`results/post_patch/`](results/post_patch/).

The sink128 default is there for a reason — on three of the seven primary
models (Qwen3-8B, Qwen2.5-7B, Phi-3.5) it recovers ~0.05-0.15 cos sim
compared to running without sink. The system-prompt tokens turn out to
matter a lot.

### Needle-in-a-haystack (retrieval accuracy)

At 1K-8K context, needle placed at 10%/50%/90% depth (12 tests per config):

| Model | FP16 baseline | K4/V2 + sink128 |
|---|---|---|
| Qwen3-8B | 12/12 | **12/12** |
| Llama-3.1-8B | 12/12 | **12/12** |
| Mistral-7B | 8/12 | 7/12 |

Qwen3-8B and Llama-3.1-8B pass perfectly. Mistral-7B fails 4/12 on the
FP16 baseline itself — TurboQuant tracks but does not exceed the model's
inherent retrieval limit.

## Troubleshooting

**Quality degradation at K4/V2 on a specific model.** Add
`fp16_sink_size=128` to your call. This single change recovers ~0.1 cos
sim on Qwen2.5-7B and ~0.1 on Qwen3-8B. If quality is still bad, check
whether `auto_detect_outliers` is disabled — turning it off drops
Qwen2.5-7B from 0.80 → 0.46 cos sim.

**"Detected N linear-attention layers in model" warning.** You're on a
hybrid architecture (e.g. Qwen3.5). Only the self-attention layers will
be compressed. This is expected — the linear-attention layers don't use
standard KV caches and are skipped automatically.

**Decode is slower than FP16 baseline.** Short contexts (≤256 tokens)
see ~5-15% overhead from the decompression path. At 2K context this
grows to 20-35% depending on model. This is inherent to the current
architecture — see [Limitations](#limitations) below.

**Memory savings smaller than advertised.** Check whether
`auto_detect_outliers` has skipped layers on your model (look for
"outlier layers detected" in stderr). On Qwen3-family models, 1-3 layers
are typically skipped and stay in FP16.

**NaN or garbage outputs.** Usually means the model has an unusually
small `head_dim` (not a power of 2) or an unsupported attention layout.
Open an issue with the model name and we'll take a look.

## Limitations

1. **Decode overhead at long context (22-36% at 2K).** Decompression cost
   scales with cache size. Two attempts to eliminate this via fused Metal
   kernels both failed to beat `mx.fast.scaled_dot_product_attention`
   end-to-end — both are preserved as documented negative results on
   branches. See [docs/INTERNALS.md](docs/INTERNALS.md) for the full
   post-mortem and the structural reason.
2. **Error compounding.** Per-vector reconstruction error compounds
   through layers. Models with fewer KV heads (e.g. Gemma-3-1B with 1
   head) are more fragile to compression.
3. **Hybrid-attention partial coverage.** Architectures that mix
   self-attention and linear attention (Qwen3.5) only get TurboQuant on
   the self-attention layers; memory savings are proportionally smaller
   and quality is bounded by the uncompressed linear-attn path.
4. **Quality depends on outlier detection.** Disabling
   `auto_detect_outliers` drops Qwen2.5-7B from 0.80 → 0.46 cos sim at
   K4/V2. Leave it on.

## CLI

```bash
# Generate with compression enabled
mlx-turboquant generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Explain quantum computing" \
  --key-bits 4 --value-bits 2

# Run benchmarks
mlx-turboquant benchmark \
  --model mlx-community/Qwen3-8B-4bit \
  --benchmarks quality memory speed
```

## Tests

```bash
# Fast unit tests (no downloads, ~10s)
python -m pytest tests/ -q --ignore=tests/test_integration.py

# Integration test (downloads a ~1GB model)
python -m pytest tests/test_integration.py -v -m slow

# Full benchmark sweep (~40 min on M1 Max 64GB, 7 models × 5 configs)
python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 1
```

## Versioning & stability

`mlx-turboquant` follows [semantic versioning](https://semver.org/) from
v1.0.0 forward. The public API is:

- `apply_turboquant(model, **kwargs)` — the primary entry point
- `enable_turboquant(model, bits, **kwargs)` — symmetric convenience wrapper
- `TurboQuantKVCache` — the cache class itself (drop-in for mlx-lm's `KVCache`)
- `TurboQuantMSE` / `TurboQuantProd` — low-level quantizer classes

Everything else (`mlx_turboquant.kernels`, `.rotation`, `.codebook`,
`.packing`, `.qjl`) is internal and may change between minor versions
without warning. In particular, the `fused_qk_scores_*` kernels and
`pre_rotate_query` utility are **research-only primitives** that are NOT
wired into the decode path. See [docs/INTERNALS.md](docs/INTERNALS.md)
for why.

Breaking changes to the public API will bump MAJOR. Deprecations will
emit a `DeprecationWarning` for at least one minor version before removal.

## How it works (brief)

TurboQuant has three steps:

1. **Random rotation.** Multiply each KV vector by a fixed random
   orthogonal matrix. This transforms the coordinates into approximately
   independent, Beta-distributed random variables.
2. **Optimal scalar quantization.** Each rotated coordinate is quantized
   using a precomputed Lloyd-Max codebook optimized for the Beta
   distribution. Lloyd-Max minimizes MSE for a given number of bits —
   it's the information-theoretically optimal scalar quantizer. Only the
   quantization indices and the vector's L2 norm are stored.
3. **Reconstruction.** Look up centroids from the codebook, apply the
   inverse rotation, rescale by the stored norm.

The per-vector MSE is within 2.7× of the Shannon information-theoretic
lower bound — you mathematically cannot do much better without changing
the bit budget.

`mlx-turboquant` adds: fused Metal kernels for quantize/dequantize,
asymmetric K/V bit allocation (keys are more sensitive than values),
attention sink (pin system prompt in FP16), outlier layer detection
(skip layers with extreme key norms), hybrid attention support, and
3.5-bit fractional quantization. See [docs/INTERNALS.md](docs/INTERNALS.md)
for kernel details, benchmark history, and the story of the fused
attention attempts.

## Part of the Eden ecosystem

- [eden](https://github.com/alex-rentel/eden) — Local AI agent framework
- [eden-flywheel](https://github.com/alex-rentel/eden-flywheel) — Claude Code sessions → training data → fine-tune → deploy
- [eden-fleet](https://github.com/alex-rentel/eden-fleet) — Distributed AI across a Mac homelab via SSH
- [eden-models](https://github.com/alex-rentel/eden-models) — Training pipeline for tool-calling LLMs on HPC
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Train ChatGPT-class models on Mac
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

MIT — crediting original authors (Zandieh, Daliri, Hadian, Mirrokni).
See [LICENSE](LICENSE).
