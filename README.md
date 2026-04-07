# mlx-turboquant

Near-optimal KV cache compression for Apple Silicon. Drop-in for any [mlx-lm](https://github.com/ml-explore/mlx-lm) model — 3-4x memory savings with no training, no calibration data, and no architecture changes.

Faithful implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google Research, ICLR 2026).

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
apply_turboquant(model, key_bits=4, value_bits=2, fp16_sink_size=128)

cache = model.make_cache()
logits = model(inputs, cache=cache)
# Older tokens are compressed to 3-4 bits automatically.
# The first 128 tokens (system prompt) stay in FP16 permanently.
# The last 128 tokens (residual window) stay in FP16 until they age out.
```

## The Problem

Every token an LLM generates stores key and value vectors for every layer. At FP16, an 8B model's KV cache eats ~576 MB at 4K context and scales linearly — 16K context means 2.3 GB just for the cache, on top of the model weights. On a 64GB Mac running a 4-bit 8B model (~4.5 GB weights), the KV cache becomes the bottleneck for long conversations, multi-turn agents, and RAG pipelines.

## What TurboQuant Does (The Paper)

Google Research published TurboQuant at ICLR 2026. The core insight: KV cache vectors have structure that naive quantization ignores. TurboQuant exploits this in three steps:

**Step 1 — Random rotation.** Multiply each KV vector by a fixed random orthogonal matrix (via QR decomposition). This transforms the coordinates into approximately independent, Beta-distributed random variables. The rotation is computed once at initialization and reused for every token.

**Step 2 — Optimal scalar quantization.** Each rotated coordinate is quantized using a precomputed Lloyd-Max codebook optimized for the Beta distribution. Lloyd-Max minimizes mean squared error for a given number of bits — it's the information-theoretically optimal scalar quantizer. Only the quantization indices and the vector's L2 norm are stored.

**Step 3 — Reconstruction.** Look up centroids from the codebook, apply the inverse rotation, rescale by the stored norm. The per-vector MSE is within 2.7x of the Shannon information-theoretic lower bound — you mathematically cannot do much better without changing the bit budget.

The paper also describes a second algorithm (TurboQuant_prod) that adds a 1-bit QJL residual correction for unbiased inner product estimation. In practice, 6+ independent implementations confirmed this hurts quality as a drop-in cache replacement — it's only useful with custom fused attention kernels. We implement it as an opt-in experimental feature.

## How It Works on MLX

This library adapts TurboQuant for Apple Silicon's unified memory architecture via MLX:

**Fused Metal kernels.** The dequantize path (unpack indices → centroid lookup → inverse rotation → norm scaling) runs as a single Metal compute dispatch. No intermediate tensor allocations, no Python overhead per-token. This brought overhead from 57% (pure Python, v0.2.0) down to 11% (v0.5.0).

**Asymmetric K/V bit allocation.** Community research confirmed that keys are far more sensitive to compression than values. Our default (K4/V2 — 4-bit keys, 2-bit values) gives ~3.6x compression with better quality than symmetric K3/V3, because attention scores depend on key precision more than value precision.

**Attention sink (v0.6.0).** The first N tokens of a conversation — typically the system prompt containing tool schemas or instructions — stay in FP16 permanently, separate from the sliding residual window. On models where the system prompt defines behavior (Qwen family, Phi-3.5), this single feature improved cosine similarity from 0.88 → 0.98 on Qwen3-8B and from 0.80 → 0.95 on Qwen2.5-7B.

**Outlier layer detection.** Some models (Qwen3, Qwen2.5) have layers where key norms spike 10-16x above median. Compressing these layers destroys quality. `auto_detect_outliers=True` (the default) identifies and skips them automatically.

**Hybrid attention support (v0.6.0).** Newer architectures like Qwen3.5 alternate between `self_attn` and `linear_attn` layers. Linear attention layers don't use standard KV caches. The library detects this and only installs TurboQuant on self-attention layers — previously this would crash.

**Fractional bit-widths.** 3.5-bit quantization splits channels: half at 4-bit, half at 3-bit. This fills the quality gap between K3 and K4 for applications that need the memory savings of 3-bit but can't tolerate the quality hit.

## Results (v0.6.0 end-to-end, v0.7.0 kernel micro-benchmarks)

Benchmarked on M1 Max 64GB, MLX 0.31.1, April 2026. **12 models across 8 architecture families**, 5 TurboQuant configurations each. Full data in [BENCHMARKS.md](BENCHMARKS.md).

### Headline: Attention Sink Quality

The biggest win in v0.6.0. Pinning the system prompt in FP16 transforms quality on models that depend on those tokens:

| Model | K4/V2 | K4/V2 + sink128 | Improvement |
|---|---|---|---|
| Qwen2.5-7B | 0.7960 | **0.9474** | **+0.151** |
| Qwen3-8B | 0.8825 | **0.9810** | **+0.098** |
| Phi-3.5-mini | 0.9411 | **0.9971** | **+0.056** |
| Llama-3.1-8B | 0.9947 | 0.9973 | +0.003 |
| Mistral-7B | 0.9944 | 0.9977 | +0.003 |
| DeepSeek-R1-8B | 0.9889 | 0.9907 | +0.002 |

Models that already scored >0.99 without the sink (Llama, Mistral, DeepSeek-R1) see negligible change — the feature helps exactly where it's needed and costs nothing where it isn't.

### KV Cache Compression (K4/V2, 4K context)

| Model | FP16 Baseline | TurboQuant K4/V2 | Compression |
|---|---|---|---|
| Phi-3.5-mini | 1,536 MB | 406 MB | **3.78x** |
| Llama-3.1-8B / Mistral-7B | 512 MB | 133 MB | **3.86x** |
| Qwen3-8B / DeepSeek-R1-8B | 576 MB | 161 MB | **3.57x** |
| Gemma3-4B | 544 MB | 165 MB | **3.30x** |

On a 64GB Mac, this means 4x longer conversations or 4x more concurrent sessions at the same memory budget.

### Decode Speed

At 256-token context (short conversations), K4/V2+sink128 adds only 1-5% overhead:

| Model | Baseline tok/s | K4/V2+sink tok/s | Overhead |
|---|---|---|---|
| Qwen3-8B | 45.1 | 44.1 | **2%** |
| Qwen3.5-9B | 42.8 | 42.4 | **1%** |
| DeepSeek-R1-8B | 44.0 | 43.2 | **2%** |
| Llama-3.1-8B | 59.2 | 56.5 | **5%** |

At 2048 tokens, overhead increases to 12-36% depending on model due to decompression cost scaling with cache size. A fused attention-from-compressed kernel (see [Next Steps](#next-steps)) would reduce this.

**Surprise finding:** K4/V2+sink128 was 30% faster than FP16 baseline on TTFT at 2K context for Qwen3-8B (6,891 ms vs 9,836 ms). The smaller compressed cache reduces memory traffic during prefill.

### Fused QK Kernel Micro-benchmarks (v0.7.0)

The fused kernel computes Q @ K^T directly from packed codebook indices, skipping the decompression step entirely. It's not yet integrated into the end-to-end decode path (see [Next Steps](#next-steps)), but micro-benchmarks show the payoff scales with context length:

| Shape | dequant + matmul | fused kernel | Speedup |
|---|---|---|---|
| T_kv=256, D=128 | 301 μs | 266 μs | 1.13x |
| T_kv=1024, D=128 | 327 μs | 231 μs | 1.42x |
| T_kv=4096, D=128 | 487 μs | 230 μs | **2.12x** |
| T_kv=1024, D=256 (Gemma) | 491 μs | 242 μs | **2.03x** |

Full data in [BENCHMARKS_v07.md](BENCHMARKS_v07.md). Design and math in [docs/FUSED_ATTENTION_DESIGN.md](docs/FUSED_ATTENTION_DESIGN.md).

### Needle-in-a-Haystack

Retrieval of a hidden fact across 1K-8K context, needle at 10%, 50%, and 90% depth (12 tests per config):

| Model | FP16 Baseline | K4/V2 | K4/V2+sink128 |
|---|---|---|---|
| Qwen3-8B | **12/12** | **12/12** | **12/12** |
| Llama-3.1-8B | **12/12** | **12/12** | **12/12** |
| Mistral-7B | 8/12 | 6/12 | 7/12 |

Qwen3-8B and Llama-3.1-8B pass perfectly — zero retrieval regression. Mistral-7B fails 4/12 on the FP16 baseline itself; TurboQuant tracks but does not exceed the model's inherent limit.

### Architecture Breadth

| Family | Models Tested | head_dim | KV Heads | Notes |
|---|---|---|---|---|
| Qwen3 | 0.6B, 1.7B, 4B, 8B | 128 | 8 | Outlier layers detected and skipped |
| Qwen3.5 | 9B | varies | varies | Hybrid attention — 24/32 layers auto-skipped |
| Qwen2.5 | 7B | 128 | 4 | Previous gen, validates backward compat |
| Llama 3.1 | 8B | 128 | 8 | GQA architecture |
| Llama 3.2 | 3B | 128 | 8 | Smaller Llama |
| Mistral | 7B v0.3 | 128 | 8 | Sliding window attention |
| Gemma 3 | 1B, 4B | 256 | 1-4 | 1-KV-head models auto-upgrade to K4/V3 |
| Phi 3.5 | mini (3.8B) | 96 | 32 | Non-power-of-2 head_dim, no GQA |
| DeepSeek R1 | 8B (Qwen3 distill) | 128 | 8 | Reasoning model, long CoT chains |

## Configuration

```python
apply_turboquant(
    model,
    key_bits=4,                # 2, 3, 3.5, or 4 bits for keys
    value_bits=2,              # 2, 3, 3.5, or 4 bits for values
    residual_window=128,       # recent tokens stay in FP16 (sliding)
    fp16_sink_size=128,        # first N tokens stay FP16 permanently (v0.6.0)
    auto_detect_outliers=True, # skip layers with extreme key norms
    skip_layers=[0, 27],       # manually pin specific layers in FP16
    chunk_size=0,              # 0 = default batch compression, >0 = fixed chunks
    qjl_correction=False,      # experimental: 1-bit QJL residual correction
)
```

### Which config to use

| Use Case | Config | Why |
|---|---|---|
| **Default / balanced** | `key_bits=4, value_bits=2` | 3.6x compression, >0.98 cos_sim on most models |
| **Tool-calling / chat** | Same + `fp16_sink_size=128` | Preserves system prompt tokens defining tool schemas |
| **Memory-constrained** | `key_bits=3, value_bits=2` | 4.0-4.4x compression, quality depends on model |
| **Conservative** | `key_bits=4, value_bits=4` | 2.5-3.1x compression, minimal quality loss |

### Low-level API

```python
from mlx_turboquant import TurboQuantMSE, TurboQuantProd

# Algorithm 1: MSE-optimized (recommended)
tq = TurboQuantMSE(d=128, bits=4)
qt = tq.quantize(vectors)
vectors_hat = tq.dequantize(qt)

# Algorithm 2: inner-product optimized (custom attention kernels only)
tq_prod = TurboQuantProd(d=128, bits=4)
qt = tq_prod.quantize(vectors)
```

## CLI

```bash
mlx-turboquant generate \
  --model mlx-community/Qwen3-8B-4bit \
  --prompt "Explain quantum computing" \
  --key-bits 4 --value-bits 2

mlx-turboquant benchmark \
  --model mlx-community/Qwen3-8B-4bit \
  --benchmarks quality memory speed
```

## Project Structure

```
mlx_turboquant/
  cache.py        # TurboQuantKVCache — drop-in for mlx-lm's KVCache
  patch.py        # apply_turboquant() monkey-patch, hybrid attention detection
  kernels.py      # Fused Metal kernels (dequantize, quantize, fused QK scores)
  quantizer.py    # TurboQuantMSE (Algo 1) + TurboQuantProd (Algo 2)
  codebook.py     # Lloyd-Max optimal scalar quantizer for Beta distributions
  rotation.py     # Random orthogonal rotation (QR) + Walsh-Hadamard
  qjl.py          # Quantized Johnson-Lindenstrauss transform
  packing.py      # 1/2/3/4-bit index packing into uint8
  cli.py          # Command-line interface
  codebooks/      # Precomputed Lloyd-Max codebooks (.npz)
benchmarks/       # Full benchmark suite (run_full_suite.py, models.yaml)
results/          # Raw JSON benchmark data (12 models × 5 configs)
tests/            # 176 unit tests + 1 integration test
docs/             # Competitive audit, fused attention design, optimization log
```

## Tests

```bash
python -m pytest tests/ -q --ignore=tests/test_integration.py   # fast, no downloads
python -m pytest tests/test_integration.py -v -m slow            # real model, ~1GB download

# Full benchmark sweep (2-4 hours)
python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 1
python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 2
```

## Version History

| Version | What Changed | Key Result |
|---|---|---|
| v0.2.0 | Real model testing, vectorized quantization | 6 model families validated |
| v0.3.0 | Fused Metal dequantize kernels | 57% → 33% decode overhead |
| v0.4.0 | 3.5-bit fractional, needle-in-haystack, PyPI packaging | 12/12 retrieval at 1K-8K |
| v0.5.0 | Batch compression, pre-allocated FP16 window | 33% → **11%** decode overhead |
| **v0.6.0** | Attention sink, hybrid attention, QJL correction, 12-model sweep | **+0.098 cos_sim** on Qwen3-8B, 12 models × 8 families validated |
| **v0.7.0** | Fused QK scores Metal kernel (4/3/2-bit) + `pre_rotate_query` utility | **2.12× speedup** on long-context decode (T_kv=4096 D=128) vs dequant+matmul. See [BENCHMARKS_v07.md](BENCHMARKS_v07.md). |

## Next Steps

### v0.7.0 shipped — fused QK scores kernel

The fused attention-from-compressed kernel landed in v0.7.0 as a first-class utility (`mlx_turboquant.kernels.fused_qk_scores_{4,3,2}bit`). Correctness is guaranteed by a 12-test suite matching the existing dequant+matmul path to `atol=1e-3` across all bit widths, head_dims 96/128/256, and decode/prefill shapes. Micro-benchmark wins:

- decode T_kv=4096 D=128: **2.12×** speedup
- decode T_kv=1024 D=256 (Gemma3-like): **2.03×** speedup
- decode T_kv=1024 D=128: **1.42×** speedup
- prefill and short-context decode: roughly tied (dispatch-bound regime)

See [docs/FUSED_ATTENTION_DESIGN.md](docs/FUSED_ATTENTION_DESIGN.md) for the math derivation and [BENCHMARKS_v07.md](BENCHMARKS_v07.md) for the full micro-benchmark.

### v0.8.0 — Full SDPA integration + long-context validation

The v0.7.0 kernels are shipped as utilities — they are NOT yet automatically used by `apply_turboquant`. Integration requires per-model-family attention patches (Llama, Qwen, Mistral, Gemma, Phi, DeepSeek each have slightly different `self_attn.__call__` implementations). That work is v0.8.0:

- **Per-family attention patches.** Replace the SDPA call with a custom path that uses the fused kernel for the compressed region and standard matmul for sink + residual.
- **SIMD reductions in the fused kernel inner loop.** Replace the serial D-element accumulation with `simd_sum` for an additional ~30-50% speedup.
- **Other deferred optimizations.** Shared-memory WHT in the quantize kernel (from v0.6.0 competitive audit).

### v0.9.0 — Long Context Validation

- Needle-in-haystack at 16K, 32K, 64K context (current tests go to 8K)
- Benchmark on Qwen3.5-35B-A3B (MoE) and Llama-3.1-70B
- Tool-calling accuracy validation on real multi-turn agent conversations

### Future

- Upstream contribution to mlx-lm as an optional KV cache backend
- Integration with [eden-fleet](https://github.com/alex-rentel/eden-fleet) for compressed KV cache transfer in distributed inference
- 1-bit weight model compatibility (Bonsai 8B) — pending PrismML's upstream kernel merge

## Limitations

1. **Decode overhead at long context (22-36% at 2K).** Decompression cost scales with cache size. The fused QK kernel (v0.7.0) achieves 2.12x speedup in isolation but is not yet integrated into the SDPA path — end-to-end decode overhead is unchanged from v0.6.0 until v0.8.0 ships the per-family attention patches.
2. **Error compounding.** Per-vector reconstruction error compounds through layers. Models with fewer KV heads (Gemma3-1B: 1 head) are more fragile.
3. **Qwen3.5 partial coverage.** Only 8/32 layers use self-attention; memory savings are proportionally smaller.
4. **Quality depends on outlier detection.** Disabling `auto_detect_outliers` drops Qwen2.5-7B from 0.80 → 0.46 cos_sim at K4/V2.

## Part of the Eden Ecosystem

- [eden](https://github.com/alex-rentel/eden) — Local AI agent framework
- [eden-flywheel](https://github.com/alex-rentel/eden-flywheel) — Claude Code sessions → training data → fine-tune → deploy
- [eden-fleet](https://github.com/alex-rentel/eden-fleet) — Distributed AI across a Mac homelab via SSH
- [eden-models](https://github.com/alex-rentel/eden-models) — Training pipeline for tool-calling LLMs on HPC
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Train ChatGPT-class models on Mac
- **mlx-turboquant** — KV cache compression for longer contexts on Apple Silicon

## Community Implementations

- [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) — V2 + V3 dual-path, best QJL analysis in the community
- [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) — Claims 98% FP16 speed via fused attention kernel (unverified)
- [rachittshah/mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) — Clean drop-in KVCache, no Metal kernels
- [helgklaizar/turboquant_mlx](https://github.com/helgklaizar/turboquant_mlx) — Attention sink + dynamic chunking, affine quantization
- [flovflo/turboquant-mlx-qwen35-kv](https://huggingface.co/flovflo/turboquant-mlx-qwen35-kv) — Qwen 3.5 35B benchmarks, honestly labeled as "TurboQuant-inspired"

This repo differentiates with: fused Metal kernels on the Lloyd-Max codebook path, asymmetric K/V bits, attention sink, outlier detection, hybrid attention support, fractional 3.5-bit, and a 12-model benchmark sweep with needle-in-haystack validation.

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
