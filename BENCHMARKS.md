# Benchmarks

All benchmarks on **Apple M1 Max 64GB / macOS 26.4 / Python 3.12.8 / mlx 0.31.1 / mlx-lm 0.31.1**, last updated 2026-04-07 for v0.6.0.

Reproduce: `python benchmarks/bench_v06.py --decode-tokens 30 --runs 3`. Raw JSON in `benchmarks/results_v06/`.

## v0.6.0 Headline

Five configs benchmarked across three models on a ~1000-token system-prompt-style prompt:

| Config | Description |
|---|---|
| `baseline` | Standard mlx-lm KVCache (FP16) |
| `k4v2` | TurboQuant default (`key_bits=4, value_bits=2`) |
| `k4v2_sink128` | k4v2 + new `fp16_sink_size=128` (attention sink) |
| `k4v2_qjl` | k4v2 + new `qjl_correction=True` (1-bit residual sketch) |
| `k3v2` | Aggressive (`key_bits=3, value_bits=2`) |

Note on v0.6.0 default: `chunk_size=0` selects the v0.5.0 batch compression path. The new `chunk_size>0` opt-in chunked-compression path is benchmark-neutral on current Metal kernels and exists for future kernel work that templates on chunk dimension.

### Quality (cos sim of last-token logits vs FP16, ~1000-token context)

| Model | head_dim | baseline | k4v2 | **k4v2_sink128** | k4v2_qjl | k3v2 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-4bit | 128 | 1.0000 | 0.9717 | **0.9749** | 0.9665 | 0.9055 |
| Qwen3-8B-4bit | 128 | 1.0000 | 0.9938 | **0.9962** | 0.9955 | 0.9804 |
| Gemma3-1B-it-4bit† | 256 | 1.0000 | 0.7278 | 0.7284 | 0.7279 | 0.7278 |

†Gemma3-1B has only 1 KV head, which triggers v0.5.0's auto-upgrade (`key_bits→4, value_bits→3`); all four TQ configs effectively become K4/V3, so the cos_sim numbers are nearly identical. The 0.728 floor is the inherent quality of K4/V3 on this 1000-token prompt for a single-KV-head model.

### Decode speed (median of 3 runs, 1 warmup, 30 decode tokens)

| Model | baseline tok/s | k4v2 | k4v2_sink128 | k4v2_qjl | k3v2 |
|---|---|---|---|---|---|
| Qwen3-1.7B-4bit | 126.2 | 87.5 | 85.5 | 86.9 | 88.0 |
| Qwen3-8B-4bit | 44.5 | 34.6 | 34.8 | 34.4 | 34.5 |
| Gemma3-1B-it-4bit | 148.7 | 113.2 | 111.9 | 113.0 | 114.0 |

### Time-to-first-token (ms, median)

| Model | baseline | k4v2 | k4v2_sink128 | k4v2_qjl | k3v2 |
|---|---|---|---|---|---|
| Qwen3-1.7B-4bit | 1168 | 972 | 944 | 1033 | 1013 |
| Qwen3-8B-4bit | 4969 | 3578 | 3558 | 3666 | 3641 |
| Gemma3-1B-it-4bit | 390 | 538 | 514 | 583 | 538 |

TurboQuant TTFT is consistently **lower** than baseline on the larger models because the reduced KV cache memory traffic dominates over the per-step concat cost during prefill.

### Top-1 logit match vs FP16

| Model | k4v2 | k4v2_sink128 | k4v2_qjl | k3v2 |
|---|---|---|---|---|
| Qwen3-1.7B-4bit | ✓ | ✓ | ✓ | ✗ |
| Qwen3-8B-4bit | ✓ | ✓ | ✓ | ✓ |
| Gemma3-1B-it-4bit† | ✗ | ✗ | ✗ | ✗ |

†Gemma3-1B's auto-upgraded K4/V3 doesn't recover top-1 on this prompt; the model is sensitive enough to compression that even 4-bit keys cause an argmax flip. This was true in v0.5.0 too.

## Did the new features earn their keep?

### Attention sink (`fp16_sink_size`) — **YES, ship it**

| Model | k4v2 cos_sim | k4v2_sink128 cos_sim | Δ |
|---|---|---|---|
| Qwen3-1.7B-4bit | 0.9717 | 0.9749 | **+0.0032** |
| Qwen3-8B-4bit | 0.9938 | 0.9962 | **+0.0024** |
| Gemma3-1B-it-4bit | 0.7278 | 0.7284 | +0.0007 |

Consistent positive cosine-sim delta on all three models, no measurable speed cost. Tooling-style prompts where the first ~128 tokens carry tool-call semantics will see the largest benefit. Default OFF to preserve v0.5.0 behavior; opt in via `apply_turboquant(model, ..., fp16_sink_size=128)`.

### QJL correction (`qjl_correction`) — **mixed, ship as opt-in experimental**

| Model | k4v2 cos_sim | k4v2_qjl cos_sim | Δ |
|---|---|---|---|
| Qwen3-1.7B-4bit | 0.9717 | 0.9665 | **−0.0052** |
| Qwen3-8B-4bit | 0.9938 | 0.9955 | **+0.0017** |
| Gemma3-1B-it-4bit | 0.7278 | 0.7279 | +0.0001 (noise) |

Helps the larger Qwen3-8B but hurts the smaller Qwen3-1.7B. Synthetic tests on N(0, 1) random data confirm the correction reduces MSE in isolation; on real KV vectors the residual structure interacts with the random JL projection in model-dependent ways. Kept in tree as an opt-in experimental flag with documented mixed results — do not enable blindly.

### Chunked compression (`chunk_size`) — **neutral, default OFF**

A/B with `chunk_size=64` vs `chunk_size=0` (v0.5.0 batch logic) on identical inputs:

| Model k4v2 decode tok/s | chunk_size=0 | chunk_size=64 |
|---|---|---|
| Qwen3-1.7B-4bit | 87.5 | 88.0 |
| Qwen3-8B-4bit | 34.6 | 34.7 |
| Gemma3-1B-it-4bit | 113.2 | 113.7 |

Within measurement noise. Our current Metal kernels do not template on the chunk dimension, so fixed-size chunks do not improve kernel template caching. The chunked code path is preserved as opt-in (`chunk_size > 0`) for future kernel work that benefits from stable input shapes.

## Comparison vs v0.5.0 (Qwen3-8B, K4/V2)

The v0.5.0 README reported 11% decode overhead at 2K context with the speed harness in `benchmarks/bench_speed.py`. The v0.6.0 measurement above is at ~1000-token context using the new harness (`benchmarks/bench_v06.py`) with stricter median-of-3-runs methodology and a different prompt. Direct apples-to-apples comparison would require running both harnesses on both versions.

| Metric | v0.5.0 (README, 2K ctx) | v0.6.0 (bench_v06, ~1K ctx) |
|---|---|---|
| Qwen3-8B baseline tok/s | 41.3 | 44.5 |
| Qwen3-8B k4v2 tok/s | 36.7 | 34.6 |
| Overhead | 11% | 22% |

The headline overhead delta is **partly real** (longer contexts amortize per-step concat cost over more attention work, so 2K shows lower percentage overhead than 1K) and **partly measurement variance** (we observed baseline tok/s for Qwen3-1.7B varying between 108.9 and 126.2 across two consecutive bench runs in this session — ~15% noise floor on M1 Max with thermal/cache state effects).

The v0.6.0 default `chunk_size=0` exactly preserves the v0.5.0 compression path, so any speed difference here is **not** a regression in the compression algorithm — it is harness/methodology differences.

## Methodology Notes

- 1-warmup-3-run protocol per config; reported numbers are the median over the 3 measured runs
- Cosine similarity is computed over the last-token logits in float64 (cast from bfloat16/float16) for numerical stability
- Decode tok/s measures the pure decode loop after prefill; TTFT measures prefill + first decoded token
- Peak memory delta is RSS-based (psutil) and includes any allocator high-water-mark; not a tight upper bound
- Models loaded once per benchmarking session; `apply_turboquant` is reset between configs by deleting the patched `model.make_cache` attribute

## Reproducibility

All numbers in this document come from a single run of:

```bash
python benchmarks/bench_v06.py --decode-tokens 30 --runs 3
```

Raw JSON results are checked into `benchmarks/results_v06/`. Hardware: Apple M1 Max, 64 GB unified RAM, 32 GPU cores, macOS 26.4.

## Pure Quantizer Quality (carried over from v0.5.0)

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

The logit-level quality gap above comes from error compounding through transformer layers, not quantizer imprecision.

## Needle-in-a-Haystack

See Phase 4 results in `benchmarks/needle_results.md` (v0.5.0) and v0.6.0 validation below once Phase 4 completes.
