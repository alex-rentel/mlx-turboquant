# Real Model Integration Test Results

Date: 2026-04-02
Machine: M1 Max 64GB

## Models Tested

| Model | Repo | head_dim | KV heads | Layers | Smoke | Quality |
|-------|------|----------|----------|--------|-------|---------|
| Qwen3-1.7B | mlx-community/Qwen3-1.7B-4bit | 128 | 8 | 28 | PASS | Good |
| Qwen3-8B | mlx-community/Qwen3-8B-4bit | 128 | 8 | 36 | PASS | Excellent |
| Gemma3-1B | mlx-community/gemma-3-1b-it-4bit | 256 | 1 | 26 | PASS | Good |
| Gemma3-4B | mlx-community/gemma-3-4b-it-4bit | 256 | 4 | 34 | PASS | Degrades >1K |

## Configuration

- key_bits=4, value_bits=2, residual_window=128 (default)
- Outlier layer auto-detection enabled

## Smoke Test (K4/V2, prompt: "Explain gravity in one sentence:")

All 4 models generate coherent English at short context.

## Quality at ~500 tokens (cosine sim vs FP16)

| Model | K4/V4 | K4/V2 | K3/V2 |
|-------|-------|-------|-------|
| Qwen3-1.7B | 0.9914 | 0.9853 | 0.9687 |
| Qwen3-8B | 0.9994 | 0.9976 | 0.9872 |
| Gemma3-1B | 0.9953 | 0.9802 | 0.9619 |
| Gemma3-4B | 0.9925 | 0.9848 | 0.9753 |

## Long Context (2K tokens)

- **Qwen3-8B:** 3.0x compression, output matches baseline at K4/V2
- **Gemma3-4B:** 2.8x compression, output degrades at 2K. Coherent up to ~1K. Recommended: use larger residual window or K4/V4 for Gemma.

## Fixes Applied During Testing

1. Fixed pyproject.toml build-backend (`setuptools.build_meta`)
2. Fixed `_get_model_config` for Gemma3 conditional models (nested `language_model.model.args` path)
