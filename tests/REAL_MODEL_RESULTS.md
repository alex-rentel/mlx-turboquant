# Real Model Smoke Test Results

Date: 2026-04-02
Machine: M1 Max 64GB

## Models Tested

| Model | Repo | head_dim | KV heads | Layers | Status |
|-------|------|----------|----------|--------|--------|
| Qwen3-1.7B | mlx-community/Qwen3-1.7B-4bit | 128 | 8 | 28 | PASS |
| Qwen3-8B | mlx-community/Qwen3-8B-4bit | 128 | 8 | 36 | PASS |
| Gemma3-1B | mlx-community/gemma-3-1b-it-4bit | 256 | 1 | 26 | PASS |
| Gemma3-4B | mlx-community/gemma-3-4b-it-4bit | 256 | ? | 34 | PASS |

## Configuration

- key_bits=4, value_bits=2, residual_window=128
- Outlier layer auto-detection enabled

## Generation Samples (K4/V2, prompt: "Explain gravity in one sentence:")

**Qwen3-1.7B:** "The force of gravity is the force that pulls objects toward the Earth and is responsible for the weight of an object o..."

**Qwen3-8B:** "Gravity is the force that attracts objects with mass toward each other, and it is responsible for keeping planets in o..."

**Gemma3-1B:** "Gravity is the force that pulls objects towards each other, caused by the mass of the universe."

**Gemma3-4B:** "Gravity is a fundamental force that attracts any two objects with mass towards each other, causing them to accelerate..."

## Fixes Required

- Fixed pyproject.toml build-backend from `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`
- No other fixes needed. All 4 models worked out of the box including:
  - Qwen3 with outlier layer detection
  - Gemma3 with head_dim=256 (non-standard)
  - GQA configurations (Qwen3 KV heads < attention heads)
