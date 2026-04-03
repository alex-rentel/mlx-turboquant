# Optimization Log

Branch: perf/optimization-pass
Model: Qwen3-8B-4bit, 2K context, K4/V2, 50 decode tokens
Machine: M1 Max 64GB

## Starting Baseline (v0.4.0)
- FP16: 41.3 tok/s
- TurboQuant: 28.4 tok/s
- Overhead: 31.2%

## Profile Results (per-layer per-step)
- Append (concat new token): 33% of cache time
- Compress (quantize + dequant): 35% of cache time
- Output concat (decompressed + FP16): 32% of cache time

## Opt 6: Batch Compression
- Hypothesis: Compress less often (threshold 2x window) to eliminate per-token quantize
- Baseline: 31.2% overhead
- After: 24.1% overhead (mean of runs 2-3)
- Delta: -7.1 percentage points
- Decision: **KEEP**
- Reason: During 50-token decode, zero compression ops (FP16 grows 128→178, threshold 256)

## Opt 2: Pre-allocated FP16 Window
- Hypothesis: Replace per-token concat with slice write into pre-allocated buffer
- Baseline: 24.1% overhead
- After: 11.2% overhead (mean of runs 2-3)
- Delta: -12.9 percentage points
- Decision: **KEEP**
- Reason: Eliminates array allocation on every decode step

## Opt 7: Argmin Distance vs Boundary Search
- Hypothesis: argmin over centroids might be faster than boundary comparison
- Baseline: 1.81ms (boundary search)
- After: 8.15ms (argmin distance)
- Delta: 4.5x SLOWER
- Decision: **REVERT** (not applied)
- Reason: Boundary search already optimal for small codebooks

## Opt 8: Float16 Rotation in Metal
- Hypothesis: Half bandwidth for rotation matrix reads
- Baseline: 0.41ms
- After: 0.42ms
- Delta: No change (kernel promotes to float32 internally)
- Decision: **SKIP**
- Reason: MLX Metal kernel operates in float32 regardless of input dtype

## Opt 3: Rotation Matrix Sharing
- Finding: Already shared via _rotation_cache dict — 1 unique object across 26 layers
- Decision: **NO ACTION NEEDED**

## Opt 9: Prefill Overhead
- Finding: TQ prefill is same speed or faster than FP16 baseline
- Decision: **NO ACTION NEEDED**

## Summary
| Optimization | Overhead | Delta |
|-------------|----------|-------|
| v0.4.0 baseline | 31.2% | — |
| + Batch compress | 24.1% | -7.1pt |
| + Pre-alloc window | 11.2% | -12.9pt |
| **Total improvement** | **11.2%** | **-20.0pt** |
