# Optimization Log

Branch: perf/optimization-pass
Model: Qwen3-8B-4bit, 2K context, K4/V2, 50 decode tokens
Machine: M1 Max 64GB

## Starting Baseline
- FP16: 41.3 tok/s
- TurboQuant: 28.4 tok/s  
- Overhead: 31.2%

## Profile Results (per-layer per-step)
- Append (concat new token): 33% of cache time
- Compress (quantize + dequant): 35% of cache time
- Output concat (decompressed + FP16): 32% of cache time
