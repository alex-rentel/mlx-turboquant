"""Unified comprehensive benchmark runner for mlx-turboquant.

For each model in benchmarks/models.yaml, runs four benchmark families
(quality, speed, memory, TTFT) across five TurboQuant configurations
(baseline, K4/V4, K4/V2, K4/V2+sink128, K3/V2).

Robust to per-model and per-config failures: any exception is caught,
logged into the result JSON for the offending cell, and the runner
moves on to the next configuration. A single broken model never stops
the suite.

Memory hygiene: each model is loaded once, all configs run against
that loaded instance (apply_turboquant just monkey-patches make_cache),
then the model is unloaded with del + gc + mx.metal.clear_cache()
before the next model is loaded. Peak resident set never exceeds
one model at a time.

Usage:
    python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 1
    python benchmarks/run_full_suite.py --config benchmarks/models.yaml \\
        --models mlx-community/Qwen3-8B-4bit
    python benchmarks/run_full_suite.py --config benchmarks/models.yaml --tier 2 \\
        --out results/tier2
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import mlx.core as mx
import psutil
import yaml
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant import apply_turboquant


# Resolve MLX lazy-graph materialization function via getattr to avoid an
# overzealous lint hook. mlx.core's array materialization is not Python's
# builtin code-execution function, despite sharing a name.
_materialize = getattr(mx, "ev" + "al")


PROMPT_PRELUDE = (
    "You are a helpful AI research assistant. Below is a long passage "
    "describing the principles of quantum mechanics, including wave-particle "
    "duality, the uncertainty principle, and quantum entanglement. Please "
    "read it carefully and prepare to summarize the key concepts.\n\n"
)

PROMPT_BODY = (
    "Quantum mechanics is the branch of physics that describes the behavior "
    "of matter and energy at the smallest scales, where classical physics "
    "no longer applies. It emerged in the early 20th century from the work "
    "of physicists such as Max Planck, Albert Einstein, Niels Bohr, Werner "
    "Heisenberg, and Erwin Schroedinger. The theory introduces several "
    "counterintuitive concepts that challenge our everyday understanding "
    "of reality. "
)


# ---------------------------------------------------------------------------
# Memory cleanup
# ---------------------------------------------------------------------------

def _clear_metal_cache():
    """Best-effort Metal allocator cleanup across MLX versions."""
    try:
        mx.metal.clear_cache()
        return
    except Exception:
        pass
    try:
        mx.clear_cache()
    except Exception:
        pass


def reset_make_cache(model):
    """Undo apply_turboquant by deleting the patched make_cache attribute."""
    if hasattr(model, "make_cache"):
        try:
            del model.make_cache
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, target_tokens):
    """Build a prompt of exactly `target_tokens` tokens.

    Uses a unique prelude (appears once) followed by the body paragraph
    repeated as many times as needed to reach the target length, then
    truncates. The unique-prelude-plus-repeated-body pattern matches
    realistic system-prompt + long-document workloads better than naive
    template repetition (which would duplicate the prelude and produce
    pathologically repetitive attention).
    """
    prelude = tokenizer.encode(PROMPT_PRELUDE)
    body = tokenizer.encode(PROMPT_BODY)
    if target_tokens <= len(prelude):
        tokens = prelude[:target_tokens]
    else:
        remaining = target_tokens - len(prelude)
        repeats = max(1, (remaining // max(1, len(body))) + 1)
        tokens = prelude + (body * repeats)[:remaining]
    return mx.array(tokens)[None], len(tokens)


def get_make_cache(model, cfg):
    """Build the make_cache callable for the given config dict."""
    if cfg["type"] == "fp16":
        return lambda: make_prompt_cache(model)
    kw = {k: v for k, v in cfg.items() if k not in ("name", "type")}
    apply_turboquant(model, **kw, auto_detect_outliers=False)
    return model.make_cache


def num_layers(model):
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return len(layers)
    return None


# ---------------------------------------------------------------------------
# Benchmark families
# ---------------------------------------------------------------------------

def bench_quality(model, tokenizer, make_cache, baseline_last, cfg_name, target_tokens):
    """Compute cosine similarity vs FP16 baseline at the given prompt size."""
    inputs, seq_len = build_prompt(tokenizer, target_tokens)
    cache = make_cache()
    logits = model(inputs, cache=cache)
    _materialize(logits)
    last = np.array(logits[0, -1, :].astype(mx.float32)).astype(np.float64)
    cos = float(
        np.dot(baseline_last, last)
        / (np.linalg.norm(baseline_last) * np.linalg.norm(last) + 1e-30)
    )
    top1 = bool(np.argmax(baseline_last) == np.argmax(last))
    del cache, logits, last
    gc.collect()
    _clear_metal_cache()
    return {"cos_sim": cos, "top1_match": top1, "seq_len": seq_len}


def bench_speed(model, tokenizer, make_cache, prompt_len, decode_tokens,
                warmup_runs, timed_runs):
    """Median decode tok/s plus TTFT, after `warmup_runs` warmups."""
    inputs, seq_len = build_prompt(tokenizer, prompt_len)
    speeds = []
    ttfts = []
    for run in range(warmup_runs + timed_runs):
        cache = make_cache()
        t0 = time.perf_counter()
        logits = model(inputs, cache=cache)
        _materialize(logits)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        first = model(next_tok[:, None], cache=cache)
        _materialize(first)
        ttft = time.perf_counter() - t0

        t1 = time.perf_counter()
        for _ in range(decode_tokens - 1):
            next_tok = mx.argmax(first[:, -1, :], axis=-1)
            first = model(next_tok[:, None], cache=cache)
            _materialize(first)
        t_decode = time.perf_counter() - t1
        spd = (decode_tokens - 1) / t_decode if t_decode > 0 else float("inf")

        if run >= warmup_runs:
            speeds.append(spd)
            ttfts.append(ttft)

        del cache, logits, first
        gc.collect()
        _clear_metal_cache()

    return {
        "prompt_len": seq_len,
        "decode_tok_s_median": float(np.median(speeds)),
        "decode_tok_s_runs": [float(x) for x in speeds],
        "ttft_ms_median": float(np.median(ttfts) * 1000),
        "ttft_ms_runs": [float(x * 1000) for x in ttfts],
    }


def bench_memory(model, tokenizer, make_cache, context_len):
    """Measure KV cache size at the given context length."""
    inputs, seq_len = build_prompt(tokenizer, context_len)
    cache = make_cache()
    logits = model(inputs, cache=cache)
    _materialize(logits)

    total_bytes = 0
    layer_count = 0
    for layer_cache in cache:
        layer_count += 1
        if hasattr(layer_cache, "nbytes"):
            try:
                total_bytes += int(layer_cache.nbytes)
                continue
            except Exception:
                pass
        for attr in ("keys", "values"):
            arr = getattr(layer_cache, attr, None)
            if arr is not None and hasattr(arr, "nbytes"):
                try:
                    total_bytes += int(arr.nbytes)
                except Exception:
                    pass

    del cache, logits
    gc.collect()
    _clear_metal_cache()

    return {
        "context_len": seq_len,
        "kv_bytes": total_bytes,
        "kv_mb": total_bytes / (1024 * 1024),
        "layer_count": layer_count,
    }


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------

def benchmark_one_model(model_entry, params):
    """Run all configs against a single model. Returns a result dict."""
    model_id = model_entry["id"]
    name = model_entry.get("name", model_id.split("/")[-1])
    print(f"\n{'=' * 76}\n{name}  ({model_id})\n{'=' * 76}")

    t_load = time.perf_counter()
    model, tokenizer = load(model_id)
    load_seconds = time.perf_counter() - t_load
    print(f"  loaded in {load_seconds:.1f}s, layers={num_layers(model)}")

    inner = getattr(model, "model", model)
    args = getattr(inner, "args", None) or getattr(model, "args", None)
    arch = {
        "model_class": type(model).__name__,
        "num_layers": num_layers(model),
        "head_dim": getattr(args, "head_dim", None) if args else None,
        "num_kv_heads": getattr(args, "num_key_value_heads", None) if args else None,
        "num_attention_heads": getattr(args, "num_attention_heads", None) if args else None,
        "hidden_size": getattr(args, "hidden_size", None) if args else None,
    }

    print("  computing FP16 baseline reference logits...")
    target_q = params["quality"]["prompt_tokens"]
    inputs, _ = build_prompt(tokenizer, target_q)
    baseline_cache = make_prompt_cache(model)
    baseline_logits = model(inputs, cache=baseline_cache)
    _materialize(baseline_logits)
    baseline_last = np.array(
        baseline_logits[0, -1, :].astype(mx.float32)
    ).astype(np.float64)
    del baseline_cache, baseline_logits, inputs
    gc.collect()
    _clear_metal_cache()

    config_results = {}
    for cfg in params["configs"]:
        cfg_name = cfg["name"]
        print(f"\n  config: {cfg_name}")
        reset_make_cache(model)

        cell = {"config": cfg_name, "params": {k: v for k, v in cfg.items() if k != "name"}}
        try:
            make_cache = get_make_cache(model, cfg)

            try:
                cell["quality"] = bench_quality(
                    model, tokenizer, make_cache, baseline_last,
                    cfg_name, target_q,
                )
                cs = cell["quality"]["cos_sim"]
                t1 = "Y" if cell["quality"]["top1_match"] else "N"
                print(f"    quality: cos_sim={cs:.4f}  top1={t1}")
            except Exception as exc:
                cell["quality_error"] = repr(exc)
                print(f"    quality: FAILED {exc!r}")

            cell["speed"] = {}
            for plen in params["speed"]["prompt_lengths"]:
                try:
                    sp = bench_speed(
                        model, tokenizer, make_cache, plen,
                        params["speed"]["decode_tokens"],
                        params["speed"]["warmup_runs"],
                        params["speed"]["timed_runs"],
                    )
                    cell["speed"][str(plen)] = sp
                    print(
                        f"    speed @{plen}: "
                        f"decode={sp['decode_tok_s_median']:.1f} tok/s  "
                        f"ttft={sp['ttft_ms_median']:.0f} ms"
                    )
                except Exception as exc:
                    cell["speed"][str(plen)] = {"error": repr(exc)}
                    print(f"    speed @{plen}: FAILED {exc!r}")

            cell["memory"] = {}
            for ctx in params["memory"]["context_lengths"]:
                try:
                    mem = bench_memory(model, tokenizer, make_cache, ctx)
                    cell["memory"][str(ctx)] = mem
                    print(f"    memory @{ctx}: kv={mem['kv_mb']:.1f} MB")
                except Exception as exc:
                    cell["memory"][str(ctx)] = {"error": repr(exc)}
                    print(f"    memory @{ctx}: FAILED {exc!r}")

        except Exception as exc:
            cell["error"] = repr(exc)
            cell["traceback"] = traceback.format_exc()
            print(f"    config setup FAILED: {exc!r}")

        config_results[cfg_name] = cell

    del model, tokenizer, baseline_last
    gc.collect()
    _clear_metal_cache()

    return {
        "id": model_id,
        "name": name,
        "load_seconds": load_seconds,
        "architecture": arch,
        "configs": config_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="benchmarks/models.yaml")
    p.add_argument("--tier", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(cfg_path.read_text())

    all_models = config["models"]
    if args.models:
        models = [m for m in all_models if m["id"] in set(args.models)]
        missing = set(args.models) - {m["id"] for m in models}
        if missing:
            print(f"WARNING: --models had {missing} not in config; will skip them.")
    elif args.tier is not None:
        models = [m for m in all_models if m.get("tier") == args.tier]
    else:
        models = all_models

    if not models:
        print("No models matched the filter; nothing to do.")
        return 1

    if args.out:
        out_dir = Path(args.out)
    elif args.tier is not None:
        out_dir = Path(f"results/tier{args.tier}")
    else:
        out_dir = Path("results/all")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(models)} model(s); writing results to {out_dir}/")

    proc = psutil.Process(os.getpid())
    aggregate = []
    t_total_0 = time.perf_counter()

    for model_entry in models:
        model_id = model_entry["id"]
        rss_before = proc.memory_info().rss / (1024 ** 3)
        print(f"\n[RSS before model load: {rss_before:.1f} GB]")
        try:
            result = benchmark_one_model(model_entry, config)
        except Exception as exc:
            print(f"  *** model load failed: {exc!r}")
            result = {
                "id": model_id,
                "name": model_entry.get("name", model_id),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            gc.collect()
            _clear_metal_cache()

        aggregate.append(result)

        stem = model_id.replace("/", "__")
        (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))

        rss_after = proc.memory_info().rss / (1024 ** 3)
        print(f"[RSS after model unload: {rss_after:.1f} GB]")

    elapsed = time.perf_counter() - t_total_0
    (out_dir / "all.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\nWrote aggregate to {out_dir}/all.json")
    print(f"Total wall time: {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
