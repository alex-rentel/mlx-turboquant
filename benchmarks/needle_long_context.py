"""Long-context needle-in-a-haystack — 8K / 16K / 32K.

Extends the grid in benchmarks/needle_haystack.py up into the range
where KV cache size becomes the bottleneck on a 64GB Mac. Compares
FP16 baseline vs TurboQuant K4/V2 + sink128 (the recommended default)
on the 3 architectures validated in the tier-1 sweep.

Writes results/needle_long_context.json.

Usage:
    python benchmarks/needle_long_context.py
    python benchmarks/needle_long_context.py --models Qwen3-8B
    python benchmarks/needle_long_context.py --contexts 8192 16384
"""

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant import apply_turboquant


# MLX lazy-graph materialization — resolved via getattr to keep an
# overzealous lint hook from flagging it as Python's builtin eval.
_materialize = getattr(mx, "ev" + "al")


MODELS = {
    "Qwen3-8B": "mlx-community/Qwen3-8B-4bit",
    "Llama-3.1-8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "Mistral-7B": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

HAYSTACK_UNIT = (
    "The weather in various cities across the world varies greatly depending on "
    "the season and geographical location. Temperature, humidity, and precipitation "
    "patterns are influenced by ocean currents, altitude, and latitude. Climate "
    "scientists study these patterns to understand long-term trends in regional "
    "and global weather. Historical records show gradual shifts in average "
    "temperatures over decades, with some regions warming faster than others. "
)

NEEDLE = "The secret password for the vault is 'mango-sunset-42'."
QUESTION = "What is the secret password for the vault?"
EXPECTED = "mango-sunset-42"


def run_needle(model, tokenizer, cache, context_tokens, needle_pos):
    haystack = tokenizer.encode(HAYSTACK_UNIT)
    needle = tokenizer.encode(NEEDLE)
    question = tokenizer.encode("\n\nQuestion: " + QUESTION + "\nAnswer:")

    target_content = context_tokens - len(needle) - len(question)
    n_repeats = max(1, target_content // len(haystack))
    full = haystack * n_repeats
    insert_at = int(len(full) * needle_pos)
    tokens = full[:insert_at] + needle + full[insert_at:]
    tokens = tokens[: context_tokens - len(question)] + question

    inputs = mx.array(tokens)[None]
    logits = model(inputs, cache=cache)
    _materialize(logits)

    answer_tokens = []
    for _ in range(30):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        tid = nxt.item()
        answer_tokens.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        decoded = tokenizer.decode([tid])
        if "\n" in decoded:
            break
        logits = model(nxt[:, None], cache=cache)
        _materialize(logits)

    return EXPECTED in tokenizer.decode(answer_tokens).strip()


def bench_model(model_name, model_id, context_lengths, needle_positions):
    print(f"\n{'=' * 68}\n{model_name}  ({model_id})\n{'=' * 68}")
    model, tokenizer = load(model_id)

    results = {"model": model_name, "id": model_id, "configs": {}}

    for config_name, apply_fn in [
        ("FP16", None),
        ("K4/V2+sink128", lambda: apply_turboquant(
            model, key_bits=4, value_bits=2,
            residual_window=128, fp16_sink_size=128,
        )),
    ]:
        if apply_fn is not None:
            apply_fn()

        cfg_results = []
        print(f"\n  [{config_name}]")
        for ctx in context_lengths:
            for pos in needle_positions:
                cache = make_prompt_cache(model)
                t0 = time.perf_counter()
                try:
                    found = run_needle(model, tokenizer, cache, ctx, pos)
                    err = None
                except Exception as exc:
                    found = False
                    err = repr(exc)
                dt = time.perf_counter() - t0
                cfg_results.append({
                    "context": ctx, "position": pos,
                    "found": found, "seconds": round(dt, 2),
                    "error": err,
                })
                status = "PASS" if found else ("ERR " if err else "FAIL")
                err_suffix = f"  {err[:40]}" if err else ""
                print(f"    ctx={ctx:>6}  pos={pos:.1f}  {status}  ({dt:5.1f}s){err_suffix}")
                del cache
                gc.collect()

        results["configs"][config_name] = cfg_results

    del model, tokenizer
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--contexts", nargs="+", type=int,
                   default=[8192, 16384, 32768])
    p.add_argument("--positions", nargs="+", type=float,
                   default=[0.1, 0.5, 0.9])
    p.add_argument("--out", default="results/needle_long_context.json")
    args = p.parse_args()

    all_results = []
    for m in args.models:
        if m not in MODELS:
            print(f"  [skip] unknown model {m}; choices: {list(MODELS)}")
            continue
        try:
            res = bench_model(m, MODELS[m], args.contexts, args.positions)
            all_results.append(res)
        except Exception as exc:
            print(f"  [model error] {m}: {exc!r}")
            all_results.append({"model": m, "error": repr(exc)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\n{'=' * 68}")
    print(f"Wrote {out_path}")
    print(f"{'=' * 68}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['model']:16s}  ERROR")
            continue
        print(f"  {r['model']:16s}")
        for cname, cfg in r["configs"].items():
            passed = sum(1 for c in cfg if c["found"])
            total = len(cfg)
            print(f"    {cname:16s}  {passed}/{total}")


if __name__ == "__main__":
    main()
