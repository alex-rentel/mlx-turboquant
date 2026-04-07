"""v0.6.0 needle-in-a-haystack validation.

Runs the needle test on Qwen3-8B with the best config from Phase 3
(k4v2 + fp16_sink_size=128) across 1K, 2K, 4K, 8K context lengths and
3 needle positions (0.1, 0.5, 0.9). 12 tests total. We must hit at
least the same 12/12 retrieval as the v0.5.0 baseline configuration.

Also runs the FP16 baseline and v0.5.0-compatible k4v2 (no sink) for
comparison so we can spot any regression introduced by sink.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant import apply_turboquant


_materialize = mx.eval


HAYSTACK_UNIT = (
    "The weather in various cities across the world varies greatly depending on "
    "the season and geographical location. Temperature, humidity, and precipitation "
    "patterns are influenced by ocean currents, altitude, and latitude. Climate "
    "scientists study these patterns to understand long-term trends. "
)

NEEDLE = "The secret password for the vault is 'mango-sunset-42'."
QUESTION = "What is the secret password for the vault?"
EXPECTED_ANSWER = "mango-sunset-42"


def run_one_needle_test(model, tokenizer, make_cache, context_tokens, needle_position):
    """Build a haystack prompt with the needle inserted, decode the answer."""
    haystack_tokens = tokenizer.encode(HAYSTACK_UNIT)
    needle_tokens = tokenizer.encode(NEEDLE)
    question_tokens = tokenizer.encode("\n\nQuestion: " + QUESTION + "\nAnswer:")

    target_content = context_tokens - len(needle_tokens) - len(question_tokens)
    n_repeats = max(1, target_content // len(haystack_tokens))
    full_haystack = haystack_tokens * n_repeats

    insert_pos = int(len(full_haystack) * needle_position)
    tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:]
    tokens = tokens[: context_tokens - len(question_tokens)]
    tokens = tokens + question_tokens

    inputs = mx.array(tokens)[None]
    cache = make_cache()

    logits = model(inputs, cache=cache)
    _materialize(logits)

    # Decode up to 30 answer tokens or until newline / EOS
    answer_token_ids = []
    for _ in range(30):
        next_id = mx.argmax(logits[:, -1, :], axis=-1)
        tid = next_id.item()
        answer_token_ids.append(tid)
        logits = model(next_id[:, None], cache=cache)
        _materialize(logits)
        decoded = tokenizer.decode([tid])
        if "\n" in decoded or tid == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(answer_token_ids).strip()
    return EXPECTED_ANSWER in answer, answer


def reset_make_cache(model):
    if hasattr(model, "make_cache"):
        try:
            del model.make_cache
        except AttributeError:
            pass


def run_config(model, tokenizer, name, make_cache, context_lengths, positions):
    """Run all (context, position) cells for one config and return a results dict."""
    rows = {}
    print(f"\n  config: {name}")
    print(f"    {'context':>8} {'position':>9} {'result':>8}  answer")
    for ctx in context_lengths:
        for pos in positions:
            t0 = time.perf_counter()
            ok, answer = run_one_needle_test(
                model, tokenizer, make_cache, ctx, pos
            )
            elapsed = time.perf_counter() - t0
            mark = "PASS" if ok else "FAIL"
            short_answer = answer[:60].replace("\n", " ")
            print(
                f"    {ctx:>8} {pos:>9.1f} {mark:>8}  "
                f"{short_answer!r:<62} ({elapsed:.1f}s)"
            )
            rows[(ctx, pos)] = ok
    n_pass = sum(1 for v in rows.values() if v)
    print(f"    -> {n_pass}/{len(rows)} pass")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    p.add_argument("--contexts", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192])
    p.add_argument("--positions", type=float, nargs="+",
                   default=[0.1, 0.5, 0.9])
    args = p.parse_args()

    print(f"Loading {args.model}...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    all_results = {}

    # 1. FP16 baseline (sanity check — should be 12/12)
    reset_make_cache(model)
    all_results["baseline"] = run_config(
        model, tokenizer, "baseline (FP16)",
        lambda: make_prompt_cache(model),
        args.contexts, args.positions,
    )

    # 2. v0.5.0 default k4v2 (validates we didn't regress core compression)
    reset_make_cache(model)
    apply_turboquant(model, key_bits=4, value_bits=2, residual_window=128,
                     auto_detect_outliers=False)
    all_results["k4v2"] = run_config(
        model, tokenizer, "k4v2 (v0.5.0 default)",
        model.make_cache,
        args.contexts, args.positions,
    )

    # 3. The Phase 3 winner: k4v2 + fp16_sink_size=128
    reset_make_cache(model)
    apply_turboquant(model, key_bits=4, value_bits=2, residual_window=128,
                     fp16_sink_size=128, auto_detect_outliers=False)
    all_results["k4v2_sink128"] = run_config(
        model, tokenizer, "k4v2_sink128 (v0.6.0 best)",
        model.make_cache,
        args.contexts, args.positions,
    )

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for name, rows in all_results.items():
        n_pass = sum(1 for v in rows.values() if v)
        print(f"  {name:<32}  {n_pass}/{len(rows)}")

    # Exit code: nonzero if k4v2_sink128 doesn't pass all 12
    target = all_results["k4v2_sink128"]
    if sum(1 for v in target.values() if v) < len(target):
        print("\n  k4v2_sink128 did NOT achieve 12/12 — regression!")
        return 1
    print("\n  k4v2_sink128 achieved full retrieval — no regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
