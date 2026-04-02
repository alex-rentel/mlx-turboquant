"""Needle-in-a-haystack test for KV cache compression quality.

Hides a fact in a long document and tests if the model can retrieve it
at various context lengths and needle positions.
"""

import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from mlx_turboquant import apply_turboquant


HAYSTACK_UNIT = (
    "The weather in various cities across the world varies greatly depending on "
    "the season and geographical location. Temperature, humidity, and precipitation "
    "patterns are influenced by ocean currents, altitude, and latitude. Climate "
    "scientists study these patterns to understand long-term trends. "
)

NEEDLE = "The secret password for the vault is 'mango-sunset-42'."

QUESTION = "What is the secret password for the vault?"
EXPECTED_ANSWER = "mango-sunset-42"


def run_needle_test(model, tokenizer, cache_factory, context_tokens: int,
                    needle_position: float) -> bool:
    """Run a single needle-in-haystack test.

    Args:
        model: The language model
        tokenizer: Tokenizer
        cache_factory: Callable that returns a list of cache objects
        context_tokens: Target context length in tokens
        needle_position: Where to place needle (0.0 = start, 1.0 = end)

    Returns:
        True if the model retrieved the needle correctly
    """
    # Build haystack
    haystack_tokens = tokenizer.encode(HAYSTACK_UNIT)
    needle_tokens = tokenizer.encode(NEEDLE)
    question_tokens = tokenizer.encode("\n\nQuestion: " + QUESTION + "\nAnswer:")

    # Calculate how many haystack repetitions we need
    target_content = context_tokens - len(needle_tokens) - len(question_tokens)
    n_repeats = max(1, target_content // len(haystack_tokens))
    full_haystack = haystack_tokens * n_repeats

    # Insert needle at specified position
    insert_pos = int(len(full_haystack) * needle_position)
    tokens = full_haystack[:insert_pos] + needle_tokens + full_haystack[insert_pos:]
    tokens = tokens[:context_tokens - len(question_tokens)]
    tokens = tokens + question_tokens

    inputs = mx.array(tokens)[None]
    cache = cache_factory()

    # Forward pass
    logits = model(inputs, cache=cache)
    mx.eval(logits)

    # Decode answer
    answer_tokens = []
    for _ in range(30):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        answer_tokens.append(next_token.item())
        logits = model(next_token[:, None], cache=cache)
        mx.eval(logits)
        # Stop at newline or EOS
        decoded = tokenizer.decode([next_token.item()])
        if "\n" in decoded or next_token.item() == tokenizer.eos_token_id:
            break

    answer = tokenizer.decode(answer_tokens).strip()
    return EXPECTED_ANSWER in answer


def needle_haystack_benchmark(model_path: str,
                               context_lengths: list[int] = [512, 1024, 2048],
                               needle_positions: list[float] = [0.1, 0.5, 0.9]):
    """Full needle-in-haystack benchmark."""
    print(f"\n{'='*60}")
    print(f"Needle-in-a-Haystack Test")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")

    model, tokenizer = load(model_path)
    num_layers = len(model.model.layers)

    configs = [
        ("FP16", lambda: [KVCache() for _ in range(num_layers)]),
    ]

    for k_bits, v_bits in [(4, 4), (4, 2), (3, 2)]:
        name = f"TQ K{k_bits}/V{v_bits}"
        apply_turboquant(model, key_bits=k_bits, value_bits=v_bits,
                        residual_window=64)

        def make_tq_cache(kb=k_bits, vb=v_bits):
            apply_turboquant(model, key_bits=kb, value_bits=vb, residual_window=64)
            return model.make_cache()

        configs.append((name, make_tq_cache))

    # Print header
    header = f"{'Context':>8} {'Position':>8}"
    for name, _ in configs:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for ctx_len in context_lengths:
        for pos in needle_positions:
            row = f"{ctx_len:>8} {pos:>8.1f}"
            for name, cache_factory in configs:
                found = run_needle_test(model, tokenizer, cache_factory, ctx_len, pos)
                row += f" {'PASS' if found else 'FAIL':>12}"
                key = (name, ctx_len, pos)
                results[key] = found
            print(row)

    # Summary
    print(f"\n{'='*60}")
    for name, _ in configs:
        total = sum(1 for k, v in results.items() if k[0] == name)
        passed = sum(1 for k, v in results.items() if k[0] == name and v)
        print(f"{name}: {passed}/{total} ({100*passed/total:.0f}%)")

    return results


if __name__ == "__main__":
    needle_haystack_benchmark("mlx-community/Qwen2.5-7B-Instruct-4bit")
