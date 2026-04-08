"""Long-context chat demo — see KV cache memory stay small under compression.

Loads Qwen3-8B with TurboQuant K4/V2 + FP16 sink of 128 tokens, runs a
multi-turn conversation, and prints KV cache memory usage after each turn.

Run:
    python examples/long_context_chat.py

Requires:
    pip install mlx-lm mlx-turboquant
    # ~4.5 GB model weights on first run

The point of the demo: on an 8B model at 4K context, the FP16 KV cache
eats ~576 MB. With the default TurboQuant config it drops to ~190 MB.
Watch the `kv ~` number in the prompt — it scales linearly with turn
count but ~3.6× slower than it would without compression.
"""

import sys

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from mlx_turboquant import apply_turboquant


MODEL = "mlx-community/Qwen3-8B-4bit"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep answers concise — aim for 2-3 "
    "sentences unless the user explicitly asks for more detail."
)

DEMO_TURNS = [
    "Explain the difference between wave-particle duality and the "
    "uncertainty principle in quantum mechanics.",
    "Could you give a concrete example of each?",
    "How does entanglement relate to either of those?",
    "If I wanted to simulate these on a classical computer, which "
    "would be hardest and why?",
    "Summarize everything you've told me in this conversation so far "
    "in three bullet points.",
]


def kv_cache_mb(cache) -> float:
    """Sum .nbytes across all layer caches — works for both FP16 KVCache
    and TurboQuantKVCache."""
    total = 0
    for c in cache:
        # Both cache types expose .nbytes; TurboQuantKVCache via @property,
        # mlx-lm's KVCache via an attribute computed from .keys/.values.
        nbytes = getattr(c, "nbytes", None)
        if nbytes is None and hasattr(c, "keys") and c.keys is not None:
            nbytes = c.keys.nbytes + c.values.nbytes
        total += nbytes or 0
    return total / (1024 * 1024)


def main():
    print(f"Loading {MODEL}...", flush=True)
    model, tokenizer = load(MODEL)

    print("Applying TurboQuant (K4/V2 + sink128)...", flush=True)
    apply_turboquant(
        model,
        key_bits=4,
        value_bits=2,
        residual_window=128,
        fp16_sink_size=128,
    )

    cache = model.make_cache()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    sampler = make_sampler(temp=0.7)

    print(f"\n{'=' * 68}")
    print(f"Long-context chat demo — {MODEL}")
    print(f"Config: key_bits=4 value_bits=2 sink=128 residual=128")
    print(f"{'=' * 68}\n")

    for turn_idx, user_text in enumerate(DEMO_TURNS, start=1):
        messages.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        print(f"--- turn {turn_idx} / {len(DEMO_TURNS)} ---")
        print(f"user: {user_text}\n")
        print("assistant: ", end="", flush=True)

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=sampler,
            prompt_cache=cache,
            verbose=False,
        )
        print(response.strip())
        messages.append({"role": "assistant", "content": response.strip()})

        kv_mb = kv_cache_mb(cache)
        ctx_tokens = sum(getattr(c, "offset", 0) for c in cache) // len(cache)
        print(f"\n  [kv ~ {kv_mb:6.1f} MB  |  ctx ~ {ctx_tokens:5d} tokens]\n")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
