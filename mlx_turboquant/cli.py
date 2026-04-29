"""CLI entry points for mlx-turboquant."""

import argparse
import sys
import time

import mlx.core as mx


def _positive_int(value: str) -> int:
    """argparse type: accept positive ints, reject 0 / negative / non-int."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {ivalue}")
    return ivalue


def _quantize_bits(value: str) -> float:
    """argparse type: accept the bit-width values supported by apply_turboquant."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a number, got {value!r}"
        ) from exc
    valid = {2, 3, 3.5, 4}
    if fvalue not in valid:
        raise argparse.ArgumentTypeError(
            f"bit-width must be one of 2, 3, 3.5, 4 — got {value!r}"
        )
    return fvalue


def _csv_positive_ints(value: str) -> list[int]:
    """argparse type: parse a comma-separated list of positive ints."""
    if not value:
        raise argparse.ArgumentTypeError("expected a non-empty CSV list of positive ints")
    parts = [p.strip() for p in value.split(",")]
    result: list[int] = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"all entries must be integers; got {p!r}"
            ) from exc
        if n <= 0:
            raise argparse.ArgumentTypeError(
                f"all entries must be positive; got {n}"
            )
        result.append(n)
    return result


def run_generate(args):
    """Generate text with TurboQuant KV cache compression."""
    from mlx_lm import load

    from .patch import apply_turboquant

    print(f"Loading model: {args.model}")
    # mlx-lm's load() returns either (model, tokenizer) or
    # (model, tokenizer, config) depending on version; index the first
    # two slots explicitly so the type checker sees a clean assignment.
    loaded = load(args.model)
    model = loaded[0]
    tokenizer = loaded[1]

    print(f"Applying TurboQuant: K{args.key_bits}/V{args.value_bits}, "
          f"residual_window={args.residual_window}")
    apply_turboquant(
        model,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        residual_window=args.residual_window,
    )

    cache = model.make_cache()
    inputs = mx.array(tokenizer.encode(args.prompt))[None]

    # Prefill
    t0 = time.perf_counter()
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    t_prefill = time.perf_counter() - t0

    # Decode
    tokens = []
    t_decode_start = time.perf_counter()
    for _ in range(args.max_tokens):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        tokens.append(token_id)
        logits = model(next_token[:, None], cache=cache)
        mx.eval(logits)

    t_decode = time.perf_counter() - t_decode_start
    output = tokenizer.decode(tokens)

    print("\n--- Generation ---")
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output}")
    print("\n--- Stats ---")
    print(f"Prefill: {inputs.shape[1]} tokens in {t_prefill:.2f}s "
          f"({inputs.shape[1]/t_prefill:.0f} tok/s)")
    print(f"Decode: {len(tokens)} tokens in {t_decode:.2f}s "
          f"({len(tokens)/t_decode:.0f} tok/s)")

    # Memory stats
    tq_bytes = sum(c.nbytes for c in cache if hasattr(c, 'nbytes'))
    print(f"Cache memory: {tq_bytes / 1e6:.1f} MB")


def run_benchmark(args):
    """Run benchmarks."""
    sys.path.insert(0, ".")

    if "quality" in args.benchmarks or "all" in args.benchmarks:
        from benchmarks.bench_quality import cosine_similarity_benchmark
        cosine_similarity_benchmark(args.model)

    if "memory" in args.benchmarks or "all" in args.benchmarks:
        from benchmarks.bench_memory import measure_cache_memory
        # args.contexts is already a list[int] thanks to _csv_positive_ints.
        measure_cache_memory(args.model, args.contexts)

    if "speed" in args.benchmarks or "all" in args.benchmarks:
        from benchmarks.bench_speed import benchmark_speed
        benchmark_speed(args.model)


def main():
    parser = argparse.ArgumentParser(
        description="mlx-turboquant: KV cache quantization for Apple Silicon"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Generate command
    gen = subparsers.add_parser("generate", help="Generate text with TurboQuant")
    gen.add_argument("--model", required=True, help="Model path or HF repo")
    gen.add_argument("--prompt", required=True, help="Input prompt")
    gen.add_argument("--max-tokens", type=_positive_int, default=200)
    gen.add_argument("--key-bits", type=_quantize_bits, default=4,
                     help="Key quantization bits: 2, 3, 3.5, or 4")
    gen.add_argument("--value-bits", type=_quantize_bits, default=2,
                     help="Value quantization bits: 2, 3, 3.5, or 4")
    gen.add_argument("--residual-window", type=_positive_int, default=128)

    # Benchmark command
    bench = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench.add_argument("--model", required=True, help="Model path or HF repo")
    bench.add_argument("--benchmarks", nargs="+", default=["all"],
                       choices=["quality", "memory", "speed", "all"])
    bench.add_argument("--contexts", type=_csv_positive_ints,
                       default=[512, 1024, 2048, 4096],
                       help="Comma-separated context lengths, e.g. 512,1024,2048")

    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
