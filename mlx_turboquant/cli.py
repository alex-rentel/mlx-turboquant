"""CLI entry points for mlx-turboquant."""

import argparse
import sys
import time

import mlx.core as mx


def run_generate(args):
    """Generate text with TurboQuant KV cache compression."""
    from mlx_lm import load

    from .patch import apply_turboquant

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

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
        contexts = [int(x) for x in args.contexts.split(",")]
        measure_cache_memory(args.model, contexts)

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
    gen.add_argument("--max-tokens", type=int, default=200)
    gen.add_argument("--key-bits", type=int, default=4)
    gen.add_argument("--value-bits", type=int, default=2)
    gen.add_argument("--residual-window", type=int, default=128)

    # Benchmark command
    bench = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench.add_argument("--model", required=True, help="Model path or HF repo")
    bench.add_argument("--benchmarks", nargs="+", default=["all"],
                       choices=["quality", "memory", "speed", "all"])
    bench.add_argument("--contexts", default="512,1024,2048,4096")

    args = parser.parse_args()
    if args.command == "generate":
        run_generate(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
