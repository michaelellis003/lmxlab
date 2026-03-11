"""Benchmark mx.compile speedup on training steps.

Measures the wall-clock time per training step with and without
mx.compile, across different model sizes. This is one of the key
MLX-specific experiments: compilation fuses the forward + backward +
optimizer update into a single optimized graph.

Usage:
    uv run python recipes/benchmark_compile.py
    uv run python recipes/benchmark_compile.py --steps 50 --sizes tiny small
"""

import argparse

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.experiments.profiling import benchmark_fn
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.llama import llama_config
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
) * 20

MODEL_SIZES = {
    "tiny": {
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 2,
        "d_ff": 128,
    },
    "small": {
        "d_model": 128,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": 4,
        "d_ff": 256,
    },
    "medium": {
        "d_model": 256,
        "n_heads": 8,
        "n_kv_heads": 4,
        "n_layers": 6,
        "d_ff": 512,
    },
}


def time_training(config, tokens, compile_step, n_steps):
    """Time n_steps of training."""
    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=n_steps,
        batch_size=4,
        compile_step=compile_step,
        warmup_steps=0,
    )

    model = LanguageModel(config)
    mx.eval(model.parameters())
    trainer = Trainer(model, train_config)

    batches = list(
        batch_iterator(tokens, batch_size=4, seq_len=32, shuffle=False)
    )
    # Take enough batches for our steps
    batches = batches[:n_steps]

    def run():
        # Reset model for fair comparison
        nonlocal trainer, model
        model = LanguageModel(config)
        mx.eval(model.parameters())
        trainer = Trainer(model, train_config)
        for batch in batches:
            trainer.train_step(batch)

    result = benchmark_fn(run, n_warmup=1, n_iter=3)
    return result


def main() -> None:
    """Benchmark compile vs no-compile."""
    parser = argparse.ArgumentParser(
        description="Benchmark mx.compile speedup"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Training steps per measurement",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(MODEL_SIZES.keys()),
        choices=list(MODEL_SIZES.keys()),
        help="Model sizes to benchmark",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    vocab = tokenizer.vocab_size

    print("=" * 65)
    print("mx.compile Training Step Benchmark")
    print("=" * 65)
    print(f"Steps per measurement: {args.steps}")
    print("3 iterations per config (mean reported)\n")

    print(
        f"{'Size':<10} {'Params':>10} "
        f"{'No compile':>12} {'Compiled':>12} {'Speedup':>10}"
    )
    print("-" * 58)

    for size_name in args.sizes:
        dims = MODEL_SIZES[size_name]
        config = llama_config(
            vocab_size=vocab,
            max_seq_len=128,
            tie_embeddings=True,
            **dims,
        )

        model = LanguageModel(config)
        mx.eval(model.parameters())
        params = model.count_parameters()

        # Benchmark without compile
        no_compile = time_training(
            config, tokens, compile_step=False, n_steps=args.steps
        )

        # Benchmark with compile
        compiled = time_training(
            config, tokens, compile_step=True, n_steps=args.steps
        )

        # Per-step times
        nc_per_step = no_compile["mean_ms"] / args.steps
        c_per_step = compiled["mean_ms"] / args.steps
        speedup = nc_per_step / c_per_step if c_per_step > 0 else 0

        print(
            f"{size_name:<10} {params:>10,} "
            f"{nc_per_step:>10.2f}ms "
            f"{c_per_step:>10.2f}ms "
            f"{speedup:>9.2f}x"
        )

    print("\n" + "=" * 65)
    print("Key insights:")
    print("  - Compilation traces the step once and reuses the graph")
    print("  - Larger models benefit more from kernel fusion")
    print("  - First compiled step is slower (tracing), subsequent are faster")
    print("  - Compilation is the default in Trainer (compile_step=True)")
    print("=" * 65)


if __name__ == "__main__":
    main()
