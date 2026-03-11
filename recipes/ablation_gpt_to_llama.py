"""Architecture ablation: GPT → LLaMA one feature at a time.

Adds LLaMA features to a GPT baseline one-by-one and measures
their individual and combined effects on training loss. This
demonstrates how the ConfigurableBlock system makes ablation
studies trivial — just change string names in BlockConfig.

Features tested (cumulative):
  1. GPT baseline (MHA + LayerNorm + Standard FFN + Sinusoidal)
  2. + RMSNorm (replace LayerNorm)
  3. + RoPE (replace sinusoidal position encoding)
  4. + GatedFFN/SwiGLU (replace standard FFN)
  5. + GQA (replace MHA with grouped-query attention)
  6. + No bias (remove all bias terms)
  = Full LLaMA

Usage:
    uv run python recipes/ablation_gpt_to_llama.py
    uv run python recipes/ablation_gpt_to_llama.py --steps 200
"""

import argparse
from dataclasses import replace

import mlx.core as mx

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

TEXT = (
    "To be, or not to be, that is the question: "
    "Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, "
    "Or to take arms against a sea of troubles, "
    "And by opposing end them. To die, to sleep; "
    "No more; and by a sleep to say we end "
    "The heart-ache and the thousand natural shocks "
    "That flesh is heir to: 'tis a consummation "
    "Devoutly to be wish'd. To die, to sleep; "
    "To sleep, perchance to dream. "
) * 5


def build_ablation_configs(vocab_size: int) -> list[tuple[str, ModelConfig]]:
    """Build a series of configs adding LLaMA features one at a time."""
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 128
    max_seq_len = 128

    # 1. GPT baseline
    gpt_block = BlockConfig(
        attention="mha",
        ffn="standard",
        norm="layer_norm",
        position="sinusoidal",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=True,
        max_seq_len=max_seq_len,
        pre_norm=True,
    )

    # 2. + RMSNorm
    rmsnorm_block = replace(gpt_block, norm="rms_norm")

    # 3. + RoPE
    rope_block = replace(rmsnorm_block, position="rope")

    # 4. + Gated FFN (SwiGLU)
    gated_block = replace(rope_block, ffn="gated")

    # 5. + GQA (2 KV heads for 4 query heads)
    gqa_block = replace(gated_block, attention="gqa", n_kv_heads=2)

    # 6. + No bias = full LLaMA
    llama_block = replace(gqa_block, bias=False)

    configs = [
        ("GPT baseline", gpt_block),
        ("+ RMSNorm", rmsnorm_block),
        ("+ RoPE", rope_block),
        ("+ SwiGLU FFN", gated_block),
        ("+ GQA", gqa_block),
        ("+ No bias (=LLaMA)", llama_block),
    ]

    return [
        (
            name,
            ModelConfig(
                block=block,
                vocab_size=vocab_size,
                n_layers=n_layers,
                tie_embeddings=True,
            ),
        )
        for name, block in configs
    ]


def train_one(
    name: str,
    config: ModelConfig,
    tokens: mx.array,
    train_config: TrainConfig,
) -> list[float]:
    """Train a model and return loss history."""
    model = LanguageModel(config)
    mx.eval(model.parameters())
    params = model.count_parameters()
    print(f"  {name}: {params:,} params")

    trainer = Trainer(model, train_config)

    def data_iter():
        yield from batch_iterator(
            tokens,
            batch_size=train_config.batch_size,
            seq_len=32,
            shuffle=True,
        )

    history = trainer.train(data_iter())
    return [m["loss"] for m in history]


def main() -> None:
    """Run the ablation study."""
    parser = argparse.ArgumentParser(description="GPT→LLaMA ablation study")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Training steps per config",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds",
    )
    args = parser.parse_args()

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    configs = build_ablation_configs(tokenizer.vocab_size)

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=5,
    )

    print("=" * 60)
    print("Architecture Ablation: GPT → LLaMA")
    print(f"{args.steps} steps, {args.seeds} seed(s)")
    print("=" * 60)

    all_results: dict[str, list[list[float]]] = {}

    for seed in range(args.seeds):
        print(f"\nSeed {seed + 42}:")
        mx.random.seed(seed + 42)
        for name, config in configs:
            mx.random.seed(seed + 42)
            losses = train_one(name, config, tokens, train_config)
            all_results.setdefault(name, []).append(losses)

    # Results table
    print("\n" + "=" * 60)
    print("Final Loss Comparison")
    print("=" * 60)
    print(f"{'Config':<25} {'Final Loss':>12} {'Improvement':>12}")
    print("-" * 50)

    baseline_loss = None
    for name, _ in configs:
        runs = all_results[name]
        avg_final = sum(r[-1] for r in runs) / len(runs)

        if baseline_loss is None:
            baseline_loss = avg_final
            imp = ""
        else:
            delta = baseline_loss - avg_final
            imp = f"{delta:+.4f}"

        print(f"{name:<25} {avg_final:>12.4f} {imp:>12}")

    # Loss curve comparison (every N steps)
    print("\n" + "=" * 60)
    print("Loss Curves (first seed, every 10 steps)")
    print("=" * 60)

    header = f"{'Step':>5}"
    short_names = []
    for name, _ in configs:
        short = name.replace("+ ", "").split("(")[0].strip()[:8]
        header += f"  {short:>10}"
        short_names.append(short)
    print(header)
    print("-" * len(header))

    n_steps = min(len(all_results[name][0]) for name, _ in configs)
    for step in range(0, n_steps, 10):
        row = f"{step + 1:>5}"
        for name, _ in configs:
            loss = all_results[name][0][step]
            row += f"  {loss:>10.4f}"
        print(row)

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("  Each row adds ONE LLaMA feature to the GPT baseline.")
    print("  This reveals which architectural choices matter most")
    print("  for convergence on this task.")
    print("  Same LanguageModel class — only BlockConfig changes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
