"""μP coordinate check: validate HP transfer across widths.

The standard μP validation (Yang et al. 2022): train models at
multiple widths with the SAME base learning rate. Under μP, loss
curves should be similar. Under SP, wider models diverge or
converge differently because the optimal LR changes with width.

Usage:
    uv run python recipes/mup_coordinate_check.py
    uv run python recipes/mup_coordinate_check.py --steps 300
"""

import argparse
import time
import urllib.request
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.training.config import TrainConfig
from lmxlab.training.optimizers import (
    create_mup_optimizer,
    create_optimizer,
)
from lmxlab.training.trainer import Trainer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/shakespeare.txt")

BASE_WIDTH = 64
WIDTHS = [64, 128, 256]
BASE_LR = 1e-3
N_LAYERS = 4
SEED = 42
BATCH_SIZE = 8
SEQ_LEN = 128
EVAL_INTERVAL = 50


def download_shakespeare() -> str:
    """Download Shakespeare text if not cached."""
    if DATA_PATH.exists():
        return DATA_PATH.read_text()
    print("Downloading Shakespeare text...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


def build_config(
    vocab_size: int,
    d_model: int,
    mup_base_width: int | None = None,
) -> ModelConfig:
    """Build a GPT config at the given width.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model hidden dimension (width).
        mup_base_width: Base width for μP. None = SP.

    Returns:
        ModelConfig.
    """
    n_heads = max(2, d_model // 32)
    d_ff = d_model * 4
    block = BlockConfig(
        attention="mha",
        ffn="standard",
        norm="layer_norm",
        position="sinusoidal",
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        bias=True,
        max_seq_len=SEQ_LEN,
        pre_norm=True,
        mup=mup_base_width is not None,
    )
    return ModelConfig(
        block=block,
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        tie_embeddings=True,
        mup_base_width=mup_base_width,
    )


def evaluate(
    model: LanguageModel,
    val_tokens: mx.array,
) -> float:
    """Run evaluation pass, return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in batch_iterator(
        val_tokens,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        shuffle=False,
    ):
        logits, _ = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = y.reshape(-1)
        loss = nn.losses.cross_entropy(
            logits,
            targets,
            reduction="mean",
        )
        mx.eval(loss)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_one(
    config: ModelConfig,
    train_tokens: mx.array,
    val_tokens: mx.array,
    lr: float,
    max_steps: int,
    seed: int,
    use_mup: bool,
) -> dict:
    """Train one model and return results.

    Args:
        config: Model configuration.
        train_tokens: Training token array.
        val_tokens: Validation token array.
        lr: Learning rate.
        max_steps: Maximum training steps.
        seed: Random seed.
        use_mup: Whether to use μP optimizer.

    Returns:
        Dict with training results.
    """
    mx.random.seed(seed)
    model = LanguageModel(config)
    n_params = model.count_parameters()

    train_cfg = TrainConfig(
        learning_rate=lr,
        warmup_steps=max(1, max_steps // 20),
        max_steps=max_steps,
        batch_size=BATCH_SIZE,
        max_grad_norm=1.0,
        lr_schedule="cosine",
        compile_step=False,
    )

    if use_mup and config.mup_base_width is not None:
        optimizer = create_mup_optimizer(
            train_cfg,
            config.width_mult,
        )
    else:
        optimizer = create_optimizer(train_cfg)

    trainer = Trainer(
        model,
        train_cfg,
        optimizer=optimizer,
    )

    # Training loop
    losses = []
    eval_losses = []
    t0 = time.time()

    for step_i, (x, y) in enumerate(
        batch_iterator(
            train_tokens,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            shuffle=True,
        )
    ):
        if step_i >= max_steps:
            break
        metrics = trainer.train_step((x, y))
        losses.append(metrics["loss"])

        if (step_i + 1) % EVAL_INTERVAL == 0:
            val_loss = evaluate(model, val_tokens)
            eval_losses.append(val_loss)

    elapsed = time.time() - t0

    # Final eval
    final_val = evaluate(model, val_tokens)
    eval_losses.append(final_val)
    best_val = min(eval_losses)

    return {
        "n_params": n_params,
        "steps": len(losses),
        "final_train_loss": losses[-1] if losses else float("nan"),
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "elapsed": elapsed,
        "losses": losses,
        "eval_losses": eval_losses,
    }


def print_results_table(
    results: dict[str, dict[str, dict]],
) -> None:
    """Print comparison table.

    Args:
        results: Nested dict [mode][width] -> result dict.
    """
    print()
    print("=" * 70)
    print("μP Coordinate Check Results")
    print("=" * 70)
    print(
        f"{'Mode':<6} {'Width':>6} {'Params':>8} "
        f"{'Train':>8} {'Val':>8} {'Best':>8} "
        f"{'Time':>6}"
    )
    print("-" * 70)

    for mode in ["SP", "μP"]:
        for width in WIDTHS:
            key = f"{mode}-{width}"
            if key not in results:
                continue
            r = results[key]
            print(
                f"{mode:<6} {width:>6} "
                f"{r['n_params']:>8,} "
                f"{r['final_train_loss']:>8.4f} "
                f"{r['final_val_loss']:>8.4f} "
                f"{r['best_val_loss']:>8.4f} "
                f"{r['elapsed']:>5.1f}s"
            )
        print("-" * 70)

    # Analysis: check if val loss spread is tighter under μP
    sp_vals = [
        results[f"SP-{w}"]["best_val_loss"]
        for w in WIDTHS
        if f"SP-{w}" in results
    ]
    mup_vals = [
        results[f"μP-{w}"]["best_val_loss"]
        for w in WIDTHS
        if f"μP-{w}" in results
    ]

    if len(sp_vals) >= 2 and len(mup_vals) >= 2:
        sp_spread = max(sp_vals) - min(sp_vals)
        mup_spread = max(mup_vals) - min(mup_vals)
        print()
        print("Val loss spread across widths:")
        print(f"  SP:  {sp_spread:.4f}")
        print(f"  μP:  {mup_spread:.4f}")

        if mup_spread < sp_spread:
            ratio = sp_spread / max(mup_spread, 1e-8)
            print(f"  μP is {ratio:.1f}x tighter — LR transfers better!")
        else:
            print(
                "  μP is NOT tighter — LR may not be transferring as expected."
            )
    print()


def main() -> None:
    """Run the μP coordinate check."""
    parser = argparse.ArgumentParser(
        description="μP coordinate check",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Training steps per run (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Load data
    text = download_shakespeare()
    tokenizer = CharTokenizer(text)
    tokens = mx.array(tokenizer.encode(text))
    vocab_size = tokenizer.vocab_size

    # 90/10 train/val split
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    print(f"Base LR: {BASE_LR}")
    print(f"Base width: {BASE_WIDTH}")
    print(f"Widths: {WIDTHS}")
    print(f"Steps: {args.steps}")
    print()

    results = {}
    total_runs = len(WIDTHS) * 2  # SP + μP
    run_i = 0

    for mode in ["SP", "μP"]:
        for width in WIDTHS:
            run_i += 1
            use_mup = mode == "μP"
            mup_base = BASE_WIDTH if use_mup else None
            width_mult = width / BASE_WIDTH

            config = build_config(
                vocab_size,
                width,
                mup_base,
            )
            n_params = LanguageModel(config).count_parameters()

            lr_label = (
                f"{BASE_LR:.0e}"
                if not use_mup or width_mult == 1.0
                else (
                    f"{BASE_LR:.0e} (embed) / "
                    f"{BASE_LR / width_mult:.0e} (hidden)"
                )
            )

            print(
                f"[{run_i}/{total_runs}] "
                f"{mode} d={width} "
                f"({n_params:,} params) "
                f"lr={lr_label}"
            )

            result = train_one(
                config,
                train_tokens,
                val_tokens,
                lr=BASE_LR,
                max_steps=args.steps,
                seed=args.seed,
                use_mup=use_mup,
            )

            key = f"{mode}-{width}"
            results[key] = result

            print(
                f"  train={result['final_train_loss']:.4f} "
                f"val={result['best_val_loss']:.4f} "
                f"({result['elapsed']:.1f}s)"
            )

    print_results_table(results)


if __name__ == "__main__":
    main()
