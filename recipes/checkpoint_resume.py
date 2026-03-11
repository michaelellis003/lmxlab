"""Checkpoint and resume training.

Demonstrates saving and loading training checkpoints: model
weights, optimizer state, and step count. Trains for N steps,
saves a checkpoint, then resumes from it and continues training.

Usage:
    uv run python recipes/checkpoint_resume.py
    uv run python recipes/checkpoint_resume.py --steps 200 --resume-steps 100
"""

import argparse
import shutil
from dataclasses import replace

import mlx.core as mx

from lmt_metal.data.batching import batch_iterator
from lmt_metal.data.tokenizer import CharTokenizer
from lmt_metal.models.base import LanguageModel
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.training.checkpoints import load_checkpoint, save_checkpoint
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.optimizers import create_optimizer
from lmt_metal.training.trainer import Trainer

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
)

CKPT_DIR = "/tmp/lmt_metal_checkpoint_demo"


def make_batches(tokens, batch_size=4, seq_len=32):
    """Create an iterator of training batches."""
    return batch_iterator(
        tokens, batch_size=batch_size, seq_len=seq_len, shuffle=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint and resume training"
    )
    parser.add_argument(
        "--steps", type=int, default=150, help="Total training steps"
    )
    parser.add_argument(
        "--resume-steps",
        type=int,
        default=75,
        help="Steps before checkpoint",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    # --- Setup ---
    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)
    config = replace(gpt_tiny(), vocab_size=tokenizer.vocab_size)

    train_config = TrainConfig(
        learning_rate=1e-3,
        max_steps=args.steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
        log_interval=25,
    )

    # ── Phase 1: Train for resume_steps, then save ──
    print(f"Phase 1: Training for {args.resume_steps} steps...")
    model = LanguageModel(config)
    mx.eval(model.parameters())
    optimizer = create_optimizer(train_config)

    trainer = Trainer(model, train_config, optimizer=optimizer)

    step = 0
    for batch in make_batches(tokens):
        if step >= args.resume_steps:
            break
        metrics = trainer.train_step(batch)
        step += 1
        if step % 25 == 0:
            print(f"  Step {step}: loss={metrics['loss']:.4f}")

    # Save checkpoint
    save_checkpoint(
        CKPT_DIR,
        model,
        optimizer=optimizer,
        step=step,
        metadata={
            "vocab_size": tokenizer.vocab_size,
            "d_model": config.block.d_model,
        },
    )
    loss_at_save = metrics["loss"]
    print(f"Saved checkpoint at step {step} (loss={loss_at_save:.4f})")

    # ── Phase 2: Resume from checkpoint ──
    remaining = args.steps - args.resume_steps
    print(f"\nPhase 2: Resuming for {remaining} more steps...")

    # Create fresh model and optimizer
    model2 = LanguageModel(config)
    mx.eval(model2.parameters())
    optimizer2 = create_optimizer(train_config)

    # Load checkpoint
    meta = load_checkpoint(CKPT_DIR, model2, optimizer=optimizer2)
    resumed_step = meta["step"]
    print(f"Loaded checkpoint from step {resumed_step}")
    print(f"  Metadata: {meta}")

    # Continue training
    trainer2 = Trainer(model2, train_config, optimizer=optimizer2)
    trainer2.step = resumed_step

    for batch in make_batches(tokens):
        if trainer2.step >= args.steps:
            break
        metrics = trainer2.train_step(batch)
        if trainer2.step % 25 == 0:
            print(f"  Step {trainer2.step}: loss={metrics['loss']:.4f}")

    final_loss = metrics["loss"]

    # ── Summary ──
    print("\nSummary:")
    ckpt_step = args.resume_steps
    print(f"  Loss at checkpoint (step {ckpt_step}): {loss_at_save:.4f}")
    print(f"  Final loss (step {args.steps}): {final_loss:.4f}")
    print(f"  Checkpoint dir: {CKPT_DIR}")

    # Cleanup
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    print("  Cleaned up checkpoint files")
    print("\nDone!")


if __name__ == "__main__":
    main()
