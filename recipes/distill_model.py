"""Distill a teacher model into a smaller student.

Demonstrates knowledge distillation:

1. Train a larger "teacher" model to convergence
2. Train a smaller "student" from scratch (baseline)
3. Train the same student with distillation from the teacher
4. Compare student quality with and without distillation

The student learns from the teacher's soft probability
distributions, which encode richer information than hard
labels alone (Hinton et al., 2015).

Usage:
    uv run python recipes/distill_model.py
    uv run python recipes/distill_model.py --temperature 4 --alpha 0.7
"""

import argparse
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from lmxlab.data.batching import batch_iterator
from lmxlab.data.tokenizer import CharTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny
from lmxlab.training.config import TrainConfig
from lmxlab.training.distillation import distillation_loss
from lmxlab.training.optimizers import create_optimizer
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


def train_standard(
    model: LanguageModel,
    tokens: mx.array,
    max_steps: int,
) -> list[float]:
    """Train with standard cross-entropy loss."""
    config = TrainConfig(
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
    )
    trainer = Trainer(model, config)

    def data():
        yield from batch_iterator(
            tokens,
            batch_size=4,
            seq_len=32,
            shuffle=True,
        )

    history = trainer.train(data())
    return [m["loss"] for m in history]


def train_distilled(
    student: LanguageModel,
    teacher: LanguageModel,
    tokens: mx.array,
    max_steps: int,
    temperature: float,
    alpha: float,
) -> list[float]:
    """Train student with distillation from teacher."""
    config = TrainConfig(
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=4,
        compile_step=False,
        warmup_steps=10,
    )
    optimizer = create_optimizer(config)

    # Custom distillation training loop
    loss_and_grad = nn.value_and_grad(
        student,
        lambda s, t: distillation_loss(
            s,
            teacher,
            t,
            temperature=temperature,
            alpha=alpha,
        ),
    )

    losses = []
    for step, (x, y) in enumerate(
        batch_iterator(
            tokens,
            batch_size=4,
            seq_len=32,
            shuffle=True,
        )
    ):
        if step >= max_steps:
            break

        # Input is tokens including last position for targets
        batch = mx.concatenate([x, y[:, -1:]], axis=1)
        loss, grads = loss_and_grad(student, batch)
        optimizer.update(student, grads)
        mx.eval(loss, student.parameters(), optimizer.state)

        losses.append(loss.item())

    return losses


def main() -> None:
    """Compare distilled vs standard training."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation demo",
    )
    parser.add_argument(
        "--teacher-steps",
        type=int,
        default=300,
        help="Steps to train teacher",
    )
    parser.add_argument(
        "--student-steps",
        type=int,
        default=200,
        help="Steps to train student",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Distillation temperature (higher = softer)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss (0=pure CE, 1=pure KL)",
    )
    args = parser.parse_args()

    mx.random.seed(42)

    tokenizer = CharTokenizer(TEXT)
    tokens = mx.array(tokenizer.encode(TEXT), dtype=mx.int32)

    # --- Step 1: Train the teacher (larger model) ---
    print("=== Training Teacher (LLaMA-tiny) ===")
    teacher_config = llama_tiny()
    teacher_config = replace(
        teacher_config,
        vocab_size=tokenizer.vocab_size,
    )
    teacher = LanguageModel(teacher_config)
    mx.eval(teacher.parameters())
    print(f"  Teacher params: {teacher.count_parameters():,}")

    teacher_losses = train_standard(
        teacher,
        tokens,
        args.teacher_steps,
    )
    print(f"  Final loss: {teacher_losses[-1]:.4f}")

    # Freeze teacher (no more training)
    teacher.freeze()

    # --- Step 2: Train student WITHOUT distillation ---
    print("\n=== Training Student Baseline (GPT-tiny) ===")
    student_config = gpt_tiny()
    student_config = replace(
        student_config,
        vocab_size=tokenizer.vocab_size,
    )

    mx.random.seed(42)
    baseline_student = LanguageModel(student_config)
    mx.eval(baseline_student.parameters())
    print(f"  Student params: {baseline_student.count_parameters():,}")

    baseline_losses = train_standard(
        baseline_student,
        tokens,
        args.student_steps,
    )
    print(f"  Final loss: {baseline_losses[-1]:.4f}")

    # --- Step 3: Train student WITH distillation ---
    print(
        f"\n=== Training Student with Distillation "
        f"(T={args.temperature}, alpha={args.alpha}) ==="
    )
    mx.random.seed(42)
    distilled_student = LanguageModel(student_config)
    mx.eval(distilled_student.parameters())

    distilled_losses = train_distilled(
        distilled_student,
        teacher,
        tokens,
        args.student_steps,
        args.temperature,
        args.alpha,
    )
    print(f"  Final loss: {distilled_losses[-1]:.4f}")

    # --- Comparison ---
    print(f"\n{'=' * 50}")
    print("Results")
    print(f"{'=' * 50}")
    print(
        f"  Teacher (LLaMA):       "
        f"{teacher.count_parameters():>8,} params, "
        f"loss={teacher_losses[-1]:.4f}"
    )
    print(
        f"  Student (baseline):    "
        f"{baseline_student.count_parameters():>8,} params, "
        f"loss={baseline_losses[-1]:.4f}"
    )
    print(
        f"  Student (distilled):   "
        f"{distilled_student.count_parameters():>8,} params, "
        f"loss={distilled_losses[-1]:.4f}"
    )

    diff = baseline_losses[-1] - distilled_losses[-1]
    if diff > 0:
        print(f"\n  Distillation improved loss by {diff:.4f}")
    else:
        print(f"\n  Baseline was better by {-diff:.4f}")

    # --- Generate from all three ---
    print("\nGeneration comparison:")
    prompt = mx.array([tokenizer.encode("To be")])

    for name, model in [
        ("Teacher", teacher),
        ("Baseline", baseline_student),
        ("Distilled", distilled_student),
    ]:
        output = generate(
            model,
            prompt,
            max_tokens=60,
            temperature=0.7,
        )
        text = tokenizer.decode(output[0].tolist())
        print(f"  {name:<10} {text}")

    print(
        "\nNote: On tiny models and data, distillation benefits "
        "are small.\nThe effect grows with teacher-student size gap "
        "and dataset complexity."
    )


if __name__ == "__main__":
    main()
