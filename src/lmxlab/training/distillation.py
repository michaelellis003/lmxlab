"""Knowledge distillation training losses.

Implements teacher-student distillation where a smaller student
model learns from a larger teacher model's soft probability
distributions (Hinton et al., 2015).

The key insight: soft targets carry more information than hard
labels. A teacher assigning 0.7 to "cat" and 0.2 to "kitten"
teaches the student about word similarity, not just correctness.

Supported modes:

- **Logit distillation**: KL divergence between temperature-scaled
  teacher and student logits. The standard approach.
- **Combined loss**: Weighted mix of distillation loss and standard
  cross-entropy on hard targets. Balances learning from teacher
  with learning from ground truth.

Example::

    from lmxlab.training.distillation import distillation_loss

    # Teacher is frozen, student is trained
    loss = distillation_loss(
        student, teacher, tokens,
        temperature=4.0, alpha=0.7,
    )
"""

import mlx.core as mx
import mlx.nn as nn

from lmxlab.models.base import LanguageModel


def distillation_loss(
    student: LanguageModel,
    teacher: LanguageModel,
    tokens: mx.array,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> mx.array:
    """Compute combined distillation + hard-target loss.

    Loss = alpha * KL(teacher || student) * T^2
         + (1 - alpha) * CE(student, targets)

    The T^2 factor compensates for the gradient magnitude
    reduction caused by temperature scaling (Hinton et al.).

    Args:
        student: Student model (being trained).
        teacher: Teacher model (frozen, no gradients).
        tokens: Input token IDs (batch, seq_len). Targets are
            tokens shifted by one position.
        temperature: Softmax temperature for soft targets.
            Higher = softer distributions, more knowledge
            transfer. Typical values: 2-10.
        alpha: Weight for distillation loss (0-1). Higher means
            more reliance on teacher, less on hard targets.

    Returns:
        Scalar combined loss.
    """
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    # Student forward pass (will receive gradients)
    student_logits, _ = student(inputs)

    # Teacher forward pass (no gradients needed)
    teacher_logits, _ = teacher(inputs)

    # Distillation component: KL divergence on soft targets
    kl = soft_target_loss(student_logits, teacher_logits, temperature)

    if alpha >= 1.0:
        return kl

    # Hard target component: standard cross-entropy
    ce = _cross_entropy(student_logits, targets)

    return alpha * kl + (1.0 - alpha) * ce


def soft_target_loss(
    student_logits: mx.array,
    teacher_logits: mx.array,
    temperature: float = 4.0,
) -> mx.array:
    """KL divergence between temperature-scaled distributions.

    KL(teacher || student) computed on softened logits.
    Multiplied by T^2 to maintain gradient scale.

    Args:
        student_logits: Student output (batch, seq_len, vocab).
        teacher_logits: Teacher output (batch, seq_len, vocab).
        temperature: Softmax temperature.

    Returns:
        Scalar KL divergence loss.
    """
    # Temperature-scaled log-softmax
    student_log_probs = nn.log_softmax(
        student_logits / temperature,
        axis=-1,
    )
    teacher_log_probs = nn.log_softmax(
        teacher_logits / temperature,
        axis=-1,
    )

    # KL(P || Q) = sum(P * (log P - log Q))
    teacher_probs = mx.exp(teacher_log_probs)
    kl = mx.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        axis=-1,
    )

    # T^2 scaling (Hinton et al.)
    return mx.mean(kl) * (temperature**2)


def _cross_entropy(
    logits: mx.array,
    targets: mx.array,
) -> mx.array:
    """Standard cross-entropy loss for language modeling.

    Args:
        logits: Model logits (batch, seq_len, vocab).
        targets: Target token IDs (batch, seq_len).

    Returns:
        Scalar cross-entropy loss.
    """
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    return nn.losses.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="mean",
    )
