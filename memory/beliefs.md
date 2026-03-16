# Belief Tracker

Bayesian-inspired belief state for key research questions.
Updated after each `/interpret` cycle with evidence and reasoning.

## Format

Each belief has a prior probability, evidence log, and current
posterior. Probabilities are subjective credences, not frequentist
p-values.

---

## B-001: Unified memory changes optimal training strategies

**Prior:** 0.60 (moderately likely)
**Current:** 0.60
**Source:** devlog "Unified Memory Changes the Trade-offs"

Unified memory eliminates CPU-GPU transfer overhead. Open questions:
does gradient accumulation matter less? Do different optimizers win?

| Date | Evidence | Direction | Updated to |
|------|----------|-----------|------------|
| — | Prior from devlog analysis | — | 0.60 |

---

## B-002: mx.compile provides >1.5x speedup for typical models

**Prior:** 0.70 (likely, based on 1.3-2x reported range)
**Current:** 0.70
**Source:** devlog "mx.compile Is Not Free"

Compilation gives 1.3-2x speedup but constrains control flow.
Small models may not benefit due to tracing overhead.

| Date | Evidence | Direction | Updated to |
|------|----------|-----------|------------|
| — | Prior from devlog (1.3-2x typical) | — | 0.70 |

---

## B-003: LoRA/QLoRA economics differ on unified memory

**Prior:** 0.55 (slightly likely)
**Current:** 0.55
**Source:** devlog "LoRA and QLoRA on Unified Memory"

Memory savings still matter (shared with OS), but performance
characteristics may differ without cross-device transfers.

| Date | Evidence | Direction | Updated to |
|------|----------|-----------|------------|
| — | Prior from devlog analysis | — | 0.55 |

---

## B-004: Behavioral tests are sufficient for ML code correctness

**Prior:** 0.75 (likely)
**Current:** 0.75
**Source:** devlog "Testing ML Code"

Invariance, directional, shape, and boundary tests cover the
important properties. Standard assertEqual doesn't work for
stochastic outputs.

| Date | Evidence | Direction | Updated to |
|------|----------|-----------|------------|
| — | Prior from testing experience | — | 0.75 |

---

## B-005: Config factories are better than class hierarchies for architectures

**Prior:** 0.85 (very likely)
**Current:** 0.85
**Source:** devlog "Config Factories Over Class Hierarchies"

8 architectures implemented, all as config factories. Trade-off:
harder to grep for individual architectures.

| Date | Evidence | Direction | Updated to |
|------|----------|-----------|------------|
| — | Prior from implementation experience | — | 0.85 |

---

## B-006: LLaMA features improve training over GPT at small scale

**Prior:** 0.70 (likely — modern features should help)
**Current:** 0.30
**Prior strength:** WEAK — all prior evidence is at >1B scale
(scale transfer tax ~50%). Should have been ~0.40 initially.
**Source:** HYP-001 experiment results + literature review

Initially expected modern LLaMA features to provide clear
improvements over GPT baseline even at small scale (d_model=256,
6 layers, char-level tokenization). HYP-001 showed the opposite:
GPT baseline outperformed all cumulative LLaMA additions.

Literature review (2026-03-11) revealed: (a) Narang et al. 2021
found most Transformer modifications don't transfer across
scales (LIT-001, Grade B); (b) our SwiGLU comparison was
parameter-confounded — d_ff=512 gave it 50% more params (LIT-002);
(c) muP (LIT-003) shows fixed-LR comparison is unreliable across
architectures. The 0.70 initial prior was poorly calibrated —
a literature-informed prior would have been ~0.40.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| — | Prior from architecture literature | C | — | 0.70 |
| 2026-03-11 | HYP-001: GPT baseline (1.65) beat full LLaMA (1.90), d=-1.17 | F | Strong against | 0.30 |
| 2026-03-11 | Narang 2021: mods don't transfer across scales | B | Against | — |
| 2026-03-11 | muP: fixed-LR comparison is confounded | C | Weakens HYP-001 evidence | — |
| 2026-03-11 | Shazeer: d_ff was wrong for SwiGLU (param mismatch) | C | Weakens HYP-001 evidence | — |

**Note:** Posterior stays at 0.30 but confidence in the HYP-001
evidence is now lower (methodology issues). HYP-001b with fixed
d_ff and LR sweep will provide a cleaner update.

| 2026-03-11 | HYP-001b: LLaMA (1.12) beats GPT (1.22) at best LR, d=+1.72, +8% | F | Strong for | 0.60 |

**Update (HYP-001b):** With parameter-matched SwiGLU (d_ff=341)
and per-config LR sweep, full LLaMA beats GPT baseline by 8%
(d=1.72, large effect). Every cumulative LLaMA feature helps.
However, CIs include zero (n=3, low power) and the experiment
used 5-min time budgets (DEC-001), which is known to be suboptimal
(SYNTH-001). Posterior moves to 0.60 — moderate evidence for,
pending replication with longer training (DEC-005).

| 2026-03-12 | HYP-001c: On val loss, LLaMA (1.670) worse than GPT (1.609), d=-5.84. All configs overfit severely (gap 0.83-0.93). HYP-001b's 8% advantage was memorization, not generalization. | F | Strong against | 0.20 |

**Update (HYP-001c):** FLOP-matched experiment at 1 PFLOPs with
proper 90/10 val split reveals the HYP-001b "advantage" was
training loss overfitting. On held-out data, GPT baseline (1.609)
beats full LLaMA (1.670) by 3.8% (d=-5.84, massive effect). All
configs cluster in a narrow 1.607-1.670 val band despite widely
varying train losses (0.68-0.86). LLaMA features improve
memorization, not generalization, at 3M-param scale with char-level
tokenization. Posterior drops to 0.20.

| 2026-03-13 | HYP-001d: With dropout=0.1, LLaMA (1.573) ties GPT (1.573), d=-0.07. With dropout=0.2, GPT (1.560) beats LLaMA (1.586), d=-3.84. Equalization exists but is dropout-rate-dependent. | F | Mixed — equalizes at 0.1, diverges at 0.2 | 0.30 |

**Update (HYP-001d):** Dropout reveals that the GPT-LLaMA gap is
partly a regularization deficit (LIT-020 confirmed). At dropout=0.1,
LLaMA catches up completely (d=-0.07). But at dropout=0.2, GPT pulls
ahead again — LLaMA appears to over-regularize at 0.2. This suggests
LLaMA features DO have comparable quality at small scale, but only
when regularization is correctly calibrated. The belief update is
nuanced: from 0.20 to 0.30, because the equalization at 0.1 partially
rehabilitates LLaMA, but the non-monotonic interaction means the
answer depends on regularization tuning, not just architecture.

| 2026-03-14 | HYP-006: At 30M with TinyStories BPE, LLaMA (2.512) massively beats GPT (3.049), d=-32.8. Hybrid-baselines: all 3 hybrid SSM architectures outperform pure attention. | F | Very strong for | 0.75 |

**Update (HYP-006 + hybrid-baselines):** The earlier negative
findings (HYP-001 through HYP-001d) were specific to 3M params
with char-level Shakespeare — a regime where architecture washes
out and regularization dominates. At 10-30M params with BPE
tokenization on TinyStories, LLaMA features provide massive
improvements (0.54 lower val loss). Hybrid SSM/attention
architectures outperform pure transformers by 0.09-0.42. The
belief now strongly favors LLaMA features helping at 10M+ scale.
The 3M char-level experiments were at a scale too small for
architecture to matter — confirming Kaplan 2020 and our own
roadmap conclusion.

---

## B-007: Test-time compute scaling works below 1B params

**Prior:** 0.25 (unlikely — prior consensus: too small)
**Current:** 0.75
**Source:** HYP-007 experiment results

At 10M params, best-of-N sampling with execution verification
shows clear pass@k scaling: pass@64 = 11.89x pass@1 overall.
No saturation at k=64. All prior literature tested TTC at 1.5B+
and assumed smaller models would be flat or quickly saturate.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| — | Prior from Snell et al. (1.5B+ only) | A | Weakly against (scale transfer) | 0.25 |
| 2026-03-15 | HYP-007: 10M model, pass@16/pass@1=5.6x, pass@64/pass@1=11.9x. No saturation. | F | Very strong for | 0.75 |
| 2026-03-15 | HYP-008: All 4 architectures (LLaMA, Falcon-H1, Jamba, Bamba) show strong TTC scaling at 10M. p@64/p@1 ranges 13.4-14.8x. Architecture-independent. | F | Very strong for | 0.90 |
| 2026-03-15 | HYP-009: During grokking, pass@64 reaches 98.9% at step 4K when pass@1 is only 14%. Pass@64 saturates 39K steps before greedy accuracy catches up. TTC reveals latent generalization hundreds of epochs early. | F | Very strong for | 0.95 |
| 2026-03-15 | HYP-010: TTC works at both 10M (p@64/p@1=14.6x) and 30M (11.9x). TTC amplification factor is stable across model sizes. BUT 30M model is worse than 10M in absolute terms due to overparameterization. | F | For (TTC works) but neutral on scaling | 0.95 |

**Update (HYP-007):** 10M LLaMA models on modular arithmetic
show robust pass@k scaling. pass@1 is low (0.55%) but pass@64
reaches 7.8% — a 14x amplification. Growth rate ~50% per
doubling of k with no saturation. This is the first evidence
of TTC working at 50-100x below prior literature's minimum
model size.

**Update (HYP-008):** Confirmed across 4 architecture families
including SSM-heavy hybrids. TTC amplification factor (~13-15x
at pass@64/pass@1) is essentially the same for pure attention,
hybrid SSM/attention, and hybrid+MoE. Posterior updated to
0.90 — TTC at small scale is now a well-replicated finding.

---

## B-008: Regularization preserves/enhances output diversity

**Prior:** 0.50 (uncertain — speculative connection)
**Current:** 0.20
**Source:** HYP-007 experiment results

Hypothesis was that dropout would promote output diversity
and steeper pass@k curves. Evidence says the opposite.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| — | Speculative: dropout → diverse subnetworks → diverse outputs | — | — | 0.50 |
| 2026-03-15 | HYP-007: dropout=0.0 has steepest pass@k curve (14.1x at p@64/p@1) vs dropout=0.1 (9.3x) and 0.2 (11.9x). Dropout HURTS both accuracy and diversity. | F | Strong against | 0.20 |

**Update (HYP-007):** Dropout reduces pass@k at every k value.
Dropout=0.0 achieves both highest pass@1 (0.55%) and highest
pass@64 (7.82%). The regularization-boosts-diversity hypothesis
is falsified. This parallels Yue et al. (2025) finding that
RLVR training narrows distributions and hurts pass@k. Verine
et al. (ICML 2025) provide the theoretical framework: dropout
improves Precision but not Recall.

---

## B-009: TTC scaling exponent is architecture-independent

**Prior:** 0.30 (H8-a prior)
**Current:** 0.85
**Source:** HYP-008 experiment results

TTC amplification factor (p@64/p@1) is determined by model
quality and task difficulty, not by architecture family. Pure
attention, hybrid SSM/attention, and hybrid+MoE all show
~13-15x amplification at 10M params on modular arithmetic.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| — | Speculative: SSMs might have less diversity due to fixed-size state | — | — | 0.30 |
| 2026-03-15 | HYP-008: p@64/p@1 ratios within 10% across 4 architecture families (13.4-14.8x). p@16/p@1 within 10% (5.8-6.4x). | F | Very strong for | 0.85 |
| 2026-03-15 | HYP-010: p@64/p@1 ratios 14.6x (10M) vs 11.9x (30M). Within 1.23x across 3x model size change. | F | Strong for | 0.90 |

**Update (HYP-008):** The prediction that fixed-size SSM state
would limit output diversity (H8-d) was strongly falsified.
SSM-heavy architectures achieve TTC exponents within 10% of
pure attention. The simplest explanation (H8-a) won: TTC scaling
depends on the model's learned distribution quality, not the
computational mechanism.

**Update (HYP-010):** Extended from architecture-independence to
size-independence. The TTC amplification factor is ~12-15x for
both 10M and 30M models on the same task. This further supports
the interpretation that TTC exponent is a task property. Belief
broadened to "TTC scaling exponent is architecture- and
size-independent" and updated to 0.90.

---

## B-010: Val loss does not predict pass@k ranking across archs

**Prior:** 0.50 (no strong prior)
**Current:** 0.75
**Source:** HYP-008 experiment results (ANOM-015)

Lower val loss does NOT imply higher pass@k when comparing
across architecture families. LLaMA has the worst val loss
(2.731) but the best pass@k at every k value.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| — | Default: lower loss should mean higher accuracy | — | — | 0.50 |
| 2026-03-15 | HYP-008: LLaMA val_loss 2.731 (worst), pass@64 8.34% (best). Jamba val_loss 2.310 (best), pass@64 3.29% (worst). Rank correlation is negative. | F | Strong for | 0.75 |
| 2026-03-15 | HYP-011: Per-token decomposition. Hybrids 29-31% worse at answer token but only 7% worse at prompt tokens. LLaMA answer entropy 2.12 vs hybrid ~1.17 nats. Mechanistic explanation confirmed: val_loss averages dilute the answer-token signal. | F | Very strong for | 0.90 |

**Update (HYP-008):** The val_loss metric is averaged over the
full vocabulary/sequence, while pass@k measures accuracy on a
specific subtask (generating the correct modular arithmetic
answer). Different architectures may achieve lower val_loss
by better predicting the prompt tokens (common patterns) while
being worse at the critical answer token. This echoes the
broader observation that perplexity and downstream task
performance don't always correlate (LIT-055, LIT-057).

**Update (HYP-011):** ANOM-015 now mechanistically explained.
Two effects: (1) answer-token loss drives pass@k but is
diluted in the average val_loss (5 prompt tokens : 1 answer
token); (2) LLaMA's answer-token distribution is more
entropic (2.12 vs ~1.17 nats) AND more accurate (0.66% vs
~0.31% P(correct)), making it ideal for best-of-N sampling.
Posterior updated to 0.90 — the mechanism is now clear.

---

## B-011: TTC reveals latent generalization before greedy accuracy

**Prior:** N/A (new belief from HYP-009)
**Current:** 0.80
**Source:** HYP-009 experiment results

During the grokking transition on modular arithmetic, pass@64
saturates to ~99% at the onset of circuit formation (step ~4K),
while greedy accuracy (pass@1) doesn't reach 99% until step 43K
— a 39,000-step (330-epoch) lead time. TTC acts as a "magnifying
glass" for latent generalization capabilities that exist in the
model's distribution but are too weak for greedy decoding.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-15 | HYP-009: Seed 42 grokked. pass@64=98.9% at step 4K when pass@1=14.1%, val_acc=19.9%. pass@1 reaches 99% at step 43K — 39K steps later. | F | Very strong for | 0.80 |

**Update (HYP-009):** Strong evidence from 1 seed that grokked.
Posterior set at 0.80 rather than higher because: (a) only 1/3
seeds grokked (seeds 43/44 stuck in oscillating plateau), (b)
only tested on modular arithmetic, (c) only one model size.
The finding is dramatic but needs replication on other tasks
and with more seeds before reaching 0.90+.
