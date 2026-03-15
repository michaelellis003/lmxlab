# Hypothesis Registry

Active, tested, and falsified hypotheses. Each links to its
experiment design and interpretation in `lab-notebook.md`.

## Status Key

| Status | Meaning |
|--------|---------|
| `active` | Pre-registered, not yet tested |
| `tested` | Experiment complete, see interpretation |
| `supported` | Evidence supports (not proven) |
| `falsified` | Evidence contradicts prediction |
| `refined` | Superseded by a revised hypothesis |

---

## HYP-001: GPT-to-LLaMA Feature Ablation

**Experiment:** 1 — GPT-to-LLaMA Feature Ablation
**Status:** tested (inconclusive)
**Question:** Which individual LLaMA-style feature contributes most
to improved training dynamics over a GPT baseline?

| ID | Hypothesis | Prediction | Falsification |
|----|-----------|------------|---------------|
| H1a | Attention dominates | GQA gives largest single improvement | +GQA improvement < 20% of total |
| H1b | FFN dominates | SwiGLU gives largest single improvement | +SwiGLU improvement < 20% of total |
| H1c | Normalization dominates | RMSNorm+no-bias gives largest improvement | +RMSNorm improvement < 20% of total |
| H1d | Interactions dominate | No single change > 20% of total | Any single feature > 50% of total |

**Design:** 6-run ablation (baseline, +GQA, +SwiGLU, +RMSNorm, +RoPE,
Full LLaMA). d_model=256, n_layers=6, 5-min budget, 3 seeds, Shakespeare.
**Analysis:** ANOVA + Cohen's d relative to baseline. Compare sum of
individual improvements to full LLaMA improvement for H1d.
**Recipe:** `ablation_gpt_to_llama.py`
**Result:** All hypotheses inconclusive. Full LLaMA (1.898) performed
worse than GPT baseline (1.652), invalidating the presupposition that
LLaMA features improve training at this scale. See ANOM-001, ANOM-002,
ANOM-003. Follow-up: HYP-001b with LR sweep and step-matched budget.

---

## HYP-002: mx.compile Coverage Scaling

**Experiment:** 2 — mx.compile Coverage Analysis
**Status:** active
**Question:** How does mx.compile speedup scale with compilation
coverage?

| ID | Hypothesis | Prediction | Falsification |
|----|-----------|------------|---------------|
| H2a | Graph size dominates | Linear speedup with compiled fraction | Non-linear relationship |
| H2b | Diminishing returns | Step compilation captures >80% of benefit | Step+eval adds >30% over step-only |
| H2c | Overhead at small scale | Tiny models slower when compiled | Tiny models faster on first step |

**Design:** 3 compilation configs x 3 model sizes x 3 seeds.
**Metrics:** steps/sec (excl. 5 warmup), peak memory, time-to-first-step.

---

## HYP-003: Optimizer Comparison on Unified Memory

**Experiment:** 3 — Optimizer Comparison
**Status:** active
**Question:** Does unified memory change which optimizers work best?

| ID | Hypothesis | Prediction | Falsification |
|----|-----------|------------|---------------|
| H3a | Same story | AdamW dominates regardless | AdamW not in top 2 |
| H3b | Memory-efficient wins | SGD/Adafactor comparatively better | SGD/Adafactor worse than CUDA ratios |
| H3c | Bandwidth matters | SGD disproportionate advantage | SGD steps/sec ratio matches CUDA |

**Design:** LLaMA-small (256d/6L), Shakespeare, AdamW/SGD+momentum/
Adafactor/Lion, LR sweep (1e-4, 3e-4, 1e-3, 3e-3), 5-min budget,
3 seeds.
**Metrics:** Best val_bpb, steps/sec, peak memory.

---

## HYP-004: KV Cache Reduction with MLA

**Experiment:** 4 — MLA KV Cache Analysis
**Status:** active
**Question:** Does MLA's ~57x KV cache compression provide practical
benefits on unified memory?

| ID | Hypothesis | Prediction | Falsification |
|----|-----------|------------|---------------|
| H4a | Memory benefit | MLA enables longer generation before OOM | KV cache not binding at <8K |
| H4b | No practical benefit | Benefit only at very long contexts | MLA helps at <4K tokens |
| H4c | Speed benefit | MLA faster per-token (bandwidth) | MLA same or slower per-token |

**Design:** MLA vs MHA at matched params. Sequence lengths: 512, 1K,
2K, 4K, 8K. Measure tokens/sec, peak memory, max sequence before OOM.

---

## HYP-005: 5-Minute Training Budget

**Experiment:** 5 — What Can You Train in 5 Minutes?
**Status:** active
**Question:** What is the best val_bpb achievable in exactly 5 minutes
of wall-clock training on M-series Mac?

This is an iterative experiment — no single hypothesis to falsify.
Each run modifies one variable and records whether val_bpb improves.

**Protocol:** Start GPT-tiny on Shakespeare. Iterate: change one thing,
train 5 min, record val_bpb. Keep improvements, discard regressions.
**Metrics:** val_bpb (primary), loss curve shape, param count.
**Simplicity bias:** At equal val_bpb, prefer fewer parameters.

---

## HYP-001b: Refined GPT-to-LLaMA Ablation

**Experiment:** 1b — GPT-to-LLaMA Feature Ablation (revised)
**Status:** tested (H1b-a supported, H1b-d falsified, B/C deferred)
**Question:** Was the GPT baseline's superiority in HYP-001 caused by
learning rate mismatch, time-budget unfairness, or tokenization choice?

| ID | Hypothesis | Prediction | Falsification |
|----|-----------|------------|---------------|
| H1b-a | LR mismatch | At optimal per-config LR, full LLaMA loss is >= 0.05 lower than GPT baseline | LLaMA still worse than GPT at all LRs tested |
| H1b-b | Time-budget unfairness | At step-matched budget (same step count), LLaMA loss is >= 0.05 lower than GPT | LLaMA still worse at equal step counts |
| H1b-c | Tokenization artifact | With BPE tokenization (TinyStories), LLaMA loss is >= 0.05 lower than GPT | LLaMA still worse with BPE tokenization |
| H1b-d | Null: scale problem | LLaMA features do not help at d_model=256 regardless of LR, budget, or tokenization | Any of H1b-a/b/c is supported |

**Prior Art (REA, 8 sources scanned):**
- [LIT-001] Narang et al. 2021 (EMNLP, Grade B): Most Transformer
  modifications don't transfer across scales. Strongest evidence
  for H1b-d (null/scale problem).
- [LIT-002] Shazeer 2020 (arXiv, Grade C): SwiGLU needs d_ff * 2/3
  for parameter matching. **HYP-001 was confounded — d_ff=512 gave
  SwiGLU 50% more FFN params.**
- [LIT-003] Yang et al. 2022 muP (arXiv, Grade C): Optimal LR
  changes with architecture under standard parametrization. Fixed-LR
  comparison is unreliable. Strongly supports H1b-a.
- [LIT-004] Touvron et al. 2023 LLaMA (arXiv, Grade C): Features
  validated only at 7B+. Scale transfer tax: ~50% discount.
- [LIT-006] Ainslie et al. 2023 GQA (EMNLP, Grade B): GQA trades
  quality for speed; may hurt quality at small scale.
- [LIT-007] Xue et al. 2022 ByT5 (TACL, Grade B): Char-level
  models need different arch tradeoffs than BPE models.
- [LIT-008] Press et al. 2022 ALiBi (ICLR, Grade B): Simple
  position encodings may beat RoPE for char-level tasks.
- **Gap:** No published ablation of individual LLaMA features
  at d_model=256 scale on char-level data.

**Literature-informed prior adjustments:**
- H1b-a (LR mismatch): prior 0.55 -> 0.70 (muP + SwiGLU evidence)
- H1b-b (time-budget): prior 0.40 -> 0.50 (FLOPs-matching standard)
- H1b-c (tokenization): prior 0.35 -> 0.45 (ByT5 + ALiBi evidence)
- H1b-d (null/scale): prior 0.50 -> 0.65 (Narang et al. strongest)

**Critical design fix from literature:** SwiGLU configs must use
d_ff = 512 * 2/3 = 341 for parameter matching (LIT-002).

**Design:**
Three sub-experiments, each isolating one confound from HYP-001:

*Sub-experiment A (LR sweep):* Same as HYP-001 but sweep LR across
{1e-4, 3e-4, 1e-3} for each of the 6 configs. Report best LR per
config. Shakespeare char-level, 5-min time budget. 3 seeds per
(config, LR) = 54 runs.

*Sub-experiment B (step-matched):* Fix step count to the minimum
steps achieved by any config in Sub-experiment A. Same 6 configs at
their best LR from A. Shakespeare char-level. 3 seeds = 18 runs.

*Sub-experiment C (BPE tokenization):* Use TinyStories dataset from
HuggingFace (`roneneldan/TinyStories`) with GPT-2 BPE tokenizer
(TiktokenTokenizer). Best LR per config from A. 5-min time budget.
3 seeds = 18 runs. Vocab size changes to 50257 so tie_embeddings
should be disabled and d_model may need adjustment.

**Protocol:** 5-min time budget (per DEC-001), 3 seeds (per DEC-002).
Sub-experiment A uses Shakespeare (per DEC-003). Sub-experiment C
uses TinyStories to test DEC-003's generalization caveat.
**Metrics:** Final training loss (primary), steps/sec (throughput),
steps completed (fairness check).
**Analysis:** For each sub-experiment: mean +/- std across seeds,
Cohen's d vs GPT baseline, 95% CI. Compare GPT vs full LLaMA at
each config's best LR.
**Explore/exploit split:** 70% exploit (running the designed
experiments) / 30% explore (if early results suggest a different
confound, adjust Sub-experiment C).
**Stopping rule:** Complete all three sub-experiments. If A resolves
the anomaly (LLaMA clearly wins at correct LR), skip B and C.

---

## HYP-001c: FLOP-Matched GPT-to-LLaMA Ablation

**Experiment:** 1c — GPT-to-LLaMA Feature Ablation (FLOP-matched)
**Status:** tested (all falsified/inconclusive)
**Question:** Does the 8% LLaMA advantage observed in HYP-001b hold,
widen, or narrow when training is FLOP-matched at Chinchilla-optimal
duration (~1 PFLOPs) instead of 5-minute time budgets?

**Prior Art (REA, 15 sources in index + 3 new scanned):**
- [LIT-009] Hoffmann et al. 2022 Chinchilla (NeurIPS, Grade A):
  Undertrained models give misleading architecture comparisons.
  The 20:1 tokens/param ratio is compute-optimal.
- [LIT-014] Porian et al. 2024 (NeurIPS, Grade B): Optimum is
  flat — 10:1 to 40:1 costs only ~1-3% in loss.
- [LIT-010] Muennighoff et al. 2023 (NeurIPS, Grade A): Data
  repetition effective up to ~4 epochs; diminishing beyond 8.
- [LIT-001] Narang et al. 2021 (EMNLP, Grade B): Most modifications
  don't transfer across scales.
- [LIT-006] Ainslie et al. 2023 (EMNLP, Grade B): GQA trades
  quality for speed; may hurt at small scale.
- [LIT-005] Kaplan et al. 2020 (arXiv, Grade C): Architecture
  details matter less than scale in broad ranges. FLOPs-matched
  is the gold standard for comparison.
- [HYP-001b] Own experiment (Grade F): LLaMA beats GPT by 8%
  at best per-config LR, but only 5-min time-budget training.
  CIs include zero (n=3). Methodology caveat: time-matched, not
  FLOP-matched; ~2-3 tokens/param (severely undertrained).
- **New scan:** Hybrid Architectures for LMs (arXiv 2024,
  Grade C): Under same FLOP budget at 100M-3B scale, hybrid
  architectures show 2.9% accuracy gain over pure transformers.
  Validates FLOP-matched methodology at moderate scale.
- **Gap:** No published FLOP-matched GPT vs LLaMA feature
  ablation at <10M params. Our work fills this gap.

| ID | Hypothesis | Prediction | Falsification | Lit. Prior |
|----|-----------|------------|---------------|-----------|
| H1c-a | Advantage widens | Full LLaMA loss >= 0.10 lower than GPT at 1 PFLOPs (>= 2x the 5-min gap) | LLaMA advantage < 0.05 at 1 PFLOPs | 0.45 |
| H1c-b | Advantage holds | Full LLaMA loss 0.05-0.10 lower than GPT (similar to 5-min result) | LLaMA advantage < 0.03 or > 0.15 | 0.30 |
| H1c-c | Advantage narrows | Full LLaMA loss < 0.05 lower than GPT, or GPT catches up | LLaMA advantage >= 0.05 | 0.15 |
| H1c-d | Null: no significant difference | GPT and LLaMA within 0.03 of each other (d < 0.5) | Cohen's d >= 0.8 for GPT vs LLaMA | 0.10 |

**Why these alternatives:**
- **H1c-a (widens):** Chinchilla (LIT-009) shows undertrained
  models distort comparisons. LLaMA features (SwiGLU, RoPE) may
  need more training to show their advantage — 5-min runs gave
  only ~2 tokens/param. At 8-9 epochs (~14 tokens/param), the
  inductive biases should have more time to compound. The Hybrid
  Architectures paper (2024) found FLOP-matched comparisons
  reveal larger efficiency differences than time-matched ones.
  Prior: 0.45.
- **H1c-b (holds):** The 8% gap from HYP-001b may already
  reflect the true architectural advantage. Porian (LIT-014)
  shows the loss optimum is flat — more training doesn't
  dramatically change relative rankings. Prior: 0.30.
- **H1c-c (narrows):** Narang (LIT-001) found most modifications
  don't transfer across scales. At 3M params, the GPT baseline
  may be near the irreducible entropy floor, leaving less room
  for architectural improvements. Ainslie (LIT-006) predicts GQA
  hurts quality at small scale. Longer training may let GPT's
  simpler optimization landscape catch up. Prior: 0.15.
- **H1c-d (null):** Kaplan (LIT-005) found architecture details
  matter less than scale. At 3M params with char-level data, all
  configs may converge to similar loss. Prior: 0.10.

**Calibration note:** HYP-001 had a poorly calibrated prior
(0.70 for B-006, should have been ~0.40). HYP-001b's result
partially corrected this. For HYP-001c, priors are informed by
both the literature AND our own HYP-001b data. The sum of priors
= 1.0 (mutually exclusive, exhaustive).

**Design:**
- 6 configs (same as HYP-001b): GPT baseline, +RMSNorm, +RoPE,
  +SwiGLU (d_ff=341), +GQA (n_kv_heads=4), +No bias (=LLaMA)
- d_model=256, n_heads=8, n_layers=6, vocab_size=96
- Each config uses its best LR from HYP-001b:
  GPT=3e-4, RMSNorm=1e-4, RoPE=3e-4, SwiGLU=3e-4,
  GQA=3e-4, LLaMA=1e-4
- FLOP budget: 1 PFLOPs (1e15) per run, enforced by FLOPCounter
  callback. Each config trains until hitting the budget — configs
  with fewer FLOPs/step will run more steps.
- Shakespeare char-level dataset, batch_size=8, seq_len=256
- Expected training: ~20,500 steps for GPT/RMSNorm/RoPE/SwiGLU,
  ~22,800 steps for GQA/LLaMA (~8-9 epochs, ~42M tokens)
- 5 seeds per config (increased from 3 to tighten CIs)
- Total: 6 configs x 5 seeds = 30 runs
- Hardware: Apple M3 Pro, 36GB unified memory, ~6.5 TFLOP/s FP32
- Estimated wall time: ~30-40 min per run, ~15-20 hours total
  (can run overnight)

**Protocol:**
- FLOP-matched budget (DEC-004), ~1 PFLOPs (DEC-005)
- 5 seeds: {42, 43, 44, 45, 46} (extends DEC-002)
- Shakespeare char-level (DEC-003)
- Per-config best LR from HYP-001b (no LR sweep — already done)
- FLOPCounter callback with flop_budget=1e15

**Metrics:**
- Primary: final training loss (lower is better)
- Secondary: loss curve shape (tokens vs loss), TFLOP/s throughput
- Tertiary: steps completed, wall time, peak memory

**Analysis:**
- Mean +/- std across 5 seeds per config
- Cohen's d for each config vs GPT baseline
- 95% confidence intervals (with n=5, CIs should be ~40% tighter
  than HYP-001b's n=3)
- One-way ANOVA across all 6 configs
- Feature contribution analysis (cumulative ablation table)
- Compare effect sizes to HYP-001b to assess training duration
  impact on architectural advantage
- Plot loss-vs-FLOPs curves for each config to check whether
  LLaMA features show earlier or later convergence

**Explore/exploit split:** 90% exploit / 10% explore. This is a
confirmatory experiment — the design is fixed from HYP-001b. The
10% explore covers: if any config shows anomalous behavior (e.g.,
divergence), investigate before continuing other seeds.

**Stopping rule:** Complete all 30 runs. No early stopping on the
experiment level (individual runs stop at the FLOP budget). If
>2 runs crash/diverge, pause and investigate before continuing.

**Result (2026-03-12):** All 30 runs completed. With proper 90/10
val split, GPT baseline (val 1.609) beats full LLaMA (val 1.670)
by 3.8% (d=-5.84). All hypotheses falsified — none predicted GPT
would be better. The HYP-001b "8% LLaMA advantage" was overfitting:
train-val gaps of 0.83-0.93 across all configs. On val loss, all
configs cluster in a narrow 1.607-1.670 band. Architecture matters
far less than overfitting at this training duration.
See ANOM-006, ANOM-007, ANOM-008. B-006 updated to 0.20.

---

## HYP-001d: Dropout Regularization × Architecture Interaction

**Experiment:** 1d — Dropout effect on GPT vs LLaMA at 1 PFLOPs
**Status:** tested (all falsified — actual outcome not predicted)
**Question:** Does dropout regularization reduce the train-val gap
and change the relative ranking of GPT vs LLaMA architectures when
training on repeated Shakespeare data at 1 PFLOPs?

**Prior Art (REA, 22 sources in index + nanoGPT reference):**
- [LIT-017] STLM Dropout Report 2024 (arXiv, Grade C): Dropout
  0.1 is standard for <100M params; uniquely alleviates multi-epoch
  Token-Crisis; linear schedule performs best.
- [LIT-018] Drop Dropout 2025 (arXiv, Grade C): Dropout NOT
  helpful for single-epoch training; confirms regime-dependence.
- [LIT-019] Dropout+Residual 2024 (arXiv, Grade C): On Tiny
  Shakespeare, dropout 0.2 optimal, best val loss 1.5531.
- [LIT-020] Re-Introducing LayerNorm 2024 (arXiv, Grade C):
  RMSNorm lacks LayerNorm's mean-subtraction implicit
  regularization. Both converge similarly at large scale, but
  small-scale overfitting regime not studied.
- [LIT-021] Small LM Trade-offs 2024 (arXiv, Grade C): Val loss
  bottoms epoch 2 in small LMs; dropout 0.1 + early stopping.
- [nanoGPT] Karpathy (Grade D): dropout=0.2 on Shakespeare
  char-level achieves val loss 1.4697 (6L/6H/384d, larger model).
- [HYP-001c] Own experiment (Grade F): Without dropout, all
  configs overfit severely (gap 0.83-0.93). GPT beats LLaMA by
  3.8% on val loss (d=-5.84).
- **Gap:** No published study of dropout × architecture interaction
  at small scale. Specifically, nobody has tested whether RMSNorm's
  implicit regularization deficit (ANOM-007) is compensated by
  explicit dropout.

**"Why hasn't anyone done this?":** The dropout-alone question is
obvious (everyone knows dropout helps with overfitting). But the
dropout × RMSNorm interaction at small scale is genuinely novel.
Our ANOM-007 finding (RMSNorm d=-5.39 worse on val loss despite
similar train loss) contradicts LIT-020's claim that the difference
is inconsequential. This experiment tests whether explicit
regularization compensates for missing implicit regularization.

| ID | Hypothesis | Prediction | Falsification | Lit. Prior |
|----|-----------|------------|---------------|-----------|
| H1d-a | Dropout equalizes | At dropout=0.2, GPT and LLaMA val loss within 0.02 (d < 0.8). Dropout compensates for RMSNorm's missing regularization. | GPT-LLaMA gap > 0.03 or d > 1.0 at dropout=0.2 | 0.35 |
| H1d-b | Helps but rankings hold | Both improve by ~0.03-0.06, but GPT remains 0.03+ better than LLaMA. RMSNorm still overfits more. | GPT-LLaMA gap < 0.02 or LLaMA beats GPT | 0.35 |
| H1d-c | Reveals LLaMA advantage | With overfitting controlled, LLaMA val loss >= 0.03 lower than GPT. HYP-001b's 8% advantage re-emerges. | LLaMA worse than or within 0.02 of GPT | 0.15 |
| H1d-d | Null: dropout ineffective | Dropout=0.2 reduces train-val gap by < 0.10. Val losses barely change (< 0.02 improvement). | Train-val gap reduces by > 0.15 | 0.15 |

**Why these alternatives:**
- **H1d-a (equalizes, 0.35):** LIT-020 shows RMSNorm lacks
  LayerNorm's implicit regularization. If ANOM-007 is purely a
  regularization deficit, explicit dropout should close the gap.
  The 0.06 GPT-LLaMA val gap in HYP-001c may be entirely due to
  differential overfitting, not architecture.
- **H1d-b (rankings hold, 0.35):** LIT-019 shows dropout helps
  (~4% improvement) but Narang (LIT-001) found architecture
  modifications don't help at small scale. If the gap is
  architectural, dropout helps both equally. B-006 at 0.20
  suggests LLaMA features don't generalize to small scale.
- **H1d-c (reveals LLaMA, 0.15):** HYP-001b showed 8% LLaMA
  advantage on train loss. This could reflect real efficiency
  masked by differential overfitting. Low prior because B-006
  is at 0.20 after 3 experiments against.
- **H1d-d (null, 0.15):** At 8-9 epochs, memorization may be too
  deep for standard dropout. LIT-022 suggests more advanced
  techniques (EntroDrop) may be needed. Low prior because LIT-017
  and LIT-019 both show dropout working in this regime.

**Calibration note:** HYP-001 series has consistently over-estimated
LLaMA's advantage (priors too high). H1d-a and H1d-b are given
equal priors (0.35 each) to avoid repeating this bias. The sum
= 1.0 (mutually exclusive, exhaustive).

**Implementation prerequisite:** The `dropout` field in ModelConfig
is a ghost field — it exists in config but `nn.Dropout` is never
instantiated in any model layer. Must wire up dropout in attention
residual, FFN residual, and post-embedding before running. This
is infrastructure work (feature, not experiment).

**Design:**
- 2 architectures: GPT baseline, full LLaMA (=No bias)
- 3 dropout rates: 0.0 (control), 0.1, 0.2
- 3 seeds: {42, 43, 44} (per DEC-002)
- d_model=256, n_heads=8, n_layers=6, vocab_size=96
- Per-config best LR from HYP-001b: GPT=3e-4, LLaMA=1e-4
- FLOP budget: 1 PFLOPs per run (matching HYP-001c)
- Shakespeare char-level, 90/10 val split, batch_size=8, seq_len=256
- Eval every 500 steps + final eval
- Total: 2 × 3 × 3 = 18 runs
- Estimated wall time: ~9-12 hours

**Protocol:**
- FLOP-matched (DEC-004), 1 PFLOPs (DEC-005)
- 3 seeds: {42, 43, 44} (DEC-002)
- Shakespeare char-level (DEC-003)
- Val loss as primary metric (DEC-008)
- dropout=0.0 runs serve as controls (must match HYP-001c)

**Metrics:**
- Primary: val_loss (best eval loss) — per DEC-008
- Secondary: train_loss, train-val gap, val_perplexity, val_accuracy
- Diagnostic: grad_norm, weight_norm, MFU, loss_spikes, peak_memory

**Analysis:**
- 2-way ANOVA: dropout rate × architecture
- Cohen's d for GPT vs LLaMA at each dropout rate
- Cohen's d for dropout=0.0 vs 0.2 within each architecture
- 95% CIs on all pairwise comparisons
- Train-val gap reduction as a function of dropout rate
- Check dropout=0.0 controls reproduce HYP-001c (within 0.01)

**Explore/exploit split:** 90% exploit / 10% explore.
If dropout=0.1 and 0.2 both show strong effects, probe
dropout=0.3 with one seed. If controls don't match HYP-001c,
investigate before continuing.

**Stopping rule:** Complete all 18 runs. If dropout=0.0 controls
deviate > 0.01 from HYP-001c, stop and debug.

**Result (2026-03-13):** All 18 runs completed. Controls match
HYP-001c (GPT 1.611 vs 1.609, LLaMA 1.671 vs 1.670). All four
hypotheses falsified — the actual outcome was not predicted:
- H1d-a (equalizes at 0.2): **Falsified.** Gap=0.026, but d=-3.84.
  However, equalization DOES occur at dropout=0.1 (gap=0.0005,
  d=-0.07). Pre-registration specified 0.2.
- H1d-b (rankings hold): **Falsified.** Gap < 0.03 at both 0.1
  and 0.2. Rankings change with dropout.
- H1d-c (reveals LLaMA): **Falsified.** LLaMA never beats GPT.
- H1d-d (null): **Falsified.** Gap reduction > 0.9 (vs < 0.10
  predicted). Dropout is extremely effective.

**Actual outcome:** Non-monotonic interaction. LLaMA optimal at
dropout=0.1 (ties GPT), GPT optimal at 0.2 (pulls ahead). LLaMA
benefits more from dropout (2.5x improvement) but over-regularizes
at 0.2. See ANOM-009, ANOM-010, ANOM-011.
B-006 updated: 0.20 → 0.30.

---

## HYP-006: Dropout × Normalization at 30M with BPE

**Experiment:** 6 — Dropout × normalization scaling validation
**Status:** tested (H6-c supported)
**Question:** Does the dropout × normalization interaction
(ANOM-009/010/011) replicate at 30M params with BPE tokenization
on TinyStories?

| ID | Hypothesis | Prediction | Falsification | Lit. Prior |
|----|-----------|------------|---------------|-----------|
| H6-a | Replicates | LLaMA and GPT have different optimal dropout rates; non-monotonic pattern preserved | Same optimal dropout for both archs (within 0.05) | 0.30 |
| H6-b | Partially | Interaction exists but optimal rates shift (e.g., both lower at 30M) | No interaction (2-way ANOVA p > 0.05) | 0.35 |
| H6-c | Null | No interaction at 30M — was a small-scale artifact of data repetition | Significant arch×dropout interaction (p < 0.05) | 0.35 |

**Why these alternatives:**
- **H6-a (replicates, 0.30):** ANOM-009/010/011 showed clear
  non-monotonic interaction at 3M with Shakespeare. If the
  interaction is architectural (RMSNorm vs LayerNorm implicit
  regularization), it should persist at 30M.
- **H6-b (partially, 0.35):** The interaction exists but
  different data regime (single-epoch TinyStories vs multi-epoch
  Shakespeare) shifts optimal rates. Dropout is less needed
  without data repetition.
- **H6-c (null, 0.35):** HYP-001d trained 8-9 epochs on
  Shakespeare — severe overfitting regime. TinyStories BPE at
  30M params trains < 1 epoch, so no overfitting to regularize.
  The interaction was a data-repetition artifact, not an
  architectural phenomenon.

**Design:**
- 2 architectures: GPT-30M (LayerNorm), LLaMA-30M (RMSNorm)
- 4 dropout rates: 0.0, 0.1, 0.2, 0.3
- 3 seeds: {42, 43, 44}
- FLOP-matched budget (shared, based on GPT-30M × 2000 steps)
- Dataset: TinyStories BPE (train/validation splits)
- LR: 3e-4 for both (standard for 30M scale)
- Eval: every 500 steps + final eval (dropout disabled)
- Total: 24 runs, ~2 hours on M3 Pro

**Recipe:** `recipes/hyp006_dropout_norm.py`

**Metrics:**
- Primary: val_loss (final, dropout disabled)
- Secondary: best_val_loss, train_loss, train-val gap
- Diagnostic: steps completed, wall time, total FLOPs

**Analysis:**
- 2-way ANOVA: architecture × dropout rate
- Cohen's d for GPT vs LLaMA at each dropout rate
- Interaction plot: dropout rate × val_loss, colored by norm
- Compare to HYP-001d results for replication assessment

**Result (2026-03-14):** 24 runs completed. H6-c (null/artifact)
supported. The dropout × normalization interaction from HYP-001d
does NOT replicate at 30M with BPE. Dropout hurts both archs
uniformly (undertrained regime, <1 epoch). LLaMA massively
outperforms GPT (gap=0.54, d=-32.8) — architecture matters at
this scale. See lab-notebook.md for full interpretation.
B-006 updated: 0.30 → 0.75.

---

## HYP-007: Test-Time Compute Scaling at Small Scale

**Experiment:** 7 — Test-time compute scaling on
verifiable tasks at <100M parameters
**Status:** tested (H7-a supported, H7-b/c/d falsified)
**Question:** Can test-time compute scaling (best-of-N
with execution verification) compensate for model size
on simple verifiable tasks at 10M parameters, and where
does this break down? Does training regularization
affect test-time scaling effectiveness?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. Finding the capability floor
  for TTC tells practitioners the minimum useful model size.
  All TTC literature claims are validated only at 1.5B+.
- Gate 2 (Scale): PASS. We are trying to FIND the floor.
  You can only find it from below. Large labs never go
  below 1B — small scale is an advantage here.
- Gate 3 (Prior coverage): PASS. Nobody has studied TTC
  below 1B. Gap is new (TTC field emerged 2024-2025).
- Gate 4 (Predictability): MILD CONCERN. A reviewer might
  predict "too small." But the EXACT floor is unknown, and
  the regularization interaction (H7-c) is NOT predictable.
- Gate 5 (Methodology): PASS. Modular arithmetic has exact
  ground truth. Unlimited test data. Pass@k is well-defined.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Why hasn't anyone done this?** Large labs focus on
frontier performance at 7B+. The TTC field is new (2024-
2025). Nobody has reason to test below 1B because they
assume it won't work. But the assumption is untested —
the capability floor is unknown. Our setup (fast
iteration on Apple Silicon, existing model infrastructure)
is uniquely suited to find it.

**Prior Art (REA, 6 sources + 5 research agents):**
- [LIT-038] Snell et al. 2024 (ICLR 2025 Oral, Grade A):
  1.5B + optimal TTC > 14x larger model. Difficulty-
  dependent strategy. Scale transfer tax applies (~50%).
- [LIT-039] Wu et al. 2024 (ICLR 2025, Grade A): Inference
  scaling law `log10(C) = 1.19*log10(N) + 2.03`. Small
  models saturate quickly. Llemma-7B+search > Llemma-34B.
- [LIT-040] GX-Chen et al. 2025 (arXiv, Grade C): KL-
  regularized RL induces mode collapse by construction.
  Diversity destruction degrades pass@k. Tested at 1.7-3B.
- [LIT-041] Yue et al. 2025 (NeurIPS 2025, Grade A):
  RLVR improves pass@1 but degrades pass@k. Base models
  achieve higher coverage at large k. Effect persists
  across 7B-32B.
- [LIT-042] Pan et al. 2025 (GitHub, Grade D): TinyZero.
  GRPO fails at 0.5B. 1.5B is minimum for RL reasoning.
- [LIT-043] Rafailov et al. 2024 (NeurIPS 2024, Grade A):
  1B models "almost immediately" over-optimize. Smaller
  models MORE vulnerable to reward hacking.
- **Gap:** No study of test-time compute scaling below 1B.
  The capability floor for best-of-N with execution
  verification is completely unknown at <100M scale.

| ID | Hypothesis | Prediction | Falsification | Lit. Prior |
|----|-----------|------------|---------------|-----------|
| H7-a | TTC helps | Best-of-N measurably improves accuracy at 10M: pass@16 >= 1.5x pass@1 on held-out modular arithmetic | pass@16 < 1.1x pass@1 | 0.25 |
| H7-b | Capability floor | pass@k saturates by k~10-20. Model either always right or always wrong — no useful diversity | pass@k still improving at k=64 | 0.30 |
| H7-c | Regularization boosts TTC | Models trained with dropout=0.1 show steeper pass@k curves (higher pass@16/pass@1 ratio) than dropout=0.0 | Dropout models have same or flatter pass@k curve | 0.20 |
| H7-d | Null: flat scaling | At 10M, best-of-N is flat. pass@k ≈ pass@1 for all k. The model has no useful output diversity | pass@k > 1.2x pass@1 at any k | 0.25 |

**Why these alternatives:**
- **H7-a (TTC helps, 0.25):** Wu et al. (LIT-039) show
  small models benefit from sampling at low compute
  budgets. If the model has ANY non-zero accuracy, best-
  of-N with a perfect verifier should improve it. Grokking
  literature shows 10M models CAN learn modular arithmetic.
  But scale transfer tax is extreme (700x from 7B to 10M).
- **H7-b (capability floor, 0.30):** Snell et al. (LIT-038)
  show TTC amplifies existing capabilities but can't create
  new ones. At 10M, the model may have a very narrow
  distribution — essentially deterministic for each input.
  TinyZero (LIT-042) shows 0.5B fails at reasoning tasks.
  Highest prior because the "too small" argument is strong.
- **H7-c (regularization, 0.20):** Our ANOM-009/010/011
  showed dropout affects output diversity. GX-Chen
  (LIT-040) proves diversity is key for pass@k. If dropout
  promotes diverse outputs, it should improve TTC. Novel
  connection — nobody has studied regularization's effect
  on TTC curves. Lower prior because the connection is
  speculative.
- **H7-d (null, 0.25):** At 10M on a simple task, the
  model may learn a near-deterministic mapping (modular
  arithmetic is a FUNCTION). If the learned function is
  either correct or incorrect for each input, sampling adds
  nothing. This differs from H7-b in that H7-b predicts
  fast saturation while H7-d predicts no improvement at
  all.

**Calibration note:** HYP-001 series showed we tend to
over-estimate effects at small scale. Priors here are
conservative: H7-a (the "exciting" outcome) gets only
0.25. The most likely outcome is H7-b or H7-d (limited
or no benefit). Sum = 1.0.

**Design:**
- **Task:** Modular arithmetic — `(a + b) mod p` with
  p=97. Training on 80% of (a,b) pairs, testing on 20%.
  BPE tokenizer (GPT-2), output is a number token.
- **Model:** 10M parameter LLaMA (llama_10m factory).
  Trained to partial accuracy (NOT to grokking).
- **Training configs:** 3 dropout rates: 0.0, 0.1, 0.2
- **TTC protocol:** For each test input, generate k
  completions with temperature=1.0. Verify each by
  computing (a+b) mod p. Report pass@k = fraction of
  test inputs where at least 1 of k completions is correct.
- **k values:** 1, 2, 4, 8, 16, 32, 64
- **Seeds:** 3 per config ({42, 43, 44})
- **Total:** 3 dropout rates × 3 seeds = 9 trained models,
  each evaluated at 7 k values = 63 pass@k measurements
- **FLOP budget for training:** Match to partial accuracy
  regime (~50-80% pass@1). Pilot run to calibrate.
- **Hardware:** Apple M3 Pro, 36GB

**Protocol:**
- 3 seeds (DEC-002)
- Val loss as primary metric during training (DEC-008)
- FLOP-matched training across dropout configs (DEC-004)
- Modular arithmetic dataset (new — not DEC-003)

**Metrics:**
- Primary: pass@k curves (k=1 through 64) on held-out
  test set
- Secondary: TTC scaling exponent (fit log-linear to
  pass@k vs k curve), pass@16/pass@1 ratio
- Diagnostic: output entropy (diversity measure),
  val_loss during training, train-val gap

**Analysis:**
- Plot pass@k curves for each (dropout, seed) combination
- Fit log-linear model: pass@k = a + b*log(k)
- Compare TTC scaling exponents across dropout rates
- Cohen's d for pass@16 between dropout=0.0 and 0.1
- 2-way ANOVA: dropout × k for pass@k
- Compare to Wu et al. inference scaling law prediction
  extrapolated to N=10M

**Explore/exploit split:** 80% exploit / 20% explore.
If pilot shows pass@1 ≈ 0 on held-out data (model didn't
generalize at all), pivot to testing at a model size
where pass@1 > 0 (e.g., 30M or adjust training duration).

**Stopping rule:** Complete all 9 models. If ALL models
have pass@1 = 0 on test set, stop and report "capability
floor is above 10M for this task." This is still a valid
finding (locates the floor).

**Experiment-Specific Metrics:**
| Callback | Rationale | Schedule |
|---|---|---|
| AttentionEntropy | H7-a vs H7-d: if model has useful diversity, `exp_attn_entropy_mean` should be moderate-high; deterministic routing (H7-d) → very low entropy | per-eval/500 |
| ActivationStats | H7-c: if dropout promotes diverse subnetworks, `exp_act_sparsity_mean` should be higher for dropout>0 models, explaining steeper pass@k curves | per-eval/500 |
| WeightStats | Baseline training diagnostic: `exp_weight_delta` tracks how far training moves from init; correlate with pass@k for post-hoc analysis | per-step/100 |

**Result (2026-03-15):** 9 runs completed (3 dropout x 3 seeds).

| Dropout | Val Loss | pass@1 | pass@16 | pass@64 | p@16/p@1 |
|---------|----------|--------|---------|---------|----------|
| 0.0 | 2.685 | 0.55% | 3.42% | 7.82% | 6.18x |
| 0.1 | 2.651 | 0.47% | 2.29% | 4.37% | 4.86x |
| 0.2 | 2.618 | 0.32% | 1.83% | 3.87% | 5.65x |

**Hypothesis adjudication:**
- H7-a (TTC helps): **Supported.** pass@16/pass@1 = 5.6x (>> 1.5x
  threshold). pass@64/pass@1 = 11.9x. TTC measurably helps at 10M.
- H7-b (capability floor): **Falsified.** pass@k still growing at
  ~45% per doubling at k=64 — no saturation detected.
- H7-c (regularization boosts TTC): **Falsified.** dropout=0.0 has
  the steepest curve (14.1x at p@64/p@1) and highest absolute pass@k
  at every k. Dropout HURTS both accuracy and diversity.
- H7-d (null: flat): **Falsified.** pass@64 = 11.9x pass@1.

**Key findings:**
1. First evidence of TTC scaling at 50-100x below prior literature
   minimum (1.5B). The capability floor is lower than assumed.
2. Dropout uniformly hurts pass@k — regularization reduces output
   diversity, paralleling Yue et al. (2025) finding that RLVR
   narrows distributions.
3. Models memorize training data (train loss ~0.002) but don't
   generalize well to held-out (a,b) pairs (pass@1 ~0.5%).
4. Despite low accuracy, the model has useful output diversity —
   best-of-64 reaches nearly 8% accuracy from 0.5% base.

B-007 created (0.25 → 0.75). B-008 created (0.50 → 0.20).
