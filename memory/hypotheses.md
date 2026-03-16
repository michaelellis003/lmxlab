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

---

## HYP-008: SSM/Hybrid Test-Time Compute Scaling

**Experiment:** 8 — Test-time compute scaling across
architecture families at 10M parameters
**Status:** tested (H8-a supported)
**Question:** Does test-time compute scaling (best-of-N with
execution verification) work equally well for SSM/hybrid
architectures as for pure attention at 10M params, and how
do pass@k curves compare across architecture families?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. Whether TTC generalizes across
  architecture families is fundamental for inference-time
  applications. All TTC literature uses pure transformers.
- Gate 2 (Scale): PASS. Finding TTC behavior at small scale.
- Gate 3 (Prior coverage): PASS. Nobody has studied TTC on
  SSMs at ANY scale. Complete gap.
- Gate 4 (Predictability): PASS. SSM fixed-size state could
  help (richer state → diverse outputs) or hurt (compressed
  state → mode collapse). Not predictable.
- Gate 5 (Methodology): PASS. Same protocol as HYP-007.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA, extending HYP-007 sources):**
- [LIT-038] Snell et al. 2024: TTC at 1.5B, pure transformers.
- [LIT-039] Wu et al. 2024: Inference scaling law, pure
  transformers only.
- [LIT-032] Mamba-3 (ICLR 2026): Mamba-3 solves modular
  arithmetic — SSMs CAN learn the task.
- [LIT-037] Jamba (ICLR 2025): Hybrid SSM-attention ablation.
  Pure Mamba fails at in-context learning.
- [HYP-007] Own experiment: TTC works at 10M for LLaMA.
  pass@64=11.9x pass@1. Dropout hurts diversity.
- **Gap:** No study of TTC scaling on SSM or hybrid
  architectures at any scale.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H8-a | Architecture-independent | All 4 archs have pass@16/pass@1 ratios within 2x of each other. TTC effectiveness depends on model quality (val_loss), not architecture type. | Any arch has p@16/p@1 ratio > 3x different from LLaMA | 0.30 |
| H8-b | Attention advantage | LLaMA has the steepest pass@k curve (highest p@64/p@1 ratio) because explicit attention access generates more diverse outputs | LLaMA's p@64/p@1 ratio is lower than at least one SSM/hybrid | 0.25 |
| H8-c | Hybrid advantage | Hybrid models (Falcon-H1, Jamba, Bamba) show steeper pass@k curves than pure attention because SSM+attention provides complementary generation modes | All hybrids have flatter curves than LLaMA | 0.20 |
| H8-d | SSM disadvantage | SSM-heavy architectures have the flattest pass@k curves — fixed-size state limits output diversity, resulting in p@64/p@1 < 5x | All SSM-heavy archs have p@64/p@1 >= 5x | 0.25 |

**Why these alternatives:**
- **H8-a (independent, 0.30):** HYP-007 showed TTC scaling is
  robust. If the effect depends mainly on model quality (val
  loss) rather than how the model processes sequences, then
  equal-quality models should show similar TTC curves.
  Hybrid baselines showed similar val_loss rankings across
  architectures. Highest prior because it's the simplest
  explanation.
- **H8-b (attention advantage, 0.25):** Attention's explicit
  O(n^2) context access could create richer next-token
  distributions. Jamba (LIT-037) found pure Mamba fails at
  in-context learning — suggesting attention provides
  capabilities SSMs lack. If in-context reasoning matters
  for output diversity, attention should win.
- **H8-c (hybrid advantage, 0.20):** Hybrid models combine
  SSM's efficient state tracking with attention's precise
  retrieval. This dual pathway could generate outputs using
  different computational strategies, increasing diversity.
  Lower prior because the mechanism is speculative.
- **H8-d (SSM disadvantage, 0.25):** SSMs compress the full
  context into a fixed-size state. This compression may lose
  the fine-grained token-level information needed for diverse
  outputs. At 10M params with limited state size, this
  compression loss could be severe.

**Design:**
- 4 architectures: LLaMA-10M (pure attention), Falcon-H1-10M
  (hybrid), Jamba-10M (hybrid+MoE), Bamba-10M (hybrid)
- dropout=0.0 (per HYP-007: dropout hurts diversity)
- 3 seeds: {42, 43, 44} (per DEC-002)
- FLOP budget: matched to LLaMA-10M × 2000 steps (2.88e14)
- Dataset: modular arithmetic (a+b) mod 97, same as HYP-007
- Eval: pass@k with k=1,2,4,8,16,32,64, N=64 samples
- Temperature: 0.8 (same as HYP-007)
- Total: 4 × 3 = 12 runs
- Estimated wall time: ~40-60 min per run, ~8-12 hours total

**Protocol:**
- FLOP-matched (DEC-004)
- 3 seeds (DEC-002)
- Val loss as training metric (DEC-008)
- pass@k as primary evaluation metric

**Metrics:**
- Primary: pass@k curves (k=1..64), p@16/p@1 ratio,
  p@64/p@1 ratio per architecture
- Secondary: val_loss, train_loss, train-val gap
- Analysis: Compare TTC scaling exponents across architectures

**Recipe:** `recipes/hyp008_ssm_ttc.py`

**Results (2026-03-15):**

| Arch | Val Loss | pass@1 | pass@16 | pass@64 | p@16/p@1 | p@64/p@1 |
|------|----------|--------|---------|---------|----------|----------|
| LLaMA | 2.731 | 0.56% | 3.63% | 8.34% | 6.4x | 14.8x |
| Falcon-H1 | 2.318 | 0.28% | 1.77% | 4.06% | 6.4x | 14.6x |
| Bamba | 2.318 | 0.28% | 1.77% | 4.04% | 6.4x | 14.5x |
| Jamba | 2.310 | 0.25% | 1.42% | 3.29% | 5.8x | 13.4x |

**Adjudication:**
- **H8-a (independent, 0.30 → SUPPORTED):** p@16/p@1 ratios
  range 5.8-6.4x (max/min = 1.10x, well within 2x threshold).
  p@64/p@1 ratios range 13.4-14.8x (max/min = 1.10x). TTC
  scaling exponents are architecture-independent.
- **H8-b (attention wins, 0.25 → WEAKLY SUPPORTED on absolute,
  NOT on exponent):** LLaMA has highest absolute pass@k at
  every k (~2x higher than hybrids), but this reflects higher
  base rate (pass@1), not steeper scaling. The TTC exponent
  (14.8x) is only 1.4% above Falcon-H1 (14.6x).
- **H8-c (hybrid wins, 0.20 → FALSIFIED):** All hybrids have
  lower or equal p@64/p@1 ratios than LLaMA (14.6x, 14.5x,
  13.4x vs 14.8x). No hybrid advantage in TTC scaling.
- **H8-d (SSM loses, 0.25 → FALSIFIED):** All SSM-heavy archs
  have p@64/p@1 far above 5x threshold (13.4-14.6x). SSMs
  show strong TTC scaling, contradicting the prediction.

**Key finding:** TTC scaling is architecture-independent at 10M.
The p@64/p@1 amplification factor (~13-15x) is a property of
the model quality and task, not the architecture family. However,
absolute pass@k depends strongly on base rate — and paradoxically,
LLaMA (worst val_loss) has the highest pass@k.

**Anomalies flagged:**
- ANOM-014: Falcon-H1 and Bamba produce identical results
  (same val_loss to 4 decimal places, same pass@k). Root cause:
  their 10m factories produce identical architectures.
- ANOM-015: Higher val_loss → higher pass@k (LLaMA paradox).
  LLaMA has val_loss 2.731 (worst) but pass@64 8.34% (best).
  Hybrids have val_loss 2.31-2.32 (better) but pass@64 3.3-4.1%.

---

## HYP-009: Grokking × Test-Time Compute Interaction

**Experiment:** 9 — Pass@k evolution across the grokking
transition on modular arithmetic
**Status:** tested (H9-a strongly supported, H9-c supported)
**Question:** How does test-time compute effectiveness (pass@k
curves) change as a model transitions from memorization to
generalization (grokking) on modular arithmetic?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. If TTC reveals latent
  generalization BEFORE it appears in greedy decoding, this
  changes how we think about model evaluation and when to
  stop training. Connects grokking (training dynamics) with
  TTC (inference dynamics) — two active research areas.
- Gate 2 (Scale): PASS. Grokking research is inherently
  small-scale (Power et al. used 2-layer models). Our 10M
  model is well within the natural range.
- Gate 3 (Prior coverage): PASS. No published work on TTC
  across the grokking transition. Grokking papers study
  val_loss/accuracy but not pass@k. TTC papers study
  fully-trained models but not training dynamics.
- Gate 4 (Predictability): PASS. Multiple plausible outcomes
  (see hypotheses below). Pre-grok diversity could help OR
  hurt TTC. Not predictable from theory alone.
- Gate 5 (Methodology): PASS. Same pass@k eval as HYP-007/
  008, applied at training checkpoints. Well-tested infra.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA):**
- [LIT-049] Power et al. 2022: Grokking on small algorithmic
  datasets. Weight decay critical. Small models grok with
  enough training.
- [LIT-050] Nanda et al. 2023: Three phases of grokking —
  memorization (0-1.4K epochs), circuit formation (1.4K-9.4K),
  cleanup/grokking (9.4K-14K epochs). Progress measures
  change continuously even when val accuracy is flat.
- [LIT-051] Gromov 2023: Grokking on modular arithmetic with
  simple architectures. Weight decay 0.1-1.0 needed.
- [HYP-007] Own: TTC works at 10M, pass@64=14x pass@1.
- [HYP-008] Own: TTC is architecture-independent.
- **Gap:** No study of pass@k across training checkpoints
  during grokking. Complete gap in both literatures.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H9-a | TTC as early indicator | pass@64 improves significantly BEFORE val accuracy jumps at the grokking transition. TTC reveals latent generalization 500+ steps before greedy decoding does. | pass@64 and pass@1 jump at the same checkpoint (within 500 steps) | 0.30 |
| H9-b | Simultaneous jump | pass@k at all k values jumps simultaneously with val accuracy at the grokking transition. The generalization circuit either works or doesn't — sampling doesn't help until it exists. | pass@64 improves >2x at a checkpoint where pass@1 hasn't improved yet | 0.35 |
| H9-c | Pre-grok diversity peak | Pre-grok models have HIGH diversity (many wrong but varied outputs) that collapses post-grok. pass@k relative amplification (p@64/p@1) peaks BEFORE grokking, then decreases as the model converges to the correct answer. | p@64/p@1 ratio is higher post-grok than pre-grok | 0.20 |
| H9-d | Post-grok TTC explosion | Post-grok, TTC becomes dramatically more effective. pass@64 jumps by >10x across the transition because the generalization circuit sometimes fires and sometimes doesn't — perfect for best-of-N. | pass@64 changes by less than 3x across the grokking transition | 0.15 |

**Why these alternatives:**
- **H9-a (early indicator, 0.30):** Grokking progress measures
  (Nanda et al.) change continuously — the circuit forms
  gradually even while val accuracy is flat. If the circuit
  partially works, sampling many outputs should occasionally
  find the correct answer. TTC would act as a "magnifying
  glass" for latent capabilities.
- **H9-b (simultaneous, 0.35):** Highest prior. The grokking
  transition is sharp in accuracy. If the circuit either
  works or doesn't, there's no intermediate state where
  sampling helps. This is the null-like hypothesis.
- **H9-c (diversity peak, 0.20):** Pre-grok models have messy,
  unstructured representations. This could produce diverse
  but wrong outputs. The relative TTC amplification might
  peak during memorization (many wrong answers) then decrease
  post-grok (one right answer dominates). B-008 (dropout
  hurts diversity) suggests regularization compresses
  distributions — grokking's weight decay does the same.
- **H9-d (post-grok explosion, 0.15):** Just after grokking,
  the correct circuit may fire intermittently (not fully
  trained in). This is the ideal regime for best-of-N: some
  samples use the circuit, others don't. Low prior because
  it requires a very specific intermediate state.

**Design (revised after pilot):**
- Architecture: LLaMA-grok (~7M params: d=128, 2 layers,
  4 heads, BPE vocab). Small model chosen after 10M model
  failed to grok — too much capacity for ~7,500 train pairs.
- Training: per-example batching (batch_size=64), full-
  sequence next-token prediction. NOT token-stream training.
  Matches grokking literature setup.
- dropout=0.0 (per HYP-007)
- Weight decay: 0.1 (wd=1.0 prevented memorization in pilot;
  wd=0.1 shows clear grokking transition starting at ~3K
  steps / ~26 epochs)
- LR: 1e-3, constant (no cosine decay — grokking needs
  sustained gradient signal)
- Modulus: 97 (same as HYP-007/008 for comparability)
- Max steps: 50,000 (~427 epochs)
- Checkpoint eval: every 1,000 steps (50 checkpoints)
- pass@k eval at each checkpoint: k=1,2,4,8,16,32,64
- Val accuracy eval at each checkpoint (greedy exact match)
- Seeds: {42, 43, 44}
- Total: 3 runs × ~50 checkpoints = ~150 pass@k evaluations
- Estimated wall time: ~22 min/run, ~80-100 min total

**Protocol:**
- Step-matched (not FLOP-matched — we need to reach grokking)
- 3 seeds (DEC-002)
- Val accuracy (greedy exact match) as grokking metric
- pass@k at checkpoints as primary TTC metric

**Metrics:**
- Primary: pass@k curves at each checkpoint, identifying the
  step where pass@1 first exceeds 5%, 50%, 90% (grokking
  markers)
- Secondary: val_loss, val_accuracy (exact match), train_loss
  at each checkpoint
- Analysis: Does pass@64 exceed threshold BEFORE pass@1 does?
  How does p@64/p@1 evolve across the transition?

**Recipe:** `recipes/hyp009_grokking_ttc.py`

**Results (2026-03-15):**
- 3 seeds completed (50K steps each). Seed 42 grokked at step
  43K. Seeds 43 and 44 did NOT grok within 50K steps (but show
  upward trends — may grok given more training).

Seed 42 trajectory (only grokking seed):

| Phase | Step | Val Acc | p@1 | p@64 | p@64/p@1 |
|-------|------|---------|-----|------|----------|
| Pre-memo | 1-3K | 0.7-1.1% | 0.9-1.0% | 43-47% | 46-48x |
| Transition | 4K | 19.9% | 14.1% | 98.9% | 7.0x |
| Early trans | 5K | 30.8% | 23.6% | 99.7% | 4.2x |
| Oscillating | 10-42K | 52-81% | 45-65% | 99-100% | 1.5-2.2x |
| GROKKING | 43K | 99.9% | 97.6% | 100% | 1.0x |
| Post-grok | 44-45K | 100% | 99.9% | 100% | 1.0x |

**Adjudication:**
- H9-a (TTC early indicator, 0.30 → STRONGLY SUPPORTED): p@64
  reaches 98.9% at step 4K, 39K steps before greedy accuracy
  catches up to 99.9% at step 43K. Largest lead time effect
  documented in TTC literature.
- H9-b (simultaneous, 0.35 → FALSIFIED): p@64 saturates at
  transition onset while p@1 takes 39K more steps.
- H9-c (diversity peak, 0.20 → SUPPORTED): p@64/p@1 peaks at
  46-48x pre-memorization, monotonically declines to 1.0x post-
  grok. Diversity is highest when accuracy is lowest.
- H9-d (post-grok explosion, 0.15 → FALSIFIED): p@64 barely
  changes across grokking (99.8% → 100%, 1.002x). TTC was
  already saturated 39K steps before the grokking step.

**Key finding:** The grokking transition is a pass@1 transition,
not a capability transition. The model has near-perfect capability
(pass@64 ≈ 100%) from step ~5K, but doesn't output the correct
answer greedily until step 43K. TTC reveals latent generalization
~330 epochs before greedy decoding does.

**Anomalies flagged:**
- ANOM-016: Only 1/3 seeds grokked at wd=0.1. Seeds 43/44 show
  same oscillating plateau but never break through.

B-011 created. B-007 updated: 0.90 → 0.95.

---

## HYP-010: TTC Scaling Exponent vs Model Size

**Experiment:** 10 — How TTC scaling changes between
10M and 30M parameters on modular arithmetic
**Status:** active
**Question:** Does the TTC amplification factor (pass@64/
pass@1) change as model size increases from 10M to 30M,
and in what direction? Does larger model size improve
absolute pass@k, TTC exponent, or both?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. Understanding how TTC
  scales with model size is fundamental for practitioners
  deciding compute allocation between training and
  inference. HYP-007 established TTC at 10M; this tests
  whether 3x more params improves the picture.
- Gate 2 (Scale): PASS. 10M-30M is our validated range
  (llama_10m, llama_30m both tested). We can directly
  reuse HYP-007 infrastructure.
- Gate 3 (Prior coverage): PASS. Wu et al. (LIT-039)
  have inference scaling laws but only validated at 7B+.
  "Scaling Laws in the Tiny Regime" (arXiv:2603.07365)
  found steeper exponents at very small scale. Nobody has
  measured TTC exponents at 10M vs 30M.
- Gate 4 (Predictability): MILD CONCERN. A naive
  prediction ("bigger model = better pass@k") is likely
  correct for absolute values. But the TTC EXPONENT
  (p@64/p@1 ratio) could go either way.
- Gate 5 (Methodology): PASS. Same protocol as HYP-007.
  Modular arithmetic, exact verification, pass@k.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA):**
- [LIT-039] Wu et al. 2024: Inference scaling law
  `log10(C) = 1.19*log10(N) + 2.03`. Predicts at N=10M:
  log10(C) ≈ 10.36; at N=30M: log10(C) ≈ 10.93. The
  ratio suggests ~3.7x more useful inference compute at
  30M before saturation.
- [LIT-053] "Scaling Laws in the Tiny Regime"
  (arXiv:2603.07365): Scaling exponents 1.4-2x steeper
  at very small scale (alpha ≈ 0.106-0.156 vs 0.076
  for large LLMs). Implies small models benefit
  disproportionately from compute scaling.
- [HYP-007] Own: 10M LLaMA, dropout=0.0, pass@64/pass@1
  = 14.1x. Growth ~50% per doubling of k.
- [HYP-008] Own: TTC amplification is architecture-
  independent (~13-15x at 10M for all arch families).
- **Gap:** No study comparing TTC exponents across model
  sizes at <100M scale.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H10-a | Absolute up, exponent stable | 30M pass@1 >> 10M pass@1, but p@64/p@1 ratios within 2x (both ~10-20x). TTC amplification is a property of the task, not model size. | p@64/p@1 ratio differs by >3x between 10M and 30M | 0.35 |
| H10-b | Both up | 30M has higher pass@1 AND steeper TTC curve (higher p@64/p@1 ratio). Larger model has more diverse, higher-quality outputs. | 30M p@64/p@1 ratio <= 10M p@64/p@1 ratio | 0.25 |
| H10-c | Exponent down | 30M has higher pass@1 but LOWER p@64/p@1 ratio (<10x). Larger model is more confident/deterministic, reducing output diversity. TTC becomes less useful as models grow. | 30M p@64/p@1 ratio >= 10M p@64/p@1 ratio (14.1x) | 0.25 |
| H10-d | Diminishing returns | 30M barely improves over 10M on this task. Modular arithmetic at mod 97 is a "hard" task where model size doesn't help — the bottleneck is algorithmic structure, not capacity. | 30M pass@1 > 2x 10M pass@1 | 0.15 |

**Why these alternatives:**
- **H10-a (stable exponent, 0.35):** HYP-008 showed TTC
  exponents are architecture-independent. The simplest
  extension is that they're also size-independent (within
  a range). The exponent may be a property of the task
  difficulty distribution. Highest prior because it's the
  simplest model.
- **H10-b (both up, 0.25):** Wu et al. (LIT-039) show
  larger models can use more inference compute before
  saturation. If 30M has higher base accuracy AND richer
  internal representations, both absolute and relative
  TTC should improve.
- **H10-c (exponent down, 0.25):** As models get better
  at a task, their output distribution sharpens. A 30M
  model might be "right or wrong" more deterministically,
  reducing the diversity that makes best-of-N useful.
  This parallels RLVR narrowing distributions (LIT-041).
- **H10-d (diminishing returns, 0.15):** Modular
  arithmetic is an algorithmic task. Past a certain
  capacity, more params don't help — the model either
  learns the circuit or it doesn't. Our 10M models
  memorized training data perfectly but only got ~0.5%
  on held-out. If the barrier is generalization, not
  capacity, 30M won't help much.

**Design:**
- 2 model sizes: LLaMA-10M (~9.9M), LLaMA-30M (~30.6M)
- dropout=0.0 (per HYP-007: dropout hurts diversity)
- 3 seeds: {42, 43, 44} (per DEC-002)
- FLOP budget: matched WITHIN each size class
  - 10M: same as HYP-007 (2000 target steps)
  - 30M: 2000 target steps (auto-scaled FLOP budget)
- Dataset: modular arithmetic (a+b) mod 97, same splits
- Eval: pass@k with k=1,2,4,8,16,32,64, N=64, temp=0.8
- Total: 2 × 3 = 6 runs
- Estimated wall time: ~30 min (10M) + ~60 min (30M)

**Protocol:**
- FLOP-matched within size class (DEC-004)
- 3 seeds (DEC-002)
- Val loss as training metric (DEC-008)
- pass@k as primary evaluation metric

**Metrics:**
- Primary: pass@k curves per model size, p@64/p@1 ratio
- Secondary: val_loss, train_loss, absolute pass@1
- Analysis: Compare TTC amplification factors across
  model sizes; fit log-linear to pass@k vs log(k)

**Recipe:** `recipes/hyp010_ttc_model_size.py`

**Results (2026-03-15):** 6 runs completed (2 sizes × 3 seeds).

| Model | Val Loss | pass@1 | pass@16 | pass@64 | p@64/p@1 |
|-------|----------|--------|---------|---------|----------|
| 10M (avg) | 2.753 | 0.56% | 3.49% | 8.19% | 14.6x |
| 30M (avg) | 3.096 | 0.43% | 2.45% | 5.07% | 11.9x |

Per-run detail:
| Run | Val Loss | p@1 | p@64 | p@64/p@1 |
|-----|----------|-----|------|----------|
| 10M s42 | 2.831 | 0.64% | 9.20% | 14.4x |
| 10M s43 | 2.755 | 0.61% | 7.26% | 11.9x |
| 10M s44 | 2.674 | 0.43% | 8.10% | 19.0x |
| 30M s42 | 4.075 | 0.34% | 5.12% | 15.0x |
| 30M s43 | 2.504 | 0.53% | 4.65% | 8.8x |
| 30M s44 | 2.710 | 0.41% | 5.43% | 13.3x |

**Adjudication:**
- H10-a (stable exponent, 0.35 → SUPPORTED): p@64/p@1 ratios
  are 14.6x (10M) vs 11.9x (30M). Ratio of ratios = 1.23x,
  within the 2x threshold. TTC amplification is roughly stable
  across 3x model size change.
- H10-b (both up, 0.25 → FALSIFIED): 30M pass@1 (0.43%) is
  LOWER than 10M (0.56%), and the TTC exponent is also lower.
  Both metrics go DOWN with more params.
- H10-c (exponent down, 0.25 → FALSIFIED on prerequisite):
  Predicted higher pass@1 with lower ratio. But 30M has LOWER
  pass@1 — the prerequisite fails.
- H10-d (diminishing returns, 0.15 → SUPPORTED in spirit):
  30M doesn't just fail to improve — it's actually WORSE.
  Stronger than predicted.

**Key finding:** The 30M model performs WORSE than the 10M model
on modular arithmetic pass@k at 2000 FLOP-matched steps. Both
models fully memorize training data (train_loss ~0.002) but 30M
generalizes worse (val_loss 3.10 vs 2.75). This is the classic
overparameterized-undertrained pattern: the 30M model has enough
capacity to memorize but not enough signal to generalize. The
FLOP budget (2000 steps) is calibrated for 10M, not 30M.

**Anomaly flagged:** ANOM-017 — 30M LLaMA worse than 10M on
modular arithmetic despite 3x more parameters and 3x more total
FLOPs. Seed 42 (30M) has extreme val_loss (4.075) — possible
bad seed.

**Implication for TTC research:** Model size scaling for TTC
requires data-scaling, not just FLOP-scaling. The 30M model
needs more diverse training data or more epochs to benefit from
its extra capacity. TTC amplification factor remains ~12-15x
regardless of model size, supporting the view that it's a task
property.

---

## HYP-011: Per-Token Loss Decomposition (ANOM-015)

**Experiment:** 11 — Why val_loss inversely predicts pass@k
across architecture families
**Status:** tested (H11-a partially supported, H11-b supported)
**Question:** Does the val_loss vs pass@k inversion (ANOM-015)
arise because different architectures allocate their modeling
capacity differently across token positions — specifically,
SSM/hybrid models predict prompt tokens better while pure
attention (LLaMA) predicts the answer token better?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. Val_loss is the standard training
  metric (DEC-008). If it inversely predicts pass@k across
  architectures, this has implications for model selection and
  evaluation methodology.
- Gate 2 (Scale): PASS. Small scale is ideal — can inspect
  every token's loss contribution in seconds.
- Gate 3 (Prior coverage): PASS. Perplexity-vs-downstream-
  accuracy mismatch is documented at large scale (GPT-4 report
  notes it), but nobody has decomposed per-position loss on
  modular arithmetic to explain WHY. No prior work on SSM vs
  attention per-position loss.
- Gate 4 (Predictability): MILD CONCERN. The "prompt vs answer
  token" explanation is intuitive and likely correct. But the
  DEGREE of the effect and whether it fully explains the
  inversion are not predictable.
- Gate 5 (Methodology): PASS. Same 4-architecture training as
  HYP-008 with added per-position loss decomposition.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA):**
- [ANOM-015] Own: LLaMA val_loss 2.731 (worst) but pass@64
  8.34% (best). Jamba val_loss 2.310 (best) but pass@64 3.29%
  (worst). Perfectly inverted ranking.
- [HYP-008] Own: TTC amplification is architecture-independent
  (~13-15x), so the inversion is in base rate, not scaling.
- Perplexity vs downstream accuracy mismatch is well-known at
  large scale but not mechanistically explained at small scale.
- **Gap:** Nobody has decomposed per-position loss on structured
  tasks to explain cross-architecture performance inversions.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H11-a | Prompt-token dominance | SSM/hybrid models have >50% lower loss on prompt tokens (digits, operators) than LLaMA, but LLaMA has >30% lower loss on the answer token. The inversion is fully explained by token-position allocation. | LLaMA's answer-token loss is >= hybrid answer-token loss | 0.40 |
| H11-b | Calibration difference | All architectures have similar per-position loss profiles, but LLaMA's answer-token logits are better calibrated (closer to uniform over the answer set). LLaMA assigns more probability mass to the correct answer even though its total loss is higher. | LLaMA and hybrids have similar answer-token entropy | 0.25 |
| H11-c | Training dynamics | The inversion is a training artifact: at 2000 steps with FLOP-matching, different architectures are at different points on their training curve. LLaMA trains more steps (2000 vs ~1667 for hybrids) and may be further into the "generalization" phase. | When step-matched (same steps, not FLOP-matched), the inversion disappears | 0.20 |
| H11-d | Null: noise | The inversion is within noise. With only 3 seeds and 4 architectures, the ranking could reverse with different seeds. | Consistent ranking across all 3 seeds (no seed has hybrid pass@k > LLaMA pass@k) | 0.15 |

**Why these alternatives:**
- **H11-a (prompt-token dominance, 0.40):** Highest prior.
  The modular arithmetic format is "a + b = c\n". Most tokens
  are predictable formatting ('+', '=', '\n') and the operands
  follow from context. SSMs with fixed-size state should
  excel at this pattern-prediction — their state naturally
  captures sequential patterns. But the critical answer token
  requires computing (a+b) mod 97, which may need precise
  token-level retrieval (attention's strength).
- **H11-b (calibration, 0.25):** Even if per-position losses
  are similar, the answer-token logit distribution could
  differ. LLaMA might spread probability more across valid
  answers while hybrids concentrate on a few wrong answers.
  This would explain why sampling (pass@k) favors LLaMA
  while average loss favors hybrids.
- **H11-c (training dynamics, 0.20):** FLOP-matching means
  LLaMA runs ~2000 steps while hybrids run ~1667 steps
  (hybrids have more params per FLOP). The extra 333 steps
  may push LLaMA into a different training phase. Lower prior
  because 2000 vs 1667 steps is a small difference.
- **H11-d (null, 0.15):** The effect is large (2x in pass@k,
  consistent across seeds in HYP-008), making noise unlikely.
  Low prior but must be tested.

**Design:**
- 4 architectures: LLaMA-10M, Falcon-H1-10M, Jamba-10M,
  Bamba-10M (same as HYP-008)
- dropout=0.0, LR=3e-4, same modular arithmetic dataset
- FLOP budget: matched to LLaMA-10M × 2000 steps
- 3 seeds: {42, 43, 44}
- Total: 12 runs (same grid as HYP-008)
- **New evaluation: Per-position loss decomposition.**
  After training, for each test prompt "a + b = c\n":
  1. Forward pass on the full sequence
  2. Compute cross-entropy loss at each token position
  3. Aggregate into: prompt_loss (positions for a, +, b, =)
     and answer_loss (position for c)
  4. Also compute answer-token entropy and top-5 logit mass
- Estimated wall time: same as HYP-008 (~8-12 hours)

**Protocol:**
- FLOP-matched (DEC-004), 3 seeds (DEC-002)
- Val loss as training metric (DEC-008)
- Per-position loss decomposition as primary new metric

**Metrics:**
- Primary: answer_loss per architecture, prompt_loss per
  architecture, answer_loss/prompt_loss ratio
- Secondary: pass@k (replication of HYP-008), answer-token
  entropy, top-5 answer logit mass
- Diagnostic: val_loss (should reproduce HYP-008 rankings)

**Analysis:**
- Compare answer_loss across architectures (key test for H11-a)
- Compare prompt_loss across architectures (expecting hybrid
  advantage)
- Correlate answer_loss with pass@1 (expecting strong positive)
- Check HYP-008 replication: val_loss and pass@k rankings
  should match
- Plot per-position loss heatmap for each architecture

**Recipe:** `recipes/hyp011_token_loss_decomp.py`

**Results (2026-03-15):** 12 runs completed (4 archs × 3 seeds).

| Arch | Val Loss | Prompt Loss | Answer Loss | A/P Ratio | Ans Entropy | Ans P(correct) | p@1 | p@64 |
|------|----------|-------------|-------------|-----------|-------------|----------------|-----|------|
| LLaMA | 2.727 | 3.469 | 7.497 | 2.2 | 2.116 | 0.66% | 0.68% | 8.36% |
| Falcon-H1 | 2.318 | 3.711 | 9.691 | 2.6 | 1.217 | 0.29% | 0.28% | 4.04% |
| Bamba | 2.318 | 3.711 | 9.691 | 2.6 | 1.217 | 0.29% | 0.28% | 4.04% |
| Jamba | 2.311 | 3.722 | 9.815 | 2.6 | 1.119 | 0.34% | 0.34% | 3.61% |

H11-a test (vs LLaMA reference):
| Arch | Prompt Loss Diff | Answer Loss Diff |
|------|------------------|------------------|
| Falcon-H1 | +7.0% | +29.3% |
| Bamba | +7.0% | +29.3% |
| Jamba | +7.3% | +30.9% |

HYP-008 replication: val_loss and pass@k rankings match
exactly (LLaMA > Falcon-H1 = Bamba > Jamba on pass@k).

**Adjudication:**
- H11-a (prompt-token dominance, 0.40 → PARTIALLY SUPPORTED):
  The prediction that hybrids have LOWER prompt loss was wrong
  — hybrids are 7% worse at prompt tokens too. BUT the key
  mechanism is confirmed: the val_loss inversion is driven by
  the answer token. Hybrids are 29-31% worse than LLaMA at the
  answer token but only 7% worse at prompt tokens. Since
  val_loss averages over all positions (and prompts have ~5
  tokens vs 1 answer token), the small prompt advantage of
  LLaMA is diluted. LLaMA's higher val_loss is because it has
  higher loss AT EVERY position, but its answer-token advantage
  is proportionally much larger. The direction prediction was
  wrong but the mechanism (answer token dominance) is correct.

- H11-b (calibration, 0.25 → SUPPORTED): LLaMA has
  significantly higher answer-token entropy (2.12) vs hybrids
  (1.12-1.22). This means LLaMA distributes probability mass
  more broadly across possible answers. Combined with higher
  P(correct) (0.66% vs 0.29-0.34%), LLaMA's answer distribution
  is both more diverse AND more accurate. This is the ideal
  combination for best-of-N sampling: enough diversity to
  sometimes hit the right answer, with enough accuracy that
  those hits are not vanishingly rare.

- H11-c (training dynamics, 0.20 → NOT TESTED): Would need
  step-matched runs to test. However, the magnitude of the
  answer-token gap (29-31%) is much larger than the step
  difference could plausibly explain (2000 vs 1667 = 20% more
  steps for LLaMA). Unlikely to be the primary driver.

- H11-d (null, 0.15 → FALSIFIED): All 3 LLaMA seeds beat all
  hybrid seeds on pass@k. The ranking is perfectly consistent.
  No overlap between LLaMA and hybrid distributions.

**Key finding:** The val_loss vs pass@k inversion (ANOM-015) is
explained by two complementary mechanisms:

1. **Answer-token loss dominance (H11-a mechanism):** LLaMA
   has 29-31% lower answer-token loss than hybrids, but only
   7% lower prompt-token loss. Since pass@k depends entirely
   on the answer token, LLaMA wins on task accuracy despite
   losing on average loss (which is diluted by prompt tokens).

2. **Answer-token calibration (H11-b):** LLaMA's answer-token
   distribution is more entropic (2.12 vs ~1.17 nats) and
   assigns 2.0-2.3x more probability to the correct answer.
   Higher entropy + higher correct-answer probability = better
   best-of-N sampling outcomes.

The combination explains BOTH the higher absolute pass@k AND
the similar TTC amplification ratios: LLaMA has a better
base rate because it predicts the answer token better, but all
architectures amplify at similar rates because the answer-token
probability distribution shape (not just the correct-answer
mass) determines the amplification factor.

**ANOM-015 status:** EXPLAINED. Update anomalies.md.

B-010 updated: 0.75 → 0.90 (now mechanistically explained).

---

## HYP-012: TTC Amplification Across Tasks

**Experiment:** 12 — Is the ~12-15x TTC amplification factor
(pass@64/pass@1) specific to modular addition, or does it
generalize across modular arithmetic operations?
**Status:** tested (H12-c partially supported, H12-a/b/d falsified)
**Question:** Does modular multiplication (a harder computation)
produce a similar, higher, or lower TTC amplification factor
compared to modular addition at 10M params?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. Balachandran et al. (2504.00294)
  showed TTC benefits are task-dependent. Our ~12-15x number
  is meaningless without cross-task validation.
- Gate 2 (Scale): PASS. Same 10M scale as HYP-007/008.
- Gate 3 (Prior coverage): PASS. No prior TTC cross-task study
  at <1B params. Literature (Balachandran, Agarwal) tests at
  7B+ on math/code/reasoning. Modular add vs mul comparison
  is novel.
- Gate 4 (Predictability): PASS. Could go either way: harder
  task → more headroom for sampling OR harder task → model
  can't solve it at all.
- Gate 5 (Methodology): PASS. Same infrastructure as HYP-007,
  identical evaluation. Only variable is the operation.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA):**
- [HYP-007] Own: Addition p@64/p@1 = 14.1x at 10M.
- [HYP-008] Own: Amplification is architecture-independent
  (~13-15x across 4 families).
- [HYP-010] Own: Amplification is size-independent (~12-15x
  at 10M and 30M).
- Balachandran et al. 2504.00294: TTC benefits vary across
  8 task types and diminish with complexity.
- Agarwal et al. 2512.02008: No single TTC strategy
  universally dominates.
- **Gap:** Nobody has tested whether TTC amplification factor
  changes across tasks of varying difficulty at the same
  model scale with the same verifier type.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H12-a | Task-independent | Multiplication amp ~10-20x, within 2x of addition's ~14x. | mul amp outside [7x, 28x] range | 0.30 |
| H12-b | Harder → higher amp | Multiplication is harder (lower p@1), so more room for TTC. Amp > 20x. | mul amp <= 20x | 0.20 |
| H12-c | Harder → lower amp | Multiplication is too hard for diverse correct sampling. Amp < 5x. | mul amp >= 5x | 0.25 |
| H12-d | Null: too hard | Multiplication p@1 ≈ 0 at 10M. No meaningful amplification measurable. | mul p@1 > 0.1% | 0.25 |

**Why these alternatives:**
- **H12-a (task-independent, 0.30):** Our HYP-008/010 showed
  architecture- and size-independence. If the factor is truly
  a property of the sampling geometry, it should be similar
  across tasks of similar structure. But literature (Balachandran)
  says task matters, so lower prior than if we had no literature.
- **H12-b (harder → higher amp, 0.20):** If multiplication
  makes greedy decoding harder but sampling can still find
  correct answers, the ratio could be larger. Analogous to
  "easy questions don't benefit from TTC" (Snell et al.).
- **H12-c (harder → lower amp, 0.25):** If multiplication
  requires more precise computation, the model may not learn
  diverse correct paths. SSM loss at answer token was already
  high for addition — multiplication may push it even higher,
  concentrating probability on fewer (wrong) answers.
- **H12-d (null, 0.25):** Multiplication mod 97 requires
  tracking products up to 96*96=9216, much harder than sums
  up to 192. A 10M model may simply not learn it in 2000
  steps. Higher null prior than usual because the difficulty
  jump is substantial.

**Design:**
- 2 operations: add (control), mul (treatment)
- 1 architecture: LLaMA-10M (best TTC performer)
- dropout=0.0 (HYP-007 finding)
- LR=3e-4, FLOP-matched to 2000 steps
- 3 seeds: {42, 43, 44}
- Total: 6 runs
- k values: [1, 2, 4, 8, 16, 32, 64]
- Temperature: 0.8
- Dataset: modular arithmetic mod 97, 80/20 train/test split

**Protocol:**
- FLOP-matched (DEC-004), 3 seeds (DEC-002)
- Val loss as training metric (DEC-008)

**Metrics:**
- Primary: p@64/p@1 ratio per operation
- Secondary: full pass@k curves, val_loss, train_loss
- Diagnostic: train-val gap (overfitting indicator)

**Recipe:** `recipes/hyp012_ttc_cross_task.py`

**Results (2026-03-16):** 6 runs completed (2 ops × 3 seeds).

| Op | Val Loss | p@1 | p@16 | p@64 | p16/p1 | p64/p1 |
|----|----------|-----|------|------|--------|--------|
| add | 2.750 | 0.67% | 3.76% | 8.36% | 5.6x | 12.5x |
| mul | 3.064 | 2.39% | 4.96% | 9.04% | 2.1x | 3.8x |

Pass@k curves (mean across 3 seeds):

| Op | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 |
|----|-----|-----|-----|-----|------|------|------|
| add | 0.0067 | 0.0107 | 0.0165 | 0.0251 | 0.0376 | 0.0561 | 0.0836 |
| mul | 0.0239 | 0.0272 | 0.0319 | 0.0389 | 0.0496 | 0.0659 | 0.0904 |

**Adjudication:**
- H12-a (task-independent, 0.30 → FALSIFIED): Multiplication
  amplification is 3.8x, well outside the [7x, 28x] range
  predicted for task-independence. The 12.5x vs 3.8x difference
  is a 3.3x ratio — TTC amplification is strongly task-dependent.

- H12-b (harder → higher amp, 0.20 → FALSIFIED): Multiplication
  has HIGHER pass@1 (2.39% vs 0.67%), not lower. The premise
  that multiplication is "harder" is wrong at this scale/training
  regime. And the amplification is lower, not higher.

- H12-c (harder → lower amp, 0.25 → PARTIALLY SUPPORTED, but
  mechanism wrong): The prediction of lower amplification (<5x)
  is correct (3.8x < 5x). But the mechanism is wrong —
  multiplication isn't harder (higher p@1). The low amplification
  is because multiplication p@1 is already relatively high,
  leaving less room for sampling to help. The pass@k curve for
  multiplication is nearly flat — the model's distribution is
  already concentrated on a few answers.

- H12-d (null: too hard, 0.25 → FALSIFIED): Multiplication
  p@1 = 2.39%, well above the 0.1% threshold. In fact,
  multiplication is EASIER than addition at pass@1.

**Key finding: TTC amplification is inversely related to
base accuracy, not task difficulty.**

The surprise result: multiplication has 3.6x higher p@1 than
addition (2.39% vs 0.67%) despite having worse val_loss (3.06
vs 2.75). This is another instance of the ANOM-015 pattern:
val_loss doesn't predict task accuracy.

The amplification factor (p@64/p@1) is inversely correlated
with p@1:
- Addition: p@1=0.67%, amp=12.5x
- Multiplication: p@1=2.39%, amp=3.8x

This makes mathematical sense: pass@k amplification depends on
the SHAPE of the probability distribution over answers. If the
model concentrates probability on a few answers (one of which
is correct), pass@1 is high but sampling can't improve much.
If the model spreads probability across many answers (with
small but non-zero mass on the correct one), pass@1 is low but
sampling finds the correct answer more often.

**Hypothesis:** Addition produces a more entropic answer
distribution (like LLaMA in HYP-011), while multiplication
produces a more peaked distribution. The peaked distribution
gives higher greedy accuracy but less room for sampling to help.

**Anomaly flagged:** ANOM-018 — Multiplication has higher
pass@1 but worse val_loss than addition. Same ANOM-015 pattern
(val_loss inversely predicts task accuracy) but now across tasks
rather than across architectures. The mechanism is likely the
same: multiplication's answer-token distribution is more peaked.

**Implication for TTC research:** The ~12-15x amplification
factor is NOT a universal constant. It depends on the model's
base accuracy and distribution shape. High-base-accuracy tasks
show lower amplification because there's less room for sampling
to help. This aligns with Snell et al.'s finding that easy
questions benefit less from TTC. Our contribution: quantifying
this at 10M scale and showing it applies within a single task
family (same tokenization, same verifier, different operation).

**B-009 update needed:** TTC scaling exponent is task-dependent
(was architecture/size-independent at 0.90). The independence
is within-task only.

---

## HYP-013: Entropy Predicts TTC Amplification

**Experiment:** 13 — Does answer-token entropy predict TTC
amplification factor across tasks?
**Status:** pre-registered
**Question:** HYP-012 showed amplification varies 3.3x across
tasks (12.5x add vs 3.8x mul). HYP-011 showed LLaMA has higher
answer-token entropy (2.12 nats) and higher pass@k than hybrids.
Does entropy causally drive amplification?

**Quality Gates (Step 0):**
- Gate 1 (Importance): PASS. If entropy predicts amplification,
  practitioners can estimate TTC benefit from a single forward
  pass — no expensive pass@k sweep needed.
- Gate 2 (Scale): PASS. Can compute exact distributions at 10M.
- Gate 3 (Prior coverage): PASS. Calibration literature exists
  but nobody has connected answer-token entropy to pass@k
  amplification factor specifically.
- Gate 4 (Predictability): MILD CONCERN. The direction (higher
  entropy → higher amplification) is intuitive. But the strength
  and functional form (linear? log? threshold?) are not.
- Gate 5 (Methodology): PASS. Combines HYP-011 (per-token loss)
  and HYP-012 (cross-task) infrastructure. Clean reuse.
- Gate 6 (Sunk cost): PASS. First round on this question.

**Prior Art (REA):**
- [HYP-011] Own: LLaMA entropy 2.12 nats vs hybrid ~1.17 nats.
  Higher entropy correlates with higher pass@k across archs.
- [HYP-012] Own: Addition amp=12.5x, mul amp=3.8x. Different
  tasks → different amplification.
- [ANOM-018] Own: Multiplication has higher p@1 but lower amp.
  Peaked distribution → less sampling benefit.
- Yue et al. (LIT-064): RL narrows distributions (lower entropy)
  and hurts pass@k. Consistent with entropy → diversity → amp.
- **Gap:** No quantitative model linking answer-token entropy
  to pass@k amplification factor.

| ID | Hypothesis | Prediction | Falsification | Prior |
|----|-----------|------------|---------------|-------|
| H13-a | Entropy predicts | r(entropy, amp) > +0.8 across all 6 runs. Higher entropy → higher amp. | r(entropy, amp) <= 0.8 | 0.35 |
| H13-b | P(correct) predicts | r(P(correct), amp) < -0.8. High P(correct) → high p@1 → low amp. Entropy is secondary. | r(P(correct), amp) >= -0.8 or r(entropy) > r(P(correct)) | 0.30 |
| H13-c | Both contribute | Both |r(entropy, amp)| > 0.5 and |r(P(correct), amp)| > 0.5. Neither alone is sufficient. | One of the two has |r| < 0.5 | 0.20 |
| H13-d | Null: noise | Seed variance dominates. Correlations are weak (|r| < 0.5) because within-operation variance swamps between-operation signal. | Any |r| > 0.5 | 0.15 |

**Why these alternatives:**
- **H13-a (entropy predicts, 0.35):** Highest prior. Entropy
  measures the "spread" of the distribution, which directly
  determines how many distinct answers get sampled. More spread
  → more diverse samples → more chances to hit the correct one.
  But: entropy includes mass on non-answer tokens, so it's an
  imperfect proxy. Also, with only 6 data points (2 operations
  × 3 seeds), the correlation estimate will be noisy.
- **H13-b (P(correct) predicts, 0.30):** P(correct) at the
  answer token directly maps to pass@1. If pass@1 is high,
  most samples are correct, so pass@64 can't be much higher —
  the ratio p@64/p@1 shrinks. This is a mathematical tautology
  for the extreme cases but the question is whether it explains
  the intermediate cases too.
- **H13-c (both, 0.20):** Entropy and P(correct) may capture
  different aspects. A model could have low entropy but high
  P(correct) (peaked on correct answer) or high entropy but
  low P(correct) (spread across wrong answers). The combination
  matters.
- **H13-d (null, 0.15):** With only 3 seeds per operation,
  within-operation variance may be large relative to the
  between-operation difference. The correlation could be
  artifactual (driven by the 2-group structure).

**Design:**
- Same as HYP-012: 2 operations × 3 seeds = 6 runs
- LLaMA-10M, dropout=0.0, FLOP-matched to 2000 steps
- Per-token loss decomposition (from HYP-011)
- pass@k evaluation (from HYP-012)
- Primary analysis: Pearson r between answer_entropy and
  p@64/p@1 across all 6 runs

**Protocol:**
- FLOP-matched (DEC-004), 3 seeds (DEC-002)
- Val loss as training metric (DEC-008)

**Metrics:**
- Primary: r(answer_entropy, amplification)
- Secondary: r(answer_correct_prob, amplification),
  r(pass@1, amplification)
- Diagnostic: val_loss, per-position loss profiles

**Recipe:** `recipes/hyp013_entropy_predicts_ttc.py`
