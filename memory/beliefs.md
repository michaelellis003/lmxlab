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

| 2026-03-16 | HYP-012: p@64/p@1 = 12.5x (addition) vs 3.8x (multiplication). 3.3x ratio across tasks. TTC amplification is strongly task-dependent. | F | Very strong against task-independence | 0.60 |

**Update (HYP-012):** CRITICAL UPDATE. Amplification factor is
NOT task-independent. Within a single task (modular add mod 97),
it's architecture- and size-independent (~12-15x). But across
tasks (add vs mul, same model, same verifier), it varies 3.3x.
Belief revised: "TTC exponent is architecture- and size-
independent WITHIN a task, but task-dependent ACROSS tasks."
This is consistent with Balachandran et al. (2504.00294).

The mechanism: amplification depends inversely on base accuracy
(p@1). Multiplication has 3.6x higher p@1 (2.39% vs 0.67%),
leaving less room for sampling to improve. The distribution
shape (peaked vs spread) determines the amplification potential.

Current posterior: 0.60 (reduced from 0.90 — the belief was
too broad; now correctly scoped to within-task only).

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
| 2026-03-16 | HYP-012: Cross-task. Multiplication val_loss 3.06 (worse) but p@1 2.39% (3.6x better than addition's 0.67%). Same ANOM-015 pattern now across tasks, not just architectures. | F | Very strong for | 0.95 |

**Update (HYP-012):** The val_loss vs task accuracy disconnect
now replicated across tasks (not just architectures). Multiplication
has worse val_loss but higher pass@1. Posterior → 0.95.
Belief broadened: "Val loss does not predict pass@k ranking
across architectures OR across tasks."

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
| 2026-03-16 | HYP-014: Replicated across 4 architectures. All 4 show pass@64 > 46% at step 2K, long before any architecture groks. Bamba pass@64=99.6% at step 2K, groks at step 20K. | C | Moderate for | 0.85 |

**Update (HYP-014):** Now replicated across 4 architecture
families. pass@64 saturates early for all architectures, long
before greedy accuracy grokking occurs. The lead time varies
(Bamba: 18K steps, LLaMA: 42K steps) but the phenomenon is
universal. Posterior → 0.85.

---

## B-012: P(correct) at answer token predicts TTC amplification

**Prior:** N/A (new belief from HYP-013)
**Current:** 0.85
**Source:** HYP-013 experiment results

The probability assigned to the correct answer token in a single
forward pass predicts TTC amplification factor with r=-0.98.
Higher P(correct) → higher pass@1 → lower amplification ratio.
Answer-token entropy is also predictive (r=+0.88) but weaker.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-16 | HYP-013: r(P(correct), amp)=-0.981, r(entropy, amp)=+0.879 across 6 runs (2 ops × 3 seeds). Mul seed 43 outlier confirms within-operation variance tracks too. | F | Very strong for | 0.85 |

**Why 0.85 and not higher:**
- Only 6 data points across 2 operations — high r values are
  easy to get with small n and 2-group structure
- Only tested on modular arithmetic with one architecture
- The near-tautological nature (pass@1 ≈ P(correct), amp =
  p@64/p@1) means the relationship may be definitional rather
  than predictive in a useful sense
- Need to test whether P(correct) predicts amplification
  across architectures (not just tasks) before 0.90+

---

## B-013: SSM layers accelerate grokking on modular arithmetic

**Prior:** N/A (new belief from HYP-014)
**Current:** 0.70
**Source:** HYP-014 experiment results

Hybrid architectures with Mamba-2 SSM layers grok modular
arithmetic (mod 97) faster than pure attention (LLaMA).
Grokking order: Bamba (20K) > Falcon-H1 (26K) > Jamba (36K)
> LLaMA (44K). The 2.2x speedup for Bamba is substantial.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-16 | HYP-014: 4 architectures, all grokked. Bamba 2.2x faster than LLaMA. All hybrids faster. Early TTC signal (step 2K) predicts grokking order. | F | Strong for | 0.70 |

**Why 0.70 and not higher:**
- Only 1 seed per architecture. Grokking is seed-dependent
  (HYP-009 showed 1/3 grokking rate). Need multi-seed
  replication.
- Only tested on modular addition mod 97. Other tasks may
  show different patterns.
- Confounds: Jamba has MoE (more params), Bamba and Falcon-H1
  have same param count but different layer patterns. Hard to
  attribute to SSM layers vs other design differences.
- Grokking instability in Jamba/Bamba complicates the picture.

**HYP-015 update:** MoE-Jamba groks 3/3 seeds vs 1/3 for
noMoE-Jamba. MoE capacity HELPS grokking. But this is
confounded by MoE having 590K more params. The SSM
acceleration claim (B-013) is unchanged because it compares
across architectures (with SSM vs pure attention), not
across MoE variants. Posterior stays at 0.70.

---

## B-014: TTC signal at early training predicts grokking order

**Prior:** N/A (new belief from HYP-014)
**Current:** 0.30
**Source:** HYP-014 experiment results, revised by HYP-016

pass@64 at step 2000 perfectly predicts the eventual grokking
ordering across 4 architectures: Bamba(99.6%) > Jamba(96.3%)
> Falcon-H1(78.5%) > LLaMA(46.2%). However, HYP-016 showed
this correlation is ZERO within a single architecture (rho=0.11,
n=10, p=0.76). The cross-architecture correlation was driven
by architectural inductive bias, not a general TTC mechanism.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-16 | HYP-014: perfect rank correlation between p@64 at step 2K and grokking onset step. Combined with HYP-009 (pass@64 as grokking indicator). | C | Moderate for | 0.65 |
| 2026-03-16 | HYP-016: rho=0.111 (p=0.76) across 10 seeds within MoE-Jamba. Zero predictive power for within-architecture grokking. Cross-architecture correlation was confounded. | F | Strong against | 0.30 |

**Why 0.30:**
- Cross-architecture prediction (rho=1.0) was real but
  confounded — architectural inductive bias drives both
  early TTC and grokking speed
- Within-architecture prediction (rho=0.11) is essentially
  zero — TTC at step 2K does not predict grokking onset
- Belief narrowed: TTC predicts architecture quality, not
  training run quality

---

## B-015: Grokking onset is massively seed-dependent

**Prior:** N/A (new belief from HYP-015)
**Current:** 0.95
**Source:** HYP-015, HYP-016, corroborated by HYP-009

Grokking onset step varies 12x across seeds for identical
architecture and hyperparameters. In HYP-016: MoE-Jamba
grokked at steps 4K, 12K, 12K, 12K, 18K, 22K, 36K, 48K,
48K across 9 seeds (1 never grokked in 50K). No early metric
(loss, accuracy, pass@64) predicts onset timing (all rho < 0.12).

This means single-seed grokking experiments are unreliable
for comparing architectures or hyperparameters. Always use
multi-seed runs for grokking studies.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-16 | HYP-015: MoE grok steps {4K, 22K, 40K}. noMoE: {4K, never, never}. | F | Very strong for | 0.85 |
| 2026-03-16 | HYP-016: 10 seeds, grok steps 4K-48K (12x range), no early metric predicts onset (all rho < 0.12). | F | Definitive | 0.95 |
| 2026-03-15 | HYP-009: 1/3 LLaMA seeds grokked at wd=0.1 | C | Supporting | — |

**Why 0.95:** Confirmed with 10 seeds. The 12x range and
zero correlation with all early metrics is definitive. Only
held from 1.0 because we've only tested one architecture
(MoE-Jamba) at 10-seed scale.

---

## B-016: MoE capacity helps grokking (more params → easier grokking)

**Prior:** N/A (new belief from HYP-015)
**Current:** 0.60
**Source:** HYP-015 experiment results

MoE-Jamba (7.6M params) grokked 3/3 seeds; noMoE-Jamba
(7.0M params) grokked 1/3 seeds. The MoE variant provides
4 expert FFNs vs 1 dense FFN, giving more capacity for the
generalization circuit to form.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-16 | HYP-015: MoE 3/3 grokked vs noMoE 1/3 grokked. | F | Strong for | 0.60 |

| 2026-03-16 | HYP-016: MoE-Jamba groks 9/10 seeds at 50K steps. | F | Corroborating | 0.70 |

**Why 0.70 (up from 0.60):**
- HYP-015: 3/3 MoE vs 1/3 noMoE. HYP-016: 9/10 MoE.
  Consistent grokking rate across 13 total MoE-Jamba seeds.
- Still confounded by param count difference (590K, 8.4%).
- MoE routing diversity (not just capacity) could be the
  mechanism. Cannot distinguish capacity vs routing effects.
- Need: param-matched comparison to isolate MoE vs capacity.

---

## Parameter Golf Beliefs

## B-017: Depth recurrence improves BPB under size constraints

**Prior:** 0.65 (moderately likely)
**Current:** 0.65
**Source:** Universal Transformers (Dehghani 2019), looped
transformers literature

Weight-sharing across transformer layers reduces unique
parameters while maintaining depth. Under a 16MB artifact
constraint, this should free budget for wider layers or
larger vocab, netting a BPB improvement.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | Prior from literature: Universal Transformers show recurrence works. Magnitude of benefit under compression constraints unknown. | C | Neutral | 0.65 |
| 2026-03-19 | HYP-024: 6L (3u×2) beats 9L (3u×3) by 0.115 BPB locally. More depth hurts: 12L and 15L both worse. BUT this is likely throughput-dominated (42% more steps at 6L). Depth recurrence still works, but optimum is fewer cycles than expected. | B | Nuanced — recurrence helps but fewer cycles may suffice | 0.70 |

## B-018: Training schedule optimization yields >0.003 BPB

**Prior:** 0.70 (likely)
**Current:** 0.85 (locally confirmed, official transfer uncertain)
**Source:** HYP-017 local experiments

The baseline training schedule is a reasonable default but
not optimized. Locally, longer warmdown produces 0.05-0.10
BPB improvement — but this is confounded by batch size
mismatch (8K local vs 524K official).

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | Prior: baseline uses conservative defaults. LR schedules have large effects in transformer training. | C | Neutral | 0.70 |
| 2026-03-18 | HYP-017: warmdown=3000 improves by 0.054 BPB locally. Monotonic trend across 5 warmdown values. But confounded by batch size mismatch — effective LR reduction, not schedule shape. | F | Strong for (locally), uncertain (officially) | 0.85 |

## B-019: Larger vocab improves BPB when paired with depth recurrence

**Prior:** 0.50 (uncertain)
**Current:** 0.50
**Source:** Trade-off between token coverage and embedding table size

Larger vocab (2048/4096) gives fewer tokens per document
(better sequence coverage) but increases embedding table.
With tied embeddings and depth recurrence freeing params,
this might net improve BPB.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | Prior: two opposing forces (better coverage vs bigger table). Outcome depends on specific constraint budget. | C | Neutral | 0.50 |
| 2026-03-18 | HYP-018: 3 unique blocks frees 11.3MB artifact budget. Massive headroom for larger embedding table now available. | C | For | 0.55 |

## B-020: Weight sharing improves BPB at small scale

**Prior:** 0.30 (unlikely — sharing loses capacity)
**Current:** 0.80
**Source:** HYP-018 depth recurrence experiments

Weight sharing (cycling N unique blocks for 9 effective layers)
improves BPB over unique blocks at the same width. 3 unique
blocks at dim=512 achieves 1.9102 vs 1.9393 baseline (+0.029).
More sharing is better (3 blocks > 5 blocks > 9 unique).

The effect combines regularization (~0.017 BPB from sharing itself)
with throughput (~0.012 BPB from 13.5% more steps due to faster
per-step time). Width reallocation fails locally because wider
models are too slow per step on Mac hardware.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | Prior: sharing typically hurts capacity; Universal Transformers showed it can work but at different scale. | C | Against | 0.30 |
| 2026-03-18 | HYP-018: 3 unique blocks beats baseline by 0.029 BPB. 5 blocks also beats baseline but by less. Monotonic trend: more sharing = better. Confound: partially explained by step count advantage. | B | Strong for | 0.80 |
| 2026-03-18 | HYP-019: U-shaped curve confirmed (1→2→**3**→5→9). 3 blocks optimal. Combined with schedule (3u+wd=5000+lr=0.03) achieves 1.8436 at 3.6MB, matching 9-unique configs. | B | Strong for | 0.85 |
| 2026-03-19 | HYP-024: With wide heads, 6L (3u×2 cycles) achieves 1.7363 vs 9L's 1.8512 — locally, fewer cycles with same 3 blocks is better. The sharing itself helps; more cycles don't add value locally. | B | For (sharing works, but optimal cycle count is low locally) | 0.85 |

---

## B-021: Wider attention heads are better at small scale

**Prior:** 0.40 (uncertain — standard practice is 64-128 head_dim)
**Current:** 0.90
**Source:** HYP-022 attention configuration experiments

At dim=512 with weight sharing (3 unique blocks), 4 heads at
head_dim=128 dramatically outperforms 8 heads at head_dim=64
(+0.072 BPB, the single largest improvement across 40+
experiments). Each head needs sufficient dimensionality for
expressive attention patterns. At head_dim=64, the attention
is too narrow to capture complex token relationships.

Full MHA (matching KV heads) is important: 4h/4kv beats 4h/2kv
by 0.042 BPB. The KV heads need to match the query head count
for the full benefit of wide heads.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | Prior: common head_dim is 64-128 across scales. No clear preference at small scale. | C | Neutral | 0.40 |
| 2026-03-18 | HYP-022: 4h/4kv (hd=128) beats 8h/4kv (hd=64) by 0.072 BPB. 4h/4kv also beats 8h/8kv (same params, different head_dim) by 0.094 BPB. | A | Definitive for | 0.90 |

---

## B-022: Local BPB is dominated by step count, not architecture quality

**Prior:** N/A (new belief from HYP-024)
**Current:** 0.90
**Source:** HYP-024 depth sweep + cumulative evidence from HYP-017 through HYP-024

On Mac with 8K batch size and 600s wallclock cap, any change that
increases per-step time hurts BPB — even if it adds representational
capacity (more layers, wider model). Conversely, any change that
reduces per-step time helps — even if it removes capacity (fewer layers).

This is because 8K batch size produces ~1000-2000 total steps, and
gradient noise is 64x higher than official 524K batch. More steps
means more gradient signal, which dominates quality at this noise level.

**Implication:** Local Mac experiments are reliable for:
- Comparing configs with SAME per-step time (e.g., head count at fixed dim)
- Smoke-testing for crashes and constraint violations
- Identifying which techniques are worth GPU validation

Local Mac experiments are UNRELIABLE for:
- Evaluating depth vs width tradeoffs
- Training schedule optimization (warmdown, warmup, LR)
- Any config that changes per-step time by >10%

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-18 | HYP-017: warmdown optimization +0.054 is confounded by batch size | C | For | — |
| 2026-03-18 | HYP-018/019: wider models fail locally despite more params | B | For | — |
| 2026-03-19 | HYP-024: 6L beats 9L by 0.115 BPB purely from 42% more steps (1996 vs 1404). 12L and 15L both worse despite identical params, solely from fewer steps. | A | Definitive for | 0.90 |

## B-023: Sliding window eval gives ~0.03 free BPB improvement

**Prior:** 0.70 (based on competition evidence PR #50)
**Current:** 0.90
**Source:** HYP-027 + competition PR #50

Sliding window evaluation with stride=256 improves BPB by +0.032
(1.7363→1.7046) at zero model size cost. This is a pure eval-time
technique unaffected by B-022. Consistent with competition results.

**Implication:** Always use EVAL_STRIDE in the final submission.
Smaller strides (128, 64) may give diminishing additional gains
but with 2-4x eval time cost.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-19 | HYP-027: stride=256 gives +0.032 BPB (1.7363→1.7046), iso-step comparison (2012 vs 1996 steps) | A | For | 0.90 |

---

## B-024: INT8 PTQ gap is negligible (~0.001 BPB)

**Prior:** 0.30 (expected ~0.05 gap based on typical INT8 quantization)
**Current:** 0.95
**Source:** HYP-029 experiment results

Post-training quantization to INT8 (via mx.quantize with group_size=64)
introduces only ~0.001 BPB degradation for our architecture (6L+3u+4h/4kv,
dim=512). This means QAT is unnecessary at INT8 precision. The prior
assumption of ~0.05 gap was based on general INT8 literature, not
architecture-specific measurement.

**Implication:** INT8 PTQ is essentially free for the competition
artifact. QAT becomes valuable only at INT4/INT6 where quantization
error is 30-295x higher. INT4+QAT could allow ~32M params in 16MB
(2x current budget), making it a high-value GPU experiment.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-19 | HYP-029: Baseline gap=0.0011, QAT gap=0.0014. Both <0.002. Prior expectation of ~0.05 was 50x too high. | A | Definitive for | 0.95 |
| 2026-03-19 | HYP-030: SWA amplifies INT8 gap 8x (0.0014→0.0111). Averaged weights are less quantization-friendly. | B | Caution — SWA interacts badly with INT8 PTQ | 0.95 |

---

## B-025: SWA hurts at high gradient noise / low step count

**Prior:** N/A (new belief from HYP-030)
**Current:** 0.75
**Source:** HYP-030 experiment results

SWA with a 25% averaging window hurts BPB when training with high
gradient noise (8K batch vs 524K official). At ~2000 steps with 8K
batch, the model is still rapidly improving — averaging over an
improving trajectory is worse than the final checkpoint. Additionally,
SWA dramatically amplifies the INT8 quantization gap (0.0014→0.0111,
8x worse) because averaged weights sit between quantization grid points.

**Implication:** Don't use SWA locally. On GPU with 524K batch and
flatter loss landscape, SWA may still help (competition SOTA uses it).
If used on GPU, try SWA_START=0.90+ to narrow the window and minimize
quantization gap amplification.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-19 | HYP-030: SWA hurt float BPB by 0.005 and INT8 BPB by 0.015. INT8 gap 8x worse. 503 checkpoints averaged. | A | Strong for | 0.75 |

---

## B-026: NorMuon improves convergence at small scale

**Prior:** 0.60 (likely based on paper, but untested at our scale)
**Current:** 0.75
**Source:** HYP-031 experiment results

NorMuon per-row adaptive normalization with correction scaling improves
BPB by ~0.012 (after B-022 step count adjustment) over standard Muon.
The improvement comes from equalizing per-neuron update magnitudes while
preserving overall gradient scale. No impact on INT8 quantization gap.

**Implication:** Use NORMUON=1 as default for competition submission.
Add to GPU-ready config alongside weight sharing and wide heads.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-19 | HYP-031: Float BPB 1.7182->1.7016 (+0.017), INT8 1.7196->1.7030 (+0.017). Step count 3.5% higher (partial B-022 confound, ~0.005). | B | Strong for | 0.75 |

---

## B-027: Input-dependent cross-layer aggregation outperforms static

**Prior:** 0.65 (likely based on AttnRes paper claims, but untested at our scale)
**Current:** 0.90
**Source:** HYP-033 experiment results + LIT-127 (arXiv 2603.15031)

Full Attention Residuals (input-dependent softmax over preceding sublayer
outputs via learned pseudo-queries) dramatically outperforms DenseFormer DWA
(static softmax weights) at iso-step. AttnRes+VR achieves +0.111 BPB over
baseline while DWA+VR is -0.017 (worse than baseline). The per-step quality
improvement is 6.5x larger than DWA's best non-iso-step result (+0.041).

This confirms the paper's Table 4 finding: static DenseFormer shows no gain
at scale, while input-dependent AttnRes provides 1.25x compute efficiency.
The key mechanism is that each sublayer's aggregation weights adapt to the
input, allowing context-dependent routing of information across layers.

**Implication:** Replace DENSE_DWA=1 with ATTN_RES=1 in all GPU submissions.
AttnRes overhead is ~2.7x on Mac (memory-bound) but expected ~5-10% on GPU
(compute-bound). AttnRes+VR is now the highest-priority novel technique.

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-20 | HYP-033: AttnRes+VR 2.303 vs baseline 2.415 (+0.111) vs DWA+VR 2.431 (-0.017). 200 iso-steps, 9L/8h/4kv (no sharing). | A | Definitive for | 0.90 |
| 2026-03-20 | LIT-127: AttnRes paper Table 4. DenseFormer 1.767 vs baseline 1.766 (no gain). AttnRes Full 1.737 (-0.03). | B | For (at scale) | — |
| 2026-03-20 | HYP-034: AttnRes FAILS on 6L/3u/4h/4kv (-0.088). Initially attributed to weight sharing. | A | Against (6L) | 0.70 |
| 2026-03-20 | HYP-035: AttnRes also fails on 6L/6u/8h/4kv (-0.052) and 6L/6u/4h/4kv (-0.099). Root cause is depth, not sharing. | A | Conditional (>=9L) | 0.75 |

---

### B-028: AttnRes Requires Depth (>=9 Layers)

**Prior:** 0.50 (no prior expectation)
**Current posterior:** 0.85
**Direction:** Depth-dependent (REVISED from "weight-sharing-dependent")

AttnRes requires sufficient depth for sublayer outputs to diverge enough to
make attention-over-depth useful. At 6 layers, all configs fail regardless of
weight sharing or head count. At 9 layers, +0.111 gain. The mechanism needs
diverse representations across depth — shallow models produce too-correlated
intermediate outputs.

Head count is a secondary factor (8h loses less than 4h at 6L).

**Implication:** GPU variant E (11 unique layers) should give even larger
per-step gains. Weight sharing variant C may actually work if depth is
sufficient (e.g., 12 layers with 3 unique). **However, untested.**

| Date | Evidence | Grade | Direction | Updated to |
|------|----------|-------|-----------|------------|
| 2026-03-20 | HYP-034: Full AttnRes -0.088 with 3u sharing at 6L. | A | For | 0.85 |
| 2026-03-20 | HYP-035: Full AttnRes -0.052 (8h) and -0.099 (4h) with 6 unique layers. Depth, not sharing, is the bottleneck. | A | Revised and strengthened | 0.85 |
