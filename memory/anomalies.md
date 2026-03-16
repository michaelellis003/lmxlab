# Anomaly Tracker

Unexpected results that don't fit current hypotheses or beliefs.
Each anomaly is tracked until explained or confirmed as artifact.

## Status Key

| Status | Meaning |
|--------|---------|
| `open` | Unexplained, needs investigation |
| `explained` | Root cause identified |
| `artifact` | Due to methodology error, not real |

---

## ANOM-001: LLaMA features degrade training vs GPT baseline

**Status:** explained
**Experiment:** HYP-001 (GPT-to-LLaMA ablation)
**Date:** 2026-03-11

**Observation:** Every cumulative LLaMA feature addition *increased*
final training loss compared to the GPT baseline. GPT baseline
achieved mean loss 1.6523 vs full LLaMA 1.8981 (d=-1.17, large
effect in the wrong direction). The degradation was cumulative
and consistent across 3 seeds.

**Expected:** LLaMA features should improve or be neutral, based
on their widespread adoption in modern architectures.

**Possible explanations (to investigate):**

1. **Learning rate mismatch:** LR=1e-3 may be too high for
   GQA/SwiGLU architectures, which have different gradient dynamics.
   GPT's standard FFN + MHA may be more stable at higher LR.
2. **Time budget disadvantage:** SwiGLU FFN runs 29% fewer steps/sec
   (32.0 vs 45.4). In a fixed 5-min budget, simpler architectures
   complete more optimization steps.
3. **Scale effect:** LLaMA features may only help at larger scale.
   At d_model=256 with char-level tokenization, the inductive biases
   of RoPE, GQA, SwiGLU may not be beneficial.
4. **Char-level tokenization:** These features were designed for BPE
   tokenization with larger vocabularies. Char-level may change which
   architectural choices matter.
5. **Seed variance:** High std (0.23-0.28) on later configs suggests
   training instability. Some seeds converged well (seed 42 GQA=1.587)
   while others did not (seed 43 GQA=2.145).

**Follow-up needed:** Re-run with (a) lower LR sweep, (b) step-matched
rather than time-matched budget, (c) BPE tokenization.

**Resolution (HYP-001b):** Primary cause was LR=1e-3 (too high
for complex architectures) combined with SwiGLU parameter mismatch
(d_ff=512 gave 50% more params). At correct per-config LR with
d_ff=341, full LLaMA beats GPT baseline by 8% (d=1.72).

---

## ANOM-002: SwiGLU FFN throughput penalty

**Status:** explained (partially)
**Experiment:** HYP-001
**Date:** 2026-03-11

**Observation:** SwiGLU FFN config ran at 32.0 steps/sec vs baseline's
45.4 steps/sec (29.5% slower), despite having only 25% more parameters
(3.97M vs 3.18M). This means in a fixed time budget, SwiGLU gets
~30% fewer gradient updates.

**Possible explanation:** GatedFFN has 3 linear projections vs
StandardFFN's 2, but parameter count alone doesn't explain the
throughput gap. May be related to MLX's handling of the gating
activation (SiLU multiply).

---

## ANOM-003: High inter-seed variance in later configs

**Status:** explained (partially)
**Experiment:** HYP-001
**Date:** 2026-03-11

**Observation:** Seed variance increased with cumulative features:
- GPT baseline: std=0.094
- + GQA: std=0.282 (3x higher)
- + No bias (=LLaMA): std=0.282 (3x higher)

Seed 43 showed particularly poor convergence for later configs
(GQA=2.145, LLaMA=2.221) while seed 42 showed decent results
(GQA=1.587, LLaMA=1.772).

**Possible explanation:** More complex architectures may need more
careful initialization or warmup. Removing bias terms and changing
attention/FFN structure simultaneously may create a larger space of
bad initializations.

**HYP-001b update:** Variance much lower in HYP-001b (std 0.003-
0.077 vs 0.094-0.282 in HYP-001). LR=1e-3 was likely the primary
cause of HYP-001's instability. At lr=1e-4 and 3e-4, all configs
converge reliably.

---

## ANOM-004: RoPE creates exceptionally stable optimization

**Status:** confirmed
**Experiment:** HYP-001b Sub-experiment A, confirmed by HYP-001c
**Date:** 2026-03-11 (confirmed 2026-03-12)

**Observation:** + RoPE config at lr=3e-4 has std=0.0029 across
3 seeds (values: 1.1970, 1.1912, 1.1943). This is 8x lower than
GPT baseline (std=0.0246) and 26x lower than full LLaMA
(std=0.0767). No other config shows anywhere near this consistency.

**HYP-001c confirmation (n=5):** RoPE val_loss std=0.0049 across
5 seeds — still the lowest of any config (GPT: 0.0077, RMSNorm:
0.0129, LLaMA: 0.0124). **Not an n=3 artifact.**

**Expected:** Similar variance to other configs (~0.02-0.07).

**Possible explanations:**
1. RoPE + RMSNorm at lr=3e-4 may create an optimization landscape
   with a very flat, well-defined minimum for char-level Shakespeare
2. The combination may enable more deterministic gradient flow
3. ~~Could be coincidence with n=3~~ Confirmed with n=5.

**Status note:** Confirmed as a real phenomenon. Interesting but
not blocking — low priority for further investigation.

---

## ANOM-005: Smoke test runs polluting results.jsonl

**Status:** open
**Experiment:** HYP-001b
**Date:** 2026-03-11

**Observation:** 18 smoke test runs from earlier testing are mixed
into results.jsonl with the same experiment name "HYP-001b-lr-sweep".
All are seed=42 with loss ~2.4-2.5 (much worse than the real
seed=42 runs). The recipe's built-in analysis correctly uses only
3 seeds per group, but raw log parsing picks up 4 per group.

**Impact:** Doesn't affect the recipe's analysis output, but makes
manual log parsing error-prone.

**Fix needed:** Add a `run_id` or `batch_id` field to results.jsonl
entries so smoke tests can be filtered. Or add a `--tag` flag to
recipes to mark runs.

---

## ANOM-006: Severe overfitting across all configs at 1 PFLOPs

**Status:** open
**Experiment:** HYP-001c (FLOP-matched ablation)
**Date:** 2026-03-12

**Observation:** All 6 configs show train-val gaps of 0.83-0.93
at 1 PFLOPs (~14 tokens/param, 8-9 epochs over Shakespeare).
Train losses range 0.68-0.86, but val losses cluster in a narrow
1.607-1.670 band. The gap means ~50% of apparent "learning" is
memorization.

**Expected:** Some overfitting is normal at 8-9 epochs, but a gap
this large (val ~2x train loss) suggests the models are well past
diminishing returns on this dataset.

**Possible explanations:**
1. **Data repetition beyond effective range.** Muennighoff 2023
   (LIT-010) found repetition effective up to ~4 epochs,
   diminishing beyond 8. We're at 8-9 epochs.
2. **No regularization.** Dropout=0.0, no data augmentation.
3. **Char-level tokenization.** Only ~96 unique tokens means
   the model can memorize character-level patterns efficiently.
4. **Small dataset.** 1.1M chars is tiny. BPE tokenization on
   a larger corpus would give more unique training signal.

**Impact:** Makes all architecture comparisons unreliable. The
"best" config depends on whether you measure train or val loss.
Must address overfitting before drawing architecture conclusions.

**Follow-up:** (a) Add dropout sweep, (b) reduce FLOP budget to
~0.25-0.5 PFLOPs, (c) switch to BPE on larger dataset.

**HYP-001d update:** Dropout substantially mitigates this. At
dropout=0.2, train-val gaps drop from 0.83-0.86 to 0.20-0.30.
Still overfitting, but far less severe. Val losses improve by
0.05-0.10. Status remains open — dropout helps but doesn't fully
resolve the issue.

---

## ANOM-007: RMSNorm and LLaMA significantly worse on val loss

**Status:** open
**Experiment:** HYP-001c
**Date:** 2026-03-12

**Observation:** Two configs stand out as significantly worse on
val loss despite being competitive on train loss:
- + RMSNorm: val 1.667 (d=-5.39 vs GPT), train 0.864
- + No bias (=LLaMA): val 1.670 (d=-5.84 vs GPT), train 0.819

Other configs (RoPE, SwiGLU, GQA) are within 0.005 of GPT on
val loss. RMSNorm and LLaMA are ~0.06 worse.

**Expected:** RMSNorm should be neutral or slightly better (it's
a simplification, not a new capability).

**Possible explanations:**
1. **RMSNorm enables faster memorization.** By removing the mean
   subtraction, RMSNorm may allow gradients to flow more freely,
   accelerating both learning AND overfitting.
2. **No-bias removal compounds the effect.** Removing bias terms
   in the LLaMA config further reduces regularization.
3. **Interaction with LR.** RMSNorm's best LR from HYP-001b was
   1e-4 (vs 3e-4 for GPT). The lower LR may not be optimal for
   longer training.

**Follow-up:** Test RMSNorm at 3e-4 LR for longer training.
Test with dropout to see if the gap closes.

**HYP-001d update:** At dropout=0.1, the RMSNorm-based LLaMA
(1.573) matches GPT (1.573) exactly (d=-0.07). This confirms
the gap was a regularization deficit, not an architectural flaw.
At dropout=0.2, the gap partially reopens (0.026) due to LLaMA
over-regularizing. Status updated: **partially explained** — the
regularization deficit hypothesis (explanation 1) is confirmed.

---

## ANOM-008: RoPE compensates for RMSNorm overfitting

**Status:** open
**Experiment:** HYP-001c
**Date:** 2026-03-12

**Observation:** The cumulative config ordering is:
- GPT baseline: val 1.6092
- + RMSNorm: val 1.6666 (+0.057, much worse)
- + RoPE: val 1.6072 (-0.059 from RMSNorm, recovers to GPT level)

Adding RoPE on top of RMSNorm recovers essentially all the val
loss degradation that RMSNorm introduced. This is surprising
because RoPE is a position encoding change and shouldn't interact
with normalization overfitting.

**Expected:** RoPE should have a modest independent effect on
position-sensitive tasks. It shouldn't counteract RMSNorm's
overfitting behavior.

**Possible explanations:**
1. **RoPE as implicit regularizer.** Rotary embeddings constrain
   the attention pattern to be relative-position-dependent, which
   may act as an inductive bias that prevents overfitting to
   absolute positions in the training data.
2. **Coincidence at this scale.** The effects may be independent
   and happen to approximately cancel.

**Follow-up:** Test RoPE alone (without RMSNorm) to separate
the effects.

---

## ANOM-009: LLaMA's optimal dropout is 0.1, not 0.2

**Status:** explained (artifact of multi-epoch char-level regime)
**Experiment:** HYP-001d (Dropout × Architecture)
**Date:** 2026-03-13

**Observation:** GPT and LLaMA respond non-monotonically to
dropout rate. GPT improves continuously from 0.0 → 0.1 → 0.2
(val 1.611 → 1.573 → 1.560). LLaMA improves at 0.1 (val 1.671
→ 1.573) but REGRESSES at 0.2 (→ 1.586). LLaMA's best dropout
rate (0.1) is lower than GPT's best (0.2).

**Expected:** Both architectures should benefit monotonically
from dropout up to 0.2 (per LIT-019). Or both should plateau.

**Possible explanations:**
1. **LLaMA already has less capacity.** GQA reduces KV heads,
   SwiGLU uses d_ff=341 (vs 512). Adding dropout=0.2 on top of
   these capacity reductions may over-regularize.
2. **No-bias removes learnable offsets.** Dropout zeros random
   activations; without bias terms, the network has fewer
   parameters to compensate for the dropped information.
3. **LR interaction.** LLaMA uses 1e-4 (vs GPT's 3e-4). Lower
   LR + higher dropout may produce insufficient gradient signal.

**Follow-up:** Test LLaMA at dropout=0.15. Or test LLaMA at
lr=3e-4 with dropout=0.2 to isolate the LR interaction.

---

## ANOM-010: Dropout equalizes GPT and LLaMA at exactly 0.1

**Status:** explained (artifact of multi-epoch char-level regime)
**Experiment:** HYP-001d
**Date:** 2026-03-13

**Observation:** At dropout=0.1, GPT (1.5729) and LLaMA (1.5734)
produce essentially identical val loss (gap = 0.0005, d = -0.07,
negligible). This is striking because at dropout=0.0 the gap is
0.060 (d=-4.61) and at dropout=0.2 the gap is 0.026 (d=-3.84).

**Expected:** Either the gap closes monotonically with more
dropout, or it stays roughly constant. A V-shaped pattern
(closes then reopens) was not predicted by any hypothesis.

**Possible explanations:**
1. **Regularization deficit compensated.** At dropout=0.1,
   explicit dropout exactly compensates for RMSNorm's missing
   implicit regularization (LIT-020, ANOM-007). At dropout=0.2,
   LLaMA is over-regularized due to its lower capacity.
2. **Coincidence.** With n=3, the equalization at exactly 0.1
   could be a statistical fluctuation. However, all 3 LLaMA
   seeds at 0.1 are consistent (std=0.008).

**Follow-up:** Replicate with n=5 to confirm. Test dropout
rates 0.05, 0.15 to trace the equalization curve.

---

## ANOM-011: LLaMA benefits MORE from dropout than GPT

**Status:** explained (artifact of multi-epoch char-level regime)
**Experiment:** HYP-001d
**Date:** 2026-03-13

**Observation:** LLaMA's val loss improvement from dropout=0.0 to
0.1 is 0.097 (d=7.77), vs GPT's 0.038 (d=5.33). LLaMA benefits
~2.5x more from the same dropout rate. Train-val gap reduction
is also larger: LLaMA 1.27 vs GPT 0.92.

**Expected:** If the architectures differ only in capacity, both
should benefit similarly from dropout.

**Possible explanations:**
1. **Confirms ANOM-007.** RMSNorm's lack of implicit regularization
   (LIT-020) means LLaMA has more headroom for improvement from
   explicit regularization.
2. **No-bias amplifies overfitting.** Without bias terms, dropout
   provides a proportionally larger regularization effect.

**Significance:** This strongly supports the hypothesis that the
GPT-LLaMA gap in HYP-001c was primarily a regularization deficit,
not an architectural quality difference.

---

## ANOM-012: LLaMA train loss much higher than val loss at 30M

**Status:** open
**Experiment:** HYP-006 (Dropout × normalization at 30M)
**Date:** 2026-03-14

**Observation:** LLaMA-30M at dropout=0.0 has train_loss=3.31
but val_loss=2.51 — train loss is 0.80 HIGHER than val loss.
This is the reverse of the typical overfitting pattern and
persists across all dropout rates (gap +0.80 to +0.85).

GPT-30M shows the expected pattern: train=3.01, val=3.05
(train slightly lower, gap -0.04).

**Expected:** Train loss should be lower than or equal to val
loss.

**Possible explanations:**
1. **TinyStories val set is easier** than train set (different
   distribution or complexity).
2. **Measurement timing:** val_loss measured at end of training
   (fully trained model), train_loss is running average or last
   batch (noisier). The model improves rapidly over 2000 steps,
   so end-of-training eval captures a better model than
   mid-training batch loss.
3. **LLaMA-specific:** LLaMA's stronger architecture may
   generalize better to the val distribution, while still
   showing high loss on challenging train batches.

**Impact:** Complicates interpretation of train-val gap as an
overfitting indicator. Need to verify how train_loss and val_loss
are measured in the HYP-006 recipe.

---

## ANOM-013: Dropout × normalization interaction was an artifact

**Status:** explained
**Experiment:** HYP-006
**Date:** 2026-03-14

**Observation:** The non-monotonic dropout × normalization
interaction (ANOM-009/010/011) found in HYP-001d at 3M params
with char-level Shakespeare does NOT replicate at 30M with BPE
on TinyStories. At 30M:
- Both GPT and LLaMA are hurt by dropout at all rates
- No equalization point exists
- No non-monotonic pattern

**Explanation:** The interaction was specific to the multi-epoch
Shakespeare regime (8-9 epochs, severe overfitting). At 30M with
TinyStories BPE, training runs <1 epoch — there is no overfitting
to regularize, so dropout only adds noise. The "novel finding"
from HYP-001d was a data-repetition artifact, not an architectural
phenomenon.

**Impact:** ANOM-009, ANOM-010, ANOM-011 should be downgraded
from "open" to "explained (artifact)" — they were real within
the 3M char-level regime but do not generalize. The dropout ×
normalization research direction (Idea A in roadmap) is weaker
than believed.

---

## ANOM-014: Falcon-H1-10M and Bamba-10M are identical

**Status:** explained
**Experiment:** HYP-008
**Date:** 2026-03-15

**Observation:** Falcon-H1 and Bamba produce identical results
across all 3 seeds — same val_loss to 4+ decimal places, same
pass@k at every k value. Seed 42: both have val_loss=2.404,
pass@64=4.39%. Seed 43: both have val_loss=2.233, pass@64=4.08%.

**Root cause:** The `falcon_h1_10m()` and `bamba_10m()` factory
functions produce architecturally identical ModelConfigs. Both
use the same hybrid pattern (MMM*MMM*MMM*), d_model=128,
n_heads=4, n_kv_heads=2, d_ff=384, and identical Mamba-2
parameters. The underlying `falcon_h1_config` and `bamba_config`
functions are functionally equivalent — they differ only in
default parameter values for full-scale models, which are
overridden by the explicit 10m factory parameters.

**Impact:** HYP-008 effectively tested 3 architectures, not 4.
Falcon-H1 and Bamba results should be treated as the same
architecture with duplicate data. The experiment still provides
valid comparisons: LLaMA (pure attention) vs hybrid (Falcon-H1/
Bamba) vs hybrid+MoE (Jamba).

**Fix needed:** Either differentiate the 10m factories (e.g.,
different hybrid patterns or SSM parameters) or remove one
as redundant at this scale.

---

## ANOM-015: Higher val_loss correlates with higher pass@k

**Status:** open
**Experiment:** HYP-008
**Date:** 2026-03-15

**Observation:** LLaMA has the highest (worst) val_loss of all
4 architectures but the highest pass@k at every k value:

| Arch | Val Loss (rank) | pass@1 (rank) | pass@64 (rank) |
|------|----------------|---------------|----------------|
| LLaMA | 2.731 (4th) | 0.56% (1st) | 8.34% (1st) |
| Falcon-H1 | 2.318 (2nd) | 0.28% (2nd) | 4.06% (2nd) |
| Bamba | 2.318 (2nd) | 0.28% (3rd) | 4.04% (3rd) |
| Jamba | 2.310 (1st) | 0.25% (4th) | 3.29% (4th) |

The ranking is perfectly inverted: best val_loss = worst pass@k.

**Expected:** Lower val_loss should indicate better next-token
prediction, which should translate to higher pass@k on the
same task.

**Possible explanations:**
1. **Val loss averages over all tokens.** Modular arithmetic
   prompts have format "X + Y = Z". Val_loss averages over
   prompt tokens (X, +, Y, =) and the answer token (Z).
   Hybrids may predict prompt tokens better (lower average
   loss) while LLaMA predicts the answer token better.
2. **SSM state compression loses precision.** SSMs compress
   context into fixed-size state. This works well for pattern
   prediction (most tokens) but may lose the precise numerical
   information needed for the final answer token.
3. **Attention enables exact retrieval.** The modular arithmetic
   task requires attending to specific operand tokens. Pure
   attention can do this directly; SSMs must recover operands
   from compressed state.
4. **Different loss landscape geometry.** LLaMA may have a
   flatter, noisier loss landscape that averages worse but
   has more diverse minima — producing higher pass@k.

**Follow-up:** Compute per-token loss breakdown (prompt tokens
vs answer token) to distinguish explanation 1 from 2-4. This
would require modifying the eval to report answer-token loss
separately.
