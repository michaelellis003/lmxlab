# Decision Records

Lightweight architecture decision records for research methodology.

---

## DEC-001: 5-Minute Time Budgets

**Date:** 2026-03-11 (migrated from devlog)
**Status:** accepted
**Context:** Need a way to make experiments directly comparable
regardless of architecture complexity.
**Decision:** All experiments use a 5-minute wall-clock time budget
(the autoresearch pattern).
**Rationale:** Fixed time eliminates timing confounds. Faster
architectures get more steps; slower ones get fewer. This naturally
rewards efficiency.
**Trade-off:** Some architectures may need longer to converge, so
5-min results may not reflect asymptotic performance.

---

## DEC-002: Minimum 3 Seeds

**Date:** 2026-03-11 (migrated from devlog)
**Status:** accepted
**Context:** ML results are stochastic; single-seed results are
unreliable.
**Decision:** Every experiment runs at least 3 seeds. Report mean
+/- std.
**Rationale:** 3 seeds is the minimum for computing standard
deviation. More seeds are better but 3 balances cost vs reliability.
**Trade-off:** 3x compute cost per experiment. With 5-min budgets,
each experiment takes ~15 minutes total.

---

## DEC-003: Shakespeare Default Dataset

**Date:** 2026-03-11 (migrated from devlog)
**Status:** accepted
**Context:** Need a standard dataset for comparing architectures.
**Decision:** Use Shakespeare (character-level) as the default
dataset for all comparison experiments.
**Rationale:** Small enough to iterate quickly, complex enough to
show meaningful differences. Character-level removes tokenizer as
a confound. Already integrated in the codebase.
**Trade-off:** Results may not generalize to BPE-tokenized or
larger datasets. Plan to validate key findings on a larger dataset
after initial experiments.

---

## DEC-004: Switch to FLOPs-Matched Comparisons

**Date:** 2026-03-11
**Status:** accepted (implement after HYP-001b completes)
**Context:** DEC-001 uses wall-clock time budgets, which is the
weakest comparison method (Kaplan et al. 2020, LIT-005). Results
depend on implementation efficiency and hardware utilization
rather than true architectural quality.
**Decision:** Switch to FLOPs-matched comparisons as the primary
methodology. Each architecture gets the same total compute budget
measured in floating-point operations.
**Rationale:** FLOPs-matched is the gold standard for architecture
comparison (Kaplan et al. 2020). It isolates architectural quality
from implementation speed, making results more meaningful and
reproducible across hardware.
**Requires:** Implement a FLOP counter per forward/backward pass
for each architecture variant. Can use analytical formulas
(e.g., 6 * N * D for a dense transformer where N=params, D=tokens)
or instrument the model.
**Supersedes:** DEC-001 for architecture comparisons. Time budgets
may still be useful for efficiency benchmarks where wall-clock
speed is the metric of interest.

---

## DEC-005: Train to Chinchilla-Optimal Duration

**Date:** 2026-03-11
**Status:** accepted (implement after HYP-001b completes)
**Context:** Our 5-minute time budgets train for ~5-10M tokens
(~2-3 tokens/param). Chinchilla scaling laws (LIT-009) recommend
~20 tokens/param for compute-optimal training. Our models are
severely undertrained, which may distort architecture comparisons
— some architectures need more steps to show their advantage.
**Decision:** Train models to near Chinchilla-optimal duration.
Default to 8 epochs on Shakespeare (~40M char tokens, ~13
tokens/param, ~33 min per run). Use 4 epochs (~17 min) for quick
iteration and 12+ epochs for final evaluation.
**Rationale:**
- Chinchilla (LIT-009, NeurIPS 2022, Grade A): 20:1 tokens/param
- Muennighoff (LIT-010, NeurIPS 2023, Grade A): 4-8 epochs is
  the sweet spot for repeated data; beyond 8 diminishes sharply
- Porian (LIT-014, NeurIPS 2024, Grade B): optimum is flat,
  anywhere from 10:1 to 40:1 costs only ~1-3% in loss
- At 3M params, LR and architecture matter more than precise
  token count, but we must not be so undertrained that we can't
  distinguish architectures
**Trade-off:** 8-epoch runs take ~33 min vs 5 min. With 3 seeds,
a single config takes ~100 min. Full sweeps (6 configs x 3 seeds)
take ~10 hours. Mitigate by running overnight or reducing sweep
breadth.
**Supersedes:** DEC-001's 5-minute budget as the default.
**Depends on:** DEC-004 (FLOPs-matched) for fair cross-architecture
comparison; use 6*N*D with SwiGLU correction for gated FFNs.
**Compute budget:** Target ~1 PFLOPs per run for 3M param models
(Chinchilla-optimal range 0.76-1.08 PFLOPs, see SYNTH-001).
Each architecture trains until hitting the FLOP budget, not a
fixed time or step count.

---

## DEC-006: Augment Dataset for Final Experiments

**Date:** 2026-03-11
**Status:** proposed (evaluate after HYP-001b interpretation)
**Context:** Shakespeare provides only ~5M unique char tokens.
Chinchilla-optimal for 3M params is 60M tokens, requiring 12
epochs of repeated data. Muennighoff (LIT-010) shows repetition
beyond 4-8 epochs has diminishing returns. Additionally, char-level
tokens carry less information than BPE tokens, so the effective
Chinchilla-optimal may be 3-4x higher (~180-240M char tokens).
**Decision:** For final evaluation experiments, augment Shakespeare
with additional public domain text (e.g., other works from Project
Gutenberg or a HuggingFace text dataset) to reach 60M+ unique
char tokens. Keep Shakespeare as the primary source for
continuity with prior experiments.
**Rationale:** More unique data is strictly better than repeating
the same data. Public domain literature has similar style
characteristics to Shakespeare, minimizing distribution shift.
**Trade-off:** Larger dataset increases data loading time and
may introduce distribution shift. Validate by comparing
Shakespeare-only vs augmented results on the same architecture.
**Status note:** Proposed, not yet accepted. Evaluate whether
data constraint is actually limiting after interpreting HYP-001b.

---

## DEC-007: Mandatory Literature Reviews at Three Checkpoints

**Date:** 2026-03-12
**Status:** accepted
**Context:** Three rounds of HYP-001 experiments could have been
shorter if we had checked the literature more frequently. The
overfitting problem (ANOM-006) is well-documented in the multi-epoch
training literature (LIT-010, LIT-017, LIT-021) but we didn't
discover these sources until the post-experiment review.
**Decision:** Literature reviews are mandatory at three points:
1. **Before experiments** (`/hypothesis` Step 2 REA) — already
   existed, search for prior work on the research question.
2. **After experiments** (`/interpret` Step 7) — NEW. Search for
   similar findings, explanations for anomalies, and suggested
   interventions. Minimum 3 targeted searches, 2-5 new sources.
3. **During reviews** (`/review` Step 6) — NEW. Check for new
   papers on open anomalies and active hypotheses.
**Rationale:** Literature checks are cheap (10-20 min) compared
to running experiments (hours). Finding that a problem is already
solved in the literature saves entire experiment cycles. The
post-HYP-001c literature review found 6 directly relevant papers
(LIT-017 through LIT-022) that would have informed the experiment
design had we checked earlier.
**Trade-off:** Adds ~15 minutes per interpretation and review
cycle. Worth it — one skipped experiment saves hours.

---

## DEC-008: Val Loss as Primary Metric

**Date:** 2026-03-12
**Status:** accepted
**Context:** HYP-001, HYP-001b, and initial HYP-001c all used
training loss as the primary metric. HYP-001c revealed this
completely masked overfitting (train-val gap ~0.85). Architecture
comparisons on training loss are unreliable.
**Decision:** All future experiment pre-registrations must use
val_loss (or val_bpb) as the primary metric. Train_loss is a
secondary diagnostic metric only.
**Rationale:** Generalization is what matters. Training loss
measures memorization capacity, not architectural quality.
**Supersedes:** Implicit convention of using "final training loss"
as primary metric in HYP-001 through HYP-001c.

---

## DEC-009: Implement μP for Cross-Scale Predictions

**Date:** 2026-03-13
**Status:** accepted
**Context:** HYP-001d showed that architecture comparisons at
3M params are confounded by regularization sensitivity. Literature
review (LIT-003, LIT-024, LIT-025) shows μP enables hyperparameter
transfer across model widths, making small-scale comparisons more
predictive of large-scale behavior. No MLX implementation exists.
**Decision:** Implement μP (Maximal Update Parameterization) in
lmxlab. Core changes: (1) scaled weight initialization,
(2) per-layer learning rate groups, (3) MuReadout output layer,
(4) attention logit rescaling (1/d_head vs 1/√d_head).
**Rationale:** Without μP, our fixed-LR comparisons are confounded
by architecture-dependent optimal LRs (LIT-003). μP removes this
confound, making our 3M-param results more generalizable.
**Trade-offs:** ~4-6 days effort. Adds complexity to config and
optimizer. Regularization HPs (dropout, weight decay) still don't
transfer via μP.

## Research Backlog: μP and Cross-Scale Transfer

**Open questions to explore after implementation:**

1. **Does μP change the GPT vs LLaMA ranking?** Re-run HYP-001c
   with μP-scaled HPs. If rankings change, our previous results
   were LR-confounded.
2. **What is the minimum proxy width?** How small can the base
   model be while still producing transferable HPs? Test d_model
   {32, 64, 128} as proxies for d_model=256.
3. **Does μP interact with dropout sensitivity?** HYP-001d found
   LLaMA optimal at dropout=0.1 vs GPT at 0.2. Does this
   differential persist under μP, or was it an LR artifact?
4. **Depth transfer:** LIT-025 (Apple 2025) extends μP to depth.
   Can we transfer HPs across n_layers too? This would let us
   predict 12-layer behavior from 2-layer experiments.
5. **Coordinate check validation:** Implement the standard μP
   validation test (verify activations don't grow with width)
   as a unit test.

---

## DEC-010: Parameter Golf Primary Metric

**Date:** 2026-03-18
**Status:** accepted
**Context:** OpenAI Parameter Golf challenge uses bits-per-byte
(BPB) on FineWeb validation as the competition metric.
**Decision:** Primary metric for all PGolf experiments is val_bpb
(lower is better). val_loss is secondary.
**Rationale:** BPB is tokenizer-agnostic and is the official
scoring metric. Different from our usual val_loss primary.

## DEC-011: Artifact Size Gate

**Date:** 2026-03-18
**Status:** accepted
**Context:** PGolf has a hard 16,000,000 byte artifact limit
(code + int8 quantized + zlib compressed model).
**Decision:** Check estimated artifact size BEFORE training.
Log runs that exceed 16MB as `constraint_violation`.
**Rationale:** Prevents wasting full training runs on models
that cannot produce valid submissions.

## DEC-012: Local MLX for Iteration

**Date:** 2026-03-18
**Status:** accepted
**Context:** Mac with Apple Silicon available for development.
8xH100 GPUs needed for official validation.
**Decision:** Use train_gpt_mlx.py locally for fast iteration.
Local BPB is for relative comparison only (not absolute).
**Rationale:** Faster iteration cycle. Architecture and
relative improvements transfer to GPU even if absolute
BPB numbers differ.

## DEC-013: Minimum Progress Threshold

**Date:** 2026-03-18
**Status:** accepted
**Context:** Need to know when to pivot vs continue tuning.
**Decision:** Minimum BPB improvement to count as progress:
0.002 for local runs, 0.005 for official submissions
(matching the competition's SOTA threshold).
**Rationale:** Below 0.002 is likely noise on local hardware.

---

## DEC-014: Agent Search vs Bayesian Optimization

**Date:** 2026-03-18
**Status:** accepted
**Context:** Ravid Shwartz Ziv's analysis of Karpathy's autoresearch
showed that Optuna TPE (Bayesian optimization) with 8 human-selected
hyperparameters outperformed an AI agent searching freely, with the
same number of trials (~85). Key insight: 8 informed parameters beat
23 blind ones because the agent's "open-ended" search is really just
hyperparameter search without a defined search space.
**Decision:** For hyperparameter tuning, prefer structured Bayesian
optimization (Optuna/TPE) with domain-informed parameter selection
over open-ended agent exploration. Use agent search for STRUCTURAL
changes (architecture, algorithm) where the search space can't be
easily parameterized. This is what our autorun loop does well:
the agent found weight sharing, head configuration changes, and
skip connection interactions that Optuna couldn't discover because
they aren't continuous hyperparameters.
**Rationale:** Agents add value through (1) code modification for
architectural changes and (2) reasoning about WHY results occurred.
For pure numeric tuning within a fixed architecture, Bayesian
methods are more sample-efficient. Our pgolf results confirm this
split: the big wins (UNIQUE_BLOCKS=3 at +0.029, NUM_HEADS=4 at
+0.072) are structural changes an optimizer can't find, while
schedule/LR tuning was confounded and low-value.
**Source:** linkedin.com/in/ravidshwartzziv — "Do LLM Coding agents
fool us?" post (2026-03)

---

## DEC-015: Local Mac Iteration Complete, Move to GPU

**Date:** 2026-03-19 (updated after HYP-031)
**Status:** accepted (FINAL)
**Context:** 60+ experiments across HYP-017 through HYP-031. B-022
(step count dominates local BPB) means local Mac experiments are
unreliable for: depth/width tradeoffs, LR/momentum tuning, and any
config that changes per-step time. After initial DEC-015, continued
with code-level/iso-step experiments (HYP-029 QAT, HYP-030 SWA,
HYP-031 NorMuon) which were productive. Now exhausted.
**Decision:** Stop ALL local Mac experiments for parameter golf.
All productive iso-step comparisons are done. GPU is required for
remaining work (mom=0.99, softcap, MLP 3x, sp2048 vocab, SWA).
**Best local INT8 BPB:** 1.7030 (6L+3u+4h/4kv+EVAL_STRIDE=256+NORMUON=1)
**GPU-ready config:** UNIQUE_BLOCKS=3, NUM_HEADS=4, NUM_KV_HEADS=4,
EVAL_STRIDE=256, MLP_TYPE=relu2, USE_SKIP=1, MUON_BACKEND_STEPS=5,
NORMUON=1, FP16_EMBED=1 (untested but free).
**Rationale:** Reliable local findings (wide heads, weight sharing,
skip connections, relu^2, NorMuon) are all iso-step comparisons.
SWA and QAT(INT8) tested and found unhelpful locally. Everything
else is confounded by the 64x batch size mismatch.
**Source:** B-022, HYP-024 through HYP-031 results
