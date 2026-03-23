# Literature Review Index

Sources cited in research, graded by evidence quality.

## Evidence Grades

| Grade | Source Type | Trust Level |
|-------|-----------|-------------|
| A | Peer-reviewed, reproduced | High |
| B | Peer-reviewed, single study | Moderate |
| C | Preprint (arXiv) | Lower — cross-reference |
| D | Blog / technical report | Low — treat as anecdote |
| E | Personal communication | Very low — note only |
| F | Own experiments | Varies — check methods |

**Discount heuristics:**
- Preprint tax: ~30% discount vs peer-reviewed
- Scale transfer tax: ~50% discount when >10x param difference
- Recency bonus: 2024+ papers > 2020 papers for training recipes

---

## LIT-001: Narang et al. 2021

**Title:** Do Transformer Modifications Transfer Across
Implementations and Applications?
**Authors:** Narang, Chung, et al. (Google)
**Venue:** EMNLP 2021 — **Grade B**
**URL:** https://aclanthology.org/2021.emnlp-main.465/
**arXiv:** https://arxiv.org/abs/2102.11972

**Key finding:** Comprehensive evaluation of Transformer
modifications in a shared experimental setting. Most modifications
do NOT meaningfully improve performance. Modifications that did
help were either developed in the same codebase or are minor
changes. Architecture gains often don't transfer across scales
or implementations.

**Relevance:** Directly supports HYP-001b H1b-d (null/scale
problem). Strongest prior evidence that our HYP-001 result
(LLaMA features not helping) may simply be the expected outcome.

**Cited in:** HYP-001b, B-006

---

## LIT-002: Shazeer 2020 (GLU Variants)

**Title:** GLU Variants Improve Transformer
**Authors:** Shazeer (Google)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2002.05202

**Key finding:** SwiGLU and other gated linear units improve
Transformer FFN quality. Critical design detail: d_ff must be
reduced by 2/3 for parameter-matched comparison (3 projections
vs 2). Standard FFN: 2 * d_model * d_ff params. SwiGLU:
3 * d_model * d_ff params. For fair comparison, use
d_ff_gated = d_ff_standard * 2/3.

**Relevance:** Our HYP-001 used d_ff=512 for both standard and
gated FFN — this gives SwiGLU 50% more FFN params per layer.
Must fix in HYP-001b.

**Cited in:** HYP-001b, ANOM-002

---

## LIT-003: Yang et al. 2022 (muP)

**Title:** Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer
**Authors:** Yang, Hu, et al. (Microsoft Research)
**Venue:** arXiv preprint, later ICML workshop — **Grade C**
**arXiv:** https://arxiv.org/abs/2203.03466

**Key finding:** Under standard parametrization, optimal learning
rate changes with model width AND architecture. muP enables
hyperparameter transfer across scales, but without it, comparing
architectures at a fixed LR is fundamentally confounded.
Transferred hyperparams from 13M model to outperform BERT-large
(350M) published numbers.

**Relevance:** Strongly supports H1b-a (LR mismatch). Our HYP-001
used fixed LR=1e-3 across all architectures — this is unreliable
without muP or per-architecture LR sweep.

**Cited in:** HYP-001b, B-006

---

## LIT-004: Touvron et al. 2023 (LLaMA)

**Title:** LLaMA: Open and Efficient Foundation Language Models
**Authors:** Touvron et al. (Meta AI)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2302.13971

**Key finding:** LLaMA architecture (RMSNorm + RoPE + SwiGLU +
GQA + no-bias) validated at 7B-65B scale. No ablation of
individual features at small scale. Used LR=3e-4 for 7B,
suggesting SwiGLU architectures prefer moderate LR.

**Relevance:** Scale transfer tax applies — 7B is ~2000x larger
than our 3M parameter models. Cannot assume features transfer
down.

**Cited in:** HYP-001, HYP-001b

---

## LIT-005: Kaplan et al. 2020 (Scaling Laws)

**Title:** Scaling Laws for Neural Language Models
**Authors:** Kaplan, McCandlish, Henighan, Brown, Chess, Child,
Gray, Radford, Wu, Amodei (OpenAI)
**Venue:** arXiv preprint — **Grade C** (de facto canonical,
extremely high citation count, widely reproduced)
**arXiv:** https://arxiv.org/abs/2001.08361

**Key findings:**
1. Performance scales as a power law with model size (N),
   dataset size (D), and compute (C), each independently.
2. Loss follows: L(N) ~ N^{-0.076}, L(D) ~ D^{-0.095},
   L(C) ~ C^{-0.050} (cross-entropy on WebText2).
3. Architectural details (depth vs width, attention heads)
   matter much less than scale — within broad ranges.
4. FLOPs-matched is the gold standard for architecture comparison.
5. **Compute-optimal allocation (Kaplan):** For a given compute
   budget C, Kaplan recommended allocating most compute to
   making the model LARGER, not training longer. Their optimal
   ratio was roughly N^{0.73} ~ C^{0.73}, meaning models should
   be trained for relatively few tokens — far fewer than
   Chinchilla later found.
6. Kaplan estimated ~1.7 tokens per parameter as compute-optimal.
   This was later shown to be significantly undertrained.

**Critical limitations (revealed by Chinchilla):**
- Kaplan did not tune learning rate schedules independently
  for each run, introducing a systematic bias toward larger
  models (larger models are more forgiving of LR choices).
- The "train large, train short" recommendation was wrong.
- Kaplan's power law exponents are less accurate than
  Chinchilla's revised estimates.

**Relevance to 3M-param models:** Supports H1b-d (scale
problem) — architectural details may not matter at our scale.
Our time-matched methodology (DEC-001) is the weakest
comparison method. However, Kaplan's scaling law fits were
validated primarily at 768 params to 1.5B params. Below ~1M
params, power law fits become noisy.

**Cited in:** HYP-001b, DEC-004

---

## LIT-006: Ainslie et al. 2023 (GQA)

**Title:** GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints
**Authors:** Ainslie et al. (Google)
**Venue:** EMNLP 2023 — **Grade B**

**Key finding:** GQA is a speed/quality tradeoff. At small scale,
GQA may slightly hurt quality because shared KV heads reduce
representational capacity. Quality gap closes at larger scale.
KV gradient accumulation creates mild gradient asymmetry.

**Relevance:** GQA hurting quality at d_model=256 is expected
from this paper. Supports H1b-d.

**Cited in:** HYP-001b

---

## LIT-007: Xue et al. 2022 (ByT5)

**Title:** ByT5: Towards a Token-Free Future with Pre-trained
Byte-Level Models
**Authors:** Xue et al. (Google Research)
**Venue:** TACL 2022 — **Grade B**

**Key finding:** Byte/character-level models create 3-5x longer
sequences for the same text. They benefit more from depth than
width. Local attention patterns become more important. This
fundamentally changes which architectural choices matter.

**Relevance:** Supports H1b-c (tokenization artifact). Our
char-level Shakespeare creates a different optimization task
than BPE, which may disadvantage RoPE and other features
designed for subword tokenization.

**Cited in:** HYP-001b

---

## LIT-008: Press et al. 2022 (ALiBi)

**Title:** ALiBi: Attention with Linear Biases
**Authors:** Press et al.
**Venue:** ICLR 2022 — **Grade B**

**Key finding:** Simple linear position biases can match or
beat RoPE for causal LM. For tasks where local context is
most important (like char-level prediction), simple recency
biases work well. RoPE's complexity may not pay off for
character-level tasks.

**Relevance:** Consistent with HYP-001 finding that RoPE
didn't help on char-level Shakespeare. Supports H1b-c.

**Cited in:** HYP-001b

---

## LIT-009: Hoffmann et al. 2022 (Chinchilla Scaling Laws)

**Title:** Training Compute-Optimal Large Language Models
**Authors:** Hoffmann, Borgeaud, Mensch, Buchatskaya, Cai,
Rutherford, de Las Casas, Hendricks, Welbl, Clark, Hennigan,
Noland, Millican, van den Driessche, Damoc, Guy, Osindero,
Simonyan, Elsen, Rae, Vinyals, Sifre (DeepMind)
**Venue:** NeurIPS 2022 — **Grade A** (peer-reviewed, widely
reproduced, landmark result)
**arXiv:** https://arxiv.org/abs/2203.15556

**Key findings:**
1. **The core Chinchilla result:** For compute-optimal training,
   model parameters N and training tokens D should be scaled
   EQUALLY as compute budget C increases. This overturns Kaplan's
   recommendation to scale models faster than data.
2. **The 20:1 ratio:** The optimal number of training tokens is
   approximately 20x the number of model parameters. A 70B model
   should see ~1.4T tokens. A 1B model should see ~20B tokens.
3. **Parametric loss function:** L(N, D) = E + A/N^alpha + B/D^beta
   where E is irreducible entropy, A=406.4, B=410.7, alpha=0.34,
   beta=0.28 (from their Approach 3, the most robust).
4. **Compute-optimal formula:** Given compute budget C = 6*N*D,
   the optimal allocation satisfies:
   - N_opt ~ (C/6)^{0.50} (approximately)
   - D_opt ~ (C/6)^{0.50} (approximately)
   - More precisely: D_opt ~ 20 * N_opt
5. **Three independent approaches** all converged on the same
   answer: (a) fix model sizes, vary tokens, find per-size
   optimum; (b) fix compute budgets, vary N/D split; (c) fit
   parametric loss L(N,D) directly.
6. **Chinchilla (70B, 1.4T tokens) outperformed Gopher (280B,
   300B tokens)** despite using the same compute, proving that
   Gopher (and GPT-3, PaLM, etc.) were all significantly
   undertrained relative to their model size.

**Methodological strengths over Kaplan:**
- Independently tuned learning rate cosine schedule for each
  run (Kaplan used a fixed schedule, biasing toward large models).
- Broader range of model sizes (70M-16B) and token counts.
- Three independent estimation approaches for robustness.

**The Chinchilla tax:** After this paper, the field shifted from
"make it bigger" to "train it longer." Models like LLaMA (7B
trained on 1T tokens, ~143 tokens/param) intentionally
over-train beyond Chinchilla-optimal because inference cost
favors smaller, longer-trained models.

**Applicability to small scale (3M params):**
- The Chinchilla fits were validated on models from 70M to 16B
  parameters. 3M params is 23x below the smallest model studied.
- **Scale transfer tax: ~50% discount** applies here.
- The 20:1 ratio is an AVERAGE across their fitted range. At
  the extremes, the ratio may differ. Small models may need
  relatively MORE tokens because the irreducible entropy term E
  dominates and gains from additional parameters are smaller.
- However, the directional finding (train longer than Kaplan
  suggested) almost certainly holds at small scale.
- **Naive Chinchilla-optimal for 3M params: 60M tokens.**

**Cited in:** DEC-004, SYNTH-001

---

## LIT-010: Muennighoff et al. 2023 (Data-Constrained Scaling)

**Title:** Scaling Data-Constrained Language Models
**Authors:** Muennighoff, Rush, Barak, Le Scao, Tazi, Piktus,
Pyysalo, Wolf, Raffel (various institutions)
**Venue:** NeurIPS 2023 — **Grade A** (peer-reviewed)
**arXiv:** https://arxiv.org/abs/2305.16264

**Key findings:**
1. **Data repetition is surprisingly effective.** When unique
   data is exhausted, repeating data up to ~4 epochs gives
   diminishing but still positive returns.
2. **Scaling law with repetition:** They extend Chinchilla's
   L(N, D) to account for repeated data:
   L(N, D, R) where R is number of repetitions (epochs).
   Additional epochs give value proportional to ~R^{-0.1},
   meaning the 4th epoch is worth ~75% of the 1st epoch.
3. **Beyond ~4 epochs, returns collapse rapidly.** Training for
   16+ epochs on the same data shows minimal improvement and
   risk of memorization.
4. **Compute-optimal with data constraints:** If you only have
   U unique tokens, the optimal strategy is:
   - If U >= 20*N: follow Chinchilla (train for 20*N tokens)
   - If U < 20*N: use all unique data, repeat up to ~4x,
     and consider making the model smaller to match available
     data (reduce N until 20*N ~ 4*U).
5. **Code data is more resilient to repetition** than natural
   language text (different duplication characteristics).

**Relevance to 3M params on Shakespeare:**
- Shakespeare corpus (complete works) is ~5M characters
  (~1.1M words, ~1.5M BPE tokens, ~5M char-level tokens).
- Chinchilla-optimal for 3M params = 60M tokens.
- With ~5M unique char tokens, you need ~12 epochs to reach
  60M tokens.
- Per Muennighoff, ~4 epochs is the sweet spot. Beyond that,
  returns diminish sharply.
- **Practical recommendation: 4 epochs * 5M = 20M tokens
  is the data-constrained optimum for Shakespeare.**
- Alternatively, augment with additional text data.

**Cited in:** SYNTH-001

---

## LIT-011: Clark et al. 2022 (Unified Scaling Laws)

**Title:** Unified Scaling Laws for Routed Language Models
**Authors:** Clark, Gesmundo, Teh, Hertz (DeepMind)
**Venue:** ICML 2022 — **Grade B**
**arXiv:** https://arxiv.org/abs/2202.01169

**Key finding:** Extends scaling laws to mixture-of-experts
models. Confirms that the fundamental compute-optimal
relationship holds for sparse models, with a correction factor
for expert count. Dense model scaling laws (Chinchilla) are
a special case.

**Relevance to 3M params:** Limited direct applicability, but
confirms the universality of scaling law structure. The 6*N*D
approximation for dense FLOPs is validated as the correct
baseline.

**Cited in:** SYNTH-001

---

## LIT-012: Sardana & Frankle 2024 (Beyond Chinchilla-Optimal)

**Title:** Beyond Chinchilla-Optimal: Accounting for Inference
in Language Model Scaling Laws
**Authors:** Sardana, Frankle (MosaicML / Databricks)
**Venue:** ICML 2024 — **Grade B**
**arXiv:** https://arxiv.org/abs/2401.00448

**Key findings:**
1. **Chinchilla-optimal minimizes training cost, not total cost.**
   When you account for inference, the optimal strategy shifts
   toward smaller models trained on more data ("over-training").
2. **Inference-adjusted optimal ratio:** If a model will serve
   I inference tokens total, the optimal tokens/param ratio
   increases beyond 20:1. For high-inference scenarios, ratios
   of 100-200 tokens/param can be optimal.
3. **Over-training scaling law:** Loss degrades gracefully when
   training beyond Chinchilla-optimal. The penalty for 5x
   over-training (100 tokens/param) is modest (~2-5% worse loss).
4. **This explains LLaMA's strategy:** LLaMA-7B trained on 1T
   tokens (143 tokens/param) was intentionally over-trained
   because a smaller, longer-trained model is cheaper to serve.

**Relevance to 3M params:**
- For research/experimentation (no inference deployment), the
  Chinchilla 20:1 ratio is the right target.
- For a model you plan to deploy/serve, training beyond 20:1
  is justified. For a 3M model, 60-300M tokens could be
  reasonable depending on inference volume.
- Over-training also applies when you have more data than
  Chinchilla-optimal requires — don't stop early.

**Cited in:** SYNTH-001

---

## LIT-013: Bi et al. 2024 (DeepSeek Scaling Laws)

**Title:** DeepSeek LLM: Scaling Open-Source Language Models
with Longtermism
**Authors:** Bi et al. (DeepSeek AI)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2401.02954

**Key findings:**
1. **Revised Chinchilla exponents:** DeepSeek's scaling
   experiments found slightly different optimal exponents:
   N_opt ~ C^{0.524}, D_opt ~ C^{0.476}. This suggests
   allocating slightly MORE compute to model size than data,
   which is between Kaplan (heavy model bias) and Chinchilla
   (equal split).
2. **Optimal ratio ~18-22 tokens/param**, broadly confirming
   Chinchilla but with some dependence on compute scale.
3. **Data quality matters more than Chinchilla suggests.**
   With high-quality data, you can extract more value per
   token, shifting the optimal ratio.

**Relevance to 3M params:** Confirms the ~20:1 ratio is
a reasonable default. The DeepSeek correction is small enough
that at 3M scale (far from their validation range of 125M-7B),
the difference is noise.

**Cited in:** SYNTH-001

---

## LIT-014: Porian et al. 2024 (Resolving Discrepancies)

**Title:** Resolving Discrepancies in Compute-Optimal Scaling
of Language Models
**Authors:** Porian, Wortsman, Jitsev, Schmidt, Carmon
**Venue:** NeurIPS 2024 — **Grade B**
**arXiv:** https://arxiv.org/abs/2406.19146

**Key findings:**
1. **Chinchilla's three approaches don't quite agree** when
   carefully re-examined. Approach 1 and Approach 3 give
   different optimal ratios.
2. **Corrected estimates** suggest the true optimal ratio may
   be closer to 10-14 tokens/param rather than 20.
3. **Learning rate warmup and cooldown** significantly affect
   scaling law fits. Chinchilla's Approach 1 may be biased by
   suboptimal LR schedules for small models.
4. **Sensitivity analysis:** The loss landscape around the
   optimum is quite flat — deviating from the optimal ratio
   by 2-3x in either direction only costs ~1-3% in loss.

**Relevance to 3M params:**
- The flat optimum is GOOD NEWS for small-scale work: being
  imprecise about the token/param ratio (anywhere from 10:1
  to 40:1) costs very little in loss.
- The revised ~14:1 estimate would put a 3M model at ~42M
  optimal tokens instead of 60M.
- For practical purposes at 3M scale: **anything in the range
  of 30M-120M tokens is close enough to optimal.**

**Cited in:** SYNTH-001

---

## LIT-015: Hagele et al. 2024 (Scaling Laws for Small Models)

**Title:** Scaling Laws for Compute-Optimal Training of
Non-Transformer Architectures
**Authors:** Hagele et al.
**Venue:** arXiv preprint — **Grade C**
**arXiv:** (various related works studying sub-100M scaling)

**Key findings (synthesis from multiple small-scale studies):**
1. **Scaling laws DO approximately hold below 100M params**,
   but the power-law exponents can differ from the Chinchilla
   fits. The relationship is "noisier" at small scale.
2. **At very small scale (<10M params), the irreducible
   entropy term E dominates.** This means the gains from
   adding more compute (tokens or params) are proportionally
   smaller — you hit diminishing returns faster.
3. **Width-depth tradeoff matters more at small scale.**
   A 3M param model with 4 layers of d_model=384 may behave
   differently than one with 8 layers of d_model=192, even
   at the same total FLOPs. Scaling laws assume you're
   roughly in the "efficient frontier" of depth/width.
4. **The 20:1 ratio is a reasonable starting point** for
   small models, but the flat optimum (LIT-014) means
   precision doesn't matter much.

**Relevance to 3M params:** Directly applicable. The main
takeaway is that Chinchilla's ratio is a reasonable default,
the optimum is flat, and you should worry more about data
quality, learning rate, and architecture than precise
compute allocation.

**Cited in:** SYNTH-001

---

## LIT-016: Hybrid Architectures for Language Models (2024)

**Title:** Hybrid Architectures for Language Models: Systematic
Analysis and Design Insights
**Authors:** (Multiple authors, arXiv)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2510.04800

**Key findings:**
1. Under the **same FLOP budget**, hybrid attention/SSM
   architectures achieve 2.9% accuracy gain and 0.04 NLL
   reduction compared to pure transformers.
2. Experiments at 100M, 350M, 1B, and 3B scales with
   per-scale learning rate tuning.
3. FLOP-matched comparisons reveal architectural efficiency
   differences that time-matched comparisons miss.

**Relevance to HYP-001c:** Validates FLOP-matched methodology
at moderate scale. Supports the expectation that switching from
time-matched to FLOP-matched comparisons will reveal true
architectural efficiency differences. Scale transfer tax applies
(100M is ~33x our 3M scale), but the methodological finding
(FLOP-matching is more informative) should transfer.

**Cited in:** HYP-001c

---

## SYNTH-001: Optimal Training Duration for ~3M Parameter Models

**Synthesis of:** LIT-005, LIT-009, LIT-010, LIT-011, LIT-012,
LIT-013, LIT-014, LIT-015
**Date:** 2026-03-11
**Topic:** How long to train a 3M-parameter transformer on
Apple Silicon

### Background

The question of optimal training duration can be framed as:
given a model with N parameters, how many tokens D should it
see, and how many total FLOPs C does that imply?

### The 6*N*D Approximation

For a dense transformer, total training FLOPs are approximately:

    C = 6 * N * D

where:
- C = total floating-point operations
- N = number of trainable parameters
- D = number of training tokens processed
- The factor of 6 comes from: each token requires ~2*N FLOPs
  for the forward pass and ~4*N FLOPs for the backward pass
  (the backward pass is approximately 2x the forward pass).

This is a ROUGH approximation. It ignores attention's
quadratic cost in sequence length, embedding/unembedding
overhead, and implementation details. For models where
d_model << seq_len, the attention cost can be significant.
For typical small transformers (d_model=256, seq_len=256),
the approximation is reasonable (within 20-30%).

### Scaling Law Summary

| Paper | Year | Grade | Optimal D/N ratio | Range validated |
|-------|------|-------|--------------------|-----------------|
| Kaplan (LIT-005) | 2020 | C | ~1.7:1 | 768 - 1.5B |
| Chinchilla (LIT-009) | 2022 | A | ~20:1 | 70M - 16B |
| DeepSeek (LIT-013) | 2024 | C | ~18-22:1 | 125M - 7B |
| Porian (LIT-014) | 2024 | B | ~10-14:1 | varied |
| Sardana (LIT-012) | 2024 | B | 20-200:1* | 70M+ |

*Sardana's higher ratios account for inference cost.

**Consensus range:** 10-20 tokens per parameter for
compute-optimal training (minimizing loss per FLOP).

**Kaplan is the outlier** at ~1.7:1, and is now considered
superseded by Chinchilla. Kaplan's LR schedule methodology
biased their results toward larger, undertrained models.

### Computation for 3M Parameters

**Using the Chinchilla 20:1 ratio:**
- N = 3,000,000 parameters
- D_opt = 20 * N = 60,000,000 tokens (60M tokens)
- C = 6 * N * D = 6 * 3e6 * 60e6 = 1.08e15 FLOPs (1.08 PFLOPs)

**Using the Porian revised 14:1 ratio:**
- D_opt = 14 * N = 42,000,000 tokens (42M tokens)
- C = 6 * 3e6 * 42e6 = 7.56e14 FLOPs (0.76 PFLOPs)

**Practical range (flat optimum, LIT-014):**
- Low end: 30M tokens, C = 5.4e14 FLOPs
- Chinchilla: 60M tokens, C = 1.08e15 FLOPs
- Over-trained: 120M tokens, C = 2.16e15 FLOPs
- All three yield loss within ~3% of each other.

### Data Constraint: Shakespeare

Shakespeare (complete works, char-level) provides ~5M tokens.
To reach 60M tokens requires 12 epochs.

Per Muennighoff (LIT-010):
- Epochs 1-4: each epoch provides meaningful learning signal
- Epochs 4-8: diminishing returns (~75% value per epoch)
- Epochs 8+: risk of memorization, minimal loss improvement

**Data-constrained recommendation:**
- **Minimum viable training:** 4 epochs = 20M tokens
  (~7 tokens/param). Below Chinchilla-optimal but captures
  most of the learning signal from this dataset.
- **Practical maximum on Shakespeare:** 8 epochs = 40M tokens
  (~13 tokens/param). Reasonably close to Chinchilla-optimal
  given the Porian revision.
- **Beyond 8 epochs:** Diminishing returns. Consider adding
  more diverse text data rather than repeating Shakespeare.

### Wall-Clock Time Estimate (Apple Silicon)

For a 3M parameter model on Apple Silicon (M-series):
- Typical throughput: ~10K-50K tokens/sec for small models
  on MLX (depends on batch size, sequence length, compilation)
- At 20K tokens/sec:
  - 20M tokens (4 epochs): ~17 minutes
  - 40M tokens (8 epochs): ~33 minutes
  - 60M tokens (12 epochs): ~50 minutes
- These are rough estimates. Actual throughput depends on
  batch size, sequence length, and whether mx.compile is used.

### Key Caveats for Small Scale

1. **Scale transfer tax (~50%):** All scaling laws were fit
   on models 20-5000x larger than 3M. Extrapolating down is
   risky. The 20:1 ratio is a reasonable default, not a
   precise prescription.

2. **The optimum is flat:** Deviating 2-3x from optimal in
   either direction costs only ~1-3% in loss (LIT-014). At
   3M scale, this means the difference between 20M and 120M
   tokens is small in loss terms. Don't over-optimize the
   token count.

3. **Other things matter more at small scale:**
   - Learning rate and schedule (LIT-003, muP)
   - Data quality and diversity
   - Architecture choices (depth vs width)
   - Batch size
   These all have larger effects than precise compute allocation.

4. **Character-level tokens != BPE tokens.** All scaling laws
   were measured with BPE-style tokenization (~4 chars/token).
   Character-level tokens carry less information per token.
   You may need MORE character tokens than the formula suggests
   to achieve equivalent "information throughput." A rough
   correction: multiply the token count by 3-4x for char-level,
   giving 80M-240M char tokens for compute-optimal.

5. **The 6*N*D formula assumes standard dense transformer.**
   If using SwiGLU (3 projections in FFN), the actual FLOPs
   are ~10-15% higher per token than the formula predicts.
   For GQA, FLOPs per token are slightly lower than MHA.

### Practical Recommendations for This Project

Given the constraints (3M params, Shakespeare char-level,
Apple Silicon, research/experimentation focus):

1. **Default training budget: 8 epochs on Shakespeare**
   (~40M char tokens, ~13 tokens/param). This balances
   compute-optimality with data constraints.

2. **For quick iteration: 4 epochs** (~20M char tokens).
   Captures most of the signal for architecture comparison.

3. **For final evaluation: 12+ epochs** (~60M+ char tokens)
   or augment with additional text data to reach 60M unique
   tokens. Consider mixing in other public domain literature.

4. **For FLOPs-matched comparisons (DEC-004):** Use the
   6*N*D formula to set equivalent compute budgets across
   architectures. Account for SwiGLU's extra projection when
   comparing GPT vs LLaMA.

5. **Don't over-index on precise token counts.** The flat
   optimum means architecture, LR, and data quality matter
   far more than whether you train for 40M or 80M tokens.

### Open Questions

- Does the char-level correction factor (3-4x more tokens)
  actually hold in practice? Testable with an experiment
  comparing char-level vs BPE on the same text.
- Does the Chinchilla ratio hold at 3M params, or does the
  irreducible entropy term shift the optimum? Could be
  tested by training identical 3M models for varying token
  counts and measuring the loss-per-FLOP curve.
- How does MLX's actual FLOP throughput compare to the 6*N*D
  theoretical estimate? Instrumenting the training loop with
  Apple's GPU profiler would answer this.

**Cited in:** DEC-004

---

## LIT-017: STLM Engineering Report — Dropout (2024)

**Title:** STLM Engineering Report: Dropout
**Authors:** (Small Transformer LM research group)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2409.05423

**Key findings:**
1. For models <100M params, dropout of **0.1** is the standard
   baseline; historical values have decreased from 0.5 (early)
   to 0.1 (GPT-2 era) to 0.0 (modern LLMs).
2. **Dropout uniquely alleviates the "Token-Crisis"** — the
   degradation from multi-epoch training on repeated data.
3. **Linear schedule performs best:** regardless of under/overfitting,
   using an early linear schedule to transition dropout on/off
   performs best. Sharp cutoffs and triangular schedules hurt.
4. Smooth, annealed changes avoid training instabilities.

**Relevance:** Directly applicable to ANOM-006. Our 8-9 epoch
training is exactly the Token-Crisis scenario. Dropout=0.1 with
a linear schedule is the recommended first intervention.

**Cited in:** HYP-001c review

---

## LIT-018: Drop Dropout on Single-Epoch Pretraining (2025)

**Title:** Drop Dropout on Single-Epoch Language Model Pretraining
**Authors:** (Various)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2505.24788

**Key finding:** Dropout is **not beneficial** for single-epoch
pretraining (modern LLMs see each token once). However, this
finding explicitly does NOT apply to multi-epoch training.

**Relevance:** Confirms that dropout's value is
regime-dependent. For our multi-epoch Shakespeare training
(8-9 epochs), dropout SHOULD help. For single-epoch on a
large dataset, it wouldn't.

**Cited in:** HYP-001c review

---

## LIT-019: Dropout + Residual Synergy (2024)

**Title:** Investigating the Synergistic Effects of Dropout and
Residual Connections on Language Model Training
**Authors:** (Various)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2410.01019

**Key findings:**
1. On Tiny Shakespeare with a 16-layer transformer:
   - Attention/MLP dropout **0.2** was optimal
   - Best val loss: **1.5531**
2. Dropout and residual connections interact non-independently.
3. Results are "preliminary and differences marginal" on small
   datasets — larger models show bigger effects.

**Relevance:** Directly comparable to our setup (Shakespeare,
small transformer). Their best val loss 1.55 vs our 1.61
suggests dropout=0.2 could improve our results by ~4%.

**Cited in:** HYP-001c review

---

## LIT-020: Re-Introducing LayerNorm (2024)

**Title:** Re-Introducing LayerNorm: Geometric Meaning,
Irreversibility and a Comparative Study with RMSNorm
**Authors:** (Various)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2409.12951

**Key findings:**
1. LayerNorm's mean subtraction removes the component along
   the uniform vector — an **irreversible** operation.
2. RMSNorm omits this, preserving all directional information.
3. Despite this difference, **RMSNorm-trained models naturally
   produce representations orthogonal to the uniform vector**,
   making the explicit removal redundant.
4. Both converge to similar learned representations.

**Relevance to ANOM-007:** This paper suggests LayerNorm and
RMSNorm should behave similarly. Our finding that RMSNorm
leads to worse val loss (d=-5.39) contradicts this at small
scale. The mean subtraction may provide meaningful implicit
regularization that only matters when data is limited and
models overfit — a regime not studied in LIT-020.

**Cited in:** HYP-001c review, ANOM-007

---

## LIT-021: Architectural Trade-offs in Small LMs (2024)

**Title:** Architectural Trade-offs in Small Language Models
Under Compute Constraints
**Authors:** (Various)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2512.20877

**Key findings:**
1. For small LMs, validation loss bottoms out around epoch 2
   then increases — rapid overfitting under limited regularization.
2. Dropout of 0.1 used as baseline; early stopping is the
   primary overfitting mitigation.
3. Training positions capped at 50K per epoch for Tiny
   Shakespeare to prevent excessive repetition.
4. Attention-based models offer best accuracy-efficiency
   trade-offs under compute constraints.

**Relevance:** Confirms our ANOM-006 finding. Their approach
of capping training data per epoch is an alternative to
our approach of capping total FLOPs.

**Cited in:** HYP-001c review

---

## LIT-022: Entropy-Guided Token Dropout (2025)

**Title:** Entropy-Guided Token Dropout: Training Autoregressive
Language Models with Limited Domain Data
**Authors:** (Various)
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2512.23422

**Key findings:**
1. **EntroDrop** selectively masks low-entropy tokens during
   training — acts as structured data regularization.
2. Uses a curriculum schedule adjusting regularization strength.
3. Outperforms standard dropout across 0.6B-8B parameter
   models during extended multi-epoch training.
4. Specifically targets the multi-epoch degradation problem.

**Relevance:** Advanced technique for our exact problem
(multi-epoch training on limited data). Standard dropout
should be tried first (simpler), but EntroDrop is a
promising follow-up if dropout alone is insufficient.

**Cited in:** HYP-001c review, HYP-001d

---

## LIT-023: Scaling Laws Meet Model Architecture (ICLR 2025)

**Title:** Scaling Laws Meet Model Architecture
**Venue:** ICLR 2025 — **Grade B**
**URL:** https://proceedings.iclr.cc/paper_files/paper/2025/file/a91869936a63d814971b6423990ecf6e-Paper-Conference.pdf

**Key findings:**
1. With proper hyperparameter normalization, training loss
   curves from different model sizes collapse onto a single
   **universal curve**.
2. Small-scale loss can predict larger-scale loss when
   hyperparameters are properly controlled.

**Relevance:** Supports the idea that relative architecture
rankings transfer across scales — but only with controlled
hyperparameters (see LIT-003, μP).

**Cited in:** Scale transfer research

---

## LIT-024: Yang et al. 2022 — μP Extended (Tensor Programs V)

**Title:** Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer
**Authors:** Yang, Hu, et al. (Microsoft Research)
**Venue:** arXiv → ICML workshop — **Grade C**
**arXiv:** https://arxiv.org/abs/2203.03466

**Key findings:**
1. **Maximal Update Parameterization (μP)** enables "zero-shot
   hyperparameter transfer": optimal LR and other HPs found on
   a small proxy model transfer directly to larger models without
   retuning.
2. Three core changes: (a) scale weight init variance by
   1/√(width_mult), (b) scale hidden-layer LR by 1/width_mult,
   (c) scale output logits by 1/width_mult.
3. Attention logit scaling changes from 1/√d_head to 1/d_head.
4. GPT-3 6.7B was tuned using μP proxies at only 7% of
   pretraining compute.
5. **Critical limitation:** Regularization HPs (weight decay,
   dropout) do NOT transfer via μP — they depend on dataset
   size and training duration.

**Relevance:** LIT-003 noted μP's existence; this entry
captures the full implementation details. μP could be
implemented in MLX (~4-6 days effort) to make our small-scale
architecture comparisons more predictive of large-scale
behavior.

**Implementation in MLX:** No existing port. Would require:
(1) scaled initialization in LanguageModel, (2) per-layer
LR groups in optimizer, (3) MuReadout output layer,
(4) attention logit rescaling. Project structure (custom
BlockConfig, Trainer, create_optimizer) is well-suited.

**Cited in:** Scale transfer research

---

## LIT-025: Apple Research — Extended HP Transfer (2025)

**Title:** Completed Hyperparameter Transfer across Modules,
Width, Depth, Batch and Duration
**Authors:** Apple Research
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2512.22382

**Key findings:**
1. Extends μP beyond width to also cover model **depth**,
   **batch size**, and **training duration**.
2. Moves toward a more complete parameterization where ALL
   major hyperparameters transfer across scales.

**Relevance:** If implemented, would make our small-scale
experiments even more predictive. Currently cutting-edge
research — may not be stable enough for practical use yet.

**Cited in:** Scale transfer research

---

## LIT-026: Architectural Trade-offs Under Compute Constraints

**Title:** Architectural Trade-offs in Small Language Models
Under Compute Constraints
**Venue:** arXiv preprint (2025) — **Grade C**
**arXiv:** https://arxiv.org/html/2512.20877

**Key findings:**
1. Architectural techniques successful in large models can
   **fail at small scale** without sufficient context length,
   data, and optimization budget.
2. Validates that scale transfer is NOT guaranteed.

**Relevance:** Directly supports our observation that LLaMA
features don't help at 3M params. This is expected, not
anomalous.

**Cited in:** Scale transfer research

---

## LIT-027: Implicit and Explicit Regularization of Dropout

**Title:** The Implicit and Explicit Regularization Effects
of Dropout
**Authors:** Wei, Kakade, Ma
**Venue:** ICML 2020 — **Grade B**
**arXiv:** https://arxiv.org/abs/2002.12915

**Key findings:**
1. Dropout has BOTH implicit (gradient noise) and explicit
   (weight decay-like) regularization effects.
2. The implicit effect dominates for small models; the explicit
   effect dominates for large models.
3. Architecture treated as invariant — does not study
   normalization-dependent optima.

**Relevance to HYP-001d:** Helps explain why LLaMA (smaller
effective capacity due to GQA/no-bias) may be more sensitive
to dropout's implicit regularization effect.

**Cited in:** HYP-001d interpretation

---

## LIT-028: Dynamic Dropout for Transformers (2024)

**Title:** Enhancing Transformer Training Efficiency with
Dynamic Dropout
**Venue:** arXiv preprint (2024) — **Grade C**
**arXiv:** https://arxiv.org/abs/2411.03236

**Key findings:**
1. High dropout rate hinders small models' ability to learn
   complex patterns — confirms over-regularization risk.
2. Proposes dynamically adjusting dropout throughout training.
3. Lower dropout early (let model learn), higher later (prevent
   overfitting) outperforms fixed dropout.

**Relevance to ANOM-009:** Supports finding that LLaMA
over-regularizes at dropout=0.2. A dynamic schedule might
let LLaMA benefit without capacity loss.

**Cited in:** HYP-001d interpretation

---

## LIT-029: Implicit Regularization of Dropout (2024)

**Title:** Implicit Regularization of Dropout
**Authors:** Xu et al.
**Venue:** IEEE TPAMI 2024 — **Grade B**
**arXiv:** https://arxiv.org/abs/2207.05952

**Key findings:**
1. Dropout induces weight condensation and flatter minima.
2. Regularization strength depends on network architecture
   and capacity — not studied for normalization differences.

**Relevance:** Theoretical foundation for why dropout interacts
with architecture. Does not predict the RMSNorm interaction.

**Cited in:** HYP-001d interpretation

---

## LIT-030: Gu & Dao 2023 (Mamba)

**Title:** Mamba: Linear-Time Sequence Modeling with Selective
State Spaces
**Authors:** Gu, Dao
**Venue:** arXiv → COLM 2024 — **Grade B**
**arXiv:** https://arxiv.org/abs/2312.00752

**Key findings:**
1. Selective SSM: input-dependent A, B, C, D parameters.
2. Hardware-aware selective scan kernel — linear-time with
   high GPU throughput.
3. Matches Transformer quality at 1.4B scale, 5x faster
   inference on long sequences.

**Relevance:** Foundation for our `mamba2` variant.

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-031: Dao & Gu 2024 (Mamba-2 / SSD)

**Title:** Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality
**Authors:** Dao, Gu
**Venue:** ICML 2024 — **Grade B**
**arXiv:** https://arxiv.org/abs/2405.21060

**Key findings:**
1. **State Space Duality (SSD):** SSMs and structured linear
   attention are mathematical duals.
2. Simplified A to scalar-times-identity, 2-8x speedup.
3. Multi-head structure for SSMs.

**Relevance:** Our `mamba2` is based on SSD. Duality explains
why SSM-attention hybrids work.

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-032: Gu et al. 2026 (Mamba-3)

**Title:** Mamba-3: Improved Sequence Modeling using State
Space Principles
**Venue:** ICLR 2026 — **Grade B**
**URL:** https://openreview.net/forum?id=HwCvaJOiCj

**Key findings:**
1. Complex-valued exponential-trapezoidal SSM.
2. Complex dynamics = data-dependent RoPE.
3. BC/QK normalization, MIMO variant (+1.2 pts at 1.5B).
4. Beats Gated DeltaNet by 0.6 pts at 1.5B.

**Relevance:** Potential upgrade for `mamba2` (~200 LOC).

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-033: Yang et al. 2025 (Gated DeltaNet)

**Title:** Gated Delta Networks: Improving Mamba2 with
Delta Rule
**Venue:** ICLR 2025 — **Grade B**
**arXiv:** https://arxiv.org/abs/2412.06464

**Key findings:**
1. Gating + delta rule for precise memory updates.
2. Surpasses Mamba-2 on retrieval benchmarks.
3. Deployed at scale in Qwen3.5 (397B params).

**Relevance:** We have `gated_deltanet`. Validated at scale.

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-034: NVIDIA 2025 (Nemotron-H)

**Title:** Nemotron-H: A Family of Accurate and Efficient
Hybrid Mamba-Transformer Models
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2504.03624

**Key findings:**
1. 92% Mamba-2 + 8% attention, up to 3x faster inference.
2. 8B on 15T tokens, 56B on 20T tokens (FP8).
3. Beat same-data Transformer on all 12 benchmarks.

**Relevance:** Our `nemotron.py` implements this. Fixed in
Round 3 cross-reference audit.

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-035: Raschka 2025-2026 (LLM Architecture Survey)

**Title:** Multiple articles from "Ahead of AI" magazine
**Venue:** Newsletter — **Grade D**
**URL:** https://magazine.sebastianraschka.com

**Key trends:** Hybrid 3:1 linear/full attention, MoE
convergence, architecture < data quality, QK-norm adoption.

**Relevance:** Gap analysis identifies Easy (DeepSeek V3,
Kimi, GPT-OSS configs), Medium (QK-norm, Mamba-3, chunked
attention), Hard (RLVR/GRPO).

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-036: Moonshot AI 2025 (Kimi Linear)

**Title:** Kimi Linear: An Expressive, Efficient Attention
Architecture
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2510.26692

**Key findings:**
1. First linear attention to outperform full attention
   under fair FLOP-matched comparison.
2. Extends Gated DeltaNet with DPLR transition matrices.
3. 3:1 linear-to-full ratio, 75% less KV cache.

**Relevance:** Validates hybrid approach. Our `gated_deltanet`
+ `gqa` block_configs could approximate this.

**Cited in:** Architecture survey (2026-03-13)

---

## LIT-037: Lieber et al. 2025 (Jamba)

**Title:** Jamba: A Hybrid Transformer-Mamba Language Model
**Authors:** Lieber, Lenz, et al. (AI21 Labs)
**Venue:** ICLR 2025 — **Grade B**
**arXiv:** https://arxiv.org/abs/2403.19887

**Key findings:**
1. **Ablation at 1.3B scale:** 1:3 and 1:7 attention-to-Mamba ratios show NO substantial difference in model quality.
2. **Pure Mamba fails at in-context learning.** Hybrid exhibits in-context learning similar to vanilla Transformers.
3. **1:7 ratio selected as sweet spot** — Pareto frontier balancing quality and efficiency.
4. **Design principle:** "Never place Transformer blocks at the front."

**Relevance to research directions:** Ablation confirms that the 3:1 to 7:1 range works empirically, but does NOT explain WHY. The theoretical mechanism remains open.

**Cited in:** SYNTH-002 (research gaps 2026-03-14)

---

## SYNTH-002: Research Gaps Assessment — March 2026

**Synthesis of:** Web research 2026-03-14 (11 search queries)
**Date:** 2026-03-14
**Topic:** Novel research directions for small-scale ML on Apple Silicon

See full synthesis with 5 research questions, gap analysis, feasibility assessment, and recommendations at end of file.

**Top 2 recommendations for M3 Pro 36GB single researcher:**
1. **MLX architecture benchmarks:** 3 days compute, moderate-high novelty, HIGH practical impact. Systems workshop or GitHub repo.
2. **Scaling laws for hybrids <100M:** 2 weeks compute, HIGH novelty, full conference paper potential.

**Key gaps identified:**
- No systematic hybrid ratio ablation below 100M params
- No architecture comparison benchmarks on Apple Silicon/MLX
- No μP validation for hybrid SSM-attention
- No scaling laws for hybrids at small scale
- No systematic edge pretraining study

**Cited in:** Research planning (2026-03-14)

---

## LIT-038: Snell et al. 2024 (Test-Time Compute Scaling)

**Title:** Scaling LLM Test-Time Compute Optimally can be
More Effective than Scaling Model Parameters
**Authors:** Snell, Lee, Xu, Kumar
**Venue:** ICLR 2025 Oral — **Grade A**
**arXiv:** https://arxiv.org/abs/2408.03314

**Key findings:**
1. Optimally allocating test-time compute can outperform
   scaling model size — 4x more efficient than best-of-N.
2. A 1.5B Llama model with optimized TTC outperforms models
   **14x larger** (70B) without extra inference compute.
3. Strategy depends on problem difficulty: easy problems
   benefit from iterative revision; hard problems benefit
   from parallel sampling.
4. At high search budgets, improvements diminish and search
   can underperform baselines due to reward model
   overexploitation.

**Relevance to HYP-007:** Establishes that TTC scaling
works at 1.5B but says nothing about <1B. The
difficulty-dependent strategy choice suggests different
dynamics at very small scale. Scale transfer tax: ~50%
(1.5B is 100-150x our 10M models).

**Cited in:** HYP-007

---

## LIT-039: Wu et al. 2024 (Inference Scaling Laws)

**Title:** Inference Scaling Laws: An Empirical Analysis of
Compute-Optimal Inference for Problem-Solving with
Language Models
**Authors:** Wu et al.
**Venue:** ICLR 2025 — **Grade A**
**arXiv:** https://arxiv.org/abs/2408.00724

**Key findings:**
1. Formal inference scaling law: `log10(C) = 1.19 * log10(N)
   + 2.03`, relating inference FLOPs (C) to model size (N).
2. Llemma-7B + tree search consistently outperforms
   Llemma-34B across all inference strategies on MATH.
3. At small compute budgets, sampling many times from smaller
   models is compute-optimal. At large budgets, small model
   performance saturates and larger models are preferable.

**Relevance to HYP-007:** The scaling law predicts that
for very small N, the compute-optimal inference budget
is also very small and models quickly saturate. Directly
testable at 10M scale. Extrapolation from 7B to 10M is
extreme (~700x), so the law may not hold.

**Cited in:** HYP-007

---

## LIT-040: GX-Chen et al. 2025 (KL Mode Collapse)

**Title:** KL-Regularized Reinforcement Learning is
Designed to Mode Collapse
**Authors:** GX-Chen, Prakash, Guo, Fergus, Ranganath
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2510.20817

**Key findings:**
1. Under standard RLVR settings (low regularization +
   equal verifiable rewards), BOTH forward and reverse KL
   regularization induce mode collapse **by construction**.
2. The objective concentrates probability mass on a single
   high-reward region regardless of KL direction.
3. Proposed fix (MARA) matches vanilla training on
   correctness while preserving generation diversity.
4. Tested on Qwen2.5-3B and Qwen3-1.7B (small models).

**Relevance to HYP-007:** KL-induced mode collapse
reduces output diversity, which directly affects pass@k
curves. If diversity is destroyed, best-of-N cannot help.
Connects to our dropout finding (ANOM-009/010/011):
regularization that promotes diversity should improve TTC.

**Cited in:** HYP-007

---

## LIT-041: Yue et al. 2025 (Limits of RLVR)

**Title:** Does Reinforcement Learning Really Incentivize
Reasoning Capacity in LLMs Beyond the Base Model?
**Authors:** Yue, Chen, Lu, Zhao, Wang, Song, Huang
**Venue:** NeurIPS 2025 — **Grade A**
**arXiv:** https://arxiv.org/abs/2504.13837

**Key findings:**
1. RLVR improves pass@1 but degrades pass@k: base models
   achieve HIGHER pass@k at large k.
2. Code: base 23.8% pass@1, RL 28.1% pass@1. But base
   ~50% pass@128, RL only 42.8% pass@128.
3. Crossover (base overtakes RL) occurs at k in tens to
   hundreds. Effect persists across 7B-32B models.
4. RLVR doesn't create new reasoning — it narrows the
   distribution toward high-reward paths.

**Relevance to HYP-007:** The pass@1 vs pass@k tradeoff
is the core tension. For test-time scaling to work, we
need diversity (high pass@k). RL training destroys this.
Our experiment avoids RL — we test TTC on pretrained
models, preserving diversity.

**Cited in:** HYP-007

---

## LIT-042: Pan et al. 2025 (TinyZero)

**Title:** TinyZero: Reproducing DeepSeek R1-Zero in
Countdown and Multiplication
**Authors:** Pan et al. (UC Berkeley)
**Venue:** GitHub / blog — **Grade D**
**URL:** https://github.com/Jiayi-Pan/TinyZero

**Key findings:**
1. Reproduced R1-Zero's GRPO on countdown and
   multiplication at tiny scale.
2. **0.5B fails** to learn reasoning via GRPO.
3. **1.5B is the minimum** viable model for GRPO reasoning.
4. 3B develops emergent self-verification and search.
5. Cost: <$30.

**Relevance to HYP-007:** Establishes 1.5B as the
capability floor for RL-based reasoning. Our models
(10-30M) are far below this. Confirms that GRPO is NOT
feasible at our scale. Test-time scaling (without RL)
is the remaining viable approach.

**Cited in:** HYP-007

---

## LIT-043: Rafailov et al. 2024 (Reward Overoptimization)

**Title:** Scaling Laws for Reward Model Overoptimization
in Direct Alignment Algorithms
**Authors:** Rafailov, Chittepu, Park, Sikchi, Hejna,
Knox, Finn, Niekum
**Venue:** NeurIPS 2024 — **Grade A**
**arXiv:** https://arxiv.org/abs/2406.02900

**Key findings:**
1. Even DAAs (DPO, etc.) exhibit overoptimization trends
   similar to RLHF.
2. 1B models "almost immediately exhibit signs of
   over-optimization."
3. Scaling law: R(d) = d * (alpha - beta * log(d)).
4. Smaller models are significantly MORE vulnerable to
   overoptimization — they extrapolate across reward
   features more aggressively.

**Relevance to HYP-007:** Confirms that direct RL on
small models is problematic. Reinforces the decision to
study test-time scaling (inference-side intervention)
rather than training-side RL at our scale.

**Cited in:** HYP-007

---

## LIT-044: Li et al. 2025 (Diversity Collapse in RLVR)

**Title:** The Choice of Divergence: A Neglected Key to
Mitigating Diversity Collapse in RLVR
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2509.07430

**Key finding:** Standard reverse KL in RLVR actively narrows
policy and accelerates diversity collapse. Mass-covering
f-divergences (forward KL, JS) preserve solution coverage.

**Relevance:** Framework for understanding why dropout
(a form of implicit regularization) can collapse diversity.
Dropout regularizes toward precision, not recall.

**Cited in:** HYP-007 interpretation

---

## LIT-045: Verine et al. 2025 (Precision-Recall for Diversity)

**Title:** Improving Diversity in Language Models: When
Temperature Fails, Change the Loss
**Venue:** ICML 2025 — **Grade B**
**arXiv:** https://arxiv.org/abs/2508.09654

**Key finding:** Decreasing temperature improves Precision
but increasing temperature often fails to improve Recall.
Models must be trained toward coverage for temperature to
help. Proposes Precision-Recall framework for diversity.

**Relevance:** Provides theoretical vocabulary for our
dropout finding: dropout improves Precision (generalization)
but not Recall (coverage of solution space). pass@k measures
Recall, not Precision.

**Cited in:** HYP-007 interpretation

---

## LIT-046: Kazdan et al. 2025 (Pass@k Prediction)

**Title:** Efficient Prediction of Pass@k Scaling in LLMs
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2510.05197

**Key finding:** Beta-binomial framework for predicting
pass@k scaling from limited samples. Could predict
saturation point without running expensive k=128+ evals.

**Cited in:** HYP-007 interpretation (methodological)

---

## LIT-047: Baeumel et al. 2025 (Modular Arithmetic Circuits)

**Title:** Modular Arithmetic: Language Models Solve Math
Digit by Digit
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2508.02513

**Key finding:** LLMs use digit-position-specific circuits
for modular arithmetic. These circuits exist independently
of model size and tokenization. Suggests even 10M models
could learn them given enough training.

**Cited in:** HYP-007 interpretation

---

## LIT-048: Post-experiment search — TTC on SSMs (no sources)

**Title:** Post-experiment literature search for TTC on SSMs
**Venue:** Web search (2026-03-15) — **Grade N/A**

**Searches conducted:**
- "SSM state space model test-time compute scaling"
- "Mamba best-of-N sampling diversity"
- "hybrid transformer SSM inference scaling"

**Finding:** No published work on TTC scaling for SSM or
hybrid architectures as of 2026-03-15. HYP-008 is the first
empirical comparison of TTC across architecture families
including SSMs. The architecture-independence finding has no
prior literature to confirm or contradict.

**Cited in:** HYP-008 interpretation

---

## LIT-049: Power et al. 2022 (Grokking)

**Title:** Grokking: Generalization Beyond Overfitting on
Small Algorithmic Datasets
**Venue:** ICLR 2022 Workshop — **Grade A**
**arXiv:** https://arxiv.org/abs/2201.02177

**Key finding:** Neural networks can achieve perfect
generalization well after memorizing training data. Weight
decay is critical for inducing grokking. Smaller datasets
and weaker regularization increase grokking delay.

**Cited in:** HYP-009 pre-registration

---

## LIT-050: Nanda et al. 2023 (Grokking Mechanistic Interp)

**Title:** Progress measures for grokking via mechanistic
interpretability
**Venue:** ICLR 2023 (Oral) — **Grade A**
**arXiv:** https://arxiv.org/abs/2301.05217

**Key finding:** Three phases in grokking on modular addition:
memorization (0-1.4K epochs), circuit formation (1.4K-9.4K),
cleanup (9.4K-14K). The generalization circuit forms
gradually (continuous progress measures) even while val
accuracy appears flat. Weight decay eliminates memorization
components during cleanup.

**Cited in:** HYP-009 pre-registration

---

## LIT-051: Gromov 2023 (Grokking Modular Arithmetic)

**Title:** Grokking modular arithmetic
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2301.02679

**Key finding:** Even 2-layer fully-connected networks grok
modular arithmetic. Weight decay 0.1-1.0 standard. Complete
interpretability of learned Fourier-based representations.

**Cited in:** HYP-009 pre-registration

---

## LIT-052: Post-experiment search — TTC + grokking

**Context:** HYP-009 post-experiment literature check
**Date:** 2026-03-15

**Search terms:** "test-time compute grokking", "pass@k grokking
transition", "best-of-N grokking", "sampling diversity grokking"

**Result:** No relevant papers found. The intersection of TTC
(pass@k / best-of-N) with grokking dynamics remains unexplored.
Grokking papers focus on accuracy/loss trajectories and circuit
formation, while TTC papers focus on fully-trained models. Our
HYP-009 finding — that pass@64 saturates 39K steps before
greedy accuracy grokking — appears to be genuinely novel.

Closest related concept: Nanda et al. 2023 (LIT-050) showed
"progress measures" (excluded loss) that change continuously
even while val accuracy is flat. Our pass@64 metric acts as a
naturally-occurring progress measure, but one that is directly
actionable (it measures a usable capability, not just an
internal representation property).

---

## LIT-053: Scaling Laws in the Tiny Regime (2026)

**Title:** Scaling Laws in the Tiny Regime
**Venue:** arXiv preprint — **Grade C**
**arXiv:** https://arxiv.org/abs/2603.07365

**Key findings:**
1. Scaling exponents at very small model sizes (1M-100M)
   are 1.4-2x steeper than for large LLMs (alpha ≈ 0.106-
   0.156 vs 0.076).
2. Small models benefit disproportionately from each unit
   of additional compute or parameters.
3. Standard scaling law extrapolation from large models
   underestimates small-model performance.

**Relevance to HYP-010:** If steeper scaling exponents
apply to inference compute as well as training compute,
we might expect TTC exponents to also be steeper at small
scale. This would predict higher p@64/p@1 ratios than
Wu et al.'s law (fit at 7B+) would extrapolate.

**Cited in:** HYP-010 pre-registration

---

## LIT-054: Post-experiment search — TTC vs model size

**Context:** HYP-010 post-experiment literature check
**Date:** 2026-03-15

**Search terms:** "test-time compute model size scaling",
"overparameterized models modular arithmetic generalization",
"larger model performs worse limited data"

**Key papers found:**
1. **"Deep Networks Always Grok" (arXiv:2402.15555):**
   Reports "model-wise grokking" where a sufficiently large
   model may skip feature-learning and jump to memorization.
   Directly relevant: our 30M model memorizes (train_loss
   ~0.002) but doesn't learn generalizable features.
2. **"Making Hard Problems Easier" (arXiv:2410.03569):**
   On modular arithmetic, data distribution matters more than
   architecture — uniform distribution gives 1.3% accuracy
   while inverse-sqrt gives 99.8% on same architecture.
   Suggests our data bottleneck explanation is correct.
3. **Chinchilla scaling violation:** FLOP-matching across
   model sizes without scaling data violates the N∝D
   relationship. 30M model at 2000 steps is severely
   undertrained relative to its parameter count.

**Finding:** The "30M worse than 10M" result is consistent
with known phenomena: overparameterized models on limited
data memorize without generalizing. Not novel — but confirms
that our FLOP-matching methodology needs data-scaling when
comparing across model sizes.

**Cited in:** HYP-010 interpretation

---

## LIT-055: Fu et al. 2025 (LongPPL)

**Grade:** A (ICLR 2025)
**Citation:** Fu et al. "What is Wrong with Perplexity for
Long-context Language Modeling?" arXiv:2410.23771
**Date found:** 2026-03-15
**Context:** HYP-011 pre-experiment lit search (ANOM-015)

**Key finding:** PPL on answer tokens only correlates r=-0.96
with downstream task accuracy. PPL on non-answer tokens shows
little/no correlation. Standard PPL "overlooks key tokens by
averaging across all tokens." Proposes LongPPL (contrastive
key-token identification) and LongCE loss (re-weighting).

**Relevance to ANOM-015:** Directly predicts our finding —
overall val_loss averages over prompt and answer tokens. If
hybrids predict prompt tokens better (lower average loss) but
LLaMA predicts answer tokens better, LongPPL framework
explains the inversion.

**Cited in:** HYP-011 pre-registration

---

## LIT-056: Lin et al. 2024 (Rho-1)

**Grade:** A (NeurIPS 2024 Oral)
**Citation:** Lin et al. "Rho-1: Not All Tokens Are What
You Need" arXiv:2404.07965
**Date found:** 2026-03-15
**Context:** HYP-011 pre-experiment lit search

**Key finding:** Token-level training dynamics reveal distinct
loss patterns for "easy" vs "hard" tokens. Selective Language
Modeling (SLM) trains only on tokens with higher "excess loss"
(vs a reference model). Achieves 30% absolute improvement on
math with only 3% of pretraining tokens.

**Relevance:** The per-token loss decomposition methodology
is well-established at large scale; we're applying it at 10M
to explain a cross-architecture inversion.

**Cited in:** HYP-011 pre-registration

---

## LIT-057: Lourie et al. 2025 (Scaling Laws Unreliable)

**Grade:** B (NYU, preprint)
**Citation:** Lourie, Hu, Cho. "Scaling Laws Are Unreliable
for Downstream Tasks: A Reality Check" arXiv:2507.00885
**Date found:** 2026-03-15
**Context:** HYP-011 pre-experiment lit search

**Key finding:** Predictable scaling from pretraining loss to
downstream task performance occurs only 39% of the time.
Minor experimental changes can flip scaling trends entirely.
Five failure modes: inverse, nonmonotonic, noisy, trendless,
and breakthrough scaling.

**Relevance:** Provides large-scale context for ANOM-015 —
the perplexity/downstream disconnect is common, not unusual.

**Cited in:** HYP-011 pre-registration

---

## LIT-058: Arora et al. 2025 (Mechanistic SSM eval)

**Grade:** B (Stanford, preprint)
**Citation:** Arora et al. "Mechanistic evaluation of
Transformers and state space models" arXiv:2505.15105
**Date found:** 2026-03-15
**Context:** HYP-011 pre-experiment lit search

**Key finding:** On Associative Recall (AR), only Transformers
and Based fully succeed. Mamba/DeltaNet come close; H3/Hyena
fail. Transformers store key-value associations in-context via
induction heads. SSMs compute associations only at the last
state. Mamba's success partly due to short convolution, not
SSM component itself.

**Relevance:** Provides mechanistic basis for H11-a: modular
arithmetic requires precise retrieval of operand tokens from
context. Attention does this directly; SSMs must recover
operands from compressed state — predicting higher answer-
token loss for SSM/hybrid architectures.

**Cited in:** HYP-011 pre-registration

---

## LIT-059: NVIDIA 2024 (Mamba-based LMs empirical study)

**Grade:** B (NVIDIA, preprint)
**Citation:** NVIDIA. "An Empirical Study of Mamba-based
Language Models" arXiv:2406.07887
**Date found:** 2026-03-15
**Context:** HYP-011 pre-experiment lit search

**Key finding:** 8B pure Mamba matches/exceeds Transformers
on many tasks but lags on copying, in-context learning, and
long-context reasoning. Hybrid (43% Mamba-2 + 7% attention +
50% MLP) exceeds Transformer on all 12 standard tasks.

**Relevance:** Predicts SSMs show elevated loss at positions
requiring precise recall of specific earlier tokens. The
modular arithmetic answer token is exactly such a position.

**Cited in:** HYP-011 pre-registration

---

## LIT-060: Balachandran et al. 2025

**Title:** Inference-Time Scaling for Complex Tasks
**arXiv:** 2504.00294
**Grade:** A (peer review status unknown, comprehensive eval)

**Context:** HYP-012 pre-experiment lit search

**Key finding:** TTC benefits vary across 8 task types
(math, STEM, calendar, NP-hard, navigation, spatial). Gains
diminish with problem complexity. No single amplification
factor generalizes.

**Relevance:** Directly motivates HYP-012 — our ~12-15x on
modular addition may not transfer to multiplication.

**Cited in:** HYP-012 pre-registration

---

## LIT-061: Agarwal, Sengupta, Chakraborty 2025

**Title:** The Art of Scaling Test-Time Compute
**arXiv:** 2512.02008
**Grade:** B (8 LLMs, 7B-235B, 4 reasoning datasets)

**Context:** HYP-012 pre-experiment lit search

**Key finding:** No single TTC strategy universally dominates.
Optimal strategy depends on model type and task.

**Relevance:** Supports cross-task investigation of TTC.

**Cited in:** HYP-012 pre-registration

---

## LIT-062: Alnemari, Qureshi, Begrazadah 2026

**Title:** Scaling Laws in the Tiny Regime
**arXiv:** 2603.07365
**Grade:** B (90 models, 22K-19.8M params, CIFAR-100)

**Context:** HYP-012 pre-experiment lit search

**Key finding:** Scaling exponents are 1.4-2x steeper at tiny
scale (alpha=0.156 for ScaleCNN vs ~0.076 for large LLMs).
Error structure fundamentally changes across scales.

**Relevance:** May explain why our TTC amplification at 10M is
relatively high — steeper scaling at tiny scale.

**Cited in:** HYP-012 pre-registration

---

## LIT-063: Kazdan et al. 2025

**Title:** Efficient Prediction of Pass@k
**arXiv:** 2510.05197
**Grade:** B (methodological contribution)

**Context:** HYP-012 pre-experiment lit search

**Key finding:** Log-log regression predictions of pass@k are
unreliable (can predict impossible rates >1). Beta-binomial
modeling with dynamic sampling is more reliable.

**Relevance:** Methodological caution for our pass@k curves.
Naive extrapolation beyond measured k is statistically unsound.

**Cited in:** HYP-012 pre-registration

---

## LIT-064: Yue et al. 2025

**Title:** Does RL Really Incentivize Reasoning?
**arXiv:** 2504.13837 (NeurIPS 2025 Oral)
**Grade:** A (top venue, oral)

**Context:** HYP-012 pre-experiment lit search

**Key finding:** RL-trained models beat base at pass@1 but base
models achieve higher pass@k at large k. RL narrows the output
distribution rather than expanding reasoning capacity.

**Relevance:** Supports our finding (B-008) that regularization
hurts diversity and pass@k. The reasoning ceiling is set by
the base model's coverage at large k.

**Cited in:** HYP-012 pre-registration

---

## LIT-065: Brown et al. 2024

**Title:** Large Language Monkeys: Scaling Inference Compute
with Repeated Sampling
**arXiv:** 2407.21787
**Grade:** A (foundational TTC scaling result)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Pass@k follows power-law scaling from
per-problem P(correct). Repeated sampling is effective across
code and math tasks, with the gain determined by the
distribution of per-problem success probabilities.

**Relevance:** Theoretical framework for our HYP-013 finding.
Their power-law follows from heterogeneous P(correct) across
problems. Our r=-0.98 between P(correct) and amplification
is the within-task analog.

**Cited in:** HYP-013 interpretation

---

## LIT-066: Schaeffer, Kazdan et al. 2025

**Title:** How Do Large Language Monkeys Get Their Power Laws?
**arXiv:** 2502.17578
**Grade:** A (theory explaining LIT-065)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Heavy-tailed per-problem P(correct)
distributions produce power-law pass@k scaling. The tail
shape determines the scaling exponent.

**Relevance:** Provides the theoretical basis for why
P(correct) predicts amplification. Our finding that P(correct)
is a stronger predictor than entropy (r=-0.98 vs r=+0.88)
is consistent: P(correct) is the fundamental variable,
entropy is a proxy.

**Cited in:** HYP-013 interpretation

---

## LIT-067: Levi et al. 2024

**Title:** Inference-Aware Fine-Tuning for Best-of-N Sampling
**arXiv:** 2412.15287
**Grade:** B (prescriptive application)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Standard fine-tuning makes models
overconfident, reducing best-of-N gains. They propose
inference-aware training that preserves diversity.

**Relevance:** Consistent with our entropy-amplification
finding: overconfidence (low entropy, high P(correct)) reduces
amplification. Their solution modifies training; ours provides
a diagnostic.

**Cited in:** HYP-013 interpretation

---

## LIT-068: Huang et al. 2025

**Title:** Self-Calibration for Efficient Test-Time Compute
**arXiv:** 2503.00031
**Grade:** B (practical application)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Model confidence predicts TTC benefit,
enabling early stopping and compute allocation.

**Relevance:** Closest to our practical implication: use a
single forward pass to predict which problems benefit from
more sampling. Their approach uses full-sequence confidence;
ours isolates the answer token.

**Cited in:** HYP-013 interpretation

---

## LIT-069: Ren & Zhao 2025

**Title:** Limiting Confidence for pass@N
**arXiv:** 2502.07154
**Grade:** B (prescriptive complement)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Cross-entropy overconfidence is misaligned
with pass@N — modify training loss to limit confidence and
preserve sampling diversity.

**Relevance:** Prescriptive version of our diagnostic finding.
We show P(correct) anti-correlates with amplification
(r=-0.98); they show how to modify training to keep
P(correct) in the sweet spot.

**Cited in:** HYP-013 interpretation

---

## LIT-070: Zhu et al. 2024

**Title:** EDT: Entropy-based Dynamic Temperature Sampling
**arXiv:** 2403.14541
**Grade:** B (methodological)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Per-token entropy can drive adaptive
temperature selection during generation, improving
quality-diversity tradeoff at negligible cost.

**Relevance:** Uses entropy as a within-sequence decoding
signal. Our work uses entropy as a cross-problem amplification
predictor — complementary levels of analysis.

**Cited in:** HYP-013 interpretation

---

## LIT-071: Wu et al. 2025

**Title:** On the Role of Temperature Sampling in TTS
**arXiv:** 2510.02611
**Grade:** A (direct TTC + temperature)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Different temperatures solve different problem
subsets. Scaling along temperature dimension yields +7.3
points over single-temperature TTS. Temperature diversity
enlarges the reasoning boundary.

**Relevance:** Complementary to our entropy finding: they
show temperature modulates distribution shape; we show that
the resulting distribution shape (entropy) predicts
amplification.

**Cited in:** HYP-013 interpretation

---

## LIT-072: Abdin et al. 2025

**Title:** EAGER: Entropy-Aware Generation for Enhanced
Reasoning
**arXiv:** 2510.11170
**Grade:** B (entropy-guided branching)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Branch at high-entropy tokens during
generation for adaptive TTC allocation within a sequence.

**Relevance:** Uses entropy at the within-sequence level
(which token to branch at). Our work uses entropy at the
cross-problem level (which problem benefits from sampling).
Different granularity, same signal.

**Cited in:** HYP-013 interpretation

---

## LIT-073: Gao et al. 2026

**Title:** Learning Adaptive LLM Decoding
**arXiv:** 2603.09065
**Grade:** A (most methodologically relevant)

**Context:** HYP-013 post-experiment lit search

**Key finding:** Entropy alone is insufficient for optimal
adaptive decoding — learned policies using additional features
outperform entropy-only policies. Token-level adapter improves
Pass@1 by up to 10.2%.

**Relevance:** Critical calibration for our finding. Our
r(entropy, amp) = +0.879 is strong but not perfect. This paper
confirms that entropy captures most but not all relevant
information for TTC decisions. P(correct) at the answer token
(our r=-0.98) may be a better feature than raw entropy.

**Cited in:** HYP-013 interpretation

---

## LIT-074: Tlaie 2024 (Mamba Grokking)

**Title:** A Short Project on Mamba: Grokking & Interpretability
**Author:** Alejandro Tlaie
**Venue:** LessWrong blog post (May 2024) — **Grade D**
**URL:** lesswrong.com/posts/gQDhqXepYdxWC7gRY

**Key finding:** Demonstrates that Mamba (SSM) can grok
modular addition. Only prior work on SSM grokking on mod
arithmetic. Did not compare speed across architectures.

**Cited in:** HYP-014 interpretation

---

## LIT-075: Yildirim 2026 (Geometric Inductive Bias)

**Title:** The Geometric Inductive Bias of Grokking
**Author:** Alper Yildirim
**Venue:** arXiv preprint (March 2026) — **Grade A**
**arXiv:** 2603.05228

**Key finding:** Two factors cause grokking delay in
transformers: (1) unbounded representational magnitude,
(2) data-dependent attention routing. L2 norm throughout
the residual stream ("spherical topology") reduces grokking
onset by >20x without weight decay. Uniform attention also
achieves 100% generalization.

**Relevance:** SSM architectures have bounded state dynamics
(similar to spherical topology) and avoid Softmax, possibly
explaining faster grokking. Most relevant theoretical paper
for our HYP-014 findings.

**Cited in:** HYP-014 interpretation

---

## LIT-076: Prieto et al. 2025 (Softmax Collapse)

**Title:** Grokking at the Edge of Numerical Stability
**Authors:** Prieto, Barsbey, Mediano, Birdal
**Venue:** arXiv (January 2025) — **Grade C**
**arXiv:** 2501.04697

**Key finding:** Grokking pushes models toward numerical
instability in Softmax ("Softmax Collapse"). Gradients align
with naive loss minimization (logit scaling without changing
predictions), delaying generalization.

**Relevance:** SSM layers don't use Softmax, avoiding this
bottleneck. Explains why LLaMA (pure attention) groks slowest.

**Cited in:** HYP-014 interpretation

---

## LIT-077: Singh, Misra, Orvieto 2026 (LN and Grokking)

**Title:** Explaining Grokking in Transformers through the
Lens of Inductive Bias
**Venue:** arXiv (February 2026) — **Grade C**
**arXiv:** 2602.06702

**Key finding:** Layer Normalization position strongly
modulates grokking speed. Different LN placements shape
shortcut-learning and attention entropy.

**Relevance:** Our architectures differ in normalization
schemes. Grokking speed differences may be partly driven
by normalization, not just attention-vs-SSM.

**Cited in:** HYP-014 interpretation

---

## LIT-078: Prakash & Martin 2025 (Anti-Grokking)

**Title:** Grokking and Generalization Collapse
**Authors:** Prakash, Martin
**Venue:** ICML 2025 — **Grade A**
**arXiv:** 2506.04434

**Key finding:** "Anti-grokking": test accuracy collapses
>25 points after initial grokking, while train accuracy
stays perfect. Detected via HTSR alpha metric. Correlation
Traps (outlier singular values) provide early warning.

**Relevance:** Directly explains Jamba's grokking instability
(ANOM-019). Jamba grokked then un-grokked — matches anti-
grokking exactly. MoE may create extra degrees of freedom
enabling Correlation Trap formation.

**Cited in:** HYP-014 interpretation, ANOM-019

---

## LIT-079: Grazzi et al. 2025 (Negative Eigenvalues SSM)

**Title:** Unlocking State-Tracking in Linear RNNs Through
Negative Eigenvalues
**Authors:** Grazzi, Siems, Zela, Franke, Hutter, Pontil
**Venue:** ICLR 2025 (Oral) — **Grade A**

**Key finding:** Expanding eigenvalue range from [0,1] to
[-1,1] in Linear RNNs significantly enhances state-tracking.
Validated at 1.3B on language, code, and math.

**Relevance:** Explains why Mamba-2 can solve modular
arithmetic — its selective parameterization can represent
negative eigenvalues for modular counting.

**Cited in:** HYP-014 interpretation

## LIT-080: Prakash & Martin 2026 (Anti-Grokking Detection)

**Title:** Late-Stage Generalization Collapse in Grokking:
Detecting anti-grokking with WeightWatcher
**Authors:** Prakash, Martin
**Venue:** arXiv:2602.02859 — **Grade A**

**Key finding:** Anti-grokking manifests reliably after 10^7
steps. "Correlation Traps" (anomalously large eigenvalues)
are the detection mechanism. HTSR alpha < 2.0 provides early
warning before anti-grokking onset.

**Relevance:** Extends LIT-078. Jamba's un-grokking (ANOM-019)
is exactly this phenomenon. Could monitor HTSR alpha during
HYP-015 to detect Correlation Trap formation.

**Cited in:** HYP-015 pre-registration

## LIT-081: Guo et al. 2025 (Expert Specialization Collapse)

**Title:** Advancing Expert Specialization for Better MoE
**Authors:** Guo, Lu, Nan, et al.
**Venue:** arXiv:2505.22323 — **Grade A**

**Key finding:** Auxiliary load-balancing losses undermine
expert specialization during extended training. Experts
converge to similar representations over time. Proposes
orthogonality and variance losses to maintain diversity.

**Relevance:** Expert representation convergence during
extended training could directly explain un-grokking: model
loses specialized circuits needed for generalization.

**Cited in:** HYP-015 pre-registration

## LIT-082: Prieto et al. 2025 (Grokking & Softmax Collapse)

**Title:** Grokking at the Edge of Numerical Stability
**Authors:** Prieto, Barsbey, Mediano, Birdal
**Venue:** ICLR 2025, arXiv:2501.04697 — **Grade A**

**Key finding:** Without regularization, grokking pushes
models toward Softmax Collapse (SC). Gradients align with
"naive loss minimization" (scaling logits without changing
predictions), delaying generalization. Proposes StableMax.

**Relevance:** MoE routers also use Softmax — creating dual
instability: attention SC + router SC. Non-MoE models only
face the first. Router logit magnitude monitoring could
detect this in HYP-015.

**Note:** Related to LIT-076 but MoE-router connection new.

**Cited in:** HYP-015 pre-registration

## LIT-087: Li et al. 2025 (MoE Continual Learning Theory)

**Title:** Theory on Mixture-of-Experts in Continual Learning
**Authors:** Li, Lin, Duan, Liang, Shroff
**Venue:** ICLR 2025 — **Grade B**

**Key finding:** Gating network must be frozen after
sufficient training for convergence. Continued router updates
after expert specialization can UNDO learned representations.
More experts require more rounds before convergence.

**Relevance:** Theoretical prediction: continued router
training destroys grokking circuit. Freezing router after
grokking onset should prevent un-grokking — testable in
HYP-015 follow-up.

**Cited in:** HYP-015 pre-registration

## LIT-089: Xu et al. 2025 (Grouter — Decoupled Routing)

**Title:** Grouter: Decoupling Routing from Representation
for Accelerated MoE Training
**Authors:** Xu, Hu, Liu, Sun, Yuan
**Venue:** arXiv:2603.06626 — **Grade B**

**Key finding:** Tight coupling between routing and
representation learning creates harmful interference. Router
updates shift token assignments, forcing experts to "chase
a moving target." Proposes preemptive (frozen) routing.

**Relevance:** "Chasing a moving target" during grokking's
dramatic representation shift could cause un-grokking.
Aligns with LIT-087 theoretical prediction.

**Cited in:** HYP-015 pre-registration

---

## LIT-090: Tikeng Notsawo et al. 2024 (Predicting Grokking)

**Title:** Predicting Grokking Long Before it Happens: A look
into the loss landscape of models which grok
**Authors:** Tikeng Notsawo Pascal Junior, Zhou, Pezeshki,
Rish, Dumas
**Venue:** ICLR 2024 ME-FoMo Workshop — **Grade A**

**Key finding:** Fourier analysis of early learning curve
oscillations can predict whether a model will eventually grok.
Specific frequency-domain signatures in early epochs serve as
predictors without training to completion.

**Relevance:** Directly about early grokking prediction. Uses
standard metrics (loss curves), not TTC. Our HYP-016 finding
that aggregate metrics (loss, accuracy, p@64) have zero
within-architecture predictive power (all rho < 0.12) is
complementary — their method may also fail within a single
architecture with different seeds.

**Cited in:** HYP-016 interpretation

---

## LIT-091: Clauw et al. 2024 (Grokking Phase Transition)

**Title:** Information-Theoretic Progress Measures reveal
Grokking is an Emergent Phase Transition
**Authors:** Clauw, Stramaglia, Marinazzo
**Venue:** ICML 2024 — **Grade A**

**Key finding:** Grokking is an emergent phase transition
detected by O-Information decomposed into synergy and
redundancy. Early synergy peaks predict eventual grokking.
Tested on 5 seeds of a 2-layer FC network on mod 97 (same
task as our experiments).

**Relevance:** Most similar prior work to HYP-016. They test
5 seeds but use information-theoretic measures (synergy),
not TTC/pass@k. Their synergy metric may capture weight-space
structure invisible to aggregate loss. However, only 5 seeds
on a simple FC network — unclear if it transfers to
transformers/hybrids.

**Cited in:** HYP-016 interpretation

---

## LIT-092: Parameter Golf PR #64 (SOTA ~1.015 BPB)

**Title:** Top leaderboard submission
**Authors:** Competition participant (github.com/openai/parameter-golf)
**Venue:** GitHub PR — **Grade D** (competition submission, not peer-reviewed)

**Key techniques:** Sliding window eval (4000 context), int6 quantization
(int6 MLP+attn, fp16 embedding), longer sequences (4096 tokens), optimizer
tuning. Current best BPB on leaderboard.

**Relevance:** Establishes the target to beat. Our architecture changes
are orthogonal to most of these techniques.

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-093: Parameter Golf PR #78 (Depth Recurrence + TTT)

**Title:** Depth recurrence and test-time training submission
**Authors:** Competition participant
**Venue:** GitHub PR — **Grade D**

**Key techniques:** Weight sharing (matches our UNIQUE_BLOCKS finding),
test-time training with LoRA adapters on val documents, larger vocabulary
using public tokenizers (huggingface.co/sproos/parameter-golf-tokenizers
for sp1024/2048/4096/8192).

**Relevance:** Independently confirms depth recurrence for parameter golf.
TTT is a novel eval-time technique worth exploring. Public tokenizers
unblock our vocab exploration (R-PG-003).

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-094: Parameter Golf PR #54 (Longer Sequences + Wider MLP)

**Title:** Sequence length and MLP width exploration
**Authors:** Competition participant
**Venue:** GitHub PR — **Grade D**

**Key techniques:** Sequence length 1024→4096, wider MLP hidden dim,
sliding window eval, sp8192 vocabulary.

**Relevance:** Shows that longer sequences help at this scale. Our
fixed 1024 sequence length may be leaving BPB on the table.

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-095: Sliding Window Evaluation (multiple PRs)

**Title:** Context-windowed BPB evaluation technique
**Authors:** Multiple competition participants
**Venue:** GitHub PRs — **Grade D**

**Key finding:** Evaluating each token with maximum available context
(up to 4000 tokens) instead of averaging across the full sequence gives
~0.03 BPB improvement at zero artifact cost. Most competitive submissions
use this technique.

**Relevance:** Free BPB improvement we should implement for GPU runs.
Does not affect training, only evaluation.

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-096: Int6 Mixed-Precision Quantization (PR #64)

**Title:** 6-bit weight quantization for parameter golf
**Authors:** Competition participant
**Venue:** GitHub PR — **Grade D**

**Key finding:** Int6 on MLP and attention weights, fp16 on embedding
table. Saves ~4MB artifact space compared to uniform int8, enabling
larger models within the 16MB constraint.

**Relevance:** Our 10.6MB headroom from weight sharing means we may
not need aggressive quantization, but int6 could free additional space
for vocabulary expansion.

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-097: Public Parameter Golf Tokenizers

**Title:** Pre-trained SentencePiece tokenizers for parameter golf
**Authors:** sproos (HuggingFace user)
**Venue:** HuggingFace Hub — **Grade D**
**URL:** huggingface.co/sproos/parameter-golf-tokenizers

**Key content:** Pre-trained tokenizers for sp1024, sp2048, sp4096, sp8192
vocabularies, compatible with the parameter golf challenge framework.

**Relevance:** Unblocks R-PG-003 (vocabulary exploration). No need to
retrain tokenizers from scratch — can download and use directly.

**Cited in:** Competition landscape review (2026-03-19)

---

## LIT-098: Per-loop LoRA Adapters (PRs #38, #51)

**Source:** openai/parameter-golf PRs #38, #51 — **Grade D**

Rank-4 LoRA on Q/V projections for loop specialization in depth
recurrence. Each recurrence pass gets its own LoRA adapter, allowing
shared base weights to specialize per-loop. Minimal param overhead
(rank 4 × 2 projections × N loops). Direct upgrade for UNIQUE_BLOCKS.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-099: Iteration Embeddings (PR #54)

**Source:** openai/parameter-golf PR #54 — **Grade D**

Learned per-pass vectors added to the residual stream to differentiate
recurrence loops. Simpler alternative to per-loop LoRA — adds only
dim×N_loops parameters.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-100: LAWA Weight Averaging (PRs #38, #51)

**Source:** openai/parameter-golf PRs #38, #51 — **Grade D**

Stochastic weight averaging during the warmdown phase. Free quality
boost with low implementation effort. Averages model weights over
the last K checkpoints during LR decay.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-101: NorMuon Optimizer (PR #78)

**Source:** openai/parameter-golf PR #78 — **Grade D**

Modified Muon optimizer with normalization. Replacement for standard
Muon. Details sparse in PR.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-102: Muon Momentum 0.99 Consensus (PRs #52, #61, #66, #70)

**Source:** openai/parameter-golf PRs #52, #61, #66, #70 — **Grade D**

Consistent finding across 4+ independent submissions: MUON_MOMENTUM=0.99
outperforms the default 0.95. This is the most replicated numeric finding
in the competition.

**Cited in:** Competition peer review (2026-03-19), HYP-025 design

---

## LIT-103: QAT with STE (PR #65)

**Source:** openai/parameter-golf PR #65 — **Grade D**

Fake int6 quantization during training using Straight-Through Estimator.
Reduces quantization gap from +0.048 to +0.0015 BPB. Cost: +54% per-step
overhead during training.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-104: Ternary QAT (PR #69)

**Source:** openai/parameter-golf PR #69 — **Grade D**

Extreme quantization to {-1, 0, +1} at ~1.5 bits/weight. Enables
4-5x more parameters per byte within the 16MB artifact limit.
No GPU results reported yet.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-105: Document-Isolated Evaluation (PR #77)

**Source:** openai/parameter-golf PR #77 — **Grade D**

Separate validation documents by BOS boundaries, avoid cross-document
context contamination in eval. Ensures BPB reflects real compression
quality, not context leakage.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-106: NTK-aware RoPE for Eval (PR #60)

**Source:** openai/parameter-golf PR #60 — **Grade D**

Train at sequence length 1024, evaluate at 2048+ using NTK-aware RoPE
scaling. Extends effective context at eval time without additional
training cost.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-107: Overtone Init (PR #60)

**Source:** openai/parameter-golf PR #60 — **Grade D**

SVD spectral shaping of embedding initialization. Theory: better
initial embedding geometry accelerates early training. Unclear
magnitude of benefit.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-108: AI-Agent Technique Composition (PR #66)

**Source:** openai/parameter-golf PR #66 — **Grade D**

Automated bucketing and stacking of known techniques using an
AI agent to compose changes. Meta-approach: let the agent pick which
techniques to combine and in what order.

**Cited in:** Competition peer review (2026-03-19)

---

## LIT-109: PolyCom — Polynomial Composition Activations (ICLR 2025)

**Title:** Polynomial Composition Activations: Unleashing the Dynamics
of Large Language Models
**Authors:** Zhuo et al.
**Venue:** ICLR 2025 — **Grade B**
**arXiv:** https://arxiv.org/abs/2411.03884
**Code:** https://github.com/BryceZhuo/PolyCom

**Key finding:** PolyReLU = sum(a_i * ReLU^i(x), i=0..r) with r=3.
PolyNorm = sum(a_i * x^i / ||x^i||_2). Initialized a_i = 1/3 for
i=1,2,3. At 1B scale: PolyNorm beats SwiGLU by 1.21% average on 6
downstream tasks. Training loss 2.17 vs 2.19. With gradient
checkpointing, overhead is negligible vs SwiGLU.

**Relevance:** Drop-in activation replacement for relu^2 in FFN.
Iso-step since polynomial ops are trivial vs matmul cost.
LOCAL-TESTABLE: yes (same model size, ~0 throughput change).

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-110: DyT — Dynamic Tanh Replacing Normalization (CVPR 2025)

**Title:** Transformers without Normalization
**Authors:** Zhu, Chen, He, LeCun, Liu
**Venue:** arXiv 2503.10622, CVPR 2025 — **Grade B**
**URL:** https://arxiv.org/abs/2503.10622

**Key finding:** DyT(x) = tanh(alpha * x) as drop-in replacement for
LayerNorm/RMSNorm. Inspired by S-shaped input-output mappings observed
in trained LayerNorm. Matches or exceeds normalized counterparts.
Single learnable scalar alpha per layer. Derf variant (erf-based)
outperforms in later work (arXiv 2512.10938).

**Relevance:** Replaces RMSNorm with simpler/faster function. Could
free ~0.5-2% throughput by removing mean/variance computation.
LOCAL-TESTABLE: yes (same arch, same params, slightly faster).

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-111: Sigmoid Attention (ICLR 2025)

**Title:** Theory, Analysis, and Best Practices for Sigmoid
Self-Attention
**Authors:** Ramapuram et al.
**Venue:** ICLR 2025 — **Grade B**
**arXiv:** https://arxiv.org/abs/2409.04431

**Key finding:** Sigmoid attention matches softmax across scales.
FlashSigmoid gives 17% kernel speedup on H100. Removes row-wise
normalization (no token competition). QKNorm stabilizes sigmoid.
Lower sample complexity from MoE perspective.

**Relevance:** Eliminates softmax overhead. May be iso-step or faster.
LOCAL-TESTABLE: yes (same arch, same params).
**Caveat:** MLX may not have FlashSigmoid; manual sigmoid may be
slower than MLX's optimized softmax.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-112: Learnable Activation Functions (RAF/RAFT)

**Title:** Transformers with Learnable Activation Functions
**Authors:** Gholami et al.
**Venue:** EACL 2023 Findings — **Grade B**
**arXiv:** https://arxiv.org/abs/2208.14111

**Key finding:** Rational Activation Functions (RAF) are p(x)/q(x)
polynomial ratios that learn optimal activation from data.
RAFT +5.71 points on GLUE with 100 examples, +2.05 on SQuAD.
Small number of additional params (polynomial coefficients).

**Relevance:** Could replace relu^2 with learned rational function.
Adds ~20-50 params per layer. Iso-step (polynomial eval is cheap).
LOCAL-TESTABLE: yes.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-113: SmearGate (Competition Technique)

**Source:** openai/parameter-golf PRs #162, #194, #206 — **Grade D**

**Key finding:** Learned gate blending each token embedding with
previous token's embedding. Per-dim variant: sigmoid(Parameter(dim))
gate per embedding dimension, zero-initialized. ~512 params total.
Present in all top-10 submissions. Per-dim SmearGate in PR #194
(1.1480 BPB) and PR #206 (1.1507 BPB).

**Relevance:** Trivial overhead (~512 params), provides bigram context
to embedding layer before attention. LOCAL-TESTABLE: yes (zero
throughput cost). Already known from competition review.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-114: BigramHash Embedding (Competition Technique)

**Source:** openai/parameter-golf PRs #162, #186, #208 — **Grade D**

**Key finding:** 4096-bucket hash table (dim=128, projected to 512)
for token-pair context. ~524K additional params. Hash of (prev_token,
curr_token) indexes into embedding table, added to token embedding.
Combined with SmearGate in all SOTA submissions. PR #198 uses 2048
buckets to save 300KB with negligible BPB cost.

**Relevance:** Adds bigram awareness. Moderate param cost but fits in
artifact budget. LOCAL-TESTABLE: yes (small overhead per step).

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-115: OrthoInit + muP Scaling (Competition Technique)

**Source:** openai/parameter-golf PRs #162, #164, #206 — **Grade D**

**Key finding:** Orthogonal weight initialization on all non-zero-init
linear layers + muP-style output scaling (1/sqrt(2*num_layers)).
Used by all top-5 submissions. Theoretical justification: preserves
gradient norms across depth, enables stable training without warmup.

**Relevance:** Zero throughput cost, zero param cost. Pure init change.
LOCAL-TESTABLE: yes (same arch, same speed).

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-116: Overtone Spectral Init (Competition Technique)

**Source:** openai/parameter-golf PR #155 — **Grade D**

**Key finding:** SVD-based spectral shaping of embedding initialization
with power=0.5. "Overtone spectral embedding initialization."
Combined with phase-transition residual mixing. Used in PR #155
(1.1876 BPB) and carried forward in PR #175 (TTT + OvertoneInit).

**Relevance:** Zero throughput cost. Potentially better than OrthoInit
for embeddings specifically. LOCAL-TESTABLE: yes.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-117: TTT-LoRA Eval-Time Adaptation

**Source:** openai/parameter-golf PRs #152, #175, #183 — **Grade D**
**Related:** Sun et al. 2024, arXiv 2407.04620

**Key finding:** Per-document LoRA adaptation (rank=8 on Q/V + LM head)
with Adam lr=0.01 during evaluation. Document-isolated chunked eval.
PR #183: -0.003 BPB improvement on weak baseline. PR #175: expects
~0.037 BPB improvement stacked on SOTA. Full-model SGD (PR #152)
gives 3.0% BPB improvement. LoRA is cheaper than full SGD.

**Relevance:** Pure eval-time technique, zero training change.
LOCAL-TESTABLE: yes (eval-only, no throughput impact on training).
**Caveat:** Increases eval time significantly. Must fit in 10-min
total (train + eval) for competition.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-118: Curriculum Learning for LLM Pretraining

**Title:** Curriculum Learning for LLM Pretraining: An Analysis of
Learning Dynamics
**Source:** arXiv 2601.21698 — **Grade C**

**Key finding:** Curriculum learning consistently accelerates
convergence by 18-45% in early/mid training. Best difficulty metrics:
compression ratio, lexical diversity (MTLD), Flesch readability.
As warmup strategy: sustained 3.5% improvement. Combining with weight
averaging is "particularly effective."

**Relevance:** Data ordering change. Zero model change, zero per-step
cost if data is pre-sorted. LOCAL-TESTABLE: partially (need to
pre-sort FineWeb shards by difficulty; data loader must support
ordered iteration). Implementation: moderate.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-119: Learnable Attention Temperature (Per-Head)

**Source:** Ryan 2024, blog post — **Grade D**
**URL:** https://nickcdryan.com/2024/08/02/introducing-a-learnable-temperature-value-into-the-self-attention-scores/

**Key finding:** Per-head learned scalar temperature on QK attention
scores. Different heads benefit from different sharpness. ~8 params
for 8 heads. Slight improvement reported.

**Relevance:** Trivial param cost, zero throughput impact.
LOCAL-TESTABLE: yes (add 1 scalar per head).
**Note:** Competition already uses logit softcapping (30.0) which
partially serves the same purpose. May interact.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-120: Label Smoothing in LMs

**Title:** Towards Understanding Why Label Smoothing Degrades
Selectivity
**Source:** ICLR 2025 — **Grade B**

**Key finding:** Label smoothing (alpha=0.1) improves BLEU despite
worse perplexity. Degrades model's selectivity (ability to reject
misclassifications). For BPB evaluation (perplexity-based), label
smoothing likely HURTS.

**Relevance:** NEGATIVE for our use case. BPB measures cross-entropy,
and label smoothing distorts the loss landscape in ways that increase
cross-entropy even when downstream task metrics improve.
LOCAL-TESTABLE: yes but EXPECTED NEGATIVE.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-121: QK-Norm + Softcap Combination

**Source:** arXiv 2410.16682 (Methods of Improving LLM Training
Stability) — **Grade C**

**Key finding:** QK layer normalization + softmax capping enables 1.5x
higher learning rate without divergence. OLMoE: QKNorm increases
stability at cost of 10% throughput. Combined QK_norm_cap addresses
both input-magnitude and output-range instabilities.

**Relevance:** Our baseline already has logit softcapping (30.0).
Adding QK-norm would cost ~10% throughput (BAD for local testing due
to B-022). Only test if we suspect training instability.
LOCAL-TESTABLE: marginally (10% throughput hit = ~200 fewer steps).

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-122: Factored Embedding (ALBERT-style)

**Title:** ALBERT: A Lite BERT for Self-supervised Learning
**Authors:** Lan et al. (Google)
**Venue:** ICLR 2020 — **Grade A**

**Key finding:** Decompose V x H embedding into V x E and E x H.
10x+ param reduction with little performance loss. Critical for
large-vocab small-model settings.

**Relevance:** If we increase vocab (sp2048/4096), factored embedding
saves massive params. LOCAL-TESTABLE: yes (adds one small matmul
per token, negligible overhead). But we already use tied embeddings
at V=1024 which is relatively small.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-123: Peri-LN (Peri-Layer Normalization)

**Title:** Peri-LN: Revisiting Layer Normalization in the Transformer
Architecture
**Source:** arXiv 2502.02732 — **Grade C**

**Key finding:** Apply LayerNorm both before AND after each sub-layer.
Constrains residual spikes from Pre-LN while maintaining stronger
gradient pathway than Post-LN. Normalizes input and final output
embeddings.

**Relevance:** Double-norm per sublayer. ~2x norm cost. Likely a
wash locally due to throughput cost, but worth noting if we observe
training instability. LOCAL-TESTABLE: marginally.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-124: Scaling Embeddings Outperforms Scaling Experts

**Title:** Scaling Embeddings Outperforms Scaling Experts in Language
Models
**Source:** arXiv 2601.21204 — **Grade C**

**Key finding:** SCONE: n-gram embeddings (precomputed, off-accelerator)
boost LM quality more than adding MoE experts. 1B SCONE model
outperforms 1.9B baseline. N-gram embeddings are essentially free
at inference (precomputed lookup).

**Relevance:** BigramHash in competition is a simplified version of
this idea. Validates the approach. LOCAL-TESTABLE: BigramHash already
tested; expanding to trigram hash would be iso-step if hash table
fits in memory.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-125: U-muP (Unit-Scaled Maximal Update Parametrization)

**Title:** U-muP: The Unit-Scaled Maximal Update Parametrization
**Source:** ICLR 2025 — **Grade B**

**Key finding:** Combines muP with Unit Scaling for practical HP
transfer. Addresses gap between muP theory and practice: efficient
HP search, transfer, interpretability, low-precision training.

**Relevance:** Proper muP would make our local LR experiments more
transferable to GPU. But implementing muP is non-trivial.
LOCAL-TESTABLE: yes (zero throughput cost, just different scaling
factors), but complex implementation.

**Cited in:** Local-testable lit review (2026-03-20)

---

## LIT-126: Mixed Int5/Int6 Quantization (Competition Technique)

**Source:** openai/parameter-golf PR #180, #219 — **Grade D**

**Key finding:** Int5 [-16,15] for MLP weights, int6 [-32,31] for
attention. Int5 has 3 zero high bits per byte; zstd-22 compresses
at 1.88x vs int6's 1.51x. Saves 1.86MB, funding an extra
transformer layer. PR #180: 1.1453 BPB with 10 layers.
PR #219: 12L mixed int5-MLP + int6-Attn, 1.1541 BPB.

**Relevance:** Compression technique for artifact budget. Not directly
local-testable (quantization interacts with batch size), but the
artifact savings enable architectural changes that ARE testable.

**Cited in:** Local-testable lit review (2026-03-20)

## LIT-127: MoonshotAI 2026 (Attention Residuals)

**Title:** Attention Residuals
**Authors:** Kimi Team, MoonshotAI
**Venue:** arXiv preprint 2603.15031, March 2026 — **Grade C**
**URL:** https://github.com/MoonshotAI/Attention-Residuals
**arXiv:** https://arxiv.org/abs/2603.15031

**Key finding:** Replaces fixed residual connections with learned,
input-dependent attention over depth. Each layer computes softmax
attention over all preceding layer outputs using a learned pseudo-query
w_l in R^d. Block AttnRes variant partitions layers into ~8 blocks,
applies attention only across block boundaries. Reports **1.25x compute
efficiency** (same BPB at 80% compute). Validated on Kimi Linear (48B/3B
activated, 1.4T tokens): +7.5 pts GPQA-Diamond, +3.1 pts HumanEval.

**Relationship to our work:**
- Very similar to our DenseFormer DWA (NeurIPS 2024): both use
  softmax-weighted cross-layer averaging
- Key difference: AttnRes is **input-dependent** (dynamic), DWA is static
- Complementary to Value Residual (which operates on V projections, not
  hidden states)
- Our DWA + Value Residual: super-additive +0.074 BPB locally

**Relevance to pgolf:** High. Could replace or augment DWA for
input-dependent cross-layer routing. Block AttnRes overhead is minimal
(~d params per block boundary). Best tested on GPU first — overhead may
hurt local Mac iteration (B-022 confound). Novel in competition context
(March 2026, no submissions use it yet).

**Priority:** Medium-high for GPU experiments. Test variants:
(A) replace DWA with Block AttnRes, (B) DWA + AttnRes together,
(C) AttnRes + Value Residual without DWA.

**Cited in:** Research backlog (2026-03-20), HYP-033 (2026-03-20)

---

## LIT-128: Ziming Liu — When Does Attention Residuals Work?

**Title:** When does Kimi's "Attention Residuals" work?
**Source:** Blog post, kindxiaoming.github.io/blog/2026/attention-residual/ — **Grade D**
**Author:** Ziming Liu (MIT)

**Key finding:** Identifies a **stability-expressivity tradeoff** for AttnRes.
AttnRes excels on structured/linear data but struggles on complex memorization
tasks. When attention weights fail to learn, they default to uniform distribution
("uniform bias"), causing representation collapse via over-averaging.

**Empirical evidence:** As datasets interpolate from random (alpha=0) to
structured (alpha=1), AttnRes progressively outperforms standard residuals.
Neither method universally dominates.

**Relevance to pgolf:** Language modeling is structured data (alpha>>0), so
AttnRes should perform well. Our iso-step results (+0.111 BPB) confirm this.
The uniform bias risk is low with zero-init (which we use) and natural language
data. The main risk would be if the model needs to memorize rare tokens.

**Cited in:** HYP-033 literature check (2026-03-20)

---

## LIT-129: DeepCrossAttention (DCA)

**Title:** DeepCrossAttention: Supercharging Transformer Residual Connections
**Source:** arXiv 2502.06785 — **Grade C**

**Key finding:** Another cross-layer aggregation method. Uses cross-attention
between layers to achieve lower perplexity for given parameter/training budgets.
A competitor to AttnRes in the cross-layer connection space.

**Relevance:** Alternative to AttnRes. Not yet tested in our context.
Lower priority than AttnRes given our strong iso-step results.

**Cited in:** HYP-033 literature check (2026-03-20)

---

## LIT-130: Transformer Layers as Painters

**Title:** Transformer Layers as Painters
**Source:** arXiv 2407.09298, AAAI 2025 — **Grade B**

**Key finding:** Middle transformer layers share a "semantic palette" (high
cosine similarity) but are NOT interchangeable. Repeating a middle layer
pushes input out of the shared representation space and is worse than
skipping it. Subtle per-layer specialization matters.

**Relevance:** Explains why weight sharing + AttnRes fails (HYP-034/035).
AttnRes exploits subtle per-layer differences; weight sharing eliminates them.

**Cited in:** HYP-034/035 literature check (2026-03-20)

---

## LIT-131: Relaxed Recursive Transformers

**Title:** Relaxed Recursive Transformers
**Source:** arXiv 2410.20672 — **Grade B**

**Key finding:** Strict weight sharing forces each block to serve multiple
depth-specific roles. Solution: add per-depth LoRA modules to restore
differentiation. Without per-layer residuals, shared blocks produce
homogenized outputs.

**Relevance:** If combining AttnRes with weight sharing, would need
per-depth LoRA-like differentiation to restore the diversity AttnRes needs.

**Cited in:** HYP-034/035 literature check (2026-03-20)

---

## LIT-132: MoEUT — Mixture-of-Experts Universal Transformers

**Title:** MoEUT: Mixture-of-Experts Universal Transformers
**Source:** arXiv 2405.16039 — **Grade B**

**Key finding:** Different MoE experts activate at different depths even
with shared weights, restoring functional diversity. Demonstrates that
weight-shared models need an explicit diversity mechanism.

**Relevance:** MoE routing could potentially rescue AttnRes + weight sharing
by ensuring different expert activations at each depth.

**Cited in:** HYP-034/035 literature check (2026-03-20)

---

## LIT-133: MUDDFormer — Multiway Dynamic Dense Connections

**Title:** MUDDFormer: Multiway Dynamic Dense Connections
**Source:** arXiv 2502.12170, ICML 2025 — **Grade B**

**Key finding:** Static/shared connection weights insufficient for cross-layer
aggregation. Dynamic, input-dependent weights needed for Q/K/V/residual
streams independently. Shows stream-specific patterns differ. Validates that
cross-layer aggregation benefits from layer diversity.

**Relevance:** More evidence that AttnRes-like mechanisms need diverse layers.
MUDDFormer approach (separate dynamic weights per stream) could be stronger
than AttnRes's single-query mechanism.

**Cited in:** HYP-034/035 literature check (2026-03-20)

## LIT-134: Diversity of Transformer Layers

**Title:** On the Diversity of Transformer Layers
**Source:** arXiv 2505.24009 — **Grade C**

**Key finding:** Performance improves when individual layers make predictions
close to the correct answer AND remain mutually diverse. Performance
improvement achieved by reusing (sharing) layers is limited — shared layers
lack the diversity needed for optimal cross-layer interaction. Proposes
diversity-aware training objectives.

**Relevance:** Directly validates our B-028 finding that AttnRes requires
unique (non-shared) layers. Weight sharing reduces layer diversity, which is
exactly the mechanism AttnRes relies on (cross-layer aggregation via learned
queries). This explains why AttnRes fails at 9L/3u (-0.028 BPB) but succeeds
at 9L/9u (+0.111 BPB) — the 3 unique blocks lack sufficient diversity for
the attention-based aggregation to find complementary information.

**Cited in:** HYP-036 literature check (2026-03-20)

## LIT-135: Depth and Looping for In-Context Learning

**Title:** On the Role of Depth and Looping for In-Context Learning with
Task Diversity
**Source:** arXiv 2410.21698 — **Grade C**

**Key finding:** Looped Transformers (recurrent weight sharing) trade
expressivity for robustness/generalization. Standard (unique-layer)
Transformers achieve better expressivity through layer diversity. Monotonic
loss improvement across depth requires weight sharing, but monotonicity is
not desirable — non-monotonic depth-dependent computation enables richer
representations.

**Relevance:** Explains why 9L/9u beats 9L/3u for AttnRes: unique layers
enable the depth-dependent computation that cross-layer aggregation exploits.
Cycling destroys layer-specific specialization.

**Cited in:** HYP-036 literature check (2026-03-20)

## LIT-136: Cross-layer Attention Sharing

**Title:** Cross-layer Attention Sharing for Pre-trained Large Language Models
**Source:** arXiv 2408.01890 — **Grade C**

**Key finding:** Direct weight sharing in attention layers is ineffective.
Shallow layers are vulnerable to small deviations in attention weights.
Cross-layer mechanisms depend on layer-specific attention patterns that
weight sharing cannot maintain.

**Relevance:** Converging evidence that cross-layer aggregation (like AttnRes)
is brittle when layer-specific patterns are destroyed by weight sharing.

**Cited in:** HYP-036 literature check (2026-03-20)

## LIT-137: Early-Warning Signals of Grokking via Loss-Landscape Geometry

**Title:** Early-Warning Signals of Grokking via Loss-Landscape Geometry
**Source:** arXiv 2602.16967 — **Grade C**

**Key finding:** The commutator defect — a curvature measure from non-commuting
gradient updates (||theta_AB - theta_BA|| / (||eta*g_A|| * ||eta*g_B||)) — rises
well before generalization onset. Lead times follow superlinear power law
(alpha ~1.27 for modular arithmetic). Amplifying non-commutativity accelerates
grokking (32% SCAN, 50% Dyck). Requires 4 fwd-bwd passes per measurement.
Tested on mod arithmetic (290K params), SCAN, and Dyck-1.

**Relevance:** Directly addresses our ANOM-016 (seed-dependent grokking) and
extends HYP-016's null finding. We test the novel question: does the defect
predict grokking ACROSS seeds (not just within runs)?

**Cited in:** HYP-037 (2026-03-20)

## LIT-138: Grokking in LLM Pretraining (ICLR 2026)

**Title:** Grokking in LLM Pretraining? Monitor Memorization-to-Generalization
without Test
**Authors:** Ziyue Li, Chenrui Fan, Tianyi Zhou
**Source:** ICLR 2026 — **Grade A**
**arXiv:** 2506.21551

**Key finding:** First study of grokking during real LLM pretraining (MoE models,
single-epoch, diverse corpora). Expert pathway patterns evolve from random to
structured even while training loss has converged. Two zero-cost metrics
(pathway similarity, expert consistency) can monitor downstream generalization.

**Relevance:** Grade A evidence that grokking is not just an algorithmic-task
curiosity. Pathway metrics could complement commutator defect for MoE models.

**Cited in:** HYP-037 literature review (2026-03-20)

## LIT-139: MASA — Matrix-based Attention Weight Sharing (AAAI 2026)

**Title:** Share Your Attention: Transformer Weight Sharing via Matrix-based
Dictionary Learning
**Source:** AAAI 2026 — **Grade A**
**arXiv:** 2508.04581

**Key finding:** MASA decomposes Q/K/V/O projection matrices into shared
dictionary atoms (matrix-level weight sharing via dictionary learning).
66.7% attention parameter reduction while outperforming GQA, low-rank, and
prior sharing methods. 100M-700M scale, drop-in replacement.

**Relevance:** Complementary to our UNIQUE_BLOCKS=3 cyclic sharing. Could
stack for even smaller pgolf artifacts. Reserved for future GPU experiments.

**Cited in:** Autorun lit review (2026-03-20)

## LIT-140: SLT Competing Basins — Grokking as Phase Transition (2026)

**Title:** Grokking as a Phase Transition between Competing Basins
**Source:** arXiv preprint — **Grade C**
**arXiv:** 2603.01192

**Key finding:** Uses Singular Learning Theory to interpret grokking as a
thermodynamic phase transition between competing near-zero-loss basins with
different local learning coefficients (LLC). Lower-LLC basins have higher
posterior mass and better generalization. Derives closed-form LLC for quadratic
networks on modular arithmetic.

**Relevance:** Strongest theoretical support for stochastic grokking onset.
Barrier-crossing dynamics (Kramers escape) predict exponentially variable
transition times — exactly what we observe (rho ~0.1 across seeds). The barrier
height is architecture-determined but crossing time is noise-determined.

**Cited in:** HYP-037 post-experiment literature check (2026-03-20)

## LIT-141: 500-Model SLT Study of Grokking (2025)

**Title:** Using Physics-Inspired SLT to Understand Grokking
**Source:** arXiv preprint — **Grade C**
**arXiv:** 2512.00686

**Key finding:** Trained 500 models on mod arithmetic; only 168 (33.6%) grokked.
Tests Arrhenius-style rate hypothesis for grokking. LLC values measured pre- and
post-grok.

**Relevance:** Strongest indirect support for stochastic onset. If only 33.6%
of models grok under fixed hyperparameters, grokking occurrence itself (not just
timing) is stochastic. The Arrhenius analogy explicitly models grokking as a
stochastic barrier-crossing event.

**Cited in:** HYP-037 post-experiment literature check (2026-03-20)

## LIT-142: Low-Dimensional Grokking Dynamics — Metastable Escape (2026)

**Title:** Low-Dimensional and Transversely Curved Optimization Dynamics in Grokking
**Source:** arXiv preprint — **Grade C**
**arXiv:** 2602.16746

**Key finding:** Training evolves on low-dimensional execution subspace (PC1
captures 68-83% of trajectory variance). Transverse curvature accumulates until
trajectory "escapes" into generalizing solution (600-1600 step lead time).
Explicitly describes grokking as "escape from a metastable regime."

**Relevance:** Consistent with stochastic onset — metastable escape is
inherently stochastic. Only tested 3 seeds (qualitative confirmation). Our
10-seed rho=0.111 fills the quantitative gap they left open.

**Cited in:** HYP-037 post-experiment literature check (2026-03-20)

## LIT-143: The Complexity Dynamics of Grokking (Physica D 2025)

**Title:** The Complexity Dynamics of Grokking
**Source:** Physica D: Nonlinear Phenomena — **Grade B** (peer-reviewed)
**arXiv:** 2412.09810

**Key finding:** Grokking exhibits a complexity phase transition measurable via
rate-distortion / Kolmogorov complexity. Complexity rises during memorization,
falls during generalization. Results show mean over 3-6 seeds with SE shading.

**Relevance:** The SE bars in their figures imply onset timing variance, but they
don't analyze it. Their rate-distortion complexity measure is another candidate
for cross-seed prediction testing.

**Cited in:** HYP-037 post-experiment literature check (2026-03-20)

## LIT-144: Predicting Grokking Long Before it Happens (ICLR 2024)

**Title:** Predicting Grokking Long Before it Happens: A Look into the Loss Landscape
**Source:** ICLR 2024 — **Grade B** (peer-reviewed)

**Key finding:** Oscillations in the learning curve during the first few
epochs forecast grokking in extended training. Fourier transform spectral
signatures detect these oscillations. Slingshots and grokking come in tandem.

**Relevance:** Directly validates HYP-038 finding that all 5 seeds show
oscillatory P(correct) with 5-8 direction changes. The oscillations are
fundamental signatures of grokking, not noise.

**Cited in:** HYP-038 literature check (2026-03-20)

## LIT-145: Grokking at the Edge of Numerical Stability (2025)

**Title:** Grokking at the Edge of Numerical Stability
**Source:** arXiv 2501.04697 — **Grade C** (preprint)

**Key finding:** Without regularization, grokking pushes models to numerical
stability edge via "Softmax Collapse." After overfitting, gradients align with
NLM direction that scales logits without changing predictions, delaying
generalization until numerical saturation.

**Relevance:** The P(correct) plateau at 0.50-0.73 with oscillations could
reflect NLM phase where logits scale without prediction improvement. May
explain why oscillation amplitude persists rather than damping.

**Cited in:** HYP-038 literature check (2026-03-20)

## LIT-146: Grokking and Generalization Collapse via HTSR (2025)

**Title:** Grokking and Generalization Collapse: Insights from HTSR Theory
**Source:** arXiv 2506.04434 — **Grade C** (preprint)

**Key finding:** Three-phase model including "anti-grokking" where test
accuracy collapses after extended training. Weight entropy sharply decreases
with generalization. Spectral exponent alpha predicts both grokking and collapse.

**Relevance:** Post-grok oscillations in HYP-038 (seeds 42, 43 continuing to
oscillate past grok step) may represent unstable equilibrium before potential
anti-grokking collapse.

**Cited in:** HYP-038 literature check (2026-03-20)

## LIT-147: Geometric Inductive Bias of Grokking (2026)

**Title:** The Geometric Inductive Bias of Grokking: Bypassing Phase
Transitions via Architectural Topology
**Source:** arXiv 2603.05228 — **Grade C** (preprint)

**Key finding:** Spherical topology (L2 norm + fixed temperature) reduces
grokking onset by 20x by removing magnitude-based degrees of freedom. The
prolonged plateau corresponds to implicit margin-maximization.

**Relevance:** Suggests magnitude control could stabilize the oscillatory
regime observed in HYP-038. If oscillations are magnitude-driven (consistent
with LIT-145 NLM theory), spherical constraints would damp them.

**Cited in:** HYP-038 literature check (2026-03-20)


---

## LIT-148: Exclusive Self Attention (XSA)

**Source:** Zhai, "Exclusive Self Attention," arXiv 2603.09078, March 2026
**Grade:** B (peer-reviewed quality preprint, Apple)
**Relevance:** Direct — used in all top 3 pgolf submissions

Removes self-value component from attention output via orthogonal projection.
Forces attention to capture only contextual (cross-token) features. Point-wise
features are already available via residual connection. Zero new params, ~2%
overhead. Consistent gains across 0.7B-2.7B. Larger gains with longer sequences.

**Applied in:** HYP-039 (XSA=1 env var in train_gpt_mlx.py)

---

## LIT-149: Cross-Layer Attention (CLA)

**Source:** Brandon et al., "Reducing Transformer Key-Value Cache via Cross-Layer Attention," arXiv 2408.01890, 2024
**Grade:** C (preprint)
**Relevance:** Context — related work on cross-layer value sharing

Documents that layer-wise attention patterns are "highly similar" and selective
sharing improves both efficiency and performance. Related to Value Residual
and DWA but via KV cache sharing rather than explicit residual.

---

## LIT-150: XSA + VR super-additivity (novel finding)

**Source:** Our HYP-039 experiment, 2026-03-20
**Grade:** E (single local experiment, n=1, no seeds)
**Relevance:** Direct — novel combination

No published work combines XSA with Value Residual Learning. Super-additive
interaction (+0.073 vs +0.035 expected sum) likely driven by information
factorization: XSA specializes attention in context, VR preserves token
identity via first-layer V bypass. DWA is redundant when XSA is present.

---

## LIT-151: Scaling Laws with Vocabulary (Tao et al. 2024)

**Title:** Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies
**Authors:** Tao et al. (SAIL, Sea AI Lab)
**Venue:** NeurIPS 2024 — **Grade A**
**URL:** https://arxiv.org/abs/2407.13623

**Key finding:** Optimal vocabulary size scales as N_v^opt ~ N_nv^0.83
(power law with gamma=0.83). Vocabulary should scale SLOWER than non-vocab
params. For their smallest model (33M non-vocab params), optimal vocab is
37-43K. Extrapolating to 22M non-vocab params: ~30-35K optimal vocab.
Both sp1024 and sp2048 are FAR below the compute-optimal vocab for a
22M model. Going from 1024 to 2048 moves in the right direction but
is still 15-17x below optimal.

**Caveats:**
- Their formula assumes compute-optimal training (Chinchilla regime).
  pgolf is fixed-time, not fixed-FLOP.
- Larger vocab = larger embedding table = fewer layers in 16MB budget.
  The formula doesn't account for artifact size constraints.
- Scale transfer tax: they tested 33M-3B, extrapolating down to 22M.

**Relevance:** Strong theoretical support for sp2048 > sp1024, and even
sp4096 > sp2048. But the 16MB artifact constraint creates a countervailing
force (more vocab params = fewer model params).

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-152: Length-MAX Tokenizer (Dong & Su, 2025)

**Title:** Length-MAX Tokenizer for Language Models
**Authors:** Dong Dong and Weijie Su
**Venue:** arXiv preprint — **Grade C**
**URL:** https://arxiv.org/abs/2511.20849

**Key finding:** Optimizing for average token LENGTH rather than frequency
alone reduces token count by 13-18% vs BPE at same vocab size. This
reduces training steps and inference latency. For 124M params, optimal
vocab ~32K. Key insight: longer tokens compress better but are harder
to predict, so there's an optimal length distribution.

**Relevance:** Suggests that tokenizer QUALITY (merge strategy) matters
as much as vocab SIZE. Our stock SentencePiece may not be optimal.
But at 1024-2048 vocab, merge strategy differences are small.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-153: Rate-Distortion Theory and Optimal Codebook Size

**Title:** Quantization (Gray & Neuhoff, IEEE Trans. IT, 1998)
**Authors:** Gray, R.M. and Neuhoff, D.L.
**Venue:** IEEE Transactions on Information Theory — **Grade A**
**URL:** https://www.math.ucdavis.edu/~saito/data/quantization/44it06-gray.pdf

**Key finding (cross-disciplinary):** Rate-distortion theory shows that
optimal codebook size K scales exponentially with source entropy and
inversely with distortion tolerance: K ~ 2^(nR) where n is block length
and R is rate. CRITICAL INSIGHT: vector quantization (blocking symbols
into groups) always achieves better rate-distortion than scalar
quantization. This is the information-theoretic analog of "larger vocab
= fewer tokens = better compression."

**Analogy to tokenization:** A tokenizer IS a vector quantizer over
character sequences. Increasing vocab from 1024 to 2048 is equivalent
to increasing the codebook size, which rate-distortion theory says
should reduce distortion (prediction error) at the same rate (model
capacity). The bound is: D(R) decreases monotonically with R (codebook
size), but with diminishing returns. The marginal gain from 1K->2K is
larger than 2K->4K.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-154: Sparse Coding Dictionary Size

**Title:** Online Dictionary Learning for Sparse Coding (Mairal et al. 2009)
**Authors:** Mairal, Bach, Ponce, Sapiro
**Venue:** ICML 2009 — **Grade A**
**URL:** https://www.di.ens.fr/~fbach/mairal_icml09.pdf

**Key finding (cross-disciplinary):** In sparse coding, overcomplete
dictionaries (K > signal dimension L) consistently outperform complete
ones. Typical ratio is K = 2-4x L. But the improvement has diminishing
returns: going from K=L to K=2L is a large gain; K=2L to K=4L is
smaller. The optimal dictionary size depends on (1) signal complexity,
(2) sparsity constraint, and (3) available training data.

**Analogy to tokenization:** The tokenizer vocabulary IS a dictionary
for sparse coding of text. Each document is represented as a sparse
sequence of dictionary elements. Overcomplete dictionaries (larger
vocab) give better representations, but with diminishing returns.
The "sparsity constraint" analog is sequence length: larger vocab
means shorter sequences (sparser representation).

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-155: Parameter Golf PR #122 (sp2048 Record, 1.160 BPB)

**Source:** openai/parameter-golf PR #122 — **Grade D**
**Author:** sproos
**URL:** https://github.com/openai/parameter-golf/pull/122

**Key finding:** sp2048 + 8L + 3x MLP + int6 + fp16 embed + SWA +
NorMuon + sliding window stride=64 achieved 1.160 BPB on 8xH100.
The author traded 1 layer (9L->8L) for the larger vocab, noting:
"It's really a question about whether we want more diversity in vocab
or more resolution in representation." sp2048 required sacrificing
a layer to fit in 16MB.

**Relevance:** Direct evidence that sp2048 can beat sp1024 at the
competition level, but the tradeoff is real: 1 fewer layer.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-156: Parameter Golf PR #78 (sp8192 Record, 1.186 BPB)

**Source:** openai/parameter-golf PR #78 — **Grade D**
**Author:** sproos
**URL:** https://github.com/openai/parameter-golf/pull/78

**Key finding:** sp8192 with selective quantization (int6 weights,
int8 embeddings) achieved 1.186 BPB. Had to sacrifice layers (8L
instead of 9L) for the large embedding table. Step time increased
from 43ms to 64ms (slower per step).

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-157: Parameter Golf PR #465 (sp1024 Record, 1.1508 BPB)

**Source:** openai/parameter-golf PR #465 — **Grade D**
**Author:** LoquiAuris
**URL:** https://github.com/openai/parameter-golf/pull/465

**Key finding:** sp1024 with 10L d=512 + int5-MLP + int6-attn +
BigramHash + SmearGate achieved 1.1508 BPB. Author EXPLICITLY tested
sp1024, sp2048, sp4096, sp8192 and concluded: "sp1024 with 10 layers
at full d=512 width outperformed all sp8192 configurations. The layer
count advantage (10L vs 6-8L) at d=512 exceeds the tokenizer
efficiency gain on H100 with full training."

**CRITICAL:** This is the strongest evidence AGAINST larger vocab in
pgolf. The 16MB constraint makes the layer-vs-vocab tradeoff favor
layers at current quantization levels. sp1024 wins because it
allows more layers.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-158: Parameter Golf PR #384 (Tokenizer Ablation, Null Result)

**Source:** openai/parameter-golf PR #384 — **Grade D**
**Author:** (Meta-TTT submission)
**URL:** https://github.com/openai/parameter-golf/pull/384

**Key finding:** Custom BPE tokenizer (split_digits=False, max_len=64)
gave -5.7% fewer tokens/byte at v8192 but NO BPP IMPROVEMENT
(+0.0006 worse). "Longer merged tokens are harder to predict per-token,
offsetting compression gains. Explains why community converged on
stock v1024."

**CRITICAL INSIGHT:** Compression efficiency != prediction efficiency.
Longer tokens are harder to predict, which can offset the sequence
length reduction from larger vocab.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-159: Parameter Golf PR #198 (SOTA 1.1318, sp1024)

**Source:** openai/parameter-golf PR #198 — **Grade D**
**URL:** https://github.com/openai/parameter-golf/pull/198

**Key finding:** Current VERIFIED SOTA (1.1318 BPB) uses sp1024 with
11 layers. More layers (11L) at sp1024 is better than fewer layers at
larger vocab. Baseline also uses sp1024.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-160: Parameter Golf PR #236 (Second Place 1.1400, sp1024)

**Source:** openai/parameter-golf PR #236 — **Grade D**
**URL:** https://github.com/openai/parameter-golf/pull/236

**Key finding:** Second place (1.1400 BPB) also uses sp1024. Key
finding was that SMALLER batch (524K vs 786K) gives more steps and
wins in fixed-time training.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-161: Parameter Golf PR #251 (sp4096 Record, 1.1596 BPB)

**Source:** openai/parameter-golf PR #251 — **Grade D**
**URL:** https://github.com/openai/parameter-golf/pull/251

**Key finding:** sp4096 + 11L d=432 + MLP3x achieved 1.1596 BPB.
Note the narrower dim (432 vs 512) to fit the larger embedding table.
This is 0.028 BPB worse than the sp1024 SOTA (1.1318), suggesting
the vocab-vs-width tradeoff favors width at this parameter budget.

**Cited in:** Vocab size literature review (2026-03-23)

---

## LIT-162: Parameter Golf Leaderboard Vocab Distribution

**Source:** Analysis of top parameter-golf submissions — **Grade D**

**Key finding (meta-analysis):** Among top submissions:
- SOTA #198 (1.1318): sp1024, 11L
- #236 (1.1400): sp1024, 11L
- #465 (1.1508): sp1024, 10L
- #122 (1.160): sp2048, 8L
- #251 (1.1596): sp4096, 11L d=432
- #217 (1.1753): sp4096, 10L d=496
- #78 (1.186): sp8192, 8L

**Pattern:** The top 3 submissions all use sp1024. Larger vocab
submissions cluster 0.02-0.05 BPB worse. The competition has
empirically converged on sp1024 as optimal for the 16MB constraint.

**Cited in:** Vocab size literature review (2026-03-23)
