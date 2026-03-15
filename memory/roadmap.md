# Research Roadmap

Updated 2026-03-14 after critical review of all directions.
See `memory/research-methodology.md` for the lessons that
drove this revision.

---

## What We Learned (HYP-001 Series Retrospective)

Four rounds of GPT-vs-LLaMA ablation (HYP-001, 001b, 001c, 001d)
produced one genuinely novel finding and several methodology
improvements. The core question ("which architecture is best at
3M params?") was wrong for the scale — this is "toy car racing,"
not "fruit fly genetics."

**Novel finding worth pursuing:**
- ANOM-009/010/011: Dropout x normalization interaction. RMSNorm
  models have different optimal dropout than LayerNorm models.
  LLaMA benefits 2.5x more from dropout. Non-monotonic: LLaMA
  optimal at 0.1, GPT at 0.2. No literature reports this.

**Methodology improvements (reusable):**
- FLOP-matched comparisons (DEC-004, R-001)
- Val loss as primary metric (DEC-008)
- μP for HP transfer (implemented and validated)
- MLflow + ExperimentRunner + FLOPCounter infrastructure

**Retired questions (answered or wrong for our scale):**
- "Which architecture is best at 3M params?" — answered: they're
  all about the same; regularization > architecture at this scale.
- HYP-002 (mx.compile) — engineering, not science.
- HYP-003 (optimizers) — well-studied, won't produce novel results.
- HYP-004 (MLA KV cache) — narrow, inference-focused.
- HYP-005 (5-min training) — superseded by FLOP-matched methodology.

---

## Ideas Evaluated (2026-03-14 Critical Review)

### REJECTED after devil's advocate review:

**Hybrid scaling laws at <100M params**
- Gate 4 FAIL: Kaplan 2020 says scaling laws are architecture-
  independent. Mamba scaling matches transformers with different
  constants. Our HYP-001 series showed architecture differences
  wash out at 3M.
- Gate 5 FAIL: Scaling law fits are unreliable below ~100M
  (noise dominates, Chinchilla validated at 70M+).
- The gap exists because experts know it's not meaningful, not
  because it's an opportunity.

**Apple Silicon inference benchmarks**
- Gate 1 FAIL: This is benchmarking, not research. The answer is
  trivially predictable (bandwidth-limited hardware favors
  bandwidth-efficient architectures).
- Gate 4 FAIL: llama.cpp already has extensive Metal benchmarks.
  Community has optimized this empirically.
- Useful as a blog post or community resource, NOT as research.

**μP for hybrid architectures**
- Gate 3 FAIL: Falcon-H1 already used customized μP for
  Mamba-Transformer hybrids. The basic question is answered.
- Gate 4 FAIL: μP theory is architecture-agnostic; the answer
  is trivially "yes" from the math.
- μP itself has limited adoption and known failure modes
  (EleutherAI: fails for standard LLaMA setups, breaks under
  FP8 and gradient clipping).

### SURVIVING ideas (pass quality gates):

**A. Dropout x normalization interaction (DEMOTED)**
- HYP-006 showed the interaction does NOT replicate at 30M with
  BPE. Dropout hurts both architectures uniformly in the
  undertrained (<1 epoch) regime. The non-monotonic interaction
  (ANOM-009/010/011) was specific to 3M params with multi-epoch
  char-level Shakespeare — a data-repetition artifact.
- Still potentially interesting in the multi-epoch regime, but
  the scope is much narrower than initially believed. Not a
  general architectural phenomenon.
- **Demoted from "strongest" to "niche."**

**B. Mechanistic understanding of hybrid architectures**
- Gate 1 PASS: "What do SSM layers learn vs attention layers?"
  is fundamental and underexplored.
- Gate 2 PASS: Small scale is an ADVANTAGE — can inspect every
  gradient, activation, attention pattern in minutes.
- Gate 3 PASS: Most hybrid work is benchmark-focused, not
  mechanistic. Limited interpretability work on hybrids.
- Gate 4 PASS: Outcome is not predictable.
- Needs: interpretability tools, probing classifiers, good
  evaluation tasks that distinguish SSM from attention behavior.
- Risk: may require ML theory expertise we don't have.

**C. Training dynamics visualization / educational contribution**
- Gate 1 PASS: nanoGPT's lasting impact was pedagogical. A clean
  codebase with 24 architectures has educational value.
- Gate 2 PASS: Small scale is ideal for teaching.
- Not traditional research, but high practical impact. Could pair
  with blog posts, tutorials, or a technical report.

**D. Method development (new technique that scales)**
- Gate 1 PASS if the technique is novel.
- Gate 2 PASS: Small-scale development, large-scale validation.
- This is how dropout, Adam, GANs started.
- The dropout x normalization finding could yield a prescriptive
  recipe: "recommended dropout by normalization scheme."
- Currently speculative — needs a concrete technique idea.

---

## Latest Experiment Findings (2026-03-14)

**HYP-006 (30M, TinyStories BPE):**
- LLaMA massively outperforms GPT at 30M (val 2.51 vs 3.05)
- Architecture matters at 10-30M scale with BPE
- Dropout hurts in undertrained regime (<1 epoch)
- The dropout × normalization interaction was a data-repetition
  artifact — does NOT replicate

**Hybrid baselines (10M, TinyStories BPE):**
- SSM hybrids (Falcon-H1, Jamba, Bamba) beat pure attention
- Ranking: Falcon-H1=Bamba (2.616) > Jamba (2.629) > LLaMA
  (2.710) >> GPT (3.132)
- Modern transformer improvements (RMSNorm, RoPE, SwiGLU, GQA)
  contribute more than SSM/attention mixing at this scale

**Experiments with proper methodology** (trust these):
- HYP-001c, HYP-001d, HYP-006, hybrid-baselines

**Superseded experiments** (exploratory value only):
- HYP-001 (no val split, wrong d_ff, fixed LR)
- HYP-001b (train loss as metric, 5-min time budget)

---

## Current Priority Order

1. **Decide what's genuinely interesting to YOU.** (Schulman,
   Nielsen: best research comes from personal curiosity, not
   gap analysis.)

2. **If pursuing understanding:** Idea B (mechanistic hybrid
   analysis) is now the strongest thread. "What do SSM layers
   learn vs attention layers?" Small scale is an advantage.
   HYP-006 and hybrid-baselines provide trained models to analyze.

3. **If pursuing community impact:** Idea C (educational
   content) has highest practical value. 5 notebooks delivered,
   24-architecture codebase is pedagogically rich.

4. **If pursuing research:** Idea A (dropout × normalization)
   is demoted — artifact of multi-epoch char-level regime.
   Could still be a niche finding for the data-repetition
   community, but not a general principle.

---

## Completed Items

| ID | Topic | Status |
|----|-------|--------|
| R-001 | FLOP counter | DONE (2026-03-12) |
| R-005 | GPT-to-LLaMA ablation | DONE (4 rounds, retired) |
| R-009 | μP implementation | DONE (2026-03-13, validated) |
| HYP-006 | Dropout × norm at 30M | DONE (2026-03-14, H6-c supported) |
| — | Hybrid baselines (5 archs) | DONE (2026-03-14) |
| — | Educational notebooks (5) | DONE (2026-03-14) |
| — | Metric callbacks library | DONE (2026-03-15) |
| HYP-007 | TTC scaling at 10M | DONE (2026-03-15, H7-a supported) |
| DEC-004 | FLOP-matched comparisons | ACCEPTED |
| DEC-005 | Chinchilla-optimal training | ACCEPTED |
| DEC-008 | Val loss as primary metric | ACCEPTED |

## Queued Ideas

**HYP-008: SSM/hybrid test-time compute scaling**
- Follow-up to HYP-007. HYP-007 showed TTC works at 10M
  (pass@64=11.9x pass@1). Does the same hold for SSMs?
- SSMs lack explicit attention patterns — failure modes on exact
  modular arithmetic may be fundamentally different.
- Architectures to test: Mamba-2 (pure SSM), Falcon-H1 / Jamba
  (hybrid), GatedDeltaNet (linear attention). All already
  implemented and cross-referenced in the codebase.
- HYP-007 finding: dropout HURTS diversity. Simplify to
  dropout=0.0 only: 4 archs x 3 seeds = 12 runs.
- **Ready to pre-register.**

**HYP-009: Grokking and TTC interaction**
- HYP-007 models have train loss ~0.002 but pass@1 ~0.5%.
  They memorize training data but don't generalize. What
  happens near the grokking transition?
- If we train much longer (10K-50K steps), the model may
  grok modular arithmetic. Does pass@k improve dramatically
  near the transition? Does TTC become more or less effective
  as the model transitions from memorization to generalization?
- Baeumel et al. 2025 shows digit-level circuits are
  model-size-independent — grokking may be achievable at 10M.
- Single architecture (LLaMA-10M, dropout=0.0), multiple
  checkpoints along the training curve, pass@k at each.

**HYP-010: TTC scaling exponent vs model size**
- HYP-007 at 10M showed ~50% growth per doubling of k.
  What is the scaling exponent at 30M, 100M?
- Wu et al. (LIT-039) provide inference scaling law
  extrapolation. Compare our empirical exponents to their
  predicted values at 10M, 30M.

## Retired Items

| ID | Topic | Reason |
|----|-------|--------|
| R-002 | Chinchilla at 3M | Scaling laws unreliable <100M |
| R-003 | Char vs BPE efficiency | Low priority |
| R-004 | MLX FLOP calibration | Engineering, not science |
| R-006 | Depth vs width | Answered by Kaplan: weak effect |
| R-007 | GQA at small scale | Answered by HYP-001c |
| R-008 | Position encoding | Answered by HYP-001 series |
| R-010 | LR schedule | Well-studied, use cosine |
| R-011 | Data repetition | Answered by Muennighoff |
| R-012 | Unified memory training | Low ROI |
| R-013 | mx.compile tradeoffs | Engineering, not science |
| HYP-002 | mx.compile coverage | Retired |
| HYP-003 | Optimizer comparison | Retired |
| HYP-004 | MLA KV cache | Retired |
| HYP-005 | 5-min training budget | Retired |
