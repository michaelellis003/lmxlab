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

**TTC research line is mature.** Four experiments (HYP-007
through HYP-010) converge on a clear story:
- TTC amplification ~12-15x on modular arithmetic (mod 97)
- Architecture-independent (4 families tested)
- Size-independent (10M-30M tested)
- Reveals latent generalization 39K steps before grokking
- The amplification factor is a task property, not model property

**Next directions (in priority order):**

1. **Write up TTC findings.** The series has a publishable
   story: "Test-time compute scaling below 1B: architecture-,
   size-, and training-phase-independent amplification on
   verifiable tasks." 4 experiments, clean methodology, novel
   findings. Blog post or short paper.

2. **If pursuing understanding:** Idea B (mechanistic hybrid
   analysis) is the strongest research thread. "What do SSM
   layers learn vs attention layers?" Small scale is an
   advantage. ANOM-015 (val_loss vs pass@k inversion) is a
   concrete starting point.

3. **If pursuing community impact:** Idea C (educational
   content) has highest practical value. TTC results could
   be a 6th notebook: "Test-Time Compute at Small Scale."

4. **If pursuing more TTC:** Test on a different task (e.g.,
   TinyStories generation, not modular arithmetic) to check
   whether the ~12-15x constant is truly task-specific or
   generalizes across verifiable tasks.

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
| HYP-008 | SSM/hybrid TTC scaling | DONE (2026-03-15, H8-a supported) |
| HYP-009 | Grokking × TTC interaction | DONE (2026-03-15, H9-a strongly supported) |
| HYP-010 | TTC exponent vs model size | DONE (2026-03-15, H10-a supported) |
| HYP-011 | Per-token loss decomposition | DONE (2026-03-16, ANOM-015 explained) |
| HYP-012 | TTC cross-task amplification | DONE (2026-03-16, amp is task-dependent) |
| DEC-004 | FLOP-matched comparisons | ACCEPTED |
| DEC-005 | Chinchilla-optimal training | ACCEPTED |
| DEC-008 | Val loss as primary metric | ACCEPTED |

## Queued Ideas

**HYP-009: Grokking and TTC interaction — COMPLETED**
- pass@64 reveals latent generalization 39K steps before greedy
  accuracy catches up. Strongest TTC result in the series.
- See HYP-009 results in hypotheses.md.

**HYP-010: TTC scaling exponent vs model size — COMPLETED**
- Result: TTC exponent is roughly size-independent
  (14.6x at 10M, 11.9x at 30M = 1.23x difference).
- 30M model performs WORSE than 10M on modular arithmetic
  due to overparameterization (data bottleneck).
- See HYP-010 results in hypotheses.md.

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
