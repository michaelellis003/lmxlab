# Experiment Results

This page presents findings from lmxlab's trusted experiments.
All results use proper methodology: train/val splits, validation
loss as the primary metric (DEC-008), and FLOP-matched compute
budgets (DEC-004).

!!! note "Methodology evolution"
    Early experiments (HYP-001, HYP-001b) had no validation split
    and reported training loss as the primary metric, which masked
    overfitting. Results from HYP-001c onward use corrected
    methodology and are the ones presented here. See
    [Methodology](methodology.md) for details.

## HYP-001 Series: GPT vs LLaMA Ablation

**Setup:** 3M params, char-level Shakespeare, 1 PFLOPs budget,
3 seeds per config.

This series tested whether LLaMA-style features (RMSNorm, RoPE,
SwiGLU, GQA, no bias) improve over a GPT-2 baseline at small
scale. After two rounds with methodology issues, HYP-001c and
HYP-001d produced reliable results.

### HYP-001c: Feature Ablation (lr=3e-4 vs 1e-4)

| Config | Val Loss | Train Loss | Gap |
|--------|----------|------------|-----|
| GPT baseline (lr=3e-4) | 1.609 +/- 0.008 | 0.781 | -0.83 |
| + RMSNorm (lr=1e-4) | 1.667 +/- 0.013 | 0.864 | -0.80 |
| + RoPE (lr=3e-4) | 1.607 +/- 0.005 | 0.767 | -0.84 |
| + SwiGLU FFN (lr=3e-4) | 1.614 +/- 0.011 | 0.682 | -0.93 |
| + GQA (lr=3e-4) | 1.608 +/- 0.012 | 0.687 | -0.92 |
| + No bias = LLaMA (lr=1e-4) | 1.670 +/- 0.012 | 0.819 | -0.85 |

All configs show massive overfitting (train-val gap ~0.83-0.93)
from 8-9 epochs of repeated Shakespeare data. Architecture
differences are within noise on val loss.

### HYP-001d: Dropout x Architecture

| Architecture | Dropout | Val Loss | Train Loss | Gap |
|--------------|---------|----------|------------|-----|
| GPT | 0.0 | 1.611 +/- 0.009 | 0.778 | -0.83 |
| GPT | 0.1 | 1.573 +/- 0.002 | 1.188 | -0.38 |
| GPT | 0.2 | **1.560 +/- 0.002** | 1.265 | -0.30 |
| LLaMA | 0.0 | 1.671 +/- 0.013 | 0.813 | -0.86 |
| LLaMA | 0.1 | **1.570 +/- 0.001** | 1.283 | -0.29 |
| LLaMA | 0.2 | 1.586 +/- 0.009 | 1.354 | -0.23 |

**Key findings:**

1. **Dropout dominates architecture.** Adding dropout=0.1 improves
   val loss by 0.05-0.10 (Cohen's d > 5), dwarfing all architecture
   effects.
2. **At dropout=0.1, architectures equalize.** GPT and LLaMA reach
   nearly identical val loss (1.573 vs 1.570).
3. **Non-monotonic interaction:** LLaMA peaks at dropout=0.1 then
   degrades at 0.2; GPT continues improving. This was later shown
   to be a data-repetition artifact (see HYP-006).

## HYP-006: Dropout x Normalization at Scale

**Setup:** 30M params, TinyStories BPE tokenization, 2000 steps
(single epoch), 3 seeds per config.

This experiment tested whether the dropout x normalization
interaction from HYP-001d replicates at larger scale with proper
tokenization.

### GPT-30M Results

| Dropout | Val Loss | Train Loss | Gap |
|---------|----------|------------|-----|
| 0.0 | **3.049 +/- 0.025** | 3.009 | -0.04 |
| 0.1 | 3.150 +/- 0.044 | 3.162 | +0.01 |
| 0.2 | 3.335 +/- 0.058 | 3.382 | +0.05 |
| 0.3 | 3.493 +/- 0.050 | 3.538 | +0.05 |

### LLaMA-30M Results

| Dropout | Val Loss | Train Loss | Gap |
|---------|----------|------------|-----|
| 0.0 | **2.512 +/- 0.013** | 3.309 | +0.80 |
| 0.1 | 2.578 +/- 0.012 | 3.408 | +0.83 |
| 0.2 | 2.656 +/- 0.008 | 3.499 | +0.84 |
| 0.3 | 2.745 +/- 0.005 | 3.593 | +0.85 |

### Architecture Comparison

| Dropout | GPT Val Loss | LLaMA Val Loss | Gap | Cohen's d |
|---------|-------------|----------------|-----|-----------|
| 0.0 | 3.049 | 2.512 | -0.54 | -32.8 |
| 0.1 | 3.150 | 2.578 | -0.57 | -17.6 |
| 0.2 | 3.335 | 2.656 | -0.68 | -16.5 |
| 0.3 | 3.493 | 2.745 | -0.75 | -21.3 |

**Key findings:**

1. **LLaMA dominates at 30M.** With BPE tokenization and realistic
   data, LLaMA outperforms GPT by 0.54 val loss, completely
   reversing the 3M char-level finding.
2. **Dropout hurts in the undertrained regime.** Both architectures
   degrade with higher dropout when training for less than one
   epoch. This is consistent with the literature: dropout only
   helps with data repetition.
3. **The HYP-001d interaction does not replicate.** The
   non-monotonic dropout x normalization phenomenon was specific
   to the multi-epoch char-level regime. It was a data-repetition
   artifact, not a fundamental architecture property.

## Hybrid Architecture Baselines

**Setup:** 10M params, TinyStories BPE tokenization, 2000 steps,
single seed.

Five architectures compared: two pure-attention (GPT, LLaMA)
and three SSM+attention hybrids (Falcon-H1, Jamba, Bamba).

| Rank | Architecture | Val Loss | Params | Wall Time |
|------|-------------|----------|--------|-----------|
| 1 | Falcon-H1 | **2.616** | 9.30M | 314s |
| 1 | Bamba | **2.616** | 9.30M | 343s |
| 3 | Jamba | 2.629 | 10.19M | 383s |
| 4 | LLaMA | 2.710 | 9.88M | 250s |
| 5 | GPT | 3.132 | 9.61M | 255s |

**Key findings:**

1. **Modern transformer features contribute more than SSM mixing.**
   The GPT-to-LLaMA gap (0.42 val loss, 16%) is much larger
   than the LLaMA-to-hybrid gap (0.09 val loss, 4%).
2. **Falcon-H1 and Bamba tie** at 2.616 val loss, with Falcon-H1
   being faster (314s vs 343s).
3. **SSM hybrids are slower** than pure-attention models due to
   sequential SSM scan operations (250-255s vs 314-383s).

## Key Findings Across Experiments

### Architecture matters at 10-30M, not at 3M

At 3M params with char-level tokenization, architecture
differences wash out entirely; regularization (dropout) is the
dominant factor. At 10-30M with BPE tokenization, architecture
produces massive effect sizes (Cohen's d > 15).

### Modern transformer features >> SSM mixing

The biggest gains come from adopting LLaMA-style features
(RMSNorm, RoPE, SwiGLU, GQA) over the GPT-2 baseline. Adding
SSM layers provides a further 4% improvement, but the bulk of
the gain is from the attention-side improvements.

### Dropout hurts in undertrained regimes

Dropout is highly beneficial when training on repeated data
(multi-epoch) but actively harmful when training for less than
one epoch. This explains the conflicting findings between
HYP-001d (8-9 epochs, dropout helps) and HYP-006 (single epoch,
dropout hurts).

### The dropout x normalization interaction was an artifact

The non-monotonic interaction between dropout rate and
normalization type (ANOM-009/010/011) was specific to the
multi-epoch char-level regime and does not generalize. It was
caused by data repetition, not a fundamental property of
LayerNorm vs RMSNorm.

## Methodology Notes

See the [Methodology](methodology.md) page for full details on
how experiments are designed, controlled, and tracked. Key
methodology improvements over the course of this work:

- **HYP-001 / HYP-001b:** No validation split. Training loss
  reported as primary metric. Results unreliable; superseded.
- **HYP-001c:** Added 90/10 train/val split. Revealed massive
  overfitting that was invisible in prior rounds.
- **HYP-001d:** Val loss as primary metric (DEC-008). Discovered
  dropout as dominant intervention.
- **HYP-006:** Scaled to 30M params with BPE tokenization.
  Showed that small-scale char-level findings don't generalize.
- **Hybrid baselines:** Extended to SSM+attention architectures.
  Confirmed modern features matter more than mixing strategy.
