# Parameter Golf: GPU Validation Plan (8xH100)

## Summary of Local Research

6 iterations, 40+ experiments on Mac (MLX, 8K batch, 600s wallclock).
Baseline: 1.2244 BPB (official), ~1.94 BPB (local).

### Confirmed Findings (Batch-Size Independent, High Confidence)

These are pure architecture changes that should transfer directly:

| Change | Local BPB Gain | Mechanism | Confidence |
|--------|---------------|-----------|------------|
| UNIQUE_BLOCKS=3 | +0.029 | Weight sharing regularization + throughput | HIGH |
| NUM_HEADS=4, NUM_KV_HEADS=4 | +0.072 | Wider head_dim (128 vs 64) | HIGH |
| Keep relu^2 (not SwiGLU) | +0.010-0.015 | Larger hidden dim (1024 vs 688) | HIGH |
| Keep skip connections | +0.034 (w/ wide heads) | Complementary to recurrence | HIGH |
| Keep MUON_BACKEND_STEPS=5 | — (default) | 3 steps destroys convergence | HIGH |

### Best Local Config

```
UNIQUE_BLOCKS=3       # 3 unique blocks, cycled for 9 layers
NUM_HEADS=4           # 4 attention heads (head_dim=128)
NUM_KV_HEADS=4        # Full MHA (no GQA)
MLP_TYPE=relu2        # Default relu^2 activation
USE_SKIP=1            # Keep encoder-decoder skip connections
```

**Local result:** 1.8512 BPB, 5.4MB artifact, 10.6MB headroom

### Confounded Findings (Need GPU Validation)

These showed improvement locally but are confounded by the 64x
batch size mismatch (8K local vs 524K official):

| Change | Local Effect | Confound | GPU Priority |
|--------|-------------|----------|-------------|
| WARMDOWN_ITERS=1500-1800 | +0.02-0.05 | Batch-size-dependent LR decay | MEDIUM |
| MATRIX_LR=0.03 | +0.005 | Lower noise → different optimal LR | LOW |

---

## GPU Experiment Plan

### Phase 1: Validate Architecture Changes (3 runs)

These test the batch-independent findings. Run on the official
8xH100 setup with the competition's standard 524K batch size.

**Run 1: Official Baseline**
```bash
# No changes — establish GPU baseline for comparison
# Expected: ~1.2244 BPB (the published baseline)
```

**Run 2: Weight Sharing Only**
```bash
UNIQUE_BLOCKS=3
# Expected: ~1.19-1.20 BPB (baseline - 0.02-0.03)
# This tests if sharing regularization transfers to large batch
```

**Run 3: Weight Sharing + Wide Heads (Best Config)**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
# Expected: ~1.15-1.18 BPB (baseline - 0.04-0.07)
# This is the money run — our best local finding
```

**Decision gate:** If Run 3 improves over Run 1 by >0.01 BPB,
proceed to Phase 2. If not, investigate why (may need LR tuning
for the different param count).

### Phase 2: Schedule Tuning (3-4 runs)

With the confirmed architecture, tune the training schedule
(this is where Bayesian optimization with Optuna would be more
sample-efficient than agent exploration — see DEC-014).

**Run 4: Conservative warmdown**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
WARMDOWN_ITERS=1500
```

**Run 5: Moderate warmdown**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
WARMDOWN_ITERS=1800
```

**Run 6 (optional): LR adjustment**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
MATRIX_LR=0.03
SCALAR_LR=0.03
TIED_EMBED_LR=0.04
```

### Phase 3: Spend Artifact Budget (2-3 runs)

The best config uses only 5.4MB of 16MB. Options to spend the
remaining 10.6MB:

**Option A: Larger Vocabulary**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
VOCAB_SIZE=4096
# Needs: sp4096 tokenizer + dataset (download via
#   python data/cached_challenge_fineweb.py --variant sp4096)
# Adds: 4096*512 = 2M embedding params (~1.5MB artifact)
# Total estimated artifact: ~7MB (still 9MB headroom)
```

**Option B: Wider Model**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
MODEL_DIM=640
# Note: need NUM_HEADS that divides 640 (4 heads → hd=160)
# Adds: ~50% more params per block
# Total estimated artifact: ~8-9MB
# WARNING: This failed locally due to throughput, but on 8xH100
# the per-step time is less of a constraint
```

**Option C: More Unique Blocks**
```bash
UNIQUE_BLOCKS=5
NUM_HEADS=4
NUM_KV_HEADS=4
# 5 unique blocks was worse locally (1.928 vs 1.910 for 3 blocks)
# but at GPU batch size, the regularization-vs-capacity tradeoff
# may be different. Worth testing if artifact budget allows.
```

### Phase 4: Optuna Sweep (if time allows)

Per DEC-014 (Shwartz Ziv's insight), once the architecture is
locked in, run Optuna TPE over the numeric hyperparameters:

**8 parameters to sweep:**
1. WARMDOWN_ITERS: [800, 2000]
2. MATRIX_LR: [0.02, 0.06]
3. SCALAR_LR: [0.02, 0.06]
4. TIED_EMBED_LR: [0.03, 0.08]
5. MUON_MOMENTUM: [0.90, 0.98]
6. LOGIT_SOFTCAP: [20.0, 50.0]
7. QK_GAIN_INIT: [1.0, 2.0]
8. WARMUP_STEPS: [10, 50]

This is the type of tuning where Bayesian optimization
outperforms agent-based search (DEC-014).

---

## Competition Intelligence (2026-03-19 Literature Review)

Current SOTA is ~1.015 BPB (PR #64). Key techniques from top submissions
that are **orthogonal to our architecture changes**:

### Must-Have (Zero/Low Cost)

1. **Sliding window evaluation** — ~0.03 BPB free. Score each token with
   up to 4000 tokens of context. All top submissions use this.
2. **fp16 embedding** — small gain, zero artifact cost.

### High Priority (Artifact Budget Dependent)

3. **Larger vocabulary (sp8192)** — 0.01-0.04 BPB. Public tokenizers
   available at huggingface.co/sproos/parameter-golf-tokenizers.
   This UNBLOCKS R-PG-003.
4. **Longer sequences (4096)** — more context per forward pass.
   Combined with sliding window for eval gains.
5. **Int6 quantization** — int6 on MLP+attn, fp16 on embedding.
   Saves ~4MB artifact. We may not need this given 10.6MB headroom.

### Lower Priority

6. **Test-Time Training (TTT)** — LoRA on val docs during eval.
   Modest gains, complex to implement.

### NOT Found in Competition

- **SSM/Mamba** — no state-space submissions. Fixed 600s training +
  16MB artifact appears to favor transformers with depth recurrence.
- **Hybrid architectures** — no submissions combining SSM + attention.

### Updated Expected Outcome

With our architecture (3u + 4h/4kv) PLUS competition techniques:
- Conservative: **1.08-1.12 BPB** (competitive with current field)
- Optimistic: **1.04-1.08 BPP** (potential new SOTA range)

### Updated Phase 3: Spend Artifact Budget

Replace Option A with the now-unblocked vocab exploration:

**Option A (revised): sp8192 Vocabulary**
```bash
UNIQUE_BLOCKS=3
NUM_HEADS=4
NUM_KV_HEADS=4
# Download: from huggingface.co/sproos/parameter-golf-tokenizers
# No need to retrain tokenizer — use pre-trained sp8192
# Estimated artifact: ~9-10MB (still under 16MB)
```

### New Phase 5: Eval-Time Optimizations (after architecture locked)

These don't affect training, only the submission script:
1. Sliding window eval (4000 context)
2. fp16 embedding storage
3. (Optional) TTT with LoRA adapters

---

## Code Changes Required

All changes are already implemented in `train_gpt_mlx.py`:

1. `UNIQUE_BLOCKS` env var — cyclic weight sharing via modulo
2. `NUM_HEADS` / `NUM_KV_HEADS` — already parameterized
3. `USE_SKIP` env var — skip connection toggle
4. `MLP_TYPE` env var — relu2 vs swiglu (keep relu2)

For the CUDA version (`train_gpt.py`), the same changes need
to be ported:
- Add `UNIQUE_BLOCKS` support to model init + forward
- Add `USE_SKIP` support
- The head count params should already work

### Porting Checklist

- [ ] Port UNIQUE_BLOCKS to train_gpt.py (cyclic `blocks[i % n]`)
- [ ] Port USE_SKIP to train_gpt.py
- [ ] Verify NUM_HEADS=4 works in CUDA version
- [ ] Download sp4096 dataset if vocab exploration planned
- [ ] Set up Optuna for Phase 4 numeric tuning

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Wide heads don't transfer to GPU | Run 3 tests this directly |
| Weight sharing hurts at 524K batch | Sharing is regularization; less needed at large batch. But throughput benefit still applies. |
| Warmdown tuning is wasted | Expected — local schedule gains were confounded. Keep conservative. |
| Vocab expansion hurts BPB | Larger vocab = fewer tokens = shorter sequences. Could reduce context quality. Test carefully. |
| sp4096 dataset unavailable | May need to download docs and retrain tokenizer. Adds ~1 hour setup. |

## Expected Outcome

Conservative estimate: **1.17-1.20 BPB** (vs 1.2244 baseline)
Best case with vocab expansion: **1.14-1.17 BPB**

The 10.6MB artifact headroom from weight sharing is the key
strategic asset — it enables experiments that no one else can run
within the 16MB constraint.
