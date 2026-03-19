# Parameter Golf Research Roadmap

Challenge: Train best LM in 16MB artifact, evaluated by val_bpb on
FineWeb. Baseline: 1.2244 BPB (9-layer, 512-dim, 1024-vocab).
Local iteration on Mac (MLX); GPU validation later.

---

## Constraint Budget

| Constraint | Limit | Baseline | Headroom |
|-----------|-------|----------|----------|
| Artifact size | 16,000,000 bytes | ~15.9 MB | ~100 KB |
| Training time | 600s on 8xH100 | ~600s | ~0s |
| Vocab size | Flexible | 1024 | — |

## Tier 1: High Impact, Moderate Risk

### R-PG-001: Training Schedule Optimization
**Priority:** Highest (zero artifact risk)
**Rationale:** Baseline uses 20-step warmup, 1200-step warmdown.
No architecture change needed. Pure optimization of LR schedule,
warmup length, warmdown curve.
**Expected gain:** 0.003-0.008 BPB
**Status:** tested (HYP-017)
**Result:** Locally, schedule changes give 0.05-0.10 BPB improvement,
but this is confounded by batch size mismatch (8K vs 524K tokens/step).
Warmup=20 is fine. Try warmdown=1500-1800 on official runs.
Conservative expected gain on official: 0.002-0.005 BPB.

### R-PG-002: Depth Recurrence / Weight Sharing
**Priority:** High (most parameter-efficient architecture change)
**Rationale:** Loop N unique blocks M times = M-layer depth with
N blocks' parameters. Universal Transformers showed this works.
Frees parameter budget for wider layers or bigger vocab.
**Expected gain:** 0.01-0.02 BPB (via reallocation of saved params)
**Status:** tested (HYP-018)
**Result:** 3 unique blocks × 3 loops at dim=512 gives 0.029 BPB
improvement AND 63% smaller artifact (4.7MB vs 12.6MB). Sharing
acts as regularization at this scale. Width reallocation fails
locally (wider models too slow per step). More sharing is better
(3 blocks > 5 blocks). Opens 11.3MB artifact headroom.
**Next:** Test deeper recurrence (3 blocks × {4,5} loops) and
combine with vocab/schedule changes.

### R-PG-003: Vocabulary Size Exploration
**Priority:** High (interacts with depth recurrence)
**Rationale:** Larger vocab (2048/4096) means fewer tokens per
document but bigger embedding table. With tied embeddings, the
table is shared. With depth recurrence saving transformer params,
a larger vocab might net improve BPB.
**Expected gain:** 0.005-0.015 BPB (with depth recurrence)
**Status:** unblocked (public tokenizers found)
**Update (2026-03-19):** Public pre-trained tokenizers for sp1024/2048/4096/8192
available at huggingface.co/sproos/parameter-golf-tokenizers (LIT-097).
No need to retrain from scratch. sp8192 used by top competitors for
0.01-0.04 BPB improvement. Test on GPU with our 10.6MB artifact headroom.

## Tier 2: Moderate Impact, Lower Risk

### R-PG-004: Low-Rank Decomposition
**Rationale:** Replace W with UV (d_model x rank, rank x d_out).
Attention QKV projections are prime targets.
**Expected gain:** 0.002-0.005 BPB (via param reallocation)
**Status:** queued

### R-PG-005: Quantization-Aware Training (QAT)
**Rationale:** Baseline uses post-training int8. QAT lets the model
adapt to quantization noise during training.
**Expected gain:** 0.002-0.005 BPB recovery
**Status:** queued

### R-PG-006: SwiGLU vs relu^2
**Rationale:** SwiGLU generally outperforms but uses 50% more params
per layer (3 projections vs 2). Must parameter-match.
**Expected gain:** 0.003-0.008 BPB
**Status:** tested (HYP-020)
**Result:** relu² beats SwiGLU by 0.01-0.015 BPB at parameter-matched
settings. The 50% larger hidden dim of relu² outweighs SwiGLU's
activation advantage. Keep relu².

## Tier 3: Speculative

### R-PG-007: MoE within 16MB
**Risk:** Expert diversity → poor zlib compression. May not fit.
**Status:** queued

### R-PG-008: Multi-Token Prediction
**Risk:** Training FLOPs overhead, unclear BPB benefit.
**Status:** queued

### R-PG-009: Skip Connection Optimization
**Rationale:** Baseline has encoder-decoder skip connections.
Could explore denser skip patterns, different mixing strategies.
**Expected gain:** 0.001-0.003 BPB
**Status:** queued

### R-PG-010: Optimizer Throughput Tuning
**Priority:** Low (already tested)
**Rationale:** Reduce Muon Newton-Schulz iterations or increase
microbatch size to gain throughput within 600s wallclock.
**Expected gain:** 0.005-0.010 BPB (via more steps)
**Status:** tested (HYP-021)
**Result:** 3 NS steps degrades BPB by 0.073 (step time only 2.4%
faster). Microbatch size is irrelevant at 8K batch. Muon's 5 NS
steps are load-bearing for convergence quality. No free throughput.

### R-PG-011: Attention Head Configuration
**Priority:** Highest (batch-independent, largest gain found)
**Rationale:** Baseline uses 8 heads at head_dim=64. Wider heads
(fewer heads, larger head_dim) may be better at small scale.
**Expected gain:** 0.03-0.07 BPB
**Status:** tested (HYP-022)
**Result:** 4 heads with head_dim=128 improves BPB by +0.072 —
THE LARGEST SINGLE IMPROVEMENT across 40+ experiments. Full MHA
(4 KV heads matching 4 query heads) is crucial. GQA with 2 KV
heads gives only +0.030. Skips remain helpful (+0.034 with wide
heads). New best: 3u + 4h/4kv = 1.8512 at 5.4MB.

## Completed Items

- R-PG-001: Schedule tuning (HYP-017) — longer warmdown helps, but
  confounded by batch size. Conservative gain: 0.002-0.005 on official.
- R-PG-002: Depth recurrence (HYP-018/019) — 3 unique blocks is
  optimal, +0.029 BPB and 63% smaller artifact. Best combined config:
  3u + wd=5000 + lr=0.03 = 1.8436 at 3.6MB. 12.4MB headroom.
- R-PG-006: SwiGLU (HYP-020) — relu² beats SwiGLU. Keep relu².
- R-PG-010: Optimizer throughput (HYP-021) — keep defaults. 5 NS
  steps are load-bearing. No free throughput gains.
- R-PG-011: Wide attention heads (HYP-022) — 4 heads at hd=128
  beats 8 heads at hd=64 by +0.072 BPB. Biggest win found.

## Recommended Competition Config

Based on 40+ local experiments (confound: 8K batch vs 524K official):

**Definite changes (batch-independent):**
- UNIQUE_BLOCKS=3 (weight sharing, +0.029 BPB, 63% smaller artifact)
- NUM_HEADS=4, NUM_KV_HEADS=4 (wide heads hd=128, +0.072 BPB)
- Keep relu² (beats parameter-matched SwiGLU)
- Keep skip connections (help +0.034 with wide heads)
- Keep MUON_BACKEND_STEPS=5 (3 steps destroys convergence)

**Combined local improvement:** +0.072 over 3u baseline, +0.101
over original 9-block baseline. New best: 1.8512 at 5.4MB.
**10.6MB artifact headroom** for vocab expansion or wider model.

**Test on official hardware:**
- WARMDOWN_ITERS=1500-1800 (conservative schedule extension)
- With 5.4MB artifact: 10.6MB free for vocab expansion (R-PG-003)

**Do NOT use locally-optimal values:**
- WARMDOWN_ITERS=5000 and MATRIX_LR=0.03 are batch-size artifacts

## Local Iteration Assessment

After 6 research iterations (HYP-017 through HYP-022) with 40+
experiments:

**Thoroughly explored:**
1. Weight sharing (U-curve, 3 blocks optimal)
2. MLP type (relu² > SwiGLU)
3. Head configuration (4 wide heads >> 8 narrow heads)
4. Skip connections (helpful with wide heads)
5. Optimizer throughput (defaults are optimal)

**Still blocked/confounded:**
6. Schedule/LR: confounded by 64x batch size mismatch
7. Vocab exploration: blocked (need sp4096 dataset)

**Recommended next steps:**
1. GPU validation: UNIQUE_BLOCKS=3 + NUM_HEADS=4 + NUM_KV_HEADS=4
2. Sliding window eval: implement for submission script (R-PG-012, free ~0.03)
3. GPU vocab exploration: sp8192 with shared blocks + wide heads (R-PG-003, unblocked)
4. Longer sequences: 1024→4096 on GPU (R-PG-013)
5. GPU QAT: if arch confirmed, quantization-aware training

### R-PG-012: Sliding Window Evaluation
**Priority:** Highest (free BPB improvement, eval-only)
**Rationale:** Score each token with up to 4000 tokens of context instead
of averaging across ~512. All top competitors use this. ~0.03 BPB free.
**Expected gain:** 0.02-0.04 BPB
**Status:** queued (implement for GPU submission script)
**Source:** LIT-095

### R-PG-013: Longer Sequences (1024→4096)
**Priority:** High (pairs with sliding window)
**Rationale:** More context per forward pass. Top competitors use 4096.
**Expected gain:** 0.01-0.02 BPB
**Status:** queued (GPU only — affects training time)

### R-PG-014: SSM/Mamba/Hybrid Architectures
**Priority:** Low (no evidence of viability)
**Rationale:** No SSM or hybrid submissions found in competition PRs.
The 600s training budget and 16MB artifact limit favor transformers with
depth recurrence. SSM's advantage (linear attention) matters less when
sequences are only 1024-4096 tokens.
**Expected gain:** Unknown — likely negative given competition evidence
**Status:** deprioritized (no competitive evidence, high risk)

## Retired Items

- R-PG-004: Low-rank — deprioritized, headroom is not the bottleneck
- R-PG-009: Skip connections — expected gain < noise floor
