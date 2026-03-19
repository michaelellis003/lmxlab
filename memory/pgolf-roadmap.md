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
**Status:** queued

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
**Status:** queued

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

## Completed Items

(none yet)

## Retired Items

(none yet)
