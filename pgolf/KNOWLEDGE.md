# Parameter Golf Competition — Complete Knowledge Base

Everything learned from 82+ hypotheses, L4 GPU validation, and cross-disciplinary research.
Competition: https://github.com/openai/parameter-golf
Deadline: April 30, 2026

## Competition Rules
- **Metric:** val BPB (bits per byte) on FineWeb, lower is better
- **Artifact limit:** 16MB (weights + code, int8+zlib compressed)
- **Training budget:** 10 minutes wall clock on 8xH100 SXM GPUs
- **Code limit:** 1500 lines in a single training script

## Competition Landscape (as of March 2026)
- **SOTA:** ~1.12 BPB (PR #287: 11L, XSA, EMA, Int6, MLP3x)
- **Official baseline:** 1.2244 BPB (9L, 512dim, sp1024, tied embeddings)
- **Winning meta:** 11 unique layers + Int6+zstd + MLP3x + SmearGate + BigramHash + SWA + WD=0.04

---

## VALIDATED FINDINGS (tested on 22M competition config)

### Tokenizer: sp2048 >> sp1024 (+1.16 BPP on 22M model)
- **Confidence: 100% — batch-invariant, pure math**
- sp2048 tokens encode ~1.6x more bytes per token
- BPP = CE × ln(2) / bytes_per_token → more bytes/token → lower BPP
- Tested on 22M model with multi-checkpoint learning curve (1000-6000 steps)
- Advantage is CONSISTENT at every checkpoint — no crossover
- **USE sp2048 ON ALL GPU SUBMISSIONS**

---

## FINDINGS THAT FAILED TO TRANSFER (tested on 6L/7M, failed on L4 GPU 22M)

### XSA (Exclusive Self Attention)
- **Local:** +0.073 BPP iso-step on 6L/3u/4h/7M
- **L4 GPU:** -0.009 BPP on 11L/9u/8h/22M (WORSE than baseline)
- **Root cause:** ~5% per-step overhead costs ~150 steps; per-step quality gain doesn't compensate at 32K batch
- **Status:** NEEDS iso-step retest on 22M config locally before trusting

### Value Residual
- **Local:** +0.027 BPP on 6L/7M
- **L4 GPU:** Included in full stack that was -0.009 overall
- **Root cause:** May be regularization (variance reduction) that only helps at 8K batch noise
- **Status:** NEEDS iso-step retest on 22M

### Z-loss + Softcap 50
- **Local:** +0.011 BPP combined on 6L/7M
- **L4 GPU:** Included in full stack, may be redundant with default WD+SWA
- **Root cause:** Z-loss controls logit scale; WD already does this. Softcap 50 was tuned for local noise regime
- **Status:** UNCERTAIN — might still help as additive to WD

### Random MLP fc (JL lemma)
- **Local:** +0.011 BPP, 3.8x lower seed variance on 6L/7M
- **L4 GPU:** -0.043 BPP on 11L/22M (MUCH WORSE)
- **Root cause:** James-Stein shrinkage helps in extreme noise (8K batch). At 32K+ batch with more data, learned fc is strictly better
- **Status:** DO NOT USE ON GPU

---

## CRITICAL LESSONS LEARNED

### 1. Test on the TARGET config, not a proxy
Our 6L/7M local proxy had ZERO rank correlation with the 11L/22M GPU config.
Every innovation that helped locally hurt on GPU. Always test on the ACTUAL
model architecture that will run on competition hardware.

### 2. Only batch-invariant techniques transfer
| Transfers ✅ | Doesn't transfer ❌ |
|---|---|
| Tokenizer choice | Regularization strength |
| Activation function | Loss function tuning |
| Head dim / count | Architectural constraints (XSA, VR) |
| Quantization precision | Variance reduction tricks (random fc) |
| Eval stride | Optimizer schedule |
| RoPE base frequency | SWA/EMA |
| Depth/width ratio (with caveats) | Weight decay magnitude |

### 3. The optimizer's curse
With N experiments, selection bias ≈ sqrt(N) × noise_floor.
At 82 experiments and σ=0.007, expected bias = ~0.06 BPP.
Our measured +0.059 improvement was approximately 50% noise.

### 4. Different batch sizes = different training regimes
- 8K batch (Mac): noise-dominated → variance reducers help
- 32K batch (L4): transitional → mixed effects
- 524K batch (8xH100): curvature-dominated → capacity + exploration help

### 5. Underfitting regime principle (LOCAL ONLY)
At ~2000 steps with 8K batch, the model is severely undertrained:
- Any regularization hurts (dropout, label smoothing, stochastic depth, WD)
- Any loss reweighting hurts (focal, MiLe) — "CE purity principle"
- Only additive auxiliary losses help (z-loss)
- More steps always beats less noise (grad accum hurts)
**CAUTION:** This principle is LOCAL-SPECIFIC. At 524K batch it may not hold.

### 6. Attention asymmetry
Sharing Q and K weights (Mahalanobis attention) hurts badly (-0.091).
Q and K serve different roles: Q = "what am I looking for", K = "what do I contain".
This IS batch-invariant — the asymmetry is architectural.

### 7. Weights are dense at 7M params
Monarch compression and magnitude pruning both catastrophically fail.
At 7M params, every weight carries unique information. No redundancy to exploit.
This is also batch-invariant — it's about model capacity.

---

## EXPERIMENT RESULTS SUMMARY (82 hypotheses)

### Architecture (batch-invariant results)
| Technique | Effect (iso-step) | Confidence | Transfer? |
|-----------|------------------|------------|-----------|
| NUM_HEADS=4 (head_dim=128) | +0.072 | High | ✅ Likely |
| UNIQUE_BLOCKS=3 (weight sharing) | +0.029 | High | ✅ Likely |
| relu² > SwiGLU | +0.015 | High | ✅ Yes |
| Full MHA > GQA | +0.042 | High | ✅ Yes |
| Keep skip connections | +0.034 | High | ✅ Yes |
| Shared QK weights | -0.091 | High | ✅ (don't do it) |
| dim=512 > dim=384 iso-step | +0.036 | High | ✅ Yes |

### Architecture (uncertain transfer)
| Technique | Effect (iso-step) | Transfer? |
|-----------|------------------|-----------|
| XSA (all layers) | +0.073 | ❓ Failed on L4, retest on 22M |
| XSA (partial, last 2/6) | +0.075 | ❓ Untested on 22M |
| Value Residual | +0.027 | ❓ Regularization in disguise? |
| AttnRes (9L+ only) | +0.111 | ❓ Needs 9L unique, depth-dependent |
| DWA | +0.041 (non-iso) | ❌ Redundant with XSA |

### Loss/Training (batch-dependent — LOCAL ONLY)
| Technique | Local Effect | GPU Effect |
|-----------|-------------|------------|
| Z-loss 1e-4 | +0.005 | Unknown (may be redundant with WD) |
| Softcap 50 (with z-loss) | +0.006 | Unknown |
| Label smoothing 0.1 | -0.054 | Skip |
| Focal loss | -0.001 to -0.004 | Skip |
| MiLe loss | -0.077 to -0.437 | Skip |
| Stochastic depth | -0.306 | Skip |
| Weight decay (any) | CATASTROPHIC with Muon at 8K | Works at 524K (competition uses 0.04) |
| SWA/EMA | All hurt at 8K | Works at 524K (competition uses it) |
| Random MLP fc | +0.011 at 8K | -0.043 at 32K — DON'T USE |

### Tokenizer (batch-invariant)
| Tokenizer | BPP (22M model) | Transfer? |
|-----------|----------------|-----------|
| sp1024 | 2.723 (at 6000 steps) | Baseline |
| **sp2048** | **1.561** (at 6000 steps) | ✅ **USE THIS** |
| sp4096 | Worse at 7M (embedding too large) | ❓ May work at 22M |

### Eval/Serialization (batch-invariant)
| Technique | Effect | Transfer? |
|-----------|--------|-----------|
| Sliding window stride=256 | +0.032 | ✅ Yes |
| FP16 embeddings | +0.001 | ✅ Yes |
| Int8 PTQ gap | 0.001 | ✅ Yes (negligible) |
| Temperature scaling | Hurts | ✅ (don't do it) |
| Post-hoc pruning | Catastrophic | ✅ (don't do it) |

---

## COMPUTE REFERENCE

| Platform | Steps in 600s | Total FLOPs | Batch |
|----------|--------------|-------------|-------|
| Mac (MLX) | ~2,857 | 3.1 PFLOPs | 8K |
| L4 GPU | ~3,000 | 13.0 PFLOPs | 32K |
| 1x H100 | ~7,500 | 129.8 PFLOPs | 128K |
| 8x H100 | ~7,407 | 768.9 PFLOPs | 786K |

Mac time to match 1xH100: 7 hours. To match 8xH100: 34.6 hours.

---

## GCP INFRASTRUCTURE

- **Project:** pgolf-lmxlab
- **Billing:** 011A81-FE9BAE-0E405B (likelilab.com)
- **GPU quota:** GPUS_ALL_REGIONS=1 (approved). H100/A100 = 0 (re-request March 24)
- **Total spend:** $3.27 of $20 budget
- **VM:** DELETED (no ongoing charges)
- **Terraform:** pgolf/infra/gcp-pgolf/
- **SSH:** `gcloud compute ssh pgolf-gpu --project=pgolf-lmxlab --zone=us-central1-a --tunnel-through-iap`
- **DELETE:** `gcloud compute instances delete pgolf-gpu --project=pgolf-lmxlab --zone=us-central1-a --quiet`

---

## GPU SUBMISSION SCRIPTS

**Location:** parameter-golf/records/track_10min_16mb/2026-03-21_XSA_VR_ZLoss_sp2048/
- `train_gpt.py` — 1343/1500 lines, PyTorch/CUDA, includes XSA+VR+z-loss+random_fc
- `deploy_runpod.sh` — automated 8xH100 deployment
- `quick_prototype.sh` — 1xA100/H100 prototyping
- `README.md` — configuration and run commands

**IMPORTANT:** The GPU script needs updating based on L4 transfer failure findings.
Remove random_fc. Re-evaluate XSA and VR on GPU directly.

---

## EXPERIMENT QUEUE (batch-invariant, run on 22M config)

Use multi-checkpoint learning curves (log every 1000 steps) with 2+ seeds.

**Done:**
1. ✅ sp2048 vs sp1024 on 22M — sp2048 wins by +1.16 BPP

**TODO (high confidence transfer):**
2. ⬜ Head dim: 4h/4kv (dim=128) vs 8h/4kv (dim=64) on 22M
3. ⬜ RoPE base: 10K vs 100K
4. ⬜ Activation: relu² vs SwiGLU on 22M (iso-params)
5. ⬜ sp4096 on 22M (enough capacity for larger embedding?)

**TODO (medium confidence, iso-step on 22M):**
6. ⬜ XSA on 22M at 200 iso-steps
7. ⬜ Value Residual on 22M at 200 iso-steps
8. ⬜ AttnRes on 22M at 200 iso-steps (needs 9 unique layers)
9. ⬜ GQA ratio: 8h/4kv vs 8h/8kv on 22M

---

## CROSS-DISCIPLINARY INSIGHTS

### From random matrix theory (JL lemma)
Random MLP fc freezing works at small scale (James-Stein shrinkage) but fails
at larger scale (more data → learned features worth the estimation cost).
The boundary depends on the noise-to-signal ratio of the gradient estimator.

### From estimation theory (James-Stein)
Fewer estimated parameters reduces variance but introduces bias. The optimal
tradeoff depends on sample size (batch × steps). At 8K×2000 = 16M samples,
shrinkage helps. At 32K×3000 = 96M samples, it hurts.

### From numerical PDE (multigrid)
Progressive training (short→long sequences) doesn't transfer because different
sequence lengths learn DIFFERENT operators, not different resolutions of the
same operator. Multigrid requires operator invariance across scales.

### From coding theory
Context mixing (combining transformer + unigram predictions) doesn't help
because the unigram prior is too uninformative. Would need a strong secondary
model (trained bigram/trigram) to provide meaningful signal.

### From information theory (rate-distortion)
At 7M params, weights are at the information-theoretic minimum — no redundancy
for compression (Monarch, pruning). More params via quantization is the path,
not better compression of existing params.

### From optimization theory (bias-variance)
In the undertrained regime (~2000 steps), bias (underfitting) dominates
variance (gradient noise). More steps always beats cleaner steps. This
flips at larger batch sizes where variance is already low.

---

## FILES IN THIS DIRECTORY

```
pgolf/
├── KNOWLEDGE.md          ← This file
├── recipes/
│   ├── pgolf_autorun.py  ← Main autonomous optimization loop
│   ├── pgolf_optuna.py   ← Bayesian hyperparameter search
│   └── autorun_template.py
├── scripts/
│   ├── run_batch_invariant.sh  ← Batch-invariant experiment runner
│   ├── monarch_compress.py     ← Post-hoc weight compression analysis
│   ├── autorun.sh
│   └── watcher.sh
├── infra/
│   └── gcp-pgolf/        ← Terraform + deployment scripts
├── experiments/
│   ├── pgolf_results.jsonl     ← All experiment results (82+ runs)
│   ├── batch_invariant/        ← 22M config comparison results
│   ├── optuna_pgolf.db
│   └── pgolf_scripts/          ← Generated training scripts
├── tests/
│   └── test_autorun.py
└── logs/
```
