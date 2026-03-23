---
name: scaling-book-notes
description: Key formulas and insights from JAX ML Scaling Book for pgolf 8xH100 training
type: reference
---

# Scaling Book Notes (jax-ml.github.io/scaling-book/)

## Key Formulas

### FLOP Counting
- **Per-token FLOPs = 6 × num_params** (forward + backward)
- Our 22M model: 6 × 22M = 132M FLOPs/token
- SwiGLU has 3 matrices per FFW block (not 2): d_model × d_ff × 3 × L

### Memory Requirements
- **Params + optimizer**: 10 bytes/param (2 bf16 + 8 fp32 Adam moments)
- Our 22M model: ~220MB total (params + optimizer)
- Activations per layer: 2 × B × (D + 2F) bytes (with gradient checkpointing)

### Compute-Bound Thresholds
- DP compute-bound when: B/X > C/W (batch per device > compute/bandwidth ratio)
- H100: C = 990 TFLOPs/s (bf16), W = 3.35 TB/s HBM bandwidth
- Alpha = C/W = 990e12 / 3.35e12 ≈ 296 FLOPs/byte

### Roofline Model
- Time = max(FLOPs / peak_FLOPs, bytes_moved / peak_bandwidth)
- If arithmetic_intensity > alpha: compute-bound
- If arithmetic_intensity < alpha: memory-bound
- Matmul intensity: min(M,N,K)/2 for bf16

## H100 Specs
| Spec | Value |
|------|-------|
| bf16 TFLOPs/s | 990 |
| fp8 TFLOPs/s | 1,980 |
| HBM | 80GB |
| HBM bandwidth | 3.35 TB/s |
| NVLink bandwidth | 900 GB/s |
| SMs | 132 |
| L2 cache | 50MB |
| SMEM per SM | 256KB |

## Pgolf-Specific Insights

### Our Setup (8xH100, 22M params)
- Model fits in single GPU (220MB << 80GB)
- **Pure data parallelism is optimal** — no need for TP or pipeline
- Each GPU processes 524K/8 = 65.5K tokens per step
- MFU ≈ 0.006% — extremely memory-bandwidth-bound at this tiny model size
- Training is dominated by memory reads, not compute
- **Key optimization: maximize memory bandwidth utilization, not FLOPs**

### Why Small Models Are Memory-Bound
- Matmul intensity for our FFW: d_ff=1024 → intensity = min(512,1024,512)/2 = 256
- H100 roofline crossover: ~296 FLOPs/byte
- 256 < 296, so we're memory-bound even for the largest matmul
- With MLP 3x (d_ff=1536): intensity = 256 (still d_model-limited)
- Batch helps: at batch=64K tokens, effective intensity >> crossover → COMPUTE-BOUND
- **At competition batch (524K), we're compute-bound on most ops** ✓

### Implications for Architecture Choices
1. Wider model (larger d_model) → higher arithmetic intensity → better GPU utilization
2. More layers same width → same intensity → more total compute, same utilization
3. MLP 3x vs 2x → more FLOPs per layer → same intensity
4. Weight sharing: reduces model size → faster reads → might help memory-bound regime
5. Int6/Int8 quantization: halves param reads → 2x speedup for memory-bound ops

## Further Reading (Most Relevant)
- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) — GPU performance first principles
- [Stanford CS336](https://stanford-cs336.github.io/spring2025/) — LLM training course
- [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — PyTorch parallelism
- [Stas Bekman ML Engineering](https://github.com/stas00/ml-engineering) — Practical infra
- [How to Optimize CUDA Matmul](https://siboehm.com/articles/22/CUDA-MMM) — Kernel optimization
