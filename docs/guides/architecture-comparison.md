# Architecture Comparison Guide

lmxlab implements **24 architectures** from a shared
`BlockConfig` + `ModelConfig` system. Every model is
assembled from registry components — swap attention, FFN,
normalization, and position encoding independently.

## Quick Reference Table

| Architecture | Attention | Norm | FFN | Position | Key Feature |
|---|---|---|---|---|---|
| **GPT** | MHA | LayerNorm | Standard | Sinusoidal | Classic baseline |
| **LLaMA** | GQA | RMSNorm | Gated (SwiGLU) | RoPE | No bias |
| **Gemma** | GQA | RMSNorm | Gated (GeGLU) | RoPE | Tied embeddings |
| **Gemma-3** | SW-GQA + GQA | RMSNorm | Gated | RoPE | Interleaved sliding window |
| **Qwen** | GQA | RMSNorm | Gated (SwiGLU) | RoPE | Bias in QKV, high theta |
| **Qwen-3.5** | DeltaNet + GQA | RMSNorm | Gated | RoPE/None | 3:1 linear:attention hybrid |
| **Mistral** | SW-GQA | RMSNorm | Gated | RoPE | 4K sliding window |
| **Mixtral** | GQA | RMSNorm | MoE (8 experts) | RoPE | Sparse MoE |
| **OLMo-2** | GQA + QK-norm | RMSNorm | Gated | RoPE | Per-head RMSNorm on Q/K |
| **GPT-OSS** | GQA + QK-norm | RMSNorm | Gated | RoPE | Tied embeddings |
| **DeepSeek** | MLA | RMSNorm | Gated | RoPE | Low-rank KV compression |
| **DeepSeek-V3** | MLA | RMSNorm | Shared MoE | RoPE | MLA + MoE |
| **Qwen-Next** | Gated GQA | RMSNorm | Gated | RoPE | Sigmoid output gate |
| **Grok** | GQA | RMSNorm | Shared MoE | RoPE | 8 experts, top-2 |
| **GLM-4.5** | MLA | RMSNorm | Gated | None | No position encoding |
| **SmolLM-3** | GQA | RMSNorm | Gated | RoPE/None | iRoPE (3:1) |
| **LLaMA-4 Scout** | Chunked + GQA | RMSNorm | Shared MoE | RoPE/None | iRoPE + MoE |
| **LLaMA-4 Maverick** | Chunked + GQA | RMSNorm | Shared MoE | RoPE/None | 128 experts |
| **Nemotron** | GQA + Mamba-2 | RMSNorm | LatentMoE | RoPE/None | Complex hybrid |
| **Kimi** | GQA + DeltaNet | RMSNorm | Shared MoE | RoPE/None | Linear attention hybrid |
| **Qwen-3-MoE** | GQA | RMSNorm | Shared MoE | RoPE | 64 experts, top-8 |
| **Falcon-H1** | Mamba-2 + GQA | RMSNorm | Gated | None/RoPE | SSM hybrid (MMM\*) |
| **Jamba** | Mamba-2 + GQA | RMSNorm | Gated + MoE | None/RoPE | SSM + MoE hybrid |
| **Bamba** | Mamba-2 + GQA | RMSNorm | Gated | None/RoPE | SSM hybrid (MMM\*) |

## Attention Mechanisms

**Standard MHA** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)):
All heads share the same number of Q, K, V projections. Simple,
well-understood.

**Grouped Query Attention** ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)):
Fewer KV heads than Q heads. Reduces KV cache size
proportionally. Most modern architectures use GQA.

**Multi-head Latent Attention** (DeepSeek, GLM): Projects KV into
a low-rank latent space. ~28x KV cache reduction. Decoupled RoPE
applied only to a subset of dimensions.

**Sliding Window** (Mistral, Gemma-3): Each token attends only to
a fixed window of past tokens. O(n * w) instead of O(n^2).

**Gated DeltaNet** (Qwen-3.5, Kimi): Linear attention variant
with data-dependent gating. O(n * d) complexity.

**Mamba-2 SSM** ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)):
Structured state space model. No explicit attention matrix —
uses selective scan over a recurrent state. O(n * d) with
hardware-efficient chunked scan.

## Normalization

**LayerNorm** (GPT only): Subtracts mean AND divides by std.
The mean subtraction provides implicit regularization.

**RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)):
Divides by root mean square only. Faster than LayerNorm, no mean
subtraction. Used by every architecture except classic GPT.

**QK-Norm** (OLMo-2, GPT-OSS): Additional per-head RMSNorm on
Q and K projections after reshape, before RoPE. Stabilizes
attention logits.

## Feed-Forward Networks

**Standard**: Two linear layers with GELU activation. `d → d_ff → d`.

**Gated (SwiGLU)** ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)):
Three projections with a gate. `d → gate * up → d`. ~50% more
parameters than standard at same `d_ff`, but typically more
effective.

**Mixture of Experts**: Routes tokens to top-k of N expert FFNs.
Increases model capacity without proportional compute increase.

**Shared MoE**: Combines a shared expert (always active) with
routed experts. Better load balancing than pure MoE.

## Position Encoding

**Sinusoidal** (GPT): Fixed, added to embeddings. Simple but
limits extrapolation.

**RoPE** ([Su et al., 2021](https://arxiv.org/abs/2104.09864)):
Rotation applied to Q and K. Enables relative position awareness
and reasonable extrapolation. The dominant choice.

**None (NoPE)**: No explicit position encoding. The model learns
position from causal attention patterns alone. Used by some layers
in iRoPE and hybrid architectures.

## Decision Tree

**"Which architecture should I study for X?"**

- **Learning transformers**: Start with **GPT** (simplest) then
  compare to **LLaMA** (modern best practices)
- **Understanding attention variants**: **GPT** (MHA) → **LLaMA**
  (GQA) → **DeepSeek** (MLA) → **Mistral** (sliding window)
- **State space models**: **Falcon-H1** or **Bamba** (cleanest
  SSM/attention hybrid pattern)
- **Mixture of Experts**: **Mixtral** (simplest MoE) → **Grok**
  (shared experts) → **DeepSeek-V3** (MoE + MLA)
- **Hybrid architectures**: **Falcon-H1** (SSM+attention) →
  **Jamba** (SSM+attention+MoE) → **Nemotron** (complex hybrid)
- **Research at small scale**: Use `_10m` or `_30m` configs
  (GPT, LLaMA, Falcon-H1, Jamba, Bamba)

## Using Scaled Research Configs

For experiments, use the pre-calibrated research configs:

```python
from lmxlab.models.gpt import gpt_10m, gpt_30m
from lmxlab.models.llama import llama_10m, llama_30m

config = gpt_10m()  # ~9.6M params, BPE vocab
model = LanguageModel(config)
```

All research configs use BPE vocab (50257 via tiktoken GPT-2),
tied embeddings, and `max_seq_len=512`.

| Config | Params | d_model | Layers | Heads |
|---|---|---|---|---|
| `gpt_10m` | ~9.6M | 128 | 16 | 4 |
| `gpt_30m` | ~30.2M | 256 | 22 | 8 |
| `llama_10m` | ~9.9M | 128 | 14 | 4 (2 KV) |
| `llama_30m` | ~30.6M | 256 | 18 | 8 (4 KV) |
| `falcon_h1_10m` | ~9.3M | 128 | 12 | 4 (SSM: 8) |
| `jamba_10m` | ~10.2M | 128 | 12 | 4 (4 experts) |
| `bamba_10m` | ~9.3M | 128 | 12 | 4 (SSM: 8) |

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI Technical Report.
- Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467). NeurIPS 2019.
- Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). arXiv preprint.
- Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). Neurocomputing, 2024.
- Gu et al. (2021). [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396). ICLR 2022.
- Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023.
- Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). arXiv preprint.
- Gu & Dao (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). COLM 2024.
- Dao & Gu (2024). [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). ICML 2024.
- Lieber et al. (2024). [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887). arXiv preprint.
- IBM et al. (2025). [Bamba: Inference-Efficient Hybrid Mamba2 Model](https://huggingface.co/blog/bamba). HuggingFace blog post.
- Zuo et al. (2025). [Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance](https://arxiv.org/abs/2507.22448). arXiv preprint.
