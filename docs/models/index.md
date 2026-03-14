# Model Architectures

lmxlab implements 24 architectures as **config factories** — functions that return a `ModelConfig`. The same `LanguageModel` class handles all of them through `ConfigurableBlock`.

## Architecture Comparison

| Architecture | Attention | FFN | Norm | Position | Bias | KV Heads | Special |
|---|---|---|---|---|---|---|---|
| **GPT** | MHA | Standard | LayerNorm | Sinusoidal | Yes | = n_heads | Baseline |
| **LLaMA** | GQA | Gated (SwiGLU) | RMSNorm | RoPE | No | < n_heads | — |
| **Gemma** | GQA (MQA) | Gated | RMSNorm | RoPE | No | 1 | Tied embeddings |
| **Gemma 3** | SlidingWindowGQA + GQA | Gated | RMSNorm | RoPE | No | < n_heads | Sliding window |
| **Qwen** | GQA | Gated | RMSNorm | RoPE (θ=1M) | Yes | < n_heads | High RoPE theta |
| **Qwen 3 MoE** | GQA | SharedExpertMoE | RMSNorm | RoPE | No | < n_heads | 64 experts, top-8 |
| **Qwen 3.5** | DeltaNet + GQA | Gated | RMSNorm | Conv + RoPE | No | < n_heads | 3:1 hybrid |
| **Qwen-Next** | GatedGQA | Gated | RMSNorm | RoPE | No | < n_heads | Sigmoid output gate |
| **Mixtral** | GQA | Gated (MoE) | RMSNorm | RoPE (θ=1M) | No | < n_heads | 8 experts, top-2 |
| **DeepSeek V2** | MLA | Gated | RMSNorm | Decoupled RoPE | No | Latent | KV compression |
| **DeepSeek V3** | MLA | SharedExpertMoE | RMSNorm | Decoupled RoPE | No | Latent | MLA + MoE |
| **Nemotron** | Mamba-2 + GQA | ReLU² / MoE | RMSNorm | RoPE | No | < n_heads | M/E/* hybrid pattern |
| **Llama 4 Scout** | ChunkedGQA | SharedExpertMoE | RMSNorm | iRoPE | No | < n_heads | Chunked attn + NoPE |
| **Llama 4 Maverick** | ChunkedGQA | SharedExpertMoE | RMSNorm | iRoPE | No | < n_heads | 128 experts, top-1 |
| **Mistral Small** | SlidingWindowGQA | Gated | RMSNorm | RoPE | No | < n_heads | All-local window |
| **OLMo 2** | GQA | Gated | RMSNorm | RoPE | No | < n_heads | QK-norm |
| **GPT-OSS** | GQA | Gated | RMSNorm | RoPE | No | < n_heads | QK-norm, tied embs |
| **Grok** | GQA | SharedExpertMoE | RMSNorm | RoPE | No | < n_heads | 8 experts + shared |
| **Kimi K2.5** | DeltaNet + GQA | SharedExpertMoE | RMSNorm | Conv + RoPE | No | < n_heads | 128 experts + shared |
| **SmolLM3** | GQA | Gated | RMSNorm | iRoPE | No | < n_heads | RoPE + NoPE layers |
| **Falcon H1** | Mamba-2 + GQA | Gated | RMSNorm | RoPE | No | < n_heads | Hybrid M/*/* pattern |
| **Jamba** | Mamba-2 + GQA | Gated / MoE | RMSNorm | RoPE | No | < n_heads | MMMA + MoE alternation |
| **Bamba** | Mamba-2 + GQA | Gated | RMSNorm | RoPE | No | < n_heads | IBM hybrid M/*/* |
| **GLM-4.5** | MLA | Gated | RMSNorm | NoPE | No | Latent | MLA without RoPE |

## GPT

The baseline architecture. Standard multi-head attention, LayerNorm, and sinusoidal positional encoding.

```python
from lmxlab.models.gpt import gpt_config

config = gpt_config()
# attention="mha", norm="layer_norm", ffn="standard"
# position="sinusoidal", bias=True
```

**Key characteristics:** Uses bias in all linear layers, pre-norm LayerNorm (matching GPT-2), and the only architecture with standard (non-gated) FFN.

## LLaMA

The modern open-source baseline. Grouped-query attention for memory efficiency, RMSNorm for speed, and SwiGLU FFN.

```python
from lmxlab.models.llama import llama_config

config = llama_config()
# attention="gqa", norm="rms_norm", ffn="gated"
# position="rope", bias=False, n_kv_heads=8
```

**Key characteristics:** No bias anywhere (simplifies the model), GQA with 8 KV heads sharing across 32 query heads, RoPE for position encoding.

## Gemma

Google's efficient variant with multi-query attention (single KV head) and tied input/output embeddings.

```python
from lmxlab.models.gemma import gemma_config

config = gemma_config()
# n_kv_heads=1 (multi-query), tie_embeddings=True
```

**Key insight:** When `n_kv_heads=1`, GQA becomes Multi-Query Attention (MQA). All query heads share a single set of keys and values.

## Qwen

Alibaba's architecture with high RoPE theta for long context and bias in QKV projections.

```python
from lmxlab.models.qwen import qwen_config

config = qwen_config()
# rope_theta=1_000_000.0, bias=True
```

**Key insight:** Higher RoPE theta extends the effective context window by changing the frequency spectrum of positional encodings.

## Mixtral (MoE)

Sparse Mixture of Experts — routes each token to 2 of 8 expert FFNs.

```python
from lmxlab.models.mixtral import mixtral_config

config = mixtral_config()
# Uses MoEFFN: 8 experts, top-2 routing
```

**Key insight:** MoE increases model capacity without proportionally increasing compute — each token only uses 2/8 of the FFN parameters.

## DeepSeek V2 (MLA)

Multi-Head Latent Attention compresses KV representations into a low-rank latent space, dramatically reducing KV cache size (~57x vs MHA).

```python
from lmxlab.models.deepseek import deepseek_config

config = deepseek_config()
# attention="mla", kv_lora_rank=512, rope_dim=64
# q_lora_rank=1536
```

**How MLA works:**

1. **Down-project** KV from `d_model` → `kv_lora_rank` (+ `rope_dim` for shared RoPE key)
2. **Cache** only the compressed latent (not full K, V)
3. **Up-project** latent → multi-head K and V at attention time
4. **Decoupled RoPE**: position info kept in a separate single-head key

Cache per token: `kv_lora_rank + rope_dim = 576` vs `2 × n_heads × head_dim = 32,768` for MHA.

## Gemma 3 (Interleaved Attention)

Mixes sliding window (local) and global attention layers. Most layers use a fixed window; every Nth layer attends to the full sequence.

```python
from lmxlab.models.gemma3 import gemma3_config

config = gemma3_config()
# Every 6th layer: global GQA (5:1 local:global ratio)
# Other layers: sliding_window_gqa with window_size=4096
# (Real Gemma 3 uses window_size=1024; adjust to match)
```

**Key insight:** Local attention is O(n × w) instead of O(n²), making long sequences tractable. Periodic global layers maintain long-range dependencies. Uses per-layer `block_configs` — a direct showcase of the ConfigurableBlock system's flexibility.

## Qwen 3.5 (Hybrid DeltaNet + GQA)

The most architecturally novel model: interleaves **Gated DeltaNet** (linear attention with delta rule) and standard **GQA** layers in a 3:1 ratio.

```python
from lmxlab.models.qwen35 import qwen35_config

config = qwen35_config()
# 75% gated_deltanet layers + 25% gqa layers
# DeltaNet: causal conv, no RoPE, fixed-size state
# GQA: standard with RoPE, growing KV cache
```

**How Gated DeltaNet works:**

1. **Delta rule**: State matrix S predicts v from k, then corrects itself based on prediction error: `S = α·S - β·(S@k - v)@k^T`
2. **Decay gate (α)**: Learned selective forgetting — when to discard old context
3. **Update gate (β)**: Learned correction strength — how much to trust new information
4. **Fixed-size state**: O(d²) per token regardless of sequence length (vs O(n) for KV cache)
5. **Short causal convolutions**: Replace RoPE for local context in DeltaNet layers

**Key insight:** Pure linear attention loses expressiveness by compressing all history into a fixed-size state. The hybrid 3:1 pattern preserves efficient long-context processing (DeltaNet) while periodic GQA layers provide global attention for tasks that need it.

## Qwen-Next (Gated Attention)

Uses **GatedGQA** — GQA with a learned sigmoid gate on the attention output. The gate modulates how much attention information passes through, improving gradient flow and representational capacity.

```python
from lmxlab.models.qwen_next import qwen_next_config

config = qwen_next_config()
# attention="gated_gqa": y = attn_out * sigmoid(W_gate @ x)
```

**Key insight:** The output gate (G1 elementwise variant from arXiv:2505.06708) adds minimal parameters but measurably improves training dynamics by giving the model a learned bypass around the attention mechanism.

## DeepSeek V3 (MLA + MoE)

Extends DeepSeek V2's MLA attention with SharedExpertMoE FFN layers, combining KV compression with sparse expert routing.

```python
from lmxlab.models.deepseek import deepseek_v3_config

config = deepseek_v3_config()
# MLA attention + SharedExpertMoE (256 routed experts + 1 shared)
# Sigmoid routing with bias correction
```

## Nemotron (Hybrid Mamba-Transformer MoE)

NVIDIA's hybrid architecture mixing three layer types encoded in a pattern string: **M** (Mamba-2 SSD), **E** (LatentMoE), and **\*** (standard attention + dense FFN with squared ReLU).

```python
from lmxlab.models.nemotron import nemotron3_config

config = nemotron3_config()
# hybrid_override_pattern: "M*M*M*M*MEMEMEMEMEMEMEMEMEME..."
# Mamba layers for sequence mixing, MoE for capacity
```

**Key insight:** Different layer types handle different aspects — Mamba-2 for efficient sequence mixing, attention for precise retrieval, and LatentMoE for routing tokens to specialized experts with reduced dimensionality.

## Llama 4 Scout (iRoPE + Chunked Attention)

Uses the **iRoPE** pattern: most layers use chunked local attention with RoPE positions resetting at chunk boundaries, while every 4th layer uses global GQA without positional encoding (NoPE) for cross-chunk information flow. All layers use SharedExpertMoE FFN.

```python
from lmxlab.models.llama4 import llama4_scout_config

config = llama4_scout_config()
# 75% chunked_gqa (8192 chunk size, local RoPE)
# 25% gqa (no position encoding — NoPE layers)
# All layers: SharedExpertMoE (16 experts + 1 shared)
```

## Mistral Small (Sliding Window)

All layers use sliding-window GQA with a fixed window size. Unlike Gemma 3's mixed approach, Mistral Small applies local attention uniformly — no global layers.

```python
from lmxlab.models.mistral import mistral_small_config

config = mistral_small_config()
# All layers: sliding_window_gqa with window_size=4096
```

## OLMo 2 (QK-Norm)

AllenAI's architecture adds **QK-norm** (per-head RMSNorm on Q and K after projection, before RoPE) to the standard LLaMA-like template. This stabilizes training at scale.

```python
from lmxlab.models.olmo import olmo2_config

config = olmo2_config()
# Standard GQA + QK-norm (per-head RMSNorm on Q, K)
```

## GPT-OSS (QK-Norm + Tied Embeddings)

OpenAI's open-source GPT uses a LLaMA-like architecture with QK-norm and tied input/output embeddings.

```python
from lmxlab.models.gpt_oss import gpt_oss_config

config = gpt_oss_config()
# GQA + QK-norm, tied embeddings, no bias
```

## Grok (SharedExpertMoE)

xAI's architecture uses GQA attention with SharedExpertMoE FFN in every layer. All layers are homogeneous (no hybrid pattern).

```python
from lmxlab.models.grok import grok_config

config = grok_config()
# GQA + SharedExpertMoE (8 routed + 1 shared)
```

## Kimi K2.5 (DeltaNet + MoE)

Moonshot AI's hybrid: interleaves GQA and Gated DeltaNet attention layers (every 4th layer is DeltaNet), all with SharedExpertMoE FFN (128 experts).

```python
from lmxlab.models.kimi import kimi_config

config = kimi_config()
# 75% gqa + 25% gated_deltanet layers
# All layers: SharedExpertMoE (128 routed + 1 shared)
```

## SmolLM3 (iRoPE)

HuggingFace's efficient model uses the iRoPE pattern: most layers use GQA with RoPE, while every 4th layer uses GQA without positional encoding (NoPE) for long-range information flow.

```python
from lmxlab.models.smollm import smollm3_config

config = smollm3_config()
# 75% gqa with RoPE + 25% gqa without position encoding
# Tied embeddings
```

## Qwen 3 MoE

Alibaba's MoE variant of Qwen 3 with GQA attention and SharedExpertMoE FFN (64 routed experts, top-8 routing, plus 1 shared expert).

```python
from lmxlab.models.qwen import qwen3_moe_config

config = qwen3_moe_config()
# GQA + SharedExpertMoE (64 experts, top_k=8, 1 shared)
```

## Llama 4 Maverick (iRoPE + 128 Experts)

The larger Llama 4 variant with the same iRoPE pattern as Scout but with 128 routed experts and top-1 routing.

```python
from lmxlab.models.llama4 import llama4_maverick_config

config = llama4_maverick_config()
# Same iRoPE pattern as Scout
# SharedExpertMoE (128 routed + 1 shared, top_k=1)
```

## Falcon H1 (Hybrid Mamba-2 + GQA)

TII's hybrid architecture with most layers using Mamba-2 SSM and periodic GQA attention layers for global context. All layers use GatedFFN (SwiGLU).

```python
from lmxlab.models.falcon import falcon_h1_config

config = falcon_h1_config()
# hybrid_pattern: "MMMM*MMM*MMM*MMM*"
# Mamba-2 layers (M) + GQA attention layers (*)
```

## Jamba (Mamba-2 + GQA + MoE)

AI21's hybrid using an MMMA pattern (3 Mamba-2 + 1 GQA per cycle) with MoE FFN on alternating attention layers.

```python
from lmxlab.models.jamba import jamba_config

config = jamba_config()
# MMMA pattern: 3 Mamba-2 + 1 GQA per cycle
# MoE (16 experts, top-2) on even attention layers
```

## Bamba (Hybrid Mamba-2 + GQA)

IBM's hybrid variant similar to Falcon H1. Mamba-2 layers handle sequence mixing with periodic GQA layers for global attention.

```python
from lmxlab.models.bamba import bamba_config

config = bamba_config()
# hybrid_pattern: "MMMM*MMM*MMM*MMM*"
# Similar to Falcon H1 with different hyperparameters
```

## GLM-4.5 (MLA NoPE)

Zhipu AI's architecture using MLA attention with `rope_dim=0` (no positional encoding). Relies entirely on learned attention patterns for position information.

```python
from lmxlab.models.glm import glm45_config

config = glm45_config()
# MLA attention with rope_dim=0 (NoPE)
# KV compression via kv_lora_rank=512
```

**Key insight:** Disabling RoPE in MLA removes the decoupled RoPE key entirely, letting the model learn position-dependent patterns implicitly through the latent representations.

## Loading Pretrained Weights

lmxlab can load pretrained weights from HuggingFace Hub for LLaMA, Gemma, Qwen2, and Mistral models.

### Quick load (requires `huggingface_hub`)

```python
from lmxlab.models.convert import load_from_hf

model, config = load_from_hf('meta-llama/Llama-3.2-1B')
```

### Manual conversion

If you have weights and config locally:

```python
import json
from lmxlab.models.convert import config_from_hf, convert_weights
from lmxlab.models.base import LanguageModel
import mlx.core as mx

# Load HF config.json
hf_config = json.loads(open('config.json').read())
model_config = config_from_hf(hf_config)

# Load and convert weights
hf_weights = mx.load('model.safetensors')
lmt_weights = convert_weights(hf_weights, 'llama')

# Build model and load
model = LanguageModel(model_config)
model.load_weights(list(lmt_weights.items()))
```

### Weight name mapping

The conversion handles the naming differences between HF and lmxlab:

| HuggingFace | lmxlab |
|---|---|
| `model.embed_tokens.weight` | `embed.weight` |
| `model.layers.{i}.self_attn.q_proj.weight` | `blocks.{i}.attention.q_proj.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `blocks.{i}.ffn.gate.weight` |
| `model.layers.{i}.mlp.up_proj.weight` | `blocks.{i}.ffn.up.weight` |
| `model.layers.{i}.mlp.down_proj.weight` | `blocks.{i}.ffn.down.weight` |
| `model.layers.{i}.input_layernorm.weight` | `blocks.{i}.attn_norm.weight` |
| `model.layers.{i}.post_attention_layernorm.weight` | `blocks.{i}.ffn_norm.weight` |
| `model.norm.weight` | `final_norm.weight` |
| `lm_head.weight` | `head.weight` |

## Quantization

lmxlab supports post-training quantization via MLX's native affine quantization. This replaces `nn.Linear` with `nn.QuantizedLinear`, reducing memory by ~4-8x.

```python
from lmxlab.core.quantize import quantize_model, dequantize_model

# Quantize to 4-bit (default)
quantize_model(model, bits=4, group_size=64)

# Or load from HF already quantized
model, config = load_from_hf('meta-llama/Llama-3.2-1B', quantize=4)

# Dequantize back to float for fine-tuning
dequantize_model(model)
```

| Bits | Memory reduction | Quality | Use case |
|------|-----------------|---------|----------|
| 8 | ~4x | Near-lossless | Fine-tuning, high-quality inference |
| 4 | ~8x | Good | Inference, fitting large models in memory |

## LoRA (Low-Rank Adaptation)

Fine-tune pretrained models efficiently by training only small low-rank matrices instead of all weights. Reduces trainable parameters by 10-100x.

```python
from lmxlab.core.lora import apply_lora, merge_lora

# Apply LoRA to attention layers (rank=8)
apply_lora(model, rank=8, targets=['attention'])

# Train — only LoRA params are trainable (~0.1% of total)
trainer = Trainer(model, train_config)
trainer.train(data)

# Merge LoRA back into base weights for inference
merge_lora(model)
```

**How it works:** Each targeted `nn.Linear` is replaced with `LoRALinear`, which computes `y = xW^T + scaling * x @ A @ B^T`. Matrix B is zero-initialized, so the model starts with the same output as the base model. Only A and B are trainable; W is frozen.

**Targets:**
- `'attention'` — q/k/v/o projections
- `'ffn'` — gate/up/down projections

## QLoRA (Quantized LoRA)

Combine quantization and LoRA for maximum memory efficiency. Keep base weights in 4-bit quantized form while training full-precision LoRA adapters. This lets you fine-tune models that nearly fill available memory.

```python
from lmxlab.core.quantize import quantize_model
from lmxlab.core.qlora import apply_qlora

# Quantize base model to 4-bit
quantize_model(model, bits=4)

# Apply LoRA on top of quantized layers
apply_qlora(model, rank=8, targets=['attention'])

# Train — only LoRA params are trainable
trainer = Trainer(model, train_config)
trainer.train(data)
```

**How it works:** Each targeted `nn.QuantizedLinear` is replaced with `LoRAQuantizedLinear`, which uses `mx.quantized_matmul` for the frozen base computation and adds a full-precision low-rank update: `y = quantized_matmul(x, W_q) + scaling * x @ A @ B^T`.

**When to use QLoRA vs LoRA:**

| Approach | Base weights | Memory | Use case |
|----------|-------------|--------|----------|
| LoRA | Float16 | Full model + LoRA | Plenty of memory |
| QLoRA | 4-bit quantized | ~25% of full + LoRA | Large models, tight memory |

## Creating a Tiny Model

Every architecture has a `_tiny()` factory for testing:

```python
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.deepseek import deepseek_tiny

config = gpt_tiny()       # d_model=64, 2 layers, 4 heads
config = deepseek_tiny()  # d_model=64, 2 layers, kv_lora_rank=16
```

These use small dimensions (d_model ≤ 128, n_layers ≤ 4, vocab ≤ 1024) to enable fast unit testing and quick experiments.
