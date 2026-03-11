# Model Architectures

lmt-metal implements eight transformer architectures as **config factories** — functions that return a `ModelConfig`. The same `LanguageModel` class handles all of them through `ConfigurableBlock`.

## Architecture Comparison

| Architecture | Attention | FFN | Norm | Position | Bias | KV Heads | Special |
|---|---|---|---|---|---|---|---|
| **GPT** | MHA | Standard | LayerNorm | Sinusoidal | Yes | = n_heads | Baseline |
| **LLaMA** | GQA | Gated (SwiGLU) | RMSNorm | RoPE | No | < n_heads | — |
| **Gemma** | GQA (MQA) | Gated | RMSNorm | RoPE | No | 1 | Tied embeddings |
| **Qwen** | GQA | Gated | RMSNorm | RoPE (θ=1M) | Yes | < n_heads | High RoPE theta |
| **Mixtral** | GQA | Gated (MoE) | RMSNorm | RoPE (θ=1M) | No | < n_heads | 8 experts, top-2 |
| **DeepSeek V2** | MLA | Gated | RMSNorm | Decoupled RoPE | No | Latent | KV compression |
| **Gemma 3** | Mixed | Gated | RMSNorm | RoPE | No | < n_heads | Sliding window |
| **Qwen 3.5** | Hybrid | Gated | RMSNorm | Conv + RoPE | No | < n_heads | DeltaNet + GQA |

## GPT

The baseline architecture. Standard multi-head attention, LayerNorm, and sinusoidal positional encoding.

```python
from lmt_metal.models.gpt import gpt_config

config = gpt_config()
# attention="mha", norm="layer_norm", ffn="standard"
# position="sinusoidal", bias=True
```

**Key characteristics:** Uses bias in all linear layers, post-LayerNorm (though our default is pre-norm), and the only architecture with standard (non-gated) FFN.

## LLaMA

The modern open-source baseline. Grouped-query attention for memory efficiency, RMSNorm for speed, and SwiGLU FFN.

```python
from lmt_metal.models.llama import llama_config

config = llama_config()
# attention="gqa", norm="rms_norm", ffn="gated"
# position="rope", bias=False, n_kv_heads=8
```

**Key characteristics:** No bias anywhere (simplifies the model), GQA with 8 KV heads sharing across 32 query heads, RoPE for position encoding.

## Gemma

Google's efficient variant with multi-query attention (single KV head) and tied input/output embeddings.

```python
from lmt_metal.models.gemma import gemma_config

config = gemma_config()
# n_kv_heads=1 (multi-query), tie_embeddings=True
```

**Key insight:** When `n_kv_heads=1`, GQA becomes Multi-Query Attention (MQA). All query heads share a single set of keys and values.

## Qwen

Alibaba's architecture with high RoPE theta for long context and bias in QKV projections.

```python
from lmt_metal.models.qwen import qwen_config

config = qwen_config()
# rope_theta=1_000_000.0, bias=True
```

**Key insight:** Higher RoPE theta extends the effective context window by changing the frequency spectrum of positional encodings.

## Mixtral (MoE)

Sparse Mixture of Experts — routes each token to 2 of 8 expert FFNs.

```python
from lmt_metal.models.mixtral import mixtral_config

config = mixtral_config()
# Uses MoEFFN: 8 experts, top-2 routing
```

**Key insight:** MoE increases model capacity without proportionally increasing compute — each token only uses 2/8 of the FFN parameters.

## DeepSeek V2 (MLA)

Multi-Head Latent Attention compresses KV representations into a low-rank latent space, dramatically reducing KV cache size (~57x vs MHA).

```python
from lmt_metal.models.deepseek import deepseek_config

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
from lmt_metal.models.gemma3 import gemma3_config

config = gemma3_config()
# Every 6th layer: global GQA
# Other layers: sliding_window_gqa with window_size=4096
```

**Key insight:** Local attention is O(n × w) instead of O(n²), making long sequences tractable. Periodic global layers maintain long-range dependencies. Uses per-layer `block_configs` — a direct showcase of the ConfigurableBlock system's flexibility.

## Qwen 3.5 (Hybrid DeltaNet + GQA)

The most architecturally novel model: interleaves **Gated DeltaNet** (linear attention with delta rule) and standard **GQA** layers in a 3:1 ratio.

```python
from lmt_metal.models.qwen35 import qwen35_config

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

## Creating a Tiny Model

Every architecture has a `_tiny()` factory for testing:

```python
from lmt_metal.models.gpt import gpt_tiny
from lmt_metal.models.deepseek import deepseek_tiny

config = gpt_tiny()       # d_model=64, 2 layers, 4 heads
config = deepseek_tiny()  # d_model=64, 2 layers, kv_lora_rank=16
```

These use small dimensions (d_model ≤ 128, n_layers ≤ 4, vocab ≤ 1024) to enable fast unit testing and quick experiments.
