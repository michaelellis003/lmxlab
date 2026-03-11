# Production Optimizations

lmxlab teaches readable implementations. But production systems use
optimizations that can deliver 2-10x speedups. This page explains
**what** those optimizations do, **why** they matter, and **how** they
relate to what lmxlab already implements.

!!! note "Educational, not implemented"
    lmxlab does not implement most of these optimizations. This page
    teaches you how they work so you can understand production systems
    like vLLM, llama.cpp, and mlx-lm.

## Flash Attention

### The problem

Standard attention computes `O = softmax(QK^T / sqrt(d)) * V`. This
materializes the full N x N attention matrix in GPU memory. For a
4096-token sequence in float16, that matrix is 32 MB **per head, per
layer**. The bottleneck is not compute — modern GPUs have enormous
FLOP/s — but **memory bandwidth**: reading and writing these large
intermediate matrices dominates wall-clock time.

### How Flash Attention solves it

Flash Attention (Dao et al., 2022) is an **IO-aware exact attention
algorithm**. "Exact" is critical: it produces mathematically identical
results, just with far fewer memory round-trips.

**Tiling.** Q, K, V are split into blocks that fit in GPU SRAM
(on-chip fast memory, ~20 MB on an A100 vs 40-80 GB HBM). Each block
of Q attends to all blocks of K, V without ever writing the full
N x N matrix to HBM.

**Online softmax.** The key algorithmic insight. Standard softmax
requires knowing `max(x_1, ..., x_N)` across the entire row before
computing any output. Online softmax maintains running statistics —
a running maximum and running denominator — updated incrementally as
each new tile of K is processed. When a new tile produces a new
maximum, previous partial results are rescaled. This makes softmax
associative over tiles.

**IO complexity.** Standard attention requires O(N*d + N^2) HBM
accesses. Flash Attention requires O(N^2 * d^2 * M^{-1}), where M
is SRAM size. For typical d=128, M~100KB, this is many-fold fewer
HBM accesses. The authors prove this is **asymptotically optimal**.

### What lmxlab does instead

lmxlab uses `mx.fast.scaled_dot_product_attention`, which is MLX's
optimized Metal kernel. It supports causal masking and computes
softmax in float32 regardless of input precision. While optimized
for Apple Silicon, it is not fully IO-aware in the Flash Attention
sense — the Metal FlashAttention project demonstrates that Flash
Attention-style tiling is feasible on Apple GPUs.

```python
# lmxlab: uses MLX's optimized kernel
output = mx.fast.scaled_dot_product_attention(
    q, k, v, scale=scale, mask=mask
)
```

**References:**
[FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135),
[FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691),
[FlashAttention-3 (Dao et al., 2024)](https://arxiv.org/abs/2407.08608)

---

## KV Cache Optimizations

### Why KV cache dominates memory

During autoregressive generation, each new token attends to all
previous tokens. The KV cache stores previously computed K and V
tensors so each step only computes projections for the new token.

The problem: KV cache scales as
`2 * n_layers * n_heads * head_dim * seq_len * bytes_per_element`.
For large models at long contexts, the KV cache can exceed the model
weights themselves. For a 70B model at 32K context in FP16, the KV
cache requires over 85 GB.

### PagedAttention (vLLM)

Traditional systems pre-allocate contiguous memory for KV cache
based on maximum sequence length. Since actual lengths vary, this
wastes 60-80% of allocated memory through fragmentation.

PagedAttention (Kwon et al., 2023) borrows from OS virtual memory:
KV cache is broken into fixed-size **blocks** (e.g., 16 tokens per
block) stored non-contiguously. Each request maintains a **block
table** mapping logical blocks to physical locations. The attention
kernel follows block table pointers instead of reading contiguous
memory.

Result: under 4% memory waste vs 60-80% in traditional systems,
enabling 2-4x throughput improvement through larger batch sizes.

### KV cache quantization

Since inference is memory-bandwidth-bound, reducing KV cache size
directly improves generation speed. Keys are typically quantized
per-channel and values per-token, because they have different
statistical properties (keys exhibit channel-wise outliers).
Quantizing KV cache to 4 bits enables 2x larger batch sizes or
4x longer sequence lengths.

### What lmxlab does

lmxlab implements a straightforward KV cache — each layer stores
K, V tensors that grow with sequence length:

```python
# lmxlab: simple cache growth
logits, cache = model(next_token, cache=cache)
mx.eval(logits, *[c for pair in cache for c in pair])
```

For educational purposes, this is the right approach — it makes
the caching mechanism explicit. Production systems add PagedAttention
and quantization on top of this same concept.

**References:**
[PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180),
[vLLM docs](https://docs.vllm.ai/en/stable/design/paged_attention/)

---

## Fused Kernels

### What kernel fusion is

A GPU kernel is a function launched on the GPU. Each kernel reads
inputs from and writes outputs to global memory (HBM). If operation A
produces a tensor that operation B consumes, the naive approach writes
A's output to HBM, then B reads it back. **Kernel fusion** merges A
and B into a single kernel so the intermediate tensor lives only in
registers or shared memory.

### Why it matters

Modern GPUs are bottlenecked by memory bandwidth, not compute. An
A100 has 312 TFLOP/s of compute but only 2 TB/s of bandwidth. For
elementwise operations (activations, normalization, residual adds),
the ratio of memory access to compute is extremely unfavorable — they
are almost entirely bandwidth-bound. Fusing them with adjacent
operations eliminates round-trips to HBM.

### Examples

**Fused attention.** Flash Attention is the canonical example — it
fuses QK^T matmul, softmax, and multiplication by V into a single
tiled kernel.

**Fused LayerNorm + Linear.** RMSNorm computes reduction statistics,
normalizes, then the next operation is typically a linear projection.
Fusing avoids writing the normalized tensor to HBM.

**Fused SwiGLU.** The gated FFN computes
`output = (xW_gate) * silu(xW_up)`, requiring two projections, an
activation, and a multiply. Fusing all four operations into one kernel
can yield 10-13% throughput improvement.

### How MLX handles this

MLX's `mx.compile` performs graph-level fusion automatically. When
you compile a function, MLX analyzes the computation graph, identifies
fusion opportunities, and generates fused Metal shaders:

```python
# lmxlab: mx.compile enables automatic fusion
self._step_fn = mx.compile(
    self._single_step,
    inputs=model.trainable_parameters(),
    outputs=model.trainable_parameters(),
)
```

For cases where automatic fusion is insufficient, MLX supports
custom Metal kernels via `mx.fast.metal_kernel()`. The unified memory
architecture eliminates CPU-GPU transfer costs, but optimizing the
intra-GPU memory hierarchy (registers, threadgroup memory, device
memory) still matters.

**References:**
[Fused SwiGLU kernels (Bitdefender Research)](https://bit-ml.github.io/blog/post/fused-swiglu-kernel/),
[MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

---

## Quantization for Inference

### How quantization works

Quantization maps high-precision values (FP16) to lower-precision
representations (INT4/INT8):
`x_q = round(x / scale) + zero_point`. This reduces model size and
increases inference speed (smaller tensors = less bandwidth = faster
for bandwidth-bound operations).

### Granularity matters

- **Per-tensor:** One scale for the entire weight tensor. Cheapest
  but crudest — outliers anywhere penalize everything.
- **Per-channel:** One scale per output channel. Captures
  channel-wise distribution differences.
- **Per-group:** Splits each channel into groups of G elements
  (commonly 32-128), each with its own scale. This is the dominant
  approach for LLMs — it balances accuracy and overhead.

### Post-training quantization methods

**GPTQ** (Frantar et al., 2022): Processes weights sequentially
within each layer. When a weight is quantized, the introduced error
is compensated by adjusting remaining weights using approximate
second-order (Hessian) information. Strong accuracy at 3-4 bits.

**AWQ** (Lin et al., 2023): Identifies "salient" weights (connected
to large activations) and protects them by applying per-channel
scaling before quantization. Faster than GPTQ to apply.

**GGUF / llama.cpp K-quants:** Uses hierarchical structure —
super-blocks of 256 weights subdivided into groups of 32, where
group scales are themselves quantized to INT8. K-quants intelligently
allocate more bits to important layers (attention) and fewer to
less important ones (some FFN layers).

### Quantization-aware training (QAT)

QAT inserts fake quantization during training — the forward pass
simulates quantized computation (with straight-through estimator for
gradients through rounding). The model learns to be robust to
quantization error. Better accuracy than PTQ at very low bits (2-3
bit), but requires full training infrastructure.

**Practical guidance:** PTQ first (fast, cheap, usually good enough
at 4-bit). QAT only if PTQ accuracy is insufficient.

### What lmxlab does

lmxlab wraps MLX's native affine quantization:

```python
from lmxlab.core.quantize import quantize_model

# Quantize to 4-bit with group size 64
quantize_model(model, bits=4, group_size=64)
```

MLX provides `mx.quantized_matmul()` which operates directly on
quantized weights, dequantizing on-the-fly during computation. This
is the key performance primitive — the Metal kernel reads compressed
weights and avoids full dequantization.

**References:**
[GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323),
[AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978),
[SqueezeLLM (Kim et al., 2023)](https://arxiv.org/abs/2306.07629),
[Visual Guide to Quantization (Grootendorst)](https://www.maartengrootendorst.com/blog/quantization/)

---

## Speculative Decoding

### The problem

Autoregressive generation is memory-bandwidth-bound: generating each
token requires reading the entire model's weights but only performs a
small matrix-vector multiplication. GPU compute units are vastly
underutilized.

### How it works

A small, fast **draft model** generates K candidate tokens. Then the
large **target model** verifies all K tokens in a **single forward
pass** (processing K tokens in parallel, like a prefill step — high
GPU utilization).

The acceptance criterion is mathematically precise: for each draft
token x sampled from draft distribution q(x), accept with probability
`min(1, p(x) / q(x))` where p(x) is the target distribution. On
rejection, sample from the residual distribution
`normalize(max(0, p(x) - q(x)))`.

This guarantees the **output distribution is identical to the target
model's** — speculative decoding is lossless.

### Variants

**Medusa** (Cai et al., 2024): Adds multiple lightweight decoding
heads to the target model itself — each head predicts a different
future position. No separate draft model needed. 2-3x speedup.

**EAGLE** (Li et al., 2024): Trains a head that predicts hidden
states (features) rather than tokens. Feature-level prediction is
easier than token-level, yielding higher acceptance rates.

**Lookahead decoding:** No additional training or models. Uses
Jacobi iteration to generate n-gram candidates and verify them.

### What lmxlab does

lmxlab implements the basic draft-then-verify paradigm:

```python
from lmxlab.inference.speculative import speculative_decode

tokens = speculative_decode(
    target_model=large_model,
    draft_model=small_model,
    prompt=prompt,
    max_tokens=100,
    n_draft=5,  # Draft 5 tokens per round
)
```

This is a pedagogically complete implementation showing the core
algorithm. Production systems add tree-structured verification,
dynamic draft length, and KV cache management for rejected tokens.

**References:**
[Speculative decoding (Leviathan et al., 2022)](https://openreview.net/pdf?id=C9NEblP8vS),
[Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774),
[EAGLE (Li et al., 2024)](https://sites.google.com/view/eagle-llm)

---

## Continuous Batching

### Why static batching wastes compute

In static batching, the server processes N requests together. The
entire batch waits until the **longest sequence** finishes. If one
request produces 500 tokens and others produce 50, those short
requests sit idle for 90% of the batch duration.

### How continuous batching works

Scheduling decisions happen **at every generation step**:

1. At each decode iteration, check if any sequence has finished.
2. Immediately evict finished sequences and insert waiting requests.
3. The batch composition changes dynamically every iteration.

GPU resources freed by a completed request are immediately used
by a new one. Throughput improvement: up to 23x over static batching
in production benchmarks.

### Supporting techniques

- **Chunked prefill:** Long prompts are processed in chunks
  interleaved with decode steps, preventing one long prompt from
  blocking the batch.
- **Ragged batching:** Variable-length sequences packed without
  padding, using offset arrays to track boundaries.
- **PagedAttention integration:** KV cache blocks freed by
  evicted sequences are reassigned to new ones.

### Relevance to Apple Silicon

Continuous batching matters less for single-user local inference
(lmxlab's primary use case) but becomes important when serving
multiple users. The mlx-lm server implements continuous batching
for MLX-based model serving.

**References:**
[Continuous batching (HuggingFace)](https://huggingface.co/blog/continuous_batching),
[Anyscale benchmark: 23x throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

## Tensor and Pipeline Parallelism

### Tensor parallelism (TP)

Splits individual weight matrices across devices. For `Y = XW`, each
device holds a shard W_i and computes Y_i = X * W_i. Results are
combined via all-reduce. Pioneered by Megatron-LM (Shoeybi et al.,
2019).

Each transformer layer requires 2 all-reduce operations. TP needs
**high-bandwidth interconnect** — works well within a node (NVLink
at 900 GB/s) but poorly across nodes.

### Pipeline parallelism (PP)

Assigns groups of consecutive layers to different devices. Device i
sends activations to device i+1. Requires less bandwidth than TP
(only activation tensors between stages, not all-reduce of full
hidden dimensions).

Microbatching with 1F1B (one forward, one backward) scheduling keeps
the pipeline full, reducing bubble overhead.

### Relevance to Apple Silicon

A single M-series chip has no CPU-GPU transfer bottleneck — unified
memory eliminates that class of problems. The M2/M3 Ultra chips
are dual-die designs connected via UltraFusion (~800 GB/s),
transparently handling a form of multi-chip parallelism.

For multi-node setups (Mac Studios via Thunderbolt 5 at ~40 Gbps),
**pipeline parallelism is preferred** because TP's all-reduce at
every layer is too bandwidth-hungry for Thunderbolt. Recent work
demonstrates TB5-connected Mac Studio clusters running distributed
inference with MLX for 1T-parameter MoE models.

**References:**
[Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053),
[Multi-node expert parallelism on Apple Silicon](https://arxiv.org/abs/2506.23635)

---

## Summary: What lmxlab Teaches vs What Production Does

| Concept | lmxlab (educational) | Production |
|---------|---------------------|------------|
| Attention | `mx.fast.scaled_dot_product_attention` | Flash Attention (tiled, IO-aware) |
| KV cache | Simple growing tensors | PagedAttention + quantized cache |
| Kernel fusion | `mx.compile` (automatic) | Hand-written fused kernels |
| Quantization | `quantize_model(bits=4)` | GPTQ/AWQ/K-quants + calibration |
| Decoding | Greedy / top-k / top-p | Speculative decoding + tree verify |
| Batching | Single sequence | Continuous batching + chunked prefill |
| Parallelism | Single device | TP + PP across devices |

The readable version teaches you **what** these operations compute.
The optimized version teaches you **how** to compute them fast. Both
are valuable — understanding the simple version makes the optimized
version comprehensible.
