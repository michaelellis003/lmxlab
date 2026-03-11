# Developer Log

A learning journal documenting design decisions, trade-offs, and lessons
learned while building lmt-metal. Following Sebastian Raschka's approach:
honest uncertainty, practical grounding, and explaining *why* not just *what*.

## Design Decisions

### Config Factories Over Class Hierarchies

The biggest early decision: GPT, LLaMA, DeepSeek, etc. are *not* separate
model classes. They're config factory functions returning `ModelConfig`.

**Why:** After studying the original LMT (PyTorch version), it became clear
that transformer architectures differ in configuration, not structure. A
`LlamaModel` and a `GPTModel` do the same thing — embed, attend, feed-forward,
project — just with different component choices. Encoding this as inheritance
(`LlamaModel(BaseModel)`) creates artificial boundaries.

With config factories, switching from GPT to LLaMA is changing a function call,
not a class hierarchy:

```python
# These produce different configs, not different model classes
config = gpt_config(d_model=512, n_heads=8, n_layers=6)
config = llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6)

# Same model class for both
model = LanguageModel(config)
```

**Trade-off:** This design makes individual architecture code harder to find
(no `class LlamaModel` to grep for). We mitigate this with well-documented
config factories in `models/*.py`.

### Registry Pattern for Components

Components (attention, FFN, norm, position) register themselves by string
name. `ConfigurableBlock` resolves them at construction time.

**Why:** This decouples component implementation from block assembly. Adding
a new attention type (MLA, sliding window) means writing the module and
registering it — no changes to `ConfigurableBlock` or any existing code.

**What we got wrong initially:** MoE FFNs needed special registration because
their constructors take extra arguments (`n_experts`, `top_k`). We solved
this by making `BlockConfig` carry MoE fields and having MoE constructors
read from the config with optional overrides.

### Explicit `mx.eval()` Boundaries

MLX operations are lazy — nothing computes until you call `mx.eval()`.
This is powerful but requires discipline. We enforce eval boundaries at
specific points:

1. After model construction (`mx.eval(model.parameters())`)
2. During generation (eval each token before feeding it back)
3. At training step boundaries (eval loss for logging)

**Lesson learned:** Forgetting `mx.eval()` after model construction can
cause the first training step to be extremely slow, because it triggers
both initialization and the first forward pass in one graph.

## Lessons Learned

### Unified Memory Changes the Trade-offs

On CUDA, the CPU-GPU memory boundary is a constant concern — batch sizes,
gradient accumulation, offloading. On Apple Silicon with MLX, unified
memory means:

- No `.to(device)` calls — arrays live everywhere at once
- No data transfer bottleneck between CPU and GPU
- The memory ceiling is system RAM (not separate GPU VRAM)
- Speculative decoding is more natural (draft and verify models share memory)

**What we don't know yet:** How much this changes optimal training strategies.
Does gradient accumulation matter less when there's no transfer overhead?
Do different optimizer choices win when memory access patterns change?
These are open questions for the experiment framework to investigate.

### `mx.compile` Is Not Free

Compiling the training step with `mx.compile` provides significant speedups
(2-4x in our benchmarks), but it constrains what you can do:

- No Python control flow that depends on array values
- Must declare `inputs` and `outputs` explicitly
- Graph changes (like modifying model structure) require recompilation

For educational code, we default to `compile_step=True` but make it easy
to disable for debugging. The `benchmark_compile.py` recipe measures the
actual speedup on your hardware.

### LoRA and QLoRA on Unified Memory

Parameter-efficient fine-tuning has different economics on unified memory.
On CUDA, QLoRA's primary value is fitting a larger model in limited VRAM.
On Apple Silicon:

- The memory savings still matter (system RAM is shared with the OS)
- But the performance characteristics may differ (no quantize-dequantize
  transfer between devices)
- Adapter save/load is valuable regardless — small files that can be
  shared independently of multi-GB base models

### Testing ML Code

Standard unit testing (`assertEqual`) doesn't work well for ML code because
outputs are stochastic. We use behavioral tests instead:

- **Invariance tests:** Does the output stay the same when it should?
  (Same input, same seed = same output)
- **Directional tests:** Does the output move in the right direction?
  (Loss decreases after training steps)
- **Shape tests:** Are the dimensions correct? (Output shape matches
  `(batch, seq_len, vocab_size)`)
- **Minimum functionality tests:** Does it produce finite values?
  (No NaN, no Inf)

## Architecture Notes

### Attention Variants Are Configurations, Not Architectures

| Variant | Key Insight |
|---------|-------------|
| MHA | Full attention: each head has independent K, V |
| GQA | Share K, V across groups of query heads |
| MLA | Compress KV into a low-rank latent, reconstruct at attention time |
| Sliding Window | Limit attention span per layer (local vs global) |
| GatedDeltaNet | Linear attention with gated delta rule (no softmax) |

These all implement the same interface: `(x, mask, cache) -> (output, cache)`.
The difference is how they compute and store key-value state.

### MoE Is Just a Different FFN

Mixture of Experts replaces the dense FFN with a routed sparse FFN.
The router selects top-k experts per token, and the outputs are combined
by router weights. From `ConfigurableBlock`'s perspective, it's just
another FFN — registered as `"moe"` instead of `"gated"`.

The interesting part is load balancing. Without it, the router collapses
to always picking the same experts. We implement bias-based balancing
(SharedExpertMoEFFN) which avoids the auxiliary loss used in some
implementations.

## Open Questions

- How does training dynamics change between Apple Silicon and CUDA for the
  same architecture? Are learning rate sensitivities the same?
- What's the optimal `mx.eval()` frequency? Too many evals waste time on
  synchronization; too few let the computation graph grow unboundedly.
- Does MLA's KV compression provide real memory benefits on unified memory,
  or is the bottleneck elsewhere?
- Can we use MLX's compilation to automatically fuse attention + FFN in a
  single block (like FlashAttention does for attention alone)?
