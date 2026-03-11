# Developer Log

A learning journal documenting design decisions, trade-offs, and lessons
learned while building lmxlab. Following Sebastian Raschka's approach:
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
(1.3-2x typical, potentially more for larger models), but it constrains
what you can do:

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

## HuggingFace Integration

### The Three-Part Story

Connecting to the HuggingFace ecosystem required three components, each
solving a different problem:

1. **`load_from_hf()`** — Download and convert pretrained weights from
   the Hub. Maps HF config keys to `ModelConfig`, converts weight names,
   handles architecture-specific quirks (rotated QKV in LLaMA, fused
   gate-up in Gemma).

2. **`HFTokenizer`** — Wraps `AutoTokenizer` with our `Tokenizer`
   protocol. Needed because pretrained models expect their own tokenizer,
   not a character or tiktoken tokenizer.

3. **`HFDataset`** — Wraps `datasets.load_dataset` with streaming
   support. Yields `(input, target)` batches by tokenizing on-the-fly
   from a token buffer.

**Design choice:** All three are lazy imports (`from transformers import ...`
inside `__init__`). This keeps the base library free of heavy dependencies —
you don't need `transformers` installed unless you actually use HF features.

**What surprised us:** Weight conversion is the hardest part. Different
architectures store weights in different formats (some fuse QKV, some
don't; some transpose FFN weights, some don't). The solution was a
mapping table per architecture, with clear error messages for missing keys.

### Streaming for Large Datasets

`HFDataset.batch_iterator()` uses a token buffer pattern: accumulate
tokens from the dataset stream until there are enough for one batch, yield
it, and keep the remainder. This means you never need the full dataset in
memory — important for multi-GB corpora.

The `streaming=True` flag enables HuggingFace's iterable dataset mode,
which downloads data on demand instead of caching the full dataset locally.

## Advanced Training Features

### DPO, GRPO, and MTP: Three Approaches to Better Models

These three training objectives each improve model quality in a different
way, and together they illustrate a useful taxonomy:

| Method | Signal Source | Key Idea |
|--------|-------------|----------|
| **DPO** | Preference pairs | Learn from "A is better than B" without a reward model |
| **GRPO** | Scalar rewards | Group-relative normalization of per-completion rewards |
| **MTP** | Same data, richer targets | Predict multiple future tokens, not just the next one |

**DPO** replaces RLHF's reward model + PPO pipeline with a single loss
function. The mathematical insight: the optimal policy under the KL-
constrained reward maximization objective has a closed-form solution that
only needs the policy and reference model log probabilities.

**GRPO** is closer to classic policy gradient but normalizes rewards within
each group of completions (zero mean, unit variance). This removes the need
for a value function baseline and makes training more stable.

**MTP** is orthogonal — it doesn't change the objective, it enriches the
training signal. Each position predicts not just the next token but the
next 2-4 tokens via lightweight auxiliary heads. This provides richer
gradients and enables speculative decoding at inference time (the auxiliary
heads serve as draft predictors).

### Curriculum Learning: Start Easy

Length curriculum (short sequences → long sequences) follows a simple
principle: let the model learn basic patterns on short context before
tackling long-range dependencies. Empirically, this often converges faster
than training on the final sequence length from the start.

The implementation is straightforward — linear interpolation of sequence
length across stages. No complex scheduling needed.

## Pre-Registered Experiment Plans

Following Platt's strong inference and Chamberlin's multiple working
hypotheses, each experiment below specifies competing hypotheses and
predictions **before** running. This guards against confirmation bias
and the garden of forking paths (Gelman & Loken, 2013).

### Experiment 1: GPT-to-LLaMA Feature Ablation

**Question:** When adding LLaMA-style features to a GPT baseline one
at a time, which individual change contributes most to improved
training dynamics?

**Competing hypotheses:**

- **H1 (Attention dominates):** GQA provides the largest improvement
  because it enables more efficient use of the parameter budget
  (sharing KV heads frees capacity for other components).
- **H2 (FFN dominates):** SwiGLU provides the largest improvement
  because the gating mechanism gives the network better gradient flow
  and expressiveness.
- **H3 (Normalization dominates):** RMSNorm + no bias provides the
  largest improvement because it stabilizes training, allowing higher
  learning rates.
- **H4 (Interactions dominate):** No single change provides more than
  20% of the total improvement; the benefit comes from combining
  features (non-linear interaction).

**Design:**

| Run | Attention | FFN | Norm | Position | Bias |
|-----|-----------|-----|------|----------|------|
| Baseline | MHA | Standard | LayerNorm | Sinusoidal | Yes |
| +GQA | GQA | Standard | LayerNorm | Sinusoidal | Yes |
| +SwiGLU | MHA | Gated | LayerNorm | Sinusoidal | Yes |
| +RMSNorm | MHA | Standard | RMSNorm | Sinusoidal | No |
| +RoPE | MHA | Standard | LayerNorm | RoPE | Yes |
| Full LLaMA | GQA | Gated | RMSNorm | RoPE | No |

**Protocol:** 5-minute time budget per run (autoresearch pattern),
3 seeds each, d_model=256, n_layers=6, Shakespeare dataset. Report
val_bpb mean +/- std. Use the `ablation_gpt_to_llama.py` recipe.

**Analysis plan:** ANOVA across single-feature runs, then compare
sum of individual improvements to Full LLaMA improvement to test H4.
Report effect sizes (Cohen's d) relative to baseline, not just
statistical significance.

**What would falsify each hypothesis:**

- H1: If +GQA improvement < 20% of total (baseline to full LLaMA)
- H2: If +SwiGLU improvement < 20% of total
- H3: If +RMSNorm improvement < 20% of total
- H4: If any single feature contributes > 50% of total

---

### Experiment 2: mx.compile Coverage Analysis

**Question:** How does the speedup from `mx.compile` scale as we
progressively compile more of the training pipeline?

**Competing hypotheses:**

- **H1 (Graph size dominates):** Speedup scales roughly linearly
  with the fraction of computation compiled, because each fused
  operation saves one memory round-trip.
- **H2 (Diminishing returns):** The first compilation (the training
  step) captures most of the benefit; additional compilation of
  evaluation or data preprocessing provides negligible speedup.
- **H3 (Overhead at small scale):** For tiny models, compilation
  overhead (tracing, first-step latency) dominates, and compiled
  code is actually slower for the first N steps.

**Design:**

| Config | What's compiled |
|--------|----------------|
| None | Nothing compiled |
| Step only | Training step (`_single_step`) |
| Step + eval | Training step + evaluation forward pass |

Measure: steps/second (excluding first 5 warmup steps), peak memory,
time-to-first-step. Three model sizes: tiny (64d/2L), small
(256d/6L), medium (512d/8L). 3 seeds each.

**Analysis plan:** Plot speedup ratio vs model size. Report
time-to-first-step separately (compilation overhead). If H3 is
correct, there should be a crossover point where compilation becomes
net positive.

---

### Experiment 3: Optimizer Comparison on Unified Memory

**Question:** Does Apple Silicon's unified memory architecture change
which optimizers work best, compared to published CUDA results?

**Competing hypotheses:**

- **H1 (Same story):** AdamW dominates regardless of hardware, because
  optimizer dynamics depend on loss landscape geometry not memory
  architecture.
- **H2 (Memory-efficient wins):** SGD with momentum or Adafactor
  perform comparatively better on Apple Silicon because they use less
  optimizer state, leaving more unified memory for larger batch sizes.
- **H3 (Bandwidth matters):** Optimizers with fewer memory accesses per
  step (SGD) gain a disproportionate advantage because Apple Silicon
  has lower memory bandwidth than datacenter GPUs.

**Design:** Train LLaMA-small (256d/6L) on Shakespeare with AdamW,
SGD+momentum, Adafactor, and Lion. Same learning rate sweep for each
(log-scale: 1e-4, 3e-4, 1e-3, 3e-3). 5-minute time budget, 3 seeds.

**Metrics:** Best val_bpb achieved, steps/second, peak memory usage.

**Analysis plan:** Compare best-run val_bpb across optimizers (paired
across seeds). Report both best-of-sweep and mean-across-sweep to
distinguish "works with tuning" from "works robustly." Compare
steps/second ratios to published CUDA ratios for the same optimizers.

---

### Experiment 4: KV Cache Reduction with MLA

**Question:** Does MLA's ~57x KV cache compression translate to
meaningful practical benefits on unified memory?

**Competing hypotheses:**

- **H1 (Memory benefit):** MLA enables substantially longer generation
  (higher max_tokens before OOM) because KV cache is the binding
  memory constraint during inference.
- **H2 (No practical benefit):** On unified memory, the total memory
  pool is large enough that KV cache is not the binding constraint
  for typical sequence lengths (< 8K tokens). MLA's benefit only
  appears at very long contexts.
- **H3 (Speed benefit):** MLA is faster per-token during generation
  because reading a smaller KV cache from memory is faster
  (bandwidth-bound operation).

**Design:** Compare DeepSeek-style MLA vs standard MHA at matched
parameter counts. Generate sequences of increasing length (512, 1K,
2K, 4K, 8K tokens). Measure: tokens/second, peak memory, and
maximum achievable sequence length before OOM.

**Analysis plan:** Plot tokens/second and memory vs sequence length
for both. If H1 is correct, there should be a clear divergence point.
If H3 is correct, MLA should be consistently faster even at short
lengths.

---

### Experiment 5: What Can You Train in 5 Minutes?

**Question:** What is the best validation BPB achievable in exactly
5 minutes of wall-clock training on an M-series Mac?

This is the autoresearch paradigm applied directly — fixed time
budget eliminates timing confounds and makes all experiments directly
comparable regardless of architecture complexity.

**Protocol:**

1. Start with GPT-tiny on Shakespeare (baseline)
2. Iterate: modify one thing (architecture, hyperparameter, data),
   train for 5 minutes, record val_bpb
3. Keep changes that improve; discard changes that don't
4. Git-as-experiment-infra: each run is a commit on
   `experiments/5min-*` branch

**Metrics:** val_bpb (primary), training loss curve shape,
parameter count (prefer simpler models at equal performance).

**Simplicity bias:** If two configurations achieve similar val_bpb,
prefer the one with fewer parameters or simpler architecture. A
0.001 improvement from deleting a feature is worth more than
the same improvement from adding complexity.

## Open Questions

- How does training dynamics change between Apple Silicon and CUDA
  for the same architecture? Are learning rate sensitivities the
  same?
- What's the optimal `mx.eval()` frequency? Too many evals waste
  time on synchronization; too few let the computation graph grow
  unboundedly.
- Does MLA's KV compression provide real memory benefits on unified
  memory, or is the bottleneck elsewhere? (See Experiment 4.)
- Can we use MLX's compilation to automatically fuse attention + FFN
  in a single block (like FlashAttention does for attention alone)?
- How does the 3:1 DeltaNet:GQA ratio in Qwen 3.5 compare to other
  ratios (2:1, 4:1) on our educational-scale models?
- At what model size does compilation speedup become worth the
  first-step overhead? (See Experiment 2.)
