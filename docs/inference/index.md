# Inference

lmxlab provides generation utilities ranging from simple greedy
decoding to advanced strategies like speculative decoding and
best-of-N sampling.

## Basic Generation

The `generate` function handles autoregressive generation with
KV caching:

```python
import mlx.core as mx
from lmxlab.models import LanguageModel, generate
from lmxlab.models.llama import llama_tiny

model = LanguageModel(llama_tiny())
mx.eval(model.parameters())

prompt = mx.array([[1, 2, 3]])
output = generate(model, prompt, max_tokens=20, temperature=0.0)
# output shape: (1, 23) -- prompt + generated
```

### Sampling Parameters

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 1.0 | Controls randomness (0 = greedy, higher = more random) |
| `top_k` | 0 | Restrict sampling to top-k tokens (0 = disabled) |
| `top_p` | 1.0 | Nucleus sampling threshold (< 1.0 to enable) |
| `repetition_penalty` | 1.0 | Penalize repeated tokens (> 1.0 to enable) |
| `stop_tokens` | None | Token IDs that halt generation |
| `max_tokens` | 100 | Maximum new tokens to generate |

### Examples

```python
# Creative writing: high temperature + nucleus sampling
output = generate(
    model, prompt, max_tokens=200,
    temperature=0.9, top_p=0.95,
)

# Focused completion: low temperature + top-k
output = generate(
    model, prompt, max_tokens=50,
    temperature=0.3, top_k=10,
)

# Chat: stop at EOS token, discourage repetition
output = generate(
    model, prompt, max_tokens=512,
    temperature=0.7, stop_tokens=[2],
    repetition_penalty=1.1,
)
```

## Streaming Generation

For interactive applications, `stream_generate` yields tokens one
at a time:

```python
from lmxlab.models import stream_generate

for token_id in stream_generate(
    model, prompt, max_tokens=100,
    temperature=0.8, stop_tokens=[2],
):
    # Process each token as it's produced
    print(token_id, end=' ', flush=True)
```

This is useful for real-time display where you want to show tokens
as they are generated rather than waiting for the full sequence.

## Best-of-N Sampling

Generate multiple candidates and select the highest-scoring one:

```python
from lmxlab.inference import best_of_n

# Generate 8 candidates, return the best by log probability
best = best_of_n(
    model, prompt, n=8,
    max_tokens=100, temperature=0.8,
)

# Length-normalized scoring (fairer for varying lengths)
best = best_of_n(
    model, prompt, n=8,
    max_tokens=100, temperature=0.8,
    score_fn='length_normalized',
)
```

Best-of-N is a simple but effective way to improve generation
quality. Each candidate is scored by its total log probability
under the model, optionally normalized by length.

## Majority Vote

For tasks with discrete answers (math, classification, code),
generate multiple completions and count the most common:

```python
from lmxlab.inference import majority_vote

results = majority_vote(
    model, prompt, n=10,
    max_tokens=20, temperature=0.8,
)
# Returns: [(token_list, count), ...] sorted by count
most_common, count = results[0]
print(f'Most common answer ({count}/10): {most_common}')
```

## Speculative Decoding

Use a small draft model to propose tokens, verified by a larger
target model in a single forward pass. This can speed up inference
when the draft model is much faster and agrees with the target
most of the time.

```python
from lmxlab.inference import speculative_decode
from lmxlab.models.gpt import gpt_tiny, gpt_config

# Small draft model
draft = LanguageModel(gpt_tiny())
mx.eval(draft.parameters())

# Larger target model
target = LanguageModel(gpt_config(d_model=256, n_heads=4, n_layers=4))
mx.eval(target.parameters())

output, stats = speculative_decode(
    target, draft, prompt,
    max_tokens=50, draft_tokens=4,
)
print(f'Acceptance rate: {stats["acceptance_rate"]:.1%}')
```

!!! note "Unified Memory Advantage"
    Speculative decoding is especially interesting on Apple Silicon.
    Both models share unified memory, so there is no data transfer
    overhead between CPU and GPU. The draft and target models can
    coexist efficiently in the same memory pool.

### How It Works

1. **Draft:** The small model generates `draft_tokens` candidate tokens
2. **Verify:** The target model processes the full sequence (prompt + drafts) in one forward pass
3. **Accept/reject:** Each drafted token is compared against what the target would have produced
4. **On match:** Accept and continue
5. **On mismatch:** Use the target's token, discard remaining drafts, restart

When the acceptance rate is high, speculative decoding generates
multiple tokens per target model forward pass, providing a
wall-clock speedup proportional to `draft_tokens * acceptance_rate`.

## Evaluation Metrics

After generation, evaluate quality with built-in metrics:

```python
from lmxlab.eval import perplexity, bits_per_byte

# Perplexity on evaluation data
ppl = perplexity(model, eval_batches)

# Bits per byte (comparable across tokenizers)
bpb = bits_per_byte(model, eval_batches, bytes_per_token=3.5)
```

For code generation tasks, use pass@k:

```python
from lmxlab.eval import pass_at_k, evaluate_pass_at_k

# Single problem: 3 correct out of 10 samples
p = pass_at_k(n=10, c=3, k=1)  # 0.3

# Evaluate across problems with a test function
results = evaluate_pass_at_k(
    completions,  # list of list of code strings
    test_fn=run_tests,  # returns True if code passes
    k_values=[1, 5, 10],
)
print(results)  # {'pass@1': 0.42, 'pass@5': 0.78, 'pass@10': 0.91}
```
