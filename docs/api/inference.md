# Inference

Advanced inference strategies for language models.

lmxlab provides three inference approaches beyond standard autoregressive
generation:

- **Best-of-N sampling**: generate multiple completions and select the
  highest-scoring one (useful when quality matters more than speed)
- **Majority vote**: generate N completions and group by content, returning
  frequency counts (useful for tasks with discrete answers like math or code)
- **Speculative decoding**: use a small draft model to propose tokens,
  verified by the target model in a single forward pass (especially
  interesting on unified memory where both models share the same memory pool)

## Usage

```python
import mlx.core as mx
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_config
from lmxlab.inference import best_of_n, majority_vote, speculative_decode

model = LanguageModel(gpt_config(vocab_size=256, d_model=128, n_heads=4, n_layers=4))
mx.eval(model.parameters())

prompt = mx.array([[1, 2, 3]])

# Best-of-N: pick the highest-scoring completion
best = best_of_n(model, prompt, n=4, max_tokens=50, temperature=0.8)

# Majority vote: group completions by content
results = majority_vote(model, prompt, n=10, max_tokens=20)
for tokens, count in results:
    print(f"  count={count}: {tokens[:5]}...")

# Speculative decoding: draft-then-verify
draft_model = LanguageModel(gpt_config(vocab_size=256, d_model=64, n_heads=2, n_layers=2))
mx.eval(draft_model.parameters())
output, stats = speculative_decode(model, draft_model, prompt, max_tokens=50)
print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
```

## Sampling

::: lmxlab.inference.sampling

## Speculative Decoding

::: lmxlab.inference.speculative
