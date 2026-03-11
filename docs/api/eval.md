# Evaluation

Metrics for language model evaluation.

## Overview

lmxlab provides standard evaluation metrics:

- **Perplexity**: exponential of the average cross-entropy loss. Lower is
  better. A perplexity of 10 means the model is as uncertain as choosing
  uniformly among 10 tokens.
- **Bits-per-byte (BPB)**: cross-entropy loss normalized by bytes. This is
  tokenizer-independent, making it useful for comparing models with different
  vocabularies.
- **pass@k**: estimates the probability that at least one of k code samples
  passes a set of tests (Chen et al., 2021). Used for code generation
  evaluation.

## Usage

```python
import mlx.core as mx
from lmxlab.models.base import LanguageModel
from lmxlab.models.gpt import gpt_tiny
from lmxlab.eval import perplexity, bits_per_byte, pass_at_k

model = LanguageModel(gpt_tiny())
mx.eval(model.parameters())

tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

# Perplexity on a sequence
ppl = perplexity(model, tokens)
print(f"Perplexity: {ppl:.1f}")

# Bits-per-byte (tokenizer-independent)
bpb = bits_per_byte(model, tokens, bytes_per_token=3.5)
print(f"BPB: {bpb:.3f}")

# pass@k for code generation
# If 3 out of 10 samples pass, estimate pass@1
p1 = pass_at_k(n=10, c=3, k=1)
print(f"pass@1: {p1:.3f}")
```

## Metrics

::: lmxlab.eval.metrics.perplexity

::: lmxlab.eval.metrics.bits_per_byte

## Code Generation Evaluation

::: lmxlab.eval.metrics.pass_at_k

::: lmxlab.eval.metrics.evaluate_pass_at_k
