# lmxlab

A research platform for language model experimentation on Apple Silicon.

## Why lmxlab?

If you're doing language model research on a Mac, you've probably hit
the limits of PyTorch's MPS backend — incomplete operator coverage,
[silent wrong results on non-contiguous tensors](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/),
and unnecessary memory copies on hardware that shares memory by design.

lmxlab is built on [MLX](https://ml-explore.github.io/mlx/) instead,
which was designed from scratch for Apple Silicon. The practical
differences for research:

- **Faster training.** MLX trains LMs
  [30-50% faster](https://github.com/LucasSte/MLX-vs-Pytorch) than
  PyTorch MPS on M-series chips, with the gap growing on newer hardware.
- **True unified memory.** MLX arrays live in shared memory — no copies
  between CPU and GPU. PyTorch MPS still duplicates tensors on device
  transfers. On a 36GB MacBook Pro, this means larger models fit.
- **No device management.** No `.to(device)`, no `.cuda()`, no
  `PYTORCH_ENABLE_MPS_FALLBACK=1`. Arrays just work on any processor.
- **Functional gradients.** `mx.value_and_grad(loss_fn)` replaces the
  `zero_grad` / `backward` / `step` ceremony and eliminates an entire
  class of gradient-accumulation bugs.

On top of MLX, lmxlab makes the experiment loop short: swap architectures
in one line, get standardized metrics automatically, and compare results
across runs without writing boilerplate.

The key idea is that GPT, LLaMA, DeepSeek, Mamba, and dozens of other
architectures are not fundamentally different models. They are different
*configurations* of the same building blocks: attention, SSMs, feed-forward
networks, normalization, and positional encoding. lmxlab makes this concrete
with **config factories** — so you can test a hypothesis across architectures
without rewriting training code.

```python
from lmxlab.models.llama import llama_config
from lmxlab.models.deepseek import deepseek_config
from lmxlab.models.base import LanguageModel

# Same LanguageModel class, different configs
llama = LanguageModel(llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6))
deepseek = LanguageModel(deepseek_config(d_model=512, n_heads=8, n_layers=6, kv_lora_rank=64))
```

No subclassing. No `LlamaModel` vs `DeepSeekModel`. One `LanguageModel` class,
assembled from registry components based on what the config asks for.

## Design principles

- **Research-first.** Designed for fast iteration on ideas, not production
  serving. Change a config, train, compare — the experiment loop should
  be the bottleneck, not the framework.
- **MLX-native.** No PyTorch translation layer. Uses MLX idioms directly:
  `nn.value_and_grad`, `mx.compile`, `mx.fast.scaled_dot_product_attention`,
  unified memory.
- **Config factories, not subclasses.** Architecture variants are configs, not
  class hierarchies. A `BlockConfig` names its components by string, and a
  typed `Registry` resolves them at construction time.
- **Progressive complexity.** Start with standard MHA and LayerNorm (GPT-style),
  then swap in GQA + RMSNorm + SwiGLU (LLaMA-style), then try MLA
  (DeepSeek-style) or Mamba SSMs. Same model class throughout.
- **Reproducible experiments.** FLOP-matched budgets, train/val splits, MLflow
  tracking, standardized metrics, and structured results logging.

## Installation

```bash
pip install lmxlab
```

Or from source:

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
pip install -e ".[dev]"
```

Requires Python 3.12+ and an Apple Silicon Mac (M1 or later) for GPU acceleration.
MLX will still run on Intel Macs and Linux using CPU, but the performance
characteristics will differ.

## What's included

- **24 architectures** as config factories: GPT, LLaMA, Gemma, Gemma 3, Qwen, Qwen 3 MoE, Qwen 3.5, Qwen-Next, Mixtral, DeepSeek V2/V3, Nemotron, Llama 4 Scout/Maverick, Mistral Small, OLMo 2, GPT-OSS, Grok, Kimi K2.5, SmolLM3, Falcon H1, Jamba, Bamba, GLM-4.5
- **Building blocks**: MHA, GQA, MLA, GatedGQA, SlidingWindowGQA, ChunkedGQA, SparseGQA (DSA), Mamba-2 SSD, Mamba-3, GatedDeltaNet, MoE, SharedExpertMoE, LatentMoE, QK-norm, SwiGLU, squared ReLU
- **Compiled training** with `mx.compile`, functional gradients, gradient clipping, cosine schedules, dropout, muP parameterization
- **Advanced training**: DPO, GRPO, multi-token prediction, curriculum learning, knowledge distillation
- **LoRA & QLoRA**: parameter-efficient fine-tuning with optional 4-bit quantization
- **Inference**: autoregressive generation, speculative decoding, best-of-N sampling, beam search, reward model scoring
- **HuggingFace integration**: load pretrained weights from the Hub
- **Experiment framework**: time/FLOP-budgeted runs, MLflow tracking, results logging, hyperparameter sweeps, MLX profiling
- **35 recipe scripts**: training, fine-tuning, ablation studies, architecture comparison, benchmarking

## Documentation overview

- **[Quickstart](getting-started/quickstart.md)** -- Build and run a model in
  under 20 lines.
- **[First Training Run](getting-started/first-training-run.md)** -- Train a
  model from scratch, step by step.
- **[Architecture Overview](architecture/overview.md)** -- How configs, registries,
  and `ConfigurableBlock` fit together.
- **[MLX Idioms](architecture/mlx-idioms.md)** -- How MLX differs from PyTorch
  and why it matters for this library.
- **[Models](models/index.md)** -- Compare all 24 architectures side-by-side.
- **[Compiled Training](architecture/compiled-training.md)** -- How `mx.compile`
  fuses the training step.
- **[Unified Memory](architecture/unified-memory.md)** -- What Apple Silicon's
  memory model means for ML.
- **[Production Optimizations](architecture/production-optimizations.md)** -- How
  production systems (vLLM, llama.cpp) optimize beyond what lmxlab teaches.
- **[Recipes](recipes/index.md)** -- All 35 ready-to-run scripts, categorized.
- **[Experiment Methodology](experiments/methodology.md)** -- How to run rigorous
  experiments with the framework.
- **[Developer Log](devlog/index.md)** -- Design decisions, lessons learned,
  and pre-registered experiment plans.
- **[API Reference](api/index.md)** -- Auto-generated module documentation.
