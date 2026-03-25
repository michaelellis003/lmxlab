# lmxlab

Language model experimentation on Apple Silicon using
[MLX](https://ml-explore.github.io/mlx/).

## Overview

Autoregressive language models (GPT, LLaMA, DeepSeek, Mamba, and
variants) differ in their sequence mixing operator (multi-head
attention, grouped-query attention, multi-latent attention, state-space
models) and feed-forward nonlinearity (SwiGLU, squared ReLU,
mixture-of-experts). The remaining components (embedding, normalization,
positional encoding, residual connections) are shared.

In lmxlab, a single `LanguageModel` class is parameterized by a
configuration object that selects components from a registry.
Architecture comparisons reduce to config changes; the training loop,
data pipeline, and evaluation code remain fixed.

```python
from lmxlab.models.llama import llama_config
from lmxlab.models.deepseek import deepseek_config
from lmxlab.models.base import LanguageModel

# Same model class, different configurations
llama = LanguageModel(llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6))
deepseek = LanguageModel(deepseek_config(d_model=512, n_heads=8, n_layers=6, kv_lora_rank=64))
```

## Motivation

PyTorch's MPS backend on Apple Silicon has incomplete operator
coverage, [silent numerical errors on non-contiguous tensors](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/),
and redundant memory copies on unified-memory hardware. MLX was
designed for Apple Silicon and
shows [30-50% higher throughput](https://github.com/LucasSte/MLX-vs-Pytorch)
than PyTorch MPS on M-series processors. It requires no explicit
device management, and its functional gradient API
(`mx.value_and_grad`) replaces the stateful
`zero_grad`/`backward`/`step` pattern.

## Design principles

1. **Composition over inheritance.** Architecture variants are
   configuration objects. Adding an architecture means registering
   components.
2. **Readability over performance.** Code is written to be read and
   changed quickly.
3. **MLX-native idioms.** Functional gradients (`nn.value_and_grad`),
   compiled execution (`mx.compile`), unified memory.
4. **Reproducibility.** Fixed time/FLOP budgets, deterministic
   train/val splits, MLflow tracking, results logging.

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
MLX also runs on Intel Macs and Linux (CPU only), but performance
characteristics will differ.

## Components

### Architectures (24 config factories)

GPT, LLaMA, Gemma, Gemma 3 (sliding window), Qwen, Qwen 3 MoE,
Qwen 3.5 (hybrid DeltaNet), Qwen-Next (gated attention), Mixtral (MoE),
DeepSeek V2/V3 (MLA + MoE), Nemotron (hybrid Mamba-Transformer MoE),
Llama 4 Scout/Maverick (iRoPE + chunked attention), Mistral Small
(sliding window), OLMo 2 (QK-norm), GPT-OSS (QK-norm), Grok
(SharedExpertMoE), Kimi K2.5 (DeltaNet + MoE), SmolLM3 (iRoPE),
Falcon H1 (hybrid Mamba-2), Jamba (Mamba-2 + MoE), Bamba (hybrid
Mamba-2), GLM-4.5 (MLA NoPE).

### Sequence mixing operators

MHA, GQA, MLA, GatedGQA, SlidingWindowGQA, ChunkedGQA, SparseGQA
(DSA), Mamba-2 SSD, Mamba-3 (trapezoidal), GatedDeltaNet.

### Feed-forward and routing

SwiGLU, squared ReLU, MoE, SharedExpertMoE, LatentMoE.

### Training

Compiled forward/backward via `mx.compile`, functional gradients,
gradient clipping, cosine annealing, dropout, muP parameterization,
DPO, GRPO, multi-token prediction, curriculum learning, knowledge
distillation. LoRA and QLoRA (4-bit quantization) for
parameter-efficient fine-tuning.

### Inference

Autoregressive generation, speculative decoding, best-of-N sampling,
beam search, reward model scoring.

### Experiment infrastructure

Runs budgeted by wall time or FLOPs, MLflow tracking, results logging,
hyperparameter sweeps, MLX profiling. HuggingFace Hub integration for
pretrained weight loading.

### Recipes

35 scripts for training, fine-tuning, alignment (DPO, GRPO),
multi-token prediction, distillation, evaluation, quantization,
architecture comparison, optimizer comparison, and KV cache analysis.

## Documentation overview

- [Quickstart](getting-started/quickstart.md): build and run a model in under 20 lines
- [First Training Run](getting-started/first-training-run.md): train a model from scratch
- [Architecture Overview](architecture/overview.md): configs, registries, and `ConfigurableBlock`
- [MLX Idioms](architecture/mlx-idioms.md): differences between MLX and PyTorch
- [Models](models/index.md): all 24 architectures compared
- [Compiled Training](architecture/compiled-training.md): how `mx.compile` fuses the training step
- [Unified Memory](architecture/unified-memory.md): Apple Silicon's memory model for ML
- [Production Optimizations](architecture/production-optimizations.md): optimizations in vLLM, llama.cpp, mlx-lm
- [Recipes](recipes/index.md): all 35 scripts, categorized
- [Experiment Methodology](experiments/methodology.md): running experiments with the framework
- [Developer Log](devlog/index.md): design decisions and pre-registered experiments
- [API Reference](api/index.md): auto-generated module documentation
