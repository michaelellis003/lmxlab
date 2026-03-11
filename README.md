# lmt-metal

An educational MLX library for transformer language models on Apple Silicon.

[![CI](https://github.com/michaelellis003/lmt-metal/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/lmt-metal/actions/workflows/ci.yml)
[![Docs](https://github.com/michaelellis003/lmt-metal/actions/workflows/docs.yml/badge.svg)](https://michaelellis003.github.io/lmt-metal/)

## Why lmt-metal?

Most transformer implementations optimize for production at the cost of readability.
lmt-metal takes the opposite approach: every layer is implemented from scratch in
[MLX](https://ml-explore.github.io/mlx/), with the explicit goal of helping you
understand how modern language models work.

The core insight is that GPT, LLaMA, and DeepSeek are not fundamentally different
architectures. They are different *configurations* of the same building blocks:
attention, feed-forward networks, normalization, and positional encoding. lmt-metal
makes this concrete by using **config factories** instead of class hierarchies.

```python
from lmt_metal.models.llama import llama_config
from lmt_metal.models.deepseek import deepseek_config
from lmt_metal.models.base import LanguageModel

# Same LanguageModel class, different configs
llama = LanguageModel(llama_config(d_model=512, n_heads=8, n_kv_heads=4, n_layers=6))
deepseek = LanguageModel(deepseek_config(d_model=512, n_heads=8, n_layers=6, kv_lora_rank=64))
```

No subclassing. No `LlamaModel` vs `DeepSeekModel`. One `LanguageModel` class,
assembled from registry components based on what the config asks for.

## What's included

- **8 architectures** as config factories: GPT, LLaMA, Gemma, Qwen, Mixtral (MoE), DeepSeek V2 (MLA), Gemma 3 (sliding window), Qwen 3.5 (hybrid DeltaNet)
- **Compiled training** with `mx.compile`, functional gradients, gradient clipping, cosine schedules
- **Advanced training**: DPO, GRPO, multi-token prediction, curriculum learning
- **LoRA & QLoRA**: parameter-efficient fine-tuning with optional 4-bit quantization
- **Inference**: autoregressive generation, speculative decoding, best-of-N sampling
- **HuggingFace integration**: load pretrained weights from the Hub
- **Experiment framework**: time-budgeted runs, results tracking, sweeps, MLX profiling
- **18 recipe scripts**: training, fine-tuning, DPO, evaluation, streaming generation, checkpointing, ablation studies, architecture comparison, benchmarking

## Quick start

```bash
pip install lmt-metal
```

```python
import mlx.core as mx
from lmt_metal.models.llama import llama_config
from lmt_metal.models.base import LanguageModel
from lmt_metal.training.config import TrainConfig
from lmt_metal.training.trainer import Trainer

# Build a small LLaMA
config = llama_config(vocab_size=256, d_model=128, n_heads=4, n_kv_heads=2, n_layers=4)
model = LanguageModel(config)
mx.eval(model.parameters())
print(f"Parameters: {model.count_parameters():,}")

# Train
trainer = Trainer(model, TrainConfig(learning_rate=1e-3, max_steps=100))
```

See the [Quickstart guide](https://michaelellis003.github.io/lmt-metal/getting-started/quickstart/) for a complete walkthrough.

## Recipes

Ready-to-run scripts in `recipes/`:

```bash
uv run python recipes/train_tiny_gpt.py              # Train a tiny GPT
uv run python recipes/train_llama_shakespeare.py      # LLaMA on Shakespeare
uv run python recipes/compare_training.py             # Compare architectures
uv run python recipes/compare_architectures.py        # Side-by-side architecture comparison
uv run python recipes/ablation_gpt_to_llama.py        # Feature ablation study
uv run python recipes/finetune_lora.py --rank 8       # LoRA fine-tuning
uv run python recipes/finetune_qlora.py --bits 4      # QLoRA (4-bit + LoRA)
uv run python recipes/train_dpo.py                    # DPO preference optimization
uv run python recipes/train_moe.py --experts 4        # Mixture of Experts
uv run python recipes/speculative_decoding.py         # Draft-then-verify generation
uv run python recipes/evaluate_model.py               # Evaluate with perplexity/BPB
uv run python recipes/interactive_generate.py         # Streaming token-by-token generation
uv run python recipes/checkpoint_resume.py            # Save and resume training
uv run python recipes/run_experiment.py               # Structured experiment with logging
uv run python recipes/sweep_learning_rate.py          # Hyperparameter sweep
uv run python recipes/load_pretrained.py              # Load HuggingFace model
uv run python recipes/profile_models.py               # Architecture profiling
uv run python recipes/benchmark_compile.py            # mx.compile speedup benchmark
```

## CLI

```bash
lmt-metal list                    # List all architectures
lmt-metal info llama --tiny       # Show config details
lmt-metal count deepseek --detail # Parameter breakdown
```

## Design principles

- **Educational first.** Code is written for clarity, not maximum performance.
- **MLX-native.** Uses MLX idioms directly: `nn.value_and_grad`, `mx.compile`, unified memory.
- **Config factories, not subclasses.** Architecture variants are configs, not class hierarchies.
- **Progressive complexity.** Start with GPT-style, swap in LLaMA-style, then try MLA. Same model class throughout.

## Requirements

- Python 3.12+
- Apple Silicon Mac (M1 or later) for GPU acceleration
- MLX will also run on Intel Macs and Linux using CPU

## Development

```bash
git clone https://github.com/michaelellis003/lmt-metal.git
cd lmt-metal
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Build docs
mkdocs serve
```

## Documentation

Full documentation at [michaelellis003.github.io/lmt-metal](https://michaelellis003.github.io/lmt-metal/).

- [Quickstart](https://michaelellis003.github.io/lmt-metal/getting-started/quickstart/)
- [Architecture Overview](https://michaelellis003.github.io/lmt-metal/architecture/overview/)
- [MLX Idioms](https://michaelellis003.github.io/lmt-metal/architecture/mlx-idioms/)
- [Models Comparison](https://michaelellis003.github.io/lmt-metal/models/)
- [Data Pipeline](https://michaelellis003.github.io/lmt-metal/data/)
- [Training](https://michaelellis003.github.io/lmt-metal/training/)
- [Inference](https://michaelellis003.github.io/lmt-metal/inference/)
- [API Reference](https://michaelellis003.github.io/lmt-metal/api/)

## License

MIT
