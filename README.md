# lmxlab

A research platform for language model experimentation on Apple Silicon.

[![CI](https://github.com/michaelellis003/lmxlab/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/lmxlab/actions/workflows/ci.yml)
[![Docs](https://github.com/michaelellis003/lmxlab/actions/workflows/docs.yml/badge.svg)](https://michaelellis003.github.io/lmxlab/)

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

## What's included

- **24 architectures** as config factories: GPT, LLaMA, Gemma, Gemma 3 (sliding window), Qwen, Qwen 3 MoE, Qwen 3.5 (hybrid DeltaNet), Qwen-Next (gated attention), Mixtral (MoE), DeepSeek V2/V3 (MLA + MoE), Nemotron (hybrid Mamba-Transformer MoE), Llama 4 Scout/Maverick (iRoPE + chunked attention), Mistral Small (sliding window), OLMo 2 (QK-norm), GPT-OSS (QK-norm), Grok (SharedExpertMoE), Kimi K2.5 (DeltaNet + MoE), SmolLM3 (iRoPE), Falcon H1 (hybrid Mamba-2), Jamba (Mamba-2 + MoE), Bamba (hybrid Mamba-2), GLM-4.5 (MLA NoPE)
- **Building blocks**: MHA, GQA, MLA, GatedGQA, SlidingWindowGQA, ChunkedGQA, SparseGQA (DSA), Mamba-2 SSD, Mamba-3 (trapezoidal), GatedDeltaNet, MoE, SharedExpertMoE, LatentMoE, QK-norm, SwiGLU, squared ReLU
- **Compiled training** with `mx.compile`, functional gradients, gradient clipping, cosine schedules, dropout, muP parameterization
- **Advanced training**: DPO, GRPO, multi-token prediction, curriculum learning, knowledge distillation
- **LoRA & QLoRA**: parameter-efficient fine-tuning with optional 4-bit quantization
- **Inference**: autoregressive generation, speculative decoding, best-of-N sampling, beam search, reward model scoring
- **HuggingFace integration**: load pretrained weights from the Hub
- **Experiment framework**: time/FLOP-budgeted runs, MLflow tracking, results logging, hyperparameter sweeps, MLX profiling
- **35 recipe scripts**: training, fine-tuning, DPO, GRPO, MTP, distillation, curriculum learning, DeltaNet hybrid, MoE, best-of-N sampling, evaluation, quantization, callbacks, optimizer comparison, KV cache analysis, experiment sweeps, benchmarking

## Quick start

```bash
pip install lmxlab
```

```python
import mlx.core as mx
from lmxlab.models.llama import llama_config
from lmxlab.models.base import LanguageModel
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

# Build a small LLaMA
config = llama_config(vocab_size=256, d_model=128, n_heads=4, n_kv_heads=2, n_layers=4)
model = LanguageModel(config)
mx.eval(model.parameters())
print(f"Parameters: {model.count_parameters():,}")

# Train
trainer = Trainer(model, TrainConfig(learning_rate=1e-3, max_steps=100))
```

See the [Quickstart guide](https://michaelellis003.github.io/lmxlab/getting-started/quickstart/) for a complete walkthrough.

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
uv run python recipes/train_grpo.py                   # GRPO reward optimization
uv run python recipes/train_curriculum.py              # Curriculum learning
uv run python recipes/train_mtp.py --n-predict 2      # Multi-token prediction
uv run python recipes/train_deltanet.py                # Hybrid DeltaNet vs GQA
uv run python recipes/train_moe.py --experts 4        # Mixture of Experts
uv run python recipes/advanced_sampling.py             # Best-of-N and majority vote
uv run python recipes/speculative_decoding.py         # Draft-then-verify generation
uv run python recipes/evaluate_model.py               # Evaluate with perplexity/BPB
uv run python recipes/interactive_generate.py         # Streaming token-by-token generation
uv run python recipes/checkpoint_resume.py            # Save and resume training
uv run python recipes/run_experiment.py               # Structured experiment with logging
uv run python recipes/sweep_learning_rate.py          # Hyperparameter sweep
uv run python recipes/load_pretrained.py              # Load HuggingFace model
uv run python recipes/profile_models.py               # Architecture profiling
uv run python recipes/benchmark_compile.py            # mx.compile speedup benchmark
uv run python recipes/distill_model.py                # Knowledge distillation
uv run python recipes/quantize_and_generate.py        # 4-bit/8-bit quantization
uv run python recipes/train_with_callbacks.py         # Logging, throughput, early stopping
uv run python recipes/train_with_datasets.py          # TextDataset vs TokenDataset
uv run python recipes/compare_schedules.py            # LR schedules and optimizers
uv run python recipes/compare_optimizers.py           # Optimizer comparison (Experiment 3)
uv run python recipes/compare_kv_cache.py             # MLA vs GQA KV cache (Experiment 4)
uv run python recipes/analyze_experiments.py          # Statistical analysis tools
```

## CLI

```bash
lmxlab list                    # List all architectures
lmxlab info llama --tiny       # Show config details
lmxlab count deepseek --detail # Parameter breakdown
```

## Design principles

- **Clarity for rapid iteration.** Code is written to be read and modified quickly, not for maximum production performance.
- **MLX-native.** Uses MLX idioms directly: `nn.value_and_grad`, `mx.compile`, unified memory.
- **Config factories, not subclasses.** Architecture variants are configs, not class hierarchies.
- **Progressive complexity.** Start with GPT-style, swap in LLaMA-style, then try MLA or Mamba. Same model class throughout.
- **Reproducible experiments.** Time/FLOP budgets, train/val splits, MLflow tracking, and structured results logging.

## Requirements

- Python 3.12+
- Apple Silicon Mac (M1 or later) for GPU acceleration
- MLX will also run on Intel Macs and Linux using CPU

## Development

```bash
git clone https://github.com/michaelellis003/lmxlab.git
cd lmxlab
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/ recipes/

# Build docs
uv run mkdocs serve
```

## Documentation

Full documentation at [michaelellis003.github.io/lmxlab](https://michaelellis003.github.io/lmxlab/).

- [Quickstart](https://michaelellis003.github.io/lmxlab/getting-started/quickstart/)
- [Architecture Overview](https://michaelellis003.github.io/lmxlab/architecture/overview/)
- [MLX Idioms](https://michaelellis003.github.io/lmxlab/architecture/mlx-idioms/)
- [Models Comparison](https://michaelellis003.github.io/lmxlab/models/)
- [Data Pipeline](https://michaelellis003.github.io/lmxlab/data/)
- [Training](https://michaelellis003.github.io/lmxlab/training/)
- [Inference](https://michaelellis003.github.io/lmxlab/inference/)
- [Recipes](https://michaelellis003.github.io/lmxlab/recipes/)
- [Production Optimizations](https://michaelellis003.github.io/lmxlab/architecture/production-optimizations/)
- [Experiment Methodology](https://michaelellis003.github.io/lmxlab/experiments/methodology/)
- [Developer Log](https://michaelellis003.github.io/lmxlab/devlog/)
- [API Reference](https://michaelellis003.github.io/lmxlab/api/)

## License

MIT
