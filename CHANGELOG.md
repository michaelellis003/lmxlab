# Changelog

All notable changes to lmxlab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-11

Initial release.

### Added

- **8 architecture config factories**: GPT, LLaMA, Gemma, Qwen, Mixtral (MoE),
  DeepSeek V2 (MLA), Gemma 3 (sliding window), Qwen 3.5 (hybrid DeltaNet)
- **Core components**: MHA, GQA, MLA, GatedDeltaNet, SlidingWindowGQA,
  StandardFFN, GatedFFN, MoEFFN, SharedExpertMoEFFN, RMSNorm, LayerNorm,
  RoPE, ALiBi, sinusoidal positional encoding
- **ConfigurableBlock** with typed Registry pattern for component resolution
- **LanguageModel** base class with tied/untied embeddings and KV cache
- **Compiled training** with `mx.compile`, `nn.value_and_grad`, gradient
  clipping, and cosine/linear/warmup learning rate schedules
- **Optimizers**: AdamW, Lion, Adafactor, SGD with momentum
- **Advanced training**: DPO, GRPO, multi-token prediction, curriculum
  learning, knowledge distillation
- **LoRA and QLoRA**: parameter-efficient fine-tuning with optional 4-bit
  quantization
- **Post-training quantization**: 4-bit and 8-bit via MLX native quantization,
  with dequantization support
- **Inference**: autoregressive generation with KV cache, streaming generation,
  top-k/top-p/temperature sampling, repetition penalty, stop tokens
- **Advanced inference**: best-of-N sampling, majority vote, speculative
  decoding
- **HuggingFace integration**: load pretrained weights (`load_from_hf`),
  tokenizer wrapper (`HFTokenizer`), streaming dataset (`HFDataset`)
- **Data pipeline**: CharTokenizer, TiktokenTokenizer, TextDataset,
  TokenDataset, batch iterator
- **Evaluation**: perplexity, bits-per-byte, pass@k for code generation
- **Experiment framework**: ExperimentRunner with time budgets, ExperimentLog,
  grid/random hyperparameter sweeps, statistical analysis (confidence
  intervals, Cohen's d, experiment comparison)
- **MLX profiling**: benchmark_fn, memory_estimate, profile_forward,
  profile_generation, count_parameters_by_module
- **CLI**: `lmxlab list`, `lmxlab info`, `lmxlab count`
- **Callbacks**: MetricsLogger, EarlyStopping, ThroughputMonitor
- **Checkpointing**: save/load via safetensors with JSON metadata
- **31 recipe scripts** covering training, fine-tuning, DPO, GRPO, MTP,
  distillation, curriculum learning, architecture comparison, ablation
  studies, quantization, speculative decoding, and more
- **Documentation site** with MkDocs Material: architecture guides, MLX
  idioms, model comparison, API reference, recipes index, devlog
- **PyPI publish workflow** via trusted publishing (OIDC)

[Unreleased]: https://github.com/michaelellis003/lmxlab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/michaelellis003/lmxlab/releases/tag/v0.1.0
