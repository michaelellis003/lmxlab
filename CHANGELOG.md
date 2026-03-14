# Changelog

All notable changes to lmxlab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **16 new architecture config factories**: DeepSeek V3 (MLA + MoE),
  Nemotron (hybrid Mamba-Transformer MoE), Llama 4 Scout/Maverick
  (iRoPE + chunked attention + MoE), Mistral Small (sliding window),
  OLMo 2 (QK-norm), GPT-OSS (QK-norm), Grok (SharedExpertMoE),
  Kimi K2.5 (DeltaNet + MoE), Qwen-Next (gated attention),
  SmolLM3 (iRoPE), Qwen 3 MoE, Falcon H1 (hybrid Mamba-2),
  Jamba (Mamba-2 + MoE), Bamba (hybrid Mamba-2), GLM-4.5 (MLA NoPE)
- **Mamba-2 SSD**: structured state-space sequence mixer with chunked
  parallel scan and recurrent inference paths
- **Mamba-3**: trapezoidal discretization, BCNorm, complex A
  (data-dependent RoPE on B/C)
- **QK-norm**: per-head RMSNorm on Q and K projections (OLMo 2 style)
- **GatedGQA**: sigmoid output gating on attention
  (arXiv:2505.06708)
- **ChunkedGQA**: fixed-size local attention with per-chunk RoPE
  (Llama 4 iRoPE pattern)
- **LatentMoE**: down-project before routing for many-expert MoE
  (arXiv:2601.18089)
- **SharedExpertMoE**: shared expert alongside routed experts
  (DeepSeek V3 style)
- **ReluSquaredFFN**: squared ReLU activation (Primer / Nemotron)
- **muP parameterization**: width-independent hyperparameter transfer
- **Dropout support**: configurable dropout in attention and FFN
- **SparseGQA (DSA)**: DeepSeek Sparse Attention with compressed tokens,
  selected tokens, and sliding window (arXiv:2512.02556)
- **GRPOTrainer**: full GRPO training loop with group sampling, reward
  scoring, and clipped surrogate objective (arXiv:2501.12948)
- **Beam search**: standard beam search with optional custom scoring
- **RewardModel**: language model + scalar head for reward scoring

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
