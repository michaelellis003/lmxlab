# Changelog

All notable changes to lmxlab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-03-15

### Fixed

- Sync `uv.lock` with v0.3.0 version bump

## [0.3.0] - 2026-03-15

### Added

- Standardized metrics callbacks (ThroughputMonitor, FLOPCounter, etc.)
- Hardware detection for Apple Silicon MFU calculation
- Analysis utilities: activation capture, attention maps, layer ablation

### Changed

- ThroughputMonitor and FLOPCounter now inject metrics into the dict
  instead of printing

### Fixed

- All 24 architecture configs registered in CLI

## [0.2.0] - 2026-03-14

### Added

- 16 new architecture configs (DeepSeek V3, Nemotron, Llama 4, Mistral Small,
  OLMo 2, GPT-OSS, Grok, Kimi K2.5, Qwen-Next, SmolLM3, Qwen 3 MoE,
  Falcon H1, Jamba, Bamba, GLM-4.5)
- Mamba-2 SSD and Mamba-3 sequence mixers
- QK-norm, GatedGQA, ChunkedGQA, SparseGQA (DSA), LatentMoE, SharedExpertMoE
- GRPOTrainer, beam search, RewardModel
- muP parameterization, dropout support

## [0.1.0] - 2026-03-11

Initial release with 8 architecture configs, compiled training, inference
with KV cache, LoRA/QLoRA, DPO/GRPO, HuggingFace integration, CLI,
and 31 recipe scripts.

[Unreleased]: https://github.com/michaelellis003/lmxlab/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/michaelellis003/lmxlab/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/michaelellis003/lmxlab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/michaelellis003/lmxlab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/michaelellis003/lmxlab/releases/tag/v0.1.0
