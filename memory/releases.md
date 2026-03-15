# Release History

Internal release notes with context beyond CHANGELOG.md.
Each release records the decision rationale, notable changes,
and lessons for future releases.

See also: `CHANGELOG.md` for the public changelog.

## Release Process

1. Ensure all CI passes on main
2. Update `CHANGELOG.md` — move items from `[Unreleased]` to new version
3. Bump version in `pyproject.toml`
4. Commit: `chore: release vX.Y.Z`
5. Create GitHub Release (triggers PyPI publish via OIDC)
6. Verify PyPI package: `pip install lmxlab==X.Y.Z`
7. Record in this file via `/release` skill

---

## v0.1.0 — 2026-03-11 (Initial Release)

**Type:** major (first public release)
**Decision:** Ship with 8 architectures, full training pipeline,
and 31 recipes to establish a comprehensive educational baseline.

**What went in:**
- 8 architecture config factories (GPT through Qwen 3.5)
- Full component registry (5 attention, 4 FFN, 2 norm, 4 position)
- Training: compiled steps, 4 optimizers, DPO/GRPO/MTP/curriculum
- LoRA/QLoRA, quantization, speculative decoding
- ExperimentRunner + analysis tools
- CLI, docs site, PyPI publishing

**What was deferred:**
- Experiment results (no pre-registered experiments run yet)
- Knowledge distillation training loop (loss function exists)
- RL with verifiable rewards (GRPO exists but no code sandbox)
- Test-time compute scaling experiments

**Lessons:**
- Comprehensive initial release establishes trust and enables
  meaningful contributions. Better to ship everything at once
  than incrementally when it's all interdependent.

---

## v0.2.0 — 2026-03-14 (Architecture Expansion)

**Type:** minor
**Decision:** Ship 16 new architectures (DeepSeek V3, Nemotron,
Llama 4, Falcon H1, Jamba, Bamba, etc.), Mamba-2/3 SSM, muP,
dropout support, SparseGQA, GRPOTrainer, beam search, RewardModel.

**What went in:** Major expansion of architecture coverage and
training capabilities. See CHANGELOG.md for full list.

**What was deferred:** Experiment results documentation, standardized
metrics pipeline.

---

## v0.3.0 — 2026-03-15 (Experiment Infrastructure)

**Type:** minor
**Decision:** Ship standardized metrics, experiment results docs,
analysis toolkit, and educational notebooks as a coherent
"experiment infrastructure" release. 9 PRs (#117-#125).

**What went in:**
- Standardized metrics pipeline (HardwareMonitor, ValTracker,
  standard_callbacks, MFU tracking, MLflow metric grouping)
- 6 experiment-specific metric callbacks
- Hardware detection (detect_peak_tflops)
- Analysis/interpretability toolkit (activations, attention, probing)
- Experiment recipes (HYP-006 at 30M, hybrid baselines at 10M)
- TinyStories BPE dataset recipe, scaled configs (10M, 30M)
- Experiment results documentation page
- 5 educational notebooks
- Bug fix: CLI registration for all 24 architectures

**What was deferred:**
- muP experiment (HYP-007 pending)
- v0.3.0 released 1 day after v0.2.0 — acceptable because the
  changes are substantial (9 PRs, 4K+ lines) and thematically
  coherent

**Lessons:**
- Rapid releases are fine when changes are substantial and CI is
  green. The v0.2.0 → v0.3.0 gap was 1 day but contained real
  work.
- Standardizing metrics before running more experiments was the
  right sequencing — future experiments automatically get
  consistent tracking.

---

*Add new release entries via `/release` skill.*
