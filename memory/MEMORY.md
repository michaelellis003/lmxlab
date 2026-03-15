# Project Memory

## Project Overview
- Package: `lmxlab` (v0.1.0), educational MLX-based transformer library
- Package manager: uv (use `uv sync --extra dev` for dev deps)
- Python 3.12, strict mypy, ruff linting
- 8 architectures: GPT, LLaMA, Gemma, Gemma3, Qwen, Qwen3.5, Mixtral, DeepSeek V2
- Advanced training: DPO, GRPO, MTP, curriculum, distillation, LoRA/QLoRA
- CLI: `lmxlab list|info|count|bench`

## Key Patterns
- Registry pattern for pluggable components (attention, FFN, norm, position)
- `ConfigurableBlock` assembles transformer blocks from registry
- Factory functions for architecture configs (e.g. `gpt_tiny()`, `llama_7b()`)

## Gotchas
- Use `uv` CLI for dependency management
- Use raw docstrings (`r"""..."""`) when including LaTeX math
- Run tests with `uv run pytest tests/ -v`
- `mx.array.item()` returns `int | float` - always wrap in `int()` for token IDs
- Most mypy errors (152) are from incomplete MLX type stubs, not real bugs

## Research Workflow

Skills: `/hypothesis` (pre-register), `/interpret` (post-experiment), `/review` (periodic)
Agent: `reviewer` (devil's advocate, read-only, sonnet)

| Memory File | Purpose |
|-------------|---------|
| `memory/hypotheses.md` | Hypothesis registry (HYP-001 to HYP-005 active) |
| `memory/beliefs.md` | Bayesian belief tracker (B-001 to B-005) |
| `memory/lab-notebook.md` | Append-only chronological record |
| `memory/anomalies.md` | Unexpected results tracker |
| `memory/literature.md` | Literature index with evidence grades (LIT-001+) |
| `memory/decisions.md` | Lightweight ADRs (DEC-001 to DEC-006) |
| `memory/roadmap.md` | Research roadmap (R-001 to R-013) |

## Design Workflow

Skills: `/design` (pre-implementation), `/critique` (post-implementation)
Agent: `architect` (SOLID/coupling reviewer, read-only, sonnet)

| Memory File | Purpose |
|-------------|---------|
| `memory/designs.md` | Design document registry (DES-001 to DES-003) |
| `memory/interfaces.md` | Interface contracts (INT-001 to INT-006) |
| `memory/patterns.md` | Pattern catalog (PAT-001 to PAT-007) |
| `memory/tech-debt.md` | Technical debt tracker (DEBT-001 to DEBT-002) |

## SDLC Workflow

Skills: `/triage` (issues), `/pr` (PRs), `/release` (versioning), `/retro` (retrospectives)
Agent: `pr-reviewer` (PR review, read-only, sonnet)

| Memory File | Purpose |
|-------------|---------|
| `memory/ci-lessons.md` | CI/CD lessons and gotchas (CI-001 to CI-005) |
| `memory/releases.md` | Release history with internal notes (v0.1.0) |

## Agent Teams (experimental)

Enabled via `.claude/settings.json`. Pre-built configs in `/team` skill:
- `/team health` — parallel project review (research + architecture + SDLC)
- `/team interpret` — one teammate per experiment
- `/team design` — parallel research + design + test planning
- `/team review-pr` — parallel PR review from 3 angles
- Teams write to `.team-output/`; lead synthesizes into memory files
- `TaskCompleted` hook enforces output file creation

## Gotchas (additional)
- `mlx.optimizers.schedulers.SchedulerBase` doesn't exist — use `Callable[[int], float]`
- `pytest` may not be in dev deps — use `uv run --with pytest pytest`
- Time-budgeted training: loop `trainer.train_step()` with `runner.is_time_up()`, don't rely on `max_steps`
- Shakespeare data: download from karpathy/char-rnn, cache to `data/shakespeare.txt`

## Literature Review Protocol

Every `/hypothesis` now includes a Rapid Evidence Assessment (REA):
1. Search arXiv/Scholar for 30-60 min, scan ~30 papers
2. Extract findings with evidence grades (A-F, see `literature.md`)
3. Apply discount heuristics: scale transfer tax (~50%), preprint
   tax (~30%), recency bonus
4. Set literature-informed priors before designing experiments
5. Record sources in `memory/literature.md` with LIT-XXX IDs

## HYP-001b (Next Experiment)

Pre-registered with lit review (8 sources). Key design fixes from
literature:
- **d_ff=341 for SwiGLU** (was 512, giving 50% more params — LIT-002)
- **LR sweep {1e-4, 3e-4, 1e-3}** per config (muP shows fixed-LR
  is unreliable — LIT-003)
- **Step-matched budget** as Sub-experiment B
- **TinyStories BPE** as Sub-experiment C
- Literature favors H1b-d (null/scale, prior 0.65) and H1b-a
  (LR mismatch, prior 0.70)

## Current State
- On main branch
- MLflow integration added (`src/lmxlab/experiments/mlflow.py`)
- HYP-001 tested: inconclusive (GPT baseline beat LLaMA — confounded)
- HYP-001b Sub-exp A tested: H1b-a SUPPORTED, H1b-d FALSIFIED
  - Full LLaMA beats GPT by 8% at correct LR (d=1.72, large effect)
  - LR was the dominant confound: LR=1e-3 bad for all configs,
    complex architectures prefer 1e-4 or 3e-4
  - SwiGLU is largest single contributor (+48.9% of total improvement)
  - CIs include zero (n=3) — directionally clear but low power
- B-006 posterior: 0.30 -> 0.60 (LLaMA features help at correct LR)
- ANOM-001 explained (LR mismatch), ANOM-004 new (RoPE low variance)
- Next: R-001 (FLOP counter), then longer training (DEC-005, ~1 PFLOPs)
