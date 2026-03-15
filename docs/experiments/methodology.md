# Experiment Methodology

lmxlab includes an experiment framework inspired by Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) patterns.
This page explains how experiments are designed, run, and tracked.

## Core Principles

### FLOP-Matched Comparisons

Architecture comparisons use FLOP-matched compute budgets (DEC-004).
Each architecture trains until it consumes the same total
floating-point operations, isolating architectural quality from
implementation speed.

```python
from lmxlab.training.callbacks import FLOPCounter

# Each architecture gets the same compute budget
flop_counter = FLOPCounter(
    flops_per_step=estimate_flops_per_step(model, seq_len, batch_size),
    flop_budget=1e15,  # 1 PFLOPs
)
```

FLOPs are estimated analytically using Megatron-LM-style formulas
(6 * N * D for dense transformers, with corrections for SwiGLU
gated FFNs). See `experiments/flops.py` for details.

!!! note "Time budgets as secondary metric"
    Wall-clock time budgets (DEC-001) are still available for
    efficiency benchmarks where speed is the metric of interest,
    but FLOP-matched is the primary method for architecture
    comparisons.

### Validation Split

Every experiment uses a train/val split (DEC-008). Validation
loss is the primary metric — training loss is a secondary
diagnostic only.

- **Shakespeare char-level:** 90/10 sequential split (~1.0M
  train tokens, ~111K val tokens), matching nanoGPT convention
- **TinyStories BPE:** Uses the dataset's built-in train/val
  splits from HuggingFace

Evaluation uses `shuffle=False` for deterministic results.
Periodic eval runs every 500 steps plus a final eval at the end
of training.

!!! warning "Superseded experiments"
    HYP-001 and HYP-001b had no validation split and reported
    training loss as the primary metric. This masked severe
    overfitting. Results from these runs are superseded —
    see [Results](results.md) for trusted findings.

### Git-as-Experiment-Infra

Each experiment records the git commit hash, so results are tied
to a specific code version. This provides natural reproducibility
without external tooling.

### Simplicity Bias

When two approaches achieve similar metrics, prefer the simpler one
(fewer parameters, less code, fewer hyperparameters). The `simplicity_score`
function quantifies this:

```python
from lmxlab.experiments.analysis import simplicity_score

# Rewards improvements that use fewer parameters
score = simplicity_score(
    entry,
    baseline_params=1_000_000,
    baseline_metric=3.5,
)
```

### Multi-Seed Runs

Single-seed results are unreliable. Run experiments with multiple
seeds and report mean +/- std:

```bash
uv run python recipes/hyp006_dropout_norm.py  # runs 3 seeds per config
```

## Tracking

### results.jsonl

All experiments log to `experiments/results.jsonl` — a line-delimited
JSON file that's easy to parse, version-control, and analyze. Each
entry records:

| Field | Description |
|-------|-------------|
| `experiment` | Name/tag |
| `commit` | Git commit hash |
| `status` | `keep`, `discard`, or `crash` |
| `val_loss` | Validation loss |
| `val_bpb` | Bits per byte |
| `train_loss` | Final training loss |
| `param_count` | Model parameters |
| `wall_time_s` | Total wall-clock time |
| `seed` | Random seed |
| `config` | Full experiment config dict |
| `metrics` | All collected metrics |

### MLflow Integration

Experiments can optionally log to MLflow for interactive
visualization. MLflow uses a local SQLite backend by default:

```python
from lmxlab.experiments.mlflow import MLflowExperimentRunner

runner = MLflowExperimentRunner(config)
runner.start()  # logs to sqlite:///mlflow.db
```

### Status: Keep vs Discard

After each experiment, compare against previous best. If the new
result improves the target metric, mark it `keep`; otherwise `discard`.
Crashed experiments are marked `crash`. This provides a quick way
to filter results.

## Sweep Utilities

### Grid Sweep

Exhaustive search over discrete parameter values:

```python
from lmxlab.experiments.sweep import grid_sweep

for params in grid_sweep({
    "lr": [1e-4, 3e-4, 1e-3],
    "d_model": [64, 128, 256],
}):
    # params = {"lr": 1e-4, "d_model": 64}, etc.
    train_model(**params)
```

### Random Sweep

Sample from continuous ranges (often more efficient than grid
search for high-dimensional spaces):

```python
from lmxlab.experiments.sweep import random_sweep

for params in random_sweep(
    param_ranges={"lr": (1e-4, 5e-3), "d_model": (32, 256)},
    n_trials=20,
):
    train_model(**params)
```

## Profiling

MLX-specific profiling tools for understanding performance on
Apple Silicon:

```python
from lmxlab.experiments.profiling import (
    benchmark_fn,
    memory_estimate,
    profile_forward,
    profile_generation,
)

# Time any function
timing = benchmark_fn(lambda: model(tokens), n_iter=10)

# Model memory footprint
mem = memory_estimate(model)

# Forward pass throughput
fwd = profile_forward(model, tokens)
print(f"{fwd['tokens_per_sec']:.0f} tokens/sec")

# Generation speed (prefill + decode)
gen = profile_generation(model, prompt, max_tokens=50)
print(f"Prefill: {gen['prefill_ms']:.1f}ms")
print(f"Decode: {gen['decode_ms_per_token']:.1f}ms/token")
```

## Recipe Scripts

Ready-to-run experiment scripts:

| Recipe | Experiment | Description |
|--------|-----------|-------------|
| `run_experiment.py` | General | Structured experiment with tracking |
| `hyp006_dropout_norm.py` | HYP-006 | Dropout x normalization at 30M params |
| `hybrid_baselines.py` | Hybrid | 5-architecture comparison at 10M params |
| `sweep_learning_rate.py` | General | Grid/random learning rate sweep |
| `benchmark_compile.py` | General | `mx.compile` speedup measurement |
| `profile_models.py` | General | Architecture profiling comparison |
| `compare_training.py` | General | Architecture training dynamics |
| `compare_architectures.py` | General | Side-by-side architecture comparison |
| `ablation_gpt_to_llama.py` | HYP-001 | Feature ablation study |
| `compare_schedules.py` | General | LR schedules and optimizer comparison |
| `analyze_experiments.py` | General | Statistical analysis tools |

## Pre-Registered Experiments

Following Platt's strong inference and Chamberlin's multiple working
hypotheses, lmxlab pre-registers experiments with competing hypotheses
and falsification criteria **before** running them. This guards against
confirmation bias and the garden of forking paths (Gelman & Loken, 2013).

Each pre-registered experiment specifies:

1. **A question** — what we want to learn
2. **Competing hypotheses** — at least 2-4 plausible explanations
3. **Design** — controlled experimental conditions
4. **Analysis plan** — how results will be interpreted
5. **Falsification criteria** — what would disprove each hypothesis

Completed experiments with trusted results:

- **HYP-001c/d:** GPT-to-LLaMA feature ablation at 3M params
- **HYP-006:** Dropout x normalization interaction at 30M params
- **Hybrid baselines:** 5-architecture comparison at 10M params

See [Results](results.md) for findings from these experiments.

### Why Pre-Registration Matters

Without pre-registration, it's easy to unconsciously adjust hypotheses
after seeing results — finding a "story" that fits the data rather than
testing a prediction against data. Pre-registration commits to the
analysis before results are known, which:

- Makes positive results more credible (they were predicted, not retrofitted)
- Makes negative results publishable (they falsify a stated hypothesis)
- Forces clearer thinking about what we actually expect and why

## Running Your Own Experiments

```python
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner

# 1. Configure
config = ExperimentConfig(
    name="my-experiment",
    description="Testing new learning rate",
    time_budget_s=300.0,
    seed=42,
)

# 2. Run
runner = ExperimentRunner(config)
runner.start()

# ... train your model ...

# 3. Check time budget
if runner.is_time_up():
    print("Time's up!")

# 4. Log results
entry = runner.finish(
    metrics={"val_loss": 2.5, "val_bpb": 1.8},
    param_count=model.count_parameters(),
    config_dict={"lr": 1e-3, "arch": "llama"},
    status="keep",
)
```
