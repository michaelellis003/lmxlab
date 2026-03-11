# Experiment Methodology

lmt-metal includes an experiment framework inspired by Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) patterns.
This page explains how experiments are designed, run, and tracked.

## Core Principles

### Fixed Time Budget

Every experiment has a wall-clock time budget (default: 5 minutes).
This eliminates timing confounds and makes all experiments directly
comparable — "what's the best loss achievable in N minutes?"

```python
from lmt_metal.experiments.runner import ExperimentConfig

config = ExperimentConfig(
    name="llama-lr-sweep",
    time_budget_s=300.0,  # 5 minutes
    seed=42,
)
```

### Git-as-Experiment-Infra

Each experiment records the git commit hash, so results are tied
to a specific code version. This provides natural reproducibility
without external tooling.

### Simplicity Bias

When two approaches achieve similar metrics, prefer the simpler one
(fewer parameters, less code, fewer hyperparameters). The `simplicity_score`
function quantifies this:

```python
from lmt_metal.experiments.analysis import simplicity_score

# Rewards improvements that use fewer parameters
score = simplicity_score(
    entry,
    baseline_params=1_000_000,
    baseline_metric=3.5,
)
```

### Multi-Seed Runs

Single-seed results are unreliable. Run experiments with multiple
seeds and report statistics:

```bash
uv run python recipes/run_experiment.py --arch llama --seeds 3
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

```python
from lmt_metal.experiments.tracking import ExperimentLog

log = ExperimentLog("experiments/results.jsonl")

# Load all entries
entries = log.load()

# Get the best result
best = log.best(metric="val_loss")

# Summary statistics
summary = log.summary()
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
from lmt_metal.experiments.sweep import grid_sweep

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
from lmt_metal.experiments.sweep import random_sweep

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
from lmt_metal.experiments.profiling import (
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

| Recipe | Description |
|--------|-------------|
| `run_experiment.py` | Structured experiment with tracking |
| `sweep_learning_rate.py` | Grid/random learning rate sweep |
| `benchmark_compile.py` | `mx.compile` speedup measurement |
| `profile_models.py` | Architecture profiling comparison |
| `compare_training.py` | Architecture training dynamics |
| `ablation_gpt_to_llama.py` | Feature ablation study |

## Running Your Own Experiments

```python
from lmt_metal.experiments.runner import ExperimentConfig, ExperimentRunner

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
