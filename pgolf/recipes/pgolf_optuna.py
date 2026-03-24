"""Optuna TPE search for Parameter Golf numeric hyperparameters.

Uses Optuna's Tree-Parzen Estimator to optimize numeric hyperparameters
while holding the architecture fixed at our best config:
  3 unique blocks, 4 heads (head_dim=128), full MHA, 6 layers (local).

Per DEC-014: agent search for structural changes, Optuna for numeric
tuning. This recipe handles the numeric side.

Usage:
    # Run N Optuna trials (default 20):
    uv run python recipes/pgolf_optuna.py --trials 20

    # Smoke test (2 trials, 200 steps each):
    uv run python recipes/pgolf_optuna.py --trials 2 --smoke

    # Resume from existing study:
    uv run python recipes/pgolf_optuna.py --trials 10 --resume
"""

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.trial import Trial

# Reuse infrastructure from pgolf_autorun
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pgolf_autorun import (
    BASELINE,
    LOCAL_OVERRIDES,
    RESULTS_FILE,
    log_result,
    run,
)

# lmxlab experiment tracking
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

STUDY_NAME = "pgolf_optuna"
DB_PATH = Path("experiments/optuna_pgolf.db")

# Architecture is fixed at our best local config
FIXED_ARCH = {
    "UNIQUE_BLOCKS": "3",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "4",
    "NUM_LAYERS": "6",
    "ITERATIONS": "5000",
}


def objective(trial: Trial, smoke: bool = False) -> float:
    """Optuna objective: minimize val_bpb.

    Searches over numeric hyperparameters while holding architecture
    fixed. Ranges informed by competition peer review (LIT-098
    through LIT-112) and our own experiments.

    Args:
        trial: Optuna trial object.
        smoke: If True, run 200-step smoke test.

    Returns:
        Validation BPB (lower is better).
    """
    # Numeric hyperparameters to tune
    muon_momentum = trial.suggest_float(
        "muon_momentum", 0.90, 0.99,
    )
    matrix_lr = trial.suggest_float(
        "matrix_lr", 0.01, 0.08, log=True,
    )
    scalar_lr = trial.suggest_float(
        "scalar_lr", 0.01, 0.08, log=True,
    )
    warmdown_iters = trial.suggest_int(
        "warmdown_iters", 500, 5000, step=500,
    )
    qk_gain_init = trial.suggest_float(
        "qk_gain_init", 0.5, 3.0,
    )
    logit_softcap = trial.suggest_float(
        "logit_softcap", 15.0, 50.0,
    )

    env_overrides = {
        **FIXED_ARCH,
        "MUON_MOMENTUM": str(muon_momentum),
        "MATRIX_LR": str(matrix_lr),
        "SCALAR_LR": str(scalar_lr),
        "WARMDOWN_ITERS": str(warmdown_iters),
        "QK_GAIN_INIT": str(qk_gain_init),
        "LOGIT_SOFTCAP": str(logit_softcap),
    }

    config = {
        "env_overrides": env_overrides,
        "description": (
            f"Optuna trial {trial.number}: "
            f"mom={muon_momentum:.3f} "
            f"lr={matrix_lr:.4f}/{scalar_lr:.4f} "
            f"wd={warmdown_iters} "
            f"qk={qk_gain_init:.2f} "
            f"sc={logit_softcap:.1f}"
        ),
        "hypothesis": f"HYP-025-optuna-t{trial.number}",
    }

    metrics = run(config, smoke=smoke)

    # Log to JSONL for compatibility with autorun results
    log_result(metrics, config)

    val_bpb = metrics.get("val_bpb", float("inf"))

    # Print summary
    print(f"\n--- Trial {trial.number} ---")
    print(f"  val_bpb: {val_bpb:.4f}")
    print(f"  params: {json.dumps(trial.params, indent=2)}")

    return val_bpb


def main() -> None:
    """Run Optuna TPE search for Parameter Golf hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Parameter Golf",
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of Optuna trials to run (default: 20)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test (200 steps per trial)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing study database",
    )
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{DB_PATH}"

    if args.resume:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage,
        )
        print(f"Resuming study with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=0,
            ),
            load_if_exists=True,
        )

    # Enqueue baseline as first trial for comparison
    if len(study.trials) == 0:
        study.enqueue_trial(
            {
                "muon_momentum": float(BASELINE["MUON_MOMENTUM"]),
                "matrix_lr": float(BASELINE["MATRIX_LR"]),
                "scalar_lr": float(BASELINE["SCALAR_LR"]),
                "warmdown_iters": int(BASELINE["WARMDOWN_ITERS"]),
                "qk_gain_init": float(BASELINE["QK_GAIN_INIT"]),
                "logit_softcap": float(BASELINE["LOGIT_SOFTCAP"]),
            },
        )

    study.optimize(
        lambda trial: objective(trial, smoke=args.smoke),
        n_trials=args.trials,
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_bpb: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Compare to baseline
    if len(study.trials) > 1:
        baseline_trial = study.trials[0]
        improvement = baseline_trial.value - study.best_value
        print(f"\nImprovement over baseline: {improvement:+.4f} BPB")

    print(f"\nAll {len(study.trials)} trials:")
    for t in sorted(study.trials, key=lambda t: t.value or float("inf")):
        status = "BEST" if t.number == study.best_trial.number else ""
        print(
            f"  Trial {t.number:3d}: "
            f"val_bpb={t.value:.4f}  "
            f"mom={t.params.get('muon_momentum', '?'):.3f}  "
            f"lr={t.params.get('matrix_lr', '?'):.4f}  "
            f"{status}"
        )


if __name__ == "__main__":
    main()
