"""Tests for experiment framework."""

from lmxlab.experiments.analysis import (
    compare_experiments,
    compute_statistics,
    simplicity_score,
)
from lmxlab.experiments.runner import ExperimentConfig, ExperimentRunner
from lmxlab.experiments.sweep import grid_sweep, random_sweep
from lmxlab.experiments.tracking import ExperimentLog, LogEntry


class TestLogEntry:
    def test_defaults(self):
        entry = LogEntry()
        assert entry.status == "keep"
        assert entry.seed == 42

    def test_custom_fields(self):
        entry = LogEntry(
            experiment="test",
            val_bpb=1.5,
            param_count=1000,
        )
        assert entry.experiment == "test"
        assert entry.val_bpb == 1.5


class TestExperimentLog:
    def test_log_and_load(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        entry = LogEntry(experiment="exp1", val_bpb=2.0)
        log.log(entry)

        loaded = log.load()
        assert len(loaded) == 1
        assert loaded[0].experiment == "exp1"
        assert loaded[0].val_bpb == 2.0

    def test_multiple_entries(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        for i in range(5):
            log.log(LogEntry(experiment=f"exp{i}", val_bpb=float(i)))
        assert len(log.load()) == 5

    def test_best(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        log.log(LogEntry(experiment="bad", val_bpb=5.0))
        log.log(LogEntry(experiment="good", val_bpb=1.0))
        log.log(LogEntry(experiment="mid", val_bpb=3.0))

        best = log.best(metric="val_bpb")
        assert best is not None
        assert best.experiment == "good"

    def test_best_empty(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        assert log.best() is None

    def test_best_ignores_discarded(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        log.log(
            LogEntry(
                experiment="discarded",
                val_bpb=0.1,
                status="discard",
            )
        )
        log.log(LogEntry(experiment="kept", val_bpb=2.0))
        best = log.best()
        assert best is not None
        assert best.experiment == "kept"

    def test_summary(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        log.log(LogEntry(experiment="a", val_bpb=1.0, status="keep"))
        log.log(LogEntry(experiment="b", val_bpb=2.0, status="discard"))
        log.log(LogEntry(experiment="c", val_bpb=3.0, status="crash"))

        s = log.summary()
        assert s["total"] == 3
        assert s["kept"] == 1
        assert s["discarded"] == 1
        assert s["crashed"] == 1

    def test_best_higher_is_better(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        log.log(LogEntry(experiment="low", val_bpb=1.0))
        log.log(LogEntry(experiment="high", val_bpb=5.0))

        best = log.best(metric="val_bpb", lower_is_better=False)
        assert best is not None
        assert best.experiment == "high"

    def test_empty_log(self, tmp_path):
        log = ExperimentLog(tmp_path / "nonexistent.jsonl")
        assert log.load() == []
        assert log.summary()["total"] == 0


class TestExperimentRunner:
    def test_start_and_time(self):
        config = ExperimentConfig(time_budget_s=10.0)
        runner = ExperimentRunner(config)
        assert runner.time_remaining() == 10.0
        runner.start()
        assert runner.time_remaining() <= 10.0
        assert not runner.is_time_up()

    def test_finish_logs_entry(self, tmp_path):
        config = ExperimentConfig(
            name="test_exp",
            output_dir=str(tmp_path),
        )
        runner = ExperimentRunner(config)
        runner.start()
        entry = runner.finish(
            metrics={"val_loss": 2.5, "val_bpb": 1.8},
            param_count=1000,
        )
        assert entry.experiment == "test_exp"
        assert entry.val_loss == 2.5
        assert entry.wall_time_s > 0

        # Check it was logged
        loaded = runner.log.load()
        assert len(loaded) == 1


class TestGridSweep:
    def test_basic(self):
        configs = list(
            grid_sweep(
                {
                    "lr": [1e-3, 1e-4],
                    "layers": [2, 4],
                }
            )
        )
        assert len(configs) == 4
        assert {"lr": 1e-3, "layers": 2} in configs
        assert {"lr": 1e-4, "layers": 4} in configs

    def test_single_param(self):
        configs = list(grid_sweep({"x": [1, 2, 3]}))
        assert len(configs) == 3

    def test_empty(self):
        configs = list(grid_sweep({}))
        assert len(configs) == 1  # one empty dict


class TestRandomSweep:
    def test_correct_number_of_trials(self):
        configs = list(
            random_sweep(
                param_ranges={"lr": (1e-4, 1e-2), "d_model": (32, 256)},
                n_trials=5,
            )
        )
        assert len(configs) == 5

    def test_values_within_range(self):
        configs = list(
            random_sweep(
                param_ranges={"lr": (0.1, 0.5), "size": (10.0, 20.0)},
                n_trials=20,
            )
        )
        for c in configs:
            assert 0.1 <= c["lr"] <= 0.5
            assert 10.0 <= c["size"] <= 20.0

    def test_contains_all_keys(self):
        configs = list(
            random_sweep(
                param_ranges={"a": (0.0, 1.0), "b": (0.0, 1.0)},
                n_trials=3,
            )
        )
        for c in configs:
            assert "a" in c
            assert "b" in c

    def test_reproducible_with_seed(self):
        c1 = list(
            random_sweep(
                param_ranges={"x": (0.0, 1.0)},
                n_trials=5,
                seed=123,
            )
        )
        c2 = list(
            random_sweep(
                param_ranges={"x": (0.0, 1.0)},
                n_trials=5,
                seed=123,
            )
        )
        for a, b in zip(c1, c2, strict=True):
            assert a["x"] == b["x"]

    def test_single_trial(self):
        configs = list(
            random_sweep(
                param_ranges={"x": (0.0, 1.0)},
                n_trials=1,
            )
        )
        assert len(configs) == 1
        assert isinstance(configs[0]["x"], float)

    def test_log_scale(self):
        configs = list(
            random_sweep(
                param_ranges={
                    "lr": (1e-5, 1e-1),
                    "d_model": (64.0, 512.0),
                },
                n_trials=50,
                log_scale={"lr"},
            )
        )
        assert len(configs) == 50
        for c in configs:
            assert 1e-5 <= c["lr"] <= 1e-1
            assert 64.0 <= c["d_model"] <= 512.0

        # Log-scale should produce values across orders
        # of magnitude. Most uniform samples of 1e-5..1e-1
        # would cluster near 1e-1; log-scale spreads them.
        lr_values = [c["lr"] for c in configs]
        small = sum(1 for v in lr_values if v < 1e-3)
        assert small > 0, "log-scale should sample small values"

    def test_log_scale_reproducible(self):
        c1 = list(
            random_sweep(
                param_ranges={"lr": (1e-4, 1e-1)},
                n_trials=5,
                seed=99,
                log_scale={"lr"},
            )
        )
        c2 = list(
            random_sweep(
                param_ranges={"lr": (1e-4, 1e-1)},
                n_trials=5,
                seed=99,
                log_scale={"lr"},
            )
        )
        for a, b in zip(c1, c2, strict=True):
            assert a["lr"] == b["lr"]


class TestAnalysis:
    def test_compute_statistics(self):
        stats = compute_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["n"] == 5

    def test_compute_statistics_empty(self):
        stats = compute_statistics([])
        assert stats["n"] == 0

    def test_compute_statistics_single_value(self):
        stats = compute_statistics([42.0])
        assert stats["mean"] == 42.0
        assert stats["std"] == 0.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["n"] == 1

    def test_compare_experiments(self, tmp_path):
        log = ExperimentLog(tmp_path / "results.jsonl")
        log.log(LogEntry(experiment="bad", val_bpb=5.0))
        log.log(LogEntry(experiment="good", val_bpb=1.0))

        compared = compare_experiments(log, metric="val_bpb")
        assert compared[0]["experiment"] == "good"
        assert compared[1]["experiment"] == "bad"

    def test_simplicity_score(self):
        # Better metric + fewer params = positive score
        entry = LogEntry(val_bpb=1.0, param_count=500)
        score = simplicity_score(
            entry,
            baseline_params=1000,
            baseline_metric=2.0,
        )
        assert score > 0

        # Worse metric = negative score
        entry2 = LogEntry(val_bpb=3.0, param_count=500)
        score2 = simplicity_score(
            entry2,
            baseline_params=1000,
            baseline_metric=2.0,
        )
        assert score2 < 0
