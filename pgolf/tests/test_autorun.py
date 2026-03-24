"""Tests for autorun recipe infrastructure."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Add recipes to path so we can import the template
@pytest.fixture(autouse=True)
def _recipes_on_path():
    recipes_dir = str(Path(__file__).parent.parent / "recipes")
    if recipes_dir not in sys.path:
        sys.path.insert(0, recipes_dir)
    yield
    if recipes_dir in sys.path:
        sys.path.remove(recipes_dir)


def test_propose_returns_valid_config():
    """Default propose([]) returns dict with all required keys."""
    from autorun_template import propose

    config = propose([])
    assert isinstance(config, dict)
    assert "arch_factory" in config
    assert "train_config" in config
    assert "description" in config
    assert isinstance(config["arch_factory"], str)
    assert isinstance(config["train_config"], dict)
    assert isinstance(config["description"], str)


def test_propose_train_config_keys():
    """Train config has expected hyperparameter keys."""
    from autorun_template import propose

    config = propose([])
    tc = config["train_config"]
    assert "learning_rate" in tc
    assert "warmup_steps" in tc
    assert isinstance(tc["learning_rate"], float)
    assert isinstance(tc["warmup_steps"], int)


def test_propose_with_past_results():
    """propose() accepts past results without error."""
    from autorun_template import propose

    past = [
        {
            "val_loss": 2.5,
            "train_loss": 2.3,
            "config": {"arch_factory": "llama_10m"},
            "description": "baseline",
        },
    ]
    config = propose(past)
    assert isinstance(config, dict)
    assert "arch_factory" in config


def test_arch_registry_covers_all_names():
    """All arch names in the registry resolve to valid factories."""
    from autorun_template import ARCH_REGISTRY, _resolve_arch

    assert len(ARCH_REGISTRY) > 0
    for name in ARCH_REGISTRY:
        config = _resolve_arch(name)
        assert hasattr(config, "vocab_size")
        assert hasattr(config, "n_layers")


def test_arch_registry_rejects_unknown():
    """Unknown arch name raises KeyError."""
    from autorun_template import _resolve_arch

    with pytest.raises(KeyError, match="Unknown arch"):
        _resolve_arch("nonexistent_arch")


def test_load_past_results_empty(tmp_path):
    """load_past_results returns [] when no results exist."""
    from autorun_template import load_past_results

    # Patch the log path to a non-existent file
    with patch("autorun_template.ExperimentLog") as mock_log_cls:
        mock_log = mock_log_cls.return_value
        mock_log.load.return_value = []
        results = load_past_results()
        assert results == []


def test_results_filtering(tmp_path):
    """Only entries matching TASK_NAME with status=keep are returned."""
    from autorun_template import TASK_NAME, load_past_results

    from lmxlab.experiments.tracking import LogEntry

    keep_entry = LogEntry(experiment=TASK_NAME, status="keep", val_loss=2.0)
    discard_entry = LogEntry(
        experiment=TASK_NAME, status="discard", val_loss=3.0
    )
    other_entry = LogEntry(
        experiment="other-task", status="keep", val_loss=1.0
    )

    with patch("autorun_template.ExperimentLog") as mock_log_cls:
        mock_log = mock_log_cls.return_value
        mock_log.load.return_value = [
            keep_entry,
            discard_entry,
            other_entry,
        ]
        results = load_past_results()

    assert len(results) == 1
    assert results[0]["val_loss"] == 2.0
    assert results[0]["experiment"] == TASK_NAME
