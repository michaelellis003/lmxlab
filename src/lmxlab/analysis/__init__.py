"""Analysis and interpretability tools for language models."""

from lmxlab.analysis.activations import ActivationCapture
from lmxlab.analysis.attention import (
    extract_attention_maps,
)
from lmxlab.analysis.probing import LinearProbe, probe_accuracy, train_probe

__all__ = [
    "ActivationCapture",
    "LinearProbe",
    "extract_attention_maps",
    "probe_accuracy",
    "train_probe",
]
