"""Convert HuggingFace model weights to lmt-metal format.

Provides weight name mappings and config extraction for loading
pretrained HuggingFace models into lmt-metal's LanguageModel.

Supported architectures:
    - LLaMA / Llama 2 / Llama 3
    - Gemma / Gemma 2
    - Qwen / Qwen 2
    - Mistral

Example::

    from lmt_metal.models.convert import (
        config_from_hf,
        convert_weights,
        load_from_hf,
    )

    # Option 1: Full pipeline (requires huggingface_hub)
    model = load_from_hf('meta-llama/Llama-3.2-1B')

    # Option 2: Manual conversion
    config = config_from_hf(hf_config_dict)
    weights = convert_weights(hf_weights_dict, 'llama')
    model = LanguageModel(config)
    model.load_weights(list(weights.items()))
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from lmt_metal.core.config import BlockConfig, ModelConfig
from lmt_metal.models.base import LanguageModel

# ── Weight name mapping functions ──────────────────────────────────


def _llama_weight_map(hf_name: str) -> str | None:
    """Map HuggingFace LLaMA weight name to lmt-metal name.

    Args:
        hf_name: HuggingFace parameter name.

    Returns:
        lmt-metal parameter name, or None if the key should
        be skipped (e.g., rotary embedding inv_freq).
    """
    # Embedding
    if hf_name == "model.embed_tokens.weight":
        return "embed.weight"

    # Final norm
    if hf_name == "model.norm.weight":
        return "final_norm.weight"

    # LM head
    if hf_name == "lm_head.weight":
        return "head.weight"

    # Layer-level parameters
    m = re.match(r"model\.layers\.(\d+)\.(.+)", hf_name)
    if not m:
        return None

    idx = m.group(1)
    rest = m.group(2)

    # Attention projections
    attn_m = re.match(
        r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(.+)",
        rest,
    )
    if attn_m:
        proj, param = attn_m.groups()
        return f"blocks.{idx}.attention.{proj}.{param}"

    # FFN projections (gate_proj, up_proj, down_proj)
    ffn_m = re.match(
        r"mlp\.(gate_proj|up_proj|down_proj)\.(.+)",
        rest,
    )
    if ffn_m:
        proj, param = ffn_m.groups()
        name_map = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }
        return f"blocks.{idx}.ffn.{name_map[proj]}.{param}"

    # Layer norms
    if rest.startswith("input_layernorm."):
        param = rest.split(".", 1)[1]
        return f"blocks.{idx}.attn_norm.{param}"

    if rest.startswith("post_attention_layernorm."):
        param = rest.split(".", 1)[1]
        return f"blocks.{idx}.ffn_norm.{param}"

    # Skip unknown (rotary_emb.inv_freq, etc.)
    return None


# Gemma uses the same HF naming as LLaMA
_gemma_weight_map = _llama_weight_map

# Qwen2 and Mistral also use the same HF naming convention
_qwen_weight_map = _llama_weight_map
_mistral_weight_map = _llama_weight_map


# Registry of weight map functions
WeightMapFn = Callable[[str], str | None]

WEIGHT_MAPS: dict[str, WeightMapFn] = {
    "llama": _llama_weight_map,
    "gemma": _gemma_weight_map,
    "gemma2": _gemma_weight_map,
    "qwen2": _qwen_weight_map,
    "mistral": _mistral_weight_map,
}


# ── Weight conversion ──────────────────────────────────────────────


def convert_weights(
    hf_weights: dict[str, mx.array],
    arch: str,
) -> dict[str, mx.array]:
    """Convert HuggingFace weight dict to lmt-metal naming.

    Args:
        hf_weights: Dictionary of HF parameter names to arrays.
        arch: Architecture name (e.g., 'llama', 'gemma').

    Returns:
        Dictionary with lmt-metal parameter names.

    Raises:
        KeyError: If arch is not in WEIGHT_MAPS.
    """
    if arch not in WEIGHT_MAPS:
        raise KeyError(
            f"Unknown architecture '{arch}'. "
            f"Available: {list(WEIGHT_MAPS.keys())}"
        )

    wmap = WEIGHT_MAPS[arch]
    converted = {}
    for hf_name, arr in hf_weights.items():
        lmt_name = wmap(hf_name)
        if lmt_name is not None:
            converted[lmt_name] = arr

    return converted


# ── Config extraction ──────────────────────────────────────────────


def config_from_hf(
    hf_config: dict,
) -> ModelConfig:
    """Create a ModelConfig from a HuggingFace config dict.

    Reads ``config.json`` fields and maps them to lmt-metal's
    BlockConfig and ModelConfig.

    Args:
        hf_config: Parsed HuggingFace config.json dict.

    Returns:
        ModelConfig matching the HF model architecture.

    Raises:
        ValueError: If model_type is not supported.
    """
    model_type = hf_config.get("model_type", "")

    # LLaMA-family (llama, gemma, qwen2, mistral)
    llama_types = {"llama", "gemma", "gemma2", "qwen2", "mistral"}
    if model_type not in llama_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {sorted(llama_types)}"
        )

    n_heads = hf_config["num_attention_heads"]
    block = BlockConfig(
        attention="gqa",
        ffn="gated",
        norm="rms_norm",
        position="rope",
        d_model=hf_config["hidden_size"],
        n_heads=n_heads,
        n_kv_heads=hf_config.get("num_key_value_heads", n_heads),
        d_ff=hf_config["intermediate_size"],
        bias=False,
        rope_theta=hf_config.get("rope_theta", 10000.0),
        max_seq_len=hf_config.get("max_position_embeddings", 4096),
        pre_norm=True,
    )

    return ModelConfig(
        block=block,
        vocab_size=hf_config["vocab_size"],
        n_layers=hf_config["num_hidden_layers"],
        tie_embeddings=hf_config.get("tie_word_embeddings", False),
    )


# ── Full loading pipeline ─────────────────────────────────────────


def load_from_hf(
    repo_id: str,
    revision: str | None = None,
    dtype: mx.Dtype = mx.float16,
    quantize: int | None = None,
) -> tuple[LanguageModel, ModelConfig]:
    """Download and load a HuggingFace model into lmt-metal.

    Requires the ``huggingface_hub`` package.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'meta-llama/Llama-3.2-1B').
        revision: Git revision (branch, tag, or commit hash).
        dtype: Target dtype for weights (default: float16).
        quantize: If set, quantize the model to this many bits
            (4 or 8) after loading. Reduces memory usage.

    Returns:
        Tuple of (loaded LanguageModel, ModelConfig).

    Raises:
        ImportError: If huggingface_hub is not installed.
        ValueError: If model_type is not supported.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for load_from_hf. "
            "Install with: pip install huggingface_hub"
        ) from e

    # Download model files
    local_dir = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    local_path = Path(local_dir)

    # Load config
    config_path = local_path / "config.json"
    hf_config = json.loads(config_path.read_text())
    model_config = config_from_hf(hf_config)

    # Load weights from all safetensors files
    weight_files = sorted(local_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files found in {local_path}")

    hf_weights: dict[str, mx.array] = {}
    for wf in weight_files:
        hf_weights.update(mx.load(str(wf)))

    # Determine architecture from model_type
    arch = hf_config["model_type"]
    if arch == "gemma2":
        arch = "gemma2"
    elif arch in ("gemma",):
        arch = "gemma"
    # else: llama, qwen2, mistral map directly

    # Convert weight names
    lmt_weights = convert_weights(hf_weights, arch)

    # Cast to target dtype
    if dtype != mx.float32:
        lmt_weights = {k: v.astype(dtype) for k, v in lmt_weights.items()}

    # Build model and load weights
    model = LanguageModel(model_config)
    model.load_weights(list(lmt_weights.items()))

    # Optional post-load quantization
    if quantize is not None:
        from lmt_metal.core.quantize import quantize_model

        quantize_model(model, bits=quantize)

    return model, model_config
