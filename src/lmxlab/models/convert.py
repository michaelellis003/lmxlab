"""Convert HuggingFace model weights to lmxlab format.

Provides weight name mappings and config extraction for loading
pretrained HuggingFace models into lmxlab's LanguageModel.

Supported architectures:
    - LLaMA / Llama 2 / Llama 3
    - Gemma / Gemma 2
    - Qwen / Qwen 2
    - Mistral

Example::

    from lmxlab.models.convert import (
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
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from lmxlab.core.config import BlockConfig, ModelConfig
from lmxlab.models.base import LanguageModel

logger = logging.getLogger(__name__)

# ── Weight name mapping functions ──────────────────────────────────


def _llama_weight_map(hf_name: str) -> str | None:
    """Map HuggingFace LLaMA weight name to lmxlab name.

    Args:
        hf_name: HuggingFace parameter name.

    Returns:
        lmxlab parameter name, or None if the key should
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


def _nemotron_weight_map(
    pattern: str,
) -> Callable[[str], str | None]:
    """Create weight map for Nemotron-H hybrid models.

    Returns a closure that maps HF weight names to lmxlab
    names, using the hybrid pattern to determine each
    layer's type (M=Mamba, E=MoE, *=attention, -=dense).

    Args:
        pattern: Hybrid override pattern string.

    Returns:
        Weight mapping function.
    """
    def _map(hf_name: str) -> str | None:
        # Embedding
        if hf_name == 'backbone.embeddings.weight':
            return 'embed.weight'
        # Final norm
        if hf_name == 'backbone.norm_f.weight':
            return 'final_norm.weight'
        # LM head
        if hf_name == 'lm_head.weight':
            return 'head.weight'

        # Layer-level parameters
        m = re.match(
            r'backbone\.layers\.(\d+)\.(.+)', hf_name,
        )
        if not m:
            return None

        idx_str = m.group(1)
        idx = int(idx_str)
        rest = m.group(2)

        # Layer norm
        if rest == 'norm.weight':
            return f'blocks.{idx_str}.attn_norm.weight'

        # Determine layer type from pattern
        if idx >= len(pattern):
            return None
        layer_type = pattern[idx]

        # M layers: Mamba-2
        if layer_type == 'M':
            return _nemotron_mamba_map(idx_str, rest)
        # * layers: attention + relu2 FFN
        if layer_type == '*':
            return _nemotron_attn_map(idx_str, rest)
        # E layers: LatentMoE
        if layer_type == 'E':
            return _nemotron_moe_map(idx_str, rest)
        # - layers: dense MLP (no attention)
        if layer_type == '-':
            return _nemotron_dense_map(idx_str, rest)

        return None

    return _map


def _nemotron_mamba_map(
    idx: str, rest: str,
) -> str | None:
    """Map Mamba-2 layer weights."""
    # mixer.in_proj -> attention.in_proj
    if rest == 'mixer.in_proj.weight':
        return f'blocks.{idx}.attention.in_proj.weight'
    # mixer.out_proj -> attention.out_proj
    if rest == 'mixer.out_proj.weight':
        return f'blocks.{idx}.attention.out_proj.weight'
    # mixer.conv1d.weight -> attention.conv_weight
    if rest == 'mixer.conv1d.weight':
        return f'blocks.{idx}.attention.conv_weight'
    # mixer.conv1d.bias -> attention.conv_bias
    if rest == 'mixer.conv1d.bias':
        return f'blocks.{idx}.attention.conv_bias'
    # mixer.A_log -> attention.A_log
    if rest == 'mixer.A_log':
        return f'blocks.{idx}.attention.A_log'
    # mixer.D -> attention.D
    if rest == 'mixer.D':
        return f'blocks.{idx}.attention.D'
    # mixer.dt_bias -> attention.dt_bias
    if rest == 'mixer.dt_bias':
        return f'blocks.{idx}.attention.dt_bias'
    # mixer.norm.weight -> attention.norm.weight
    if rest == 'mixer.norm.weight':
        return f'blocks.{idx}.attention.norm.weight'
    return None


def _nemotron_attn_map(
    idx: str, rest: str,
) -> str | None:
    """Map attention-only layer weights.

    Nemotron-H attention (*) layers contain only Q/K/V/O
    projections — no FFN. The FFN is in separate dense (-)
    or MoE (E) layers.
    """
    # Attention projections
    attn_m = re.match(
        r'mixer\.(q_proj|k_proj|v_proj|o_proj)\.(.+)',
        rest,
    )
    if attn_m:
        proj, param = attn_m.groups()
        return f'blocks.{idx}.attention.{proj}.{param}'

    return None


def _nemotron_moe_map(
    idx: str, rest: str,
) -> str | None:
    """Map LatentMoE layer weights."""
    # Router
    if rest.startswith('mlp.router.'):
        param = rest.split('mlp.router.', 1)[1]
        return f'blocks.{idx}.ffn.router.{param}'

    # Down/up projections for latent space
    if rest.startswith('mlp.down_proj.'):
        param = rest.split('mlp.down_proj.', 1)[1]
        return f'blocks.{idx}.ffn.down_proj.{param}'
    if rest.startswith('mlp.up_proj.'):
        param = rest.split('mlp.up_proj.', 1)[1]
        return f'blocks.{idx}.ffn.up_proj.{param}'

    # Per-expert weights
    expert_m = re.match(
        r'mlp\.experts\.(\d+)\.(.+)', rest,
    )
    if expert_m:
        e_idx, param = expert_m.groups()
        return f'blocks.{idx}.ffn.experts.{e_idx}.{param}'

    # Shared expert
    if rest.startswith('mlp.shared_expert.'):
        param = rest.split('mlp.shared_expert.', 1)[1]
        return (
            f'blocks.{idx}.ffn.shared_expert.{param}'
        )

    # Score correction bias
    if rest == 'mlp.score_correction_bias':
        return (
            f'blocks.{idx}.ffn.score_correction_bias'
        )

    return None


def _nemotron_dense_map(
    idx: str, rest: str,
) -> str | None:
    """Map dense MLP layer weights (no attention).

    Dense (-) layers use ``mixer.up_proj`` /
    ``mixer.down_proj`` (not ``mlp.*``).
    """
    ffn_m = re.match(
        r'mixer\.(up_proj|down_proj)\.(.+)', rest,
    )
    if ffn_m:
        proj, param = ffn_m.groups()
        name_map = {'up_proj': 'up', 'down_proj': 'down'}
        return f'blocks.{idx}.ffn.{name_map[proj]}.{param}'
    return None


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
    pattern: str | None = None,
) -> dict[str, mx.array]:
    """Convert HuggingFace weight dict to lmxlab naming.

    Args:
        hf_weights: Dictionary of HF parameter names to arrays.
        arch: Architecture name (e.g., 'llama', 'nemotron_h').
        pattern: Hybrid override pattern (required for
            nemotron_h architecture).

    Returns:
        Dictionary with lmxlab parameter names.

    Raises:
        KeyError: If arch is not supported.
        ValueError: If pattern is required but not provided.
    """
    if arch == 'nemotron_h':
        if pattern is None:
            raise ValueError(
                "pattern is required for nemotron_h"
            )
        wmap = _nemotron_weight_map(pattern)
    elif arch in WEIGHT_MAPS:
        wmap = WEIGHT_MAPS[arch]
    else:
        raise KeyError(
            f"Unknown architecture '{arch}'. "
            f"Available: {list(WEIGHT_MAPS.keys()) + ['nemotron_h']}"
        )

    converted = {}
    for hf_name, arr in hf_weights.items():
        lmt_name = wmap(hf_name)
        if lmt_name is not None:
            converted[lmt_name] = arr

    return converted


# ── Config extraction ──────────────────────────────────────────────


def config_from_hf(
    hf_config: dict[str, Any],
) -> ModelConfig:
    """Create a ModelConfig from a HuggingFace config dict.

    Reads ``config.json`` fields and maps them to lmxlab's
    BlockConfig and ModelConfig.

    Args:
        hf_config: Parsed HuggingFace config.json dict.

    Returns:
        ModelConfig matching the HF model architecture.

    Raises:
        ValueError: If model_type is not supported.
    """
    model_type = hf_config.get("model_type", "")

    # Nemotron-H hybrid architecture
    if model_type == 'nemotron_h':
        return _config_from_nemotron_h(hf_config)

    # LLaMA-family (llama, gemma, qwen2, mistral)
    llama_types = {"llama", "gemma", "gemma2", "qwen2", "mistral"}
    if model_type not in llama_types:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {sorted(llama_types | {'nemotron_h'})}"
        )

    # Validate required keys with clear error messages
    required = [
        "num_attention_heads",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_hidden_layers",
    ]
    missing = [k for k in required if k not in hf_config]
    if missing:
        raise ValueError(f"HF config missing required keys: {missing}")

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


def _config_from_nemotron_h(
    hf_config: dict[str, Any],
) -> ModelConfig:
    """Extract ModelConfig from a Nemotron-H HF config.

    Args:
        hf_config: HF config dict with model_type='nemotron_h'.

    Returns:
        ModelConfig for Nemotron-H.
    """
    from lmxlab.models.nemotron import nemotron3_config

    pattern = hf_config.get('hybrid_override_pattern', '')
    d_model = hf_config.get('hidden_size', 4096)
    n_heads = hf_config.get('num_attention_heads', 32)
    n_kv_heads = hf_config.get('num_key_value_heads', n_heads)
    d_ff = hf_config.get('intermediate_size', 16384)

    # Mamba-2 SSM params (flat fields in HF config)
    mamba_n_heads = hf_config.get('mamba_num_heads', 128)
    ssm_state_size = hf_config.get('ssm_state_size', 128)
    mamba_expand = hf_config.get('expand', 2)
    mamba_head_dim = hf_config.get(
        'mamba_head_dim',
        (d_model * mamba_expand) // mamba_n_heads,
    )
    mamba_n_groups = hf_config.get('n_groups', 8)
    mamba_chunk_size = hf_config.get('chunk_size', 128)
    conv_kernel_size = hf_config.get('conv_kernel', 4)

    # MoE params
    n_experts = hf_config.get('num_local_experts', 64)
    top_k_experts = hf_config.get('num_experts_per_tok', 8)
    moe_latent_size = hf_config.get(
        'moe_latent_size', 1024,
    )
    moe_d_ff = hf_config.get(
        'moe_intermediate_size', 1024,
    )
    shared_expert_d_ff = hf_config.get(
        'shared_expert_intermediate_size', d_ff,
    )
    scaling = hf_config.get(
        'routed_scaling_factor', 5.0,
    )
    moe_n_groups = hf_config.get('n_group', 8)
    moe_topk_groups = hf_config.get('topk_group', 4)

    return nemotron3_config(
        hybrid_override_pattern=pattern,
        vocab_size=hf_config.get('vocab_size', 256000),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        mamba_n_heads=mamba_n_heads,
        mamba_head_dim=mamba_head_dim,
        ssm_state_size=ssm_state_size,
        mamba_expand=mamba_expand,
        mamba_n_groups=mamba_n_groups,
        mamba_chunk_size=mamba_chunk_size,
        n_experts=n_experts,
        top_k_experts=top_k_experts,
        moe_latent_size=moe_latent_size,
        moe_d_ff=moe_d_ff,
        shared_expert_d_ff=shared_expert_d_ff,
        moe_routed_scaling_factor=scaling,
        moe_n_groups=moe_n_groups,
        moe_topk_groups=moe_topk_groups,
        max_seq_len=hf_config.get(
            'max_position_embeddings', 8192,
        ),
        rope_theta=hf_config.get('rope_theta', 500000.0),
        tie_embeddings=hf_config.get(
            'tie_word_embeddings', False,
        ),
        conv_kernel_size=conv_kernel_size,
    )


# ── Full loading pipeline ─────────────────────────────────────────


def load_from_hf(
    repo_id: str,
    revision: str | None = None,
    dtype: mx.Dtype = mx.float16,
    quantize: int | None = None,
) -> tuple[LanguageModel, ModelConfig]:
    """Download and load a HuggingFace model into lmxlab.

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
        loaded = mx.load(str(wf))
        if isinstance(loaded, dict):
            hf_weights.update(loaded)

    # Determine architecture for weight mapping
    arch = hf_config["model_type"]

    # Convert weight names
    pattern = hf_config.get('hybrid_override_pattern')
    lmt_weights = convert_weights(
        hf_weights, arch, pattern=pattern,
    )

    # Cast to target dtype
    if dtype != mx.float32:
        lmt_weights = {k: v.astype(dtype) for k, v in lmt_weights.items()}

    # Build model and load weights
    model = LanguageModel(model_config)

    # Warn if converted weights don't cover all model parameters
    import mlx.utils

    model_keys = set(dict(mlx.utils.tree_flatten(model.parameters())).keys())
    loaded_keys = set(lmt_weights.keys())
    missing = model_keys - loaded_keys
    if missing:
        logger.warning(
            "Missing %d model parameters after conversion: %s",
            len(missing),
            sorted(missing)[:10],
        )

    model.load_weights(list(lmt_weights.items()))

    # Optional post-load quantization
    if quantize is not None:
        from lmxlab.core.quantize import quantize_model

        quantize_model(model, bits=quantize)

    return model, model_config
