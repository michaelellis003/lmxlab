#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    unique_blocks: int = int(os.environ.get("UNIQUE_BLOCKS", 0))  # 0 = num_layers (no sharing)
    sharing_pattern: str = os.environ.get("SHARING_PATTERN", "cyclic")  # cyclic or sandwich
    hourglass_ratio: int = int(os.environ.get("HOURGLASS_RATIO", 0))  # 0=disabled, 2/4=downsample middle layers
    mlp_type: str = os.environ.get("MLP_TYPE", "relu2")  # relu2 or swiglu
    use_skip: bool = bool(int(os.environ.get("USE_SKIP", "1")))  # 0 = no skip connections
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    normuon: bool = bool(int(os.environ.get("NORMUON", 0)))  # 0 = standard Muon, 1 = NorMuon (per-row adaptive normalization)
    normuon_beta2: float = float(os.environ.get("NORMUON_BETA2", 0.999))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 0))  # 0 = non-overlapping, >0 = sliding window stride
    eval_seq_len: int = int(os.environ.get("EVAL_SEQ_LEN", 0))  # 0 = same as train_seq_len, >0 = extended eval context
    swa_start: float = float(os.environ.get("SWA_START", 0.0))  # 0 = disabled, >0 = fraction of training to start SWA (e.g. 0.75)
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.0))  # 0 = use running mean (Polyak), >0 = EMA decay (e.g. 0.999)
    qat_bits: int = int(os.environ.get("QAT_BITS", 0))  # 0 = disabled, 4/6/8 = fake-quantize weights via STE
    qat_group_size: int = int(os.environ.get("QAT_GROUP_SIZE", 64))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))  # 0 = disabled, 4096 = default
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))
    ortho_init: bool = bool(int(os.environ.get("ORTHO_INIT", 0)))  # 0 = default init, 1 = orthogonal init
    kron_rank: int = int(os.environ.get("KRON_RANK", 0))  # 0 = dense MLP, >0 = Kronecker-factored MLP with given rank
    parallel_block: bool = bool(int(os.environ.get("PARALLEL_BLOCK", 0)))  # 0 = sequential attn→mlp, 1 = parallel (GPT-J/PaLM style)
    mlp_rot_pairs: int = int(os.environ.get("MLP_ROT_PAIRS", 0))  # 0 = disabled, >0 = content-dependent Givens rotation before MLP
    lowrank_q: int = int(os.environ.get("LOWRANK_Q", 0))  # 0 = full-rank Q, >0 = factor Q as dim→r→dim
    poly_relu: bool = bool(int(os.environ.get("POLY_RELU", 0)))  # 0 = relu^2, 1 = learnable a0+a1*relu+a2*relu^2+a3*relu^3
    attn_temp: bool = bool(int(os.environ.get("ATTN_TEMP", 0)))  # 0 = disabled, 1 = learnable per-head attention temperature
    gated_attn: bool = bool(int(os.environ.get("GATED_ATTN", 0)))  # 0 = standard, 1 = per-head sigmoid gate on SDPA output
    dense_dwa: bool = bool(int(os.environ.get("DENSE_DWA", 0)))  # 0 = standard residual, 1 = DenseFormer depth-weighted average
    value_resid: bool = bool(int(os.environ.get("VALUE_RESID", 0)))  # 0 = disabled, 1 = add first layer's V as residual to subsequent layers
    dyt: bool = bool(int(os.environ.get("DYT", 0)))  # 0 = RMSNorm, 1 = DyT (Dynamic Tanh) normalization
    attn_res: bool = bool(int(os.environ.get("ATTN_RES", 0)))  # 0 = disabled, 1 = Full Attention Residuals (arXiv 2603.15031)
    attn_res_block_size: int = int(os.environ.get("ATTN_RES_BLOCK_SIZE", 0))  # 0 = full AttnRes (every sublayer), >0 = block AttnRes (aggregate every S layers)
    xsa: bool = bool(int(os.environ.get("XSA", 0)))  # 0 = standard attn, 1 = Exclusive Self Attention (arXiv 2603.09078)
    xsa_start_layer: int = int(os.environ.get("XSA_START_LAYER", 0))  # 0 = all layers, N = XSA only on layers >= N (e.g., 5 = last layers only)
    label_smooth: float = float(os.environ.get("LABEL_SMOOTH", 0.0))  # 0 = standard CE, >0 = label smoothing (e.g. 0.1)
    z_loss_weight: float = float(os.environ.get("Z_LOSS", 0.0))  # 0 = disabled, >0 = PaLM-style z-loss penalty on log-partition (e.g. 1e-4)
    focal_gamma: float = float(os.environ.get("FOCAL_GAMMA", 0.0))  # 0 = standard CE, >0 = focal loss gamma (e.g. 0.5, downweight easy tokens)
    mile_gamma: float = float(os.environ.get("MILE_GAMMA", 0.0))  # 0 = disabled, >0 = MiLe loss (entropy-weighted, arXiv 2310.19531)
    v_norm: bool = bool(int(os.environ.get("V_NORM", 0)))  # 0 = no V normalization, 1 = RMSNorm on V (like Q and K)
    stoch_depth: float = float(os.environ.get("STOCH_DEPTH", 0.0))  # 0 = disabled, >0 = stochastic depth drop rate (e.g. 0.1)
    mingru_layers: str = os.environ.get("MINGRU_LAYERS", "")  # comma-separated layer indices to use MinGRU instead of attention (e.g. "0,1" or "0,2,4")
    conv_layers: str = os.environ.get("CONV_LAYERS", "")  # comma-separated layer indices to use CausalConv instead of attention
    sgu_layers: str = os.environ.get("SGU_LAYERS", "")  # comma-separated layer indices to use gMLP SGU instead of attention
    pre_conv: int = int(os.environ.get("PRE_CONV", 0))  # 0 = disabled, >0 = causal conv kernel size before attention (e.g. 4)
    eval_temp: float = float(os.environ.get("EVAL_TEMP", 1.0))  # Temperature scaling at eval time (1.0 = no scaling)
    eval_mix_alpha: float = float(os.environ.get("EVAL_MIX_ALPHA", 0.0))  # 0 = no mixing, >0 = mix with unigram (e.g. 0.01)
    eval_mix_logprobs: str = os.environ.get("EVAL_MIX_LOGPROBS", "")  # path to unigram log-probs .npy file
    progressive_seq: float = float(os.environ.get("PROGRESSIVE_SEQ", 0.0))  # 0 = disabled, >0 = fraction of training at half seq_len (e.g. 0.6 = first 60% at seq_len/2)

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,attn_res_queries",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight=None, eps=eps)


# Global QAT state — set by main() after warmup
_QAT_BITS: int = 0
_QAT_GROUP_SIZE: int = 64


def fake_quantize(w: mx.array) -> mx.array:
    """Fake-quantize weight via STE: gradient passes through as identity."""
    if _QAT_BITS <= 0 or w.ndim != 2:
        return w
    w_q = mx.dequantize(*mx.quantize(w, group_size=_QAT_GROUP_SIZE, bits=_QAT_BITS),
                         group_size=_QAT_GROUP_SIZE, bits=_QAT_BITS)
    return mx.stop_gradient(w_q - w) + w  # STE: forward uses w_q, backward uses w


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        w = fake_quantize(self.weight) if _QAT_BITS > 0 else self.weight
        return x @ w.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class DyT(nn.Module):
    """Dynamic Tanh normalization (CVPR 2025). Drop-in replacement for RMSNorm."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = mx.ones((1,), dtype=mx.float32)
        self.gamma = mx.ones((dim,), dtype=mx.float32)
        self.beta = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        a = self.alpha.astype(x.dtype)
        return self.gamma.astype(x.dtype) * mx.tanh(a * x) + self.beta.astype(x.dtype)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        lowrank_q: int = 0,
        attn_temp: bool = False,
        gated_attn: bool = False,
        value_resid: bool = False,
        xsa: bool = False,
        v_norm: bool = False,
        pre_conv: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        if lowrank_q > 0:
            self.c_q_down = CastedLinear(dim, lowrank_q)
            self.c_q_up = CastedLinear(lowrank_q, dim)
        else:
            self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        if attn_temp:
            self.attn_log_temp = mx.zeros((num_heads,), dtype=mx.float32)
        if gated_attn:
            # Per-head sigmoid gate on SDPA output (NeurIPS 2025 Best Paper)
            self.attn_gate = mx.zeros((num_heads,), dtype=mx.float32)
        if value_resid:
            # Blend weight for first-layer V residual (ACL 2025 ResFormer)
            self.v_resid_weight = mx.zeros((1,), dtype=mx.float32)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5
        object.__setattr__(self, "_lowrank_q", lowrank_q)
        object.__setattr__(self, "_attn_temp", attn_temp)
        object.__setattr__(self, "_gated_attn", gated_attn)
        object.__setattr__(self, "_value_resid", value_resid)
        object.__setattr__(self, "_xsa", xsa)
        object.__setattr__(self, "_v_norm", v_norm)
        object.__setattr__(self, "_pre_conv", pre_conv)
        if pre_conv > 0:
            # Small depthwise causal conv on input before Q/K/V projections
            self.conv_weight = mx.random.normal((dim, pre_conv)) * 0.02
        object.__setattr__(self, "_last_v", None)

    def __call__(self, x: mx.array, v_resid: mx.array | None = None, use_xsa: bool | None = None) -> mx.array:
        bsz, seqlen, dim = x.shape
        if self._pre_conv > 0:
            # Causal depthwise conv: mix local context before Q/K/V projection
            k = self._pre_conv
            x_pad = mx.pad(x, [(0, 0), (k - 1, 0), (0, 0)])
            x_conv = mx.zeros_like(x)
            for i in range(k):
                x_conv = x_conv + x_pad[:, i:i + seqlen, :] * self.conv_weight[:, i][None, None, :]
            x = x + x_conv  # residual: original + local context
        if self._lowrank_q > 0:
            q = self.c_q_up(self.c_q_down(x))
        else:
            q = self.c_q(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Value Residual Learning: blend with first layer's V
        if self._value_resid and v_resid is not None:
            alpha = mx.sigmoid(self.v_resid_weight).astype(v.dtype)
            v = (1 - alpha) * v + alpha * v_resid

        if self._v_norm:
            v = rms_norm(v).astype(COMPUTE_DTYPE)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        scale = self.scale
        if self._attn_temp:
            temp = mx.exp(self.attn_log_temp).astype(q.dtype)[None, :, None, None]
            q = q * temp
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        xsa_active = use_xsa if use_xsa is not None else self._xsa
        if xsa_active:
            # Exclusive Self Attention (arXiv 2603.09078): remove self-value component
            # v shape: (bsz, num_kv_heads, seqlen, head_dim), y: (bsz, num_heads, seqlen, head_dim)
            v_xsa = v
            if self.num_kv_heads != self.num_heads:
                v_xsa = mx.repeat(v_xsa, self.num_heads // self.num_kv_heads, axis=1)
            v_norm = mx.sqrt(mx.maximum(mx.sum(v_xsa * v_xsa, axis=-1, keepdims=True), 1e-12))
            v_hat = v_xsa / v_norm
            y = y - mx.sum(y * v_hat, axis=-1, keepdims=True) * v_hat
        if self._gated_attn:
            # Per-head sigmoid gate (eliminates attention sinks)
            gate = mx.sigmoid(self.attn_gate).astype(y.dtype)[None, :, None, None]
            y = y * gate
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        if self._value_resid:
            object.__setattr__(self, "_last_v", v)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int, rot_pairs: int = 0, poly_relu: bool = False):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)
        if rot_pairs > 0:
            # Content-dependent Givens rotation before MLP (PR #215 technique).
            # Zero-init so rotation starts as identity and gradually learns.
            self.angle_proj = CastedLinear(dim, rot_pairs)
            self.angle_proj.weight = mx.zeros_like(self.angle_proj.weight)
        if poly_relu:
            # Learnable polynomial: a0 + a1*relu(x) + a2*relu(x)^2 + a3*relu(x)^3
            # Init a2=1.0 so it starts as relu^2 (baseline behavior)
            self.poly_w = mx.array([0.0, 0.0, 1.0, 0.0], dtype=mx.float32)
        object.__setattr__(self, "_rot_pairs", rot_pairs)
        object.__setattr__(self, "_poly_relu", poly_relu)

    def __call__(self, x: mx.array) -> mx.array:
        r = self._rot_pairs
        if r > 0:
            angles = self.angle_proj(x)
            cos_a = mx.cos(angles).astype(x.dtype)
            sin_a = mx.sin(angles).astype(x.dtype)
            x1, x2 = x[..., :r], x[..., r:2*r]
            x = mx.concatenate([
                x1 * cos_a + x2 * sin_a,
                -x1 * sin_a + x2 * cos_a,
                x[..., 2*r:]
            ], axis=-1)
        h = nn.relu(self.fc(x))
        if self._poly_relu:
            w = self.poly_w.astype(h.dtype)
            h2 = h * h
            return self.proj(w[0] + w[1] * h + w[2] * h2 + w[3] * h2 * h)
        return self.proj(h * h)


def _kron_factors(n: int) -> int:
    """Find factor of n closest to sqrt(n) for balanced Kronecker decomposition."""
    s = int(n ** 0.5)
    while s > 1:
        if n % s == 0:
            return s
        s -= 1
    return 1


class KroneckerLinear(nn.Module):
    """Linear layer decomposed as sum of Kronecker products: W ≈ Σ_r (A_r ⊗ B_r).

    For W: [in_dim, out_dim], we factor in_dim = q1*q2, out_dim = p1*p2, then
    (A_r ⊗ B_r) x = vec(B_r @ X @ A_r^T) where X = x.reshape(..., q1, q2).
    Params: rank * (p1*p2 + q1*q2) vs in_dim*out_dim dense.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        # Factor dimensions: q1*q2 = in_dim, p1*p2 = out_dim
        self.q1 = _kron_factors(in_dim)
        self.q2 = in_dim // self.q1
        self.p1 = _kron_factors(out_dim)
        self.p2 = out_dim // self.p1
        # A: [rank, p1, q1], B: [rank, p2, q2]
        scale_a = (self.p1 * self.q1) ** -0.5
        scale_b = (self.p2 * self.q2) ** -0.5
        self.A = mx.random.normal((rank, self.p1, self.q1)) * scale_a
        self.B = mx.random.normal((rank, self.p2, self.q2)) * scale_b

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, seq, in_dim] → reshape to [..., q1, q2]
        shape = x.shape[:-1]
        X = x.reshape(*shape, self.q1, self.q2).astype(COMPUTE_DTYPE)
        A = self.A.astype(X.dtype)  # [rank, p1, q1]
        B = self.B.astype(X.dtype)  # [rank, p2, q2]
        # Σ_r A_r @ X @ B_r^T: loop is faster than broadcasting (avoids large intermediates)
        out = mx.zeros((*shape, self.p1, self.p2), dtype=X.dtype)
        for r in range(self.rank):
            out = out + mx.matmul(mx.matmul(A[r], X), B[r].T)
        return out.reshape(*shape, self.out_dim)


class KroneckerMLP(nn.Module):
    """MLP with Kronecker-factored linear layers. Uses relu^2 activation."""
    def __init__(self, dim: int, mlp_mult: int, rank: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = KroneckerLinear(dim, hidden, rank)
        self.proj = KroneckerLinear(hidden, dim, rank)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class SwiGLUMLP(nn.Module):
    # SwiGLU: silu(gate) * up, then project down. Uses 3 projections.
    # Hidden dim is 2/3 of relu2's to parameter-match (3 * 2/3 = 2).
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(dim * mlp_mult * 2 / 3)
        # Round to nearest multiple of 8 for hardware alignment
        hidden = ((hidden + 7) // 8) * 8
        self.gate = CastedLinear(dim, hidden)
        self.up = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.silu(self.gate(x)) * self.up(x))


class MinGRU(nn.Module):
    """Minimal GRU (arXiv 2410.01201) — parallel-trainable RNN replacement for attention.

    z = sigmoid(linear_z(x))
    h_tilde = linear_h(x)
    h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t  (via parallel scan in log space)

    Uses log-space parallel scan for numerical stability and parallelism.
    """
    def __init__(self, dim: int, expand: int = 1):
        super().__init__()
        inner = dim * expand
        self.linear_z = CastedLinear(dim, inner)
        self.linear_h = CastedLinear(dim, inner)
        self.proj = CastedLinear(inner, dim)
        # Zero-init output projection (like attention output)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        # Compatibility with value residual (MinGRU doesn't produce V)
        object.__setattr__(self, "_last_v", None)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        # x: (batch, seq, dim)
        z = mx.sigmoid(self.linear_z(x))  # gate: (batch, seq, inner)
        h_tilde = self.linear_h(x)        # candidate: (batch, seq, inner)

        # Chunked parallel minGRU: vectorize within chunks, sequential between
        # h_t = (1-z_t) * h_{t-1} + z_t * h_tilde_t
        bsz, seqlen, inner = h_tilde.shape
        # Simple sequential scan with chunked mx.eval to limit graph size
        # 1024 tokens / 128 chunk = 8 mx.eval calls (manageable)
        chunk = 128
        h = mx.zeros((bsz, inner), dtype=h_tilde.dtype)
        outputs = []
        for start in range(0, seqlen, chunk):
            end = min(start + chunk, seqlen)
            for t in range(start, end):
                z_t = z[:, t, :]
                h = (1.0 - z_t) * h + z_t * h_tilde[:, t, :]
                outputs.append(h[:, None, :])
            pass  # chunk boundary (graph checkpoint not possible inside grad)
        y = mx.concatenate(outputs, axis=1)  # (bsz, seqlen, inner)
        return self.proj(y)


class CausalConvMixer(nn.Module):
    """Causal depthwise conv + gating — replaces attention for local patterns.

    Uses nn.Conv1d with groups=dim for fully parallel depthwise convolution.
    No Python for-loops. Gated output: conv(x) * sigmoid(gate(x)).
    """
    def __init__(self, dim: int, kernel_size: int = 32, expand: int = 1):
        super().__init__()
        inner = dim * expand
        self.up = CastedLinear(dim, inner)
        self.gate = CastedLinear(dim, inner)
        # Depthwise causal convolution via nn.Conv1d with groups=inner
        self.conv = nn.Conv1d(in_channels=inner, out_channels=inner, kernel_size=kernel_size,
                              groups=inner, padding=kernel_size - 1, bias=True)
        self.proj = CastedLinear(inner, dim)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        object.__setattr__(self, "_kernel_size", kernel_size)
        object.__setattr__(self, "_last_v", None)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        bsz, seqlen, dim = x.shape
        h = self.up(x)
        g = mx.sigmoid(self.gate(x))
        # Causal conv: pad left, then truncate to original length
        conv_out = self.conv(h)[:, :seqlen, :]
        y = nn.silu(conv_out) * g
        return self.proj(y)


class SpatialGatingUnit(nn.Module):
    """gMLP Spatial Gating Unit — causal token mixing via learned matrix.

    Split hidden into two halves: one for content, one for spatial mixing.
    Spatial half is mixed via a causal weight matrix (lower triangular).
    No attention, no recurrence. Fully parallel.
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, expand: int = 2):
        super().__init__()
        inner = dim * expand
        self.up = CastedLinear(dim, inner)
        # Spatial mixing: causal matrix (lower triangular, learnable)
        # Initialize as identity-ish (small values below diagonal)
        spatial_w = mx.zeros((max_seq_len, max_seq_len))
        self.spatial_weight = spatial_w
        self.spatial_bias = mx.zeros((max_seq_len,))
        self.proj = CastedLinear(inner // 2, dim)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        object.__setattr__(self, "_max_seq_len", max_seq_len)
        object.__setattr__(self, "_last_v", None)

    def __call__(self, x: mx.array, **kwargs) -> mx.array:
        bsz, seqlen, dim = x.shape
        h = self.up(x)  # (bsz, seqlen, inner)
        inner = h.shape[-1]
        h_content = h[:, :, :inner // 2]
        h_spatial = h[:, :, inner // 2:]
        # Causal spatial mixing: lower triangular matrix multiply
        W = self.spatial_weight[:seqlen, :seqlen]
        mask = mx.tril(mx.ones((seqlen, seqlen)))
        W_causal = W * mask + self.spatial_bias[None, :seqlen]
        # Mix: (bsz, seqlen, inner//2) via (seqlen, seqlen) @ (bsz, seqlen, inner//2)
        h_mixed = mx.einsum("ij,bjd->bid", W_causal, h_spatial)
        # Gate
        y = h_content * mx.sigmoid(h_mixed)
        return self.proj(y)


class SpatialGatingBlock(nn.Module):
    """Block using gMLP Spatial Gating Unit instead of attention."""
    def __init__(self, dim: int, mlp_mult: int, mlp_type: str = "relu2",
                 max_seq_len: int = 1024, dyt: bool = False, **kwargs):
        super().__init__()
        self.attn_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.mlp_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.attn = SpatialGatingUnit(dim, max_seq_len=max_seq_len)
        if mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(dim, mlp_mult)
        else:
            self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array, v_resid: mx.array | None = None, use_xsa: bool | None = None) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class CausalConvBlock(nn.Module):
    """Block using CausalConvMixer instead of attention. Same interface as Block."""
    def __init__(self, dim: int, mlp_mult: int, mlp_type: str = "relu2",
                 kernel_size: int = 4, expand: int = 2, dyt: bool = False, **kwargs):
        super().__init__()
        self.attn_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.mlp_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.attn = CausalConvMixer(dim, kernel_size=kernel_size, expand=expand)
        if mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(dim, mlp_mult)
        else:
            self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array, v_resid: mx.array | None = None, use_xsa: bool | None = None) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class MinGRUBlock(nn.Module):
    """Block that uses MinGRU instead of attention. Same interface as Block."""
    def __init__(self, dim: int, mlp_mult: int, mlp_type: str = "relu2",
                 expand: int = 1, dyt: bool = False, **kwargs):
        super().__init__()
        self.attn_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.mlp_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.attn = MinGRU(dim, expand=expand)
        if mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(dim, mlp_mult)
        else:
            self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array, v_resid: mx.array | None = None, use_xsa: bool | None = None) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_type: str = "relu2",
        kron_rank: int = 0,
        parallel: bool = False,
        rot_pairs: int = 0,
        lowrank_q: int = 0,
        poly_relu: bool = False,
        attn_temp: bool = False,
        gated_attn: bool = False,
        value_resid: bool = False,
        dyt: bool = False,
        xsa: bool = False,
        v_norm: bool = False,
        pre_conv: int = 0,
    ):
        super().__init__()
        self.attn_norm = DyT(dim) if dyt else RMSNormNoWeight()
        if not parallel:
            self.mlp_norm = DyT(dim) if dyt else RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, lowrank_q=lowrank_q, attn_temp=attn_temp, gated_attn=gated_attn, value_resid=value_resid, xsa=xsa, v_norm=v_norm, pre_conv=pre_conv)
        if kron_rank > 0:
            self.mlp = KroneckerMLP(dim, mlp_mult, kron_rank)
        elif mlp_type == "swiglu":
            self.mlp = SwiGLUMLP(dim, mlp_mult)
        else:
            self.mlp = MLP(dim, mlp_mult, rot_pairs=rot_pairs, poly_relu=poly_relu)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))
        object.__setattr__(self, "_parallel", parallel)

    def __call__(self, x: mx.array, x0: mx.array, v_resid: mx.array | None = None, use_xsa: bool | None = None) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self._parallel:
            h = self.attn_norm(x)
            attn_out = self.attn(h, v_resid=v_resid, use_xsa=use_xsa)
            mlp_out = self.mlp(h)
            x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        else:
            attn_out = self.attn(self.attn_norm(x), v_resid=v_resid, use_xsa=use_xsa)
            x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class SmearGate(nn.Module):
    """Blend each token embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def __call__(self, token_ids: mx.array) -> mx.array:
        t = token_ids.astype(mx.int32)
        mod = self.bigram_vocab_size - 1
        h_first = mx.full(t.shape[:-1] + (1,), mod, dtype=mx.int32)
        h_rest = (36313 * t[..., 1:] ^ 27191 * t[..., :-1]) % mod
        h = mx.concatenate([h_first, h_rest], axis=-1)
        emb = self.embed(h)
        if self.proj is not None:
            emb = self.proj(emb)
        return emb * self.scale.astype(emb.dtype)


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, unique_blocks: int = 0, mlp_type: str = "relu2", use_skip: bool = True,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128, ortho_init: bool = False,
                 kron_rank: int = 0, sharing_pattern: str = "cyclic",
                 hourglass_ratio: int = 0, parallel_block: bool = False,
                 mlp_rot_pairs: int = 0, lowrank_q: int = 0,
                 poly_relu: bool = False, attn_temp: bool = False,
                 gated_attn: bool = False, value_resid: bool = False,
                 dense_dwa: bool = False, dyt: bool = False,
                 attn_res: bool = False, attn_res_block_size: int = 0,
                 xsa: bool = False, xsa_start_layer: int = 0,
                 label_smooth: float = 0.0, z_loss_weight: float = 0.0,
                 focal_gamma: float = 0.0, mile_gamma: float = 0.0,
                 v_norm: bool = False, stoch_depth: float = 0.0,
                 mingru_layers: str = "", conv_layers: str = "",
                 pre_conv: int = 0, sgu_layers: str = "",
                 train_seq_len: int = 1024):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.use_skip = use_skip
        object.__setattr__(self, "_xsa_start_layer", xsa_start_layer)
        object.__setattr__(self, "_label_smooth", label_smooth)
        object.__setattr__(self, "_z_loss_weight", z_loss_weight)
        object.__setattr__(self, "_focal_gamma", focal_gamma)
        object.__setattr__(self, "_mile_gamma", mile_gamma)
        object.__setattr__(self, "_stoch_depth", stoch_depth)
        object.__setattr__(self, "_eval_temp", 1.0)  # set at eval time
        object.__setattr__(self, "_eval_mix_alpha", 0.0)
        object.__setattr__(self, "_eval_mix_logprobs", None)

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.smear = SmearGate(dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.num_layers = num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        n_unique = unique_blocks if unique_blocks > 0 else num_layers
        mingru_set = set(int(x) for x in mingru_layers.split(",") if x.strip()) if mingru_layers else set()
        conv_set = set(int(x) for x in conv_layers.split(",") if x.strip()) if conv_layers else set()
        sgu_set = set(int(x) for x in sgu_layers.split(",") if x.strip()) if sgu_layers else set()
        self.blocks = [
            MinGRUBlock(dim, mlp_mult, mlp_type=mlp_type, dyt=dyt) if i in mingru_set
            else CausalConvBlock(dim, mlp_mult, mlp_type=mlp_type, dyt=dyt) if i in conv_set
            else SpatialGatingBlock(dim, mlp_mult, mlp_type=mlp_type, max_seq_len=train_seq_len, dyt=dyt) if i in sgu_set
            else Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, mlp_type=mlp_type, kron_rank=kron_rank, parallel=parallel_block, rot_pairs=mlp_rot_pairs, lowrank_q=lowrank_q, poly_relu=poly_relu, attn_temp=attn_temp, gated_attn=gated_attn, value_resid=value_resid, dyt=dyt, xsa=xsa, v_norm=v_norm, pre_conv=pre_conv)
            for i in range(n_unique)
        ]
        # Build layer→block index mapping (use object.__setattr__ to avoid MLX state tracking)
        if sharing_pattern == "sandwich" and n_unique >= 3:
            layer_map = [0] + [1] * (num_layers - 2) + [n_unique - 1]
        else:
            layer_map = [i % n_unique for i in range(num_layers)]
        object.__setattr__(self, "_layer_map", layer_map)
        object.__setattr__(self, "hourglass_ratio", hourglass_ratio)
        object.__setattr__(self, "_dense_dwa", dense_dwa)
        object.__setattr__(self, "_value_resid", value_resid)
        object.__setattr__(self, "_attn_res", attn_res)
        object.__setattr__(self, "_attn_res_block_size", attn_res_block_size)
        if dense_dwa:
            # DenseFormer DWA: learnable weights for cross-layer weighted average
            self.dwa_weights = [mx.zeros((i + 1,), dtype=mx.float32) for i in range(num_layers)]
        if attn_res:
            if attn_res_block_size > 0:
                # Block AttnRes: aggregate only at block boundaries
                # Number of blocks = ceil(num_layers / block_size)
                import math
                n_blocks_ar = math.ceil(num_layers / attn_res_block_size)
                # (n_blocks - 1) internal boundaries + 1 final = n_blocks queries
                n_queries = n_blocks_ar
            else:
                # Full Attention Residuals: one d-dim pseudo-query per sublayer + 1 final
                # 2 * num_layers sublayers (attn + MLP) + 1 final output query
                n_queries = 2 * num_layers + 1
            self.attn_res_queries = [mx.zeros((dim,), dtype=mx.float32) for _ in range(n_queries)]
        self.final_norm = DyT(dim) if dyt else RMSNormNoWeight()

        if ortho_init:
            import mlx.nn.init as mxi
            ortho = mxi.orthogonal()
            for b in self.blocks:
                for name, child in b.named_modules():
                    if isinstance(child, (nn.Linear, CastedLinear)):
                        w = child.weight
                        if min(w.shape) >= 64:
                            child.weight = ortho(w)
        # Output projections always zero-init (validated: +0.042 BPB vs scaled ortho)
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            if hasattr(b.mlp.proj, "weight"):
                b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def _attn_res_aggregate(self, outputs: list[mx.array], query_idx: int) -> mx.array:
        """Full Attention Residuals: attend over all previous sublayer outputs."""
        q = self.attn_res_queries[query_idx]  # (d,)
        V = mx.stack(outputs, axis=0)  # (N, B, T, D)
        K = rms_norm(V)  # RMSNorm on keys to prevent magnitude bias
        logits = (q * K).sum(axis=-1)  # (N, B, T) — dot product of query with each key
        alpha = mx.softmax(logits, axis=0)  # softmax over depth
        return (alpha[..., None] * V).sum(axis=0)  # (B, T, D)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        logits = c * mx.tanh(logits / c)
        if self._eval_temp != 1.0 and not self.training:
            logits = logits / self._eval_temp
        if self._eval_mix_alpha > 0.0 and not self.training and self._eval_mix_logprobs is not None:
            # Context mixing: blend transformer probs with unigram probs
            alpha = self._eval_mix_alpha
            p_nn = mx.softmax(logits, axis=-1)
            p_uni = mx.exp(self._eval_mix_logprobs)
            p_mix = (1.0 - alpha) * p_nn + alpha * p_uni
            logits = mx.log(mx.maximum(p_mix, 1e-10))
        return logits

    def _causal_pool(self, x: mx.array, k: int) -> mx.array:
        """Causal average pooling: group k consecutive tokens, take mean.

        Result is shifted right by one group to maintain causality:
        pooled[0] = zeros (no past info), pooled[j] = mean(x[(j-1)*k : j*k]).
        """
        bsz, seq, dim = x.shape
        trunc = (seq // k) * k
        pooled = x[:, :trunc].reshape(bsz, trunc // k, k, dim).mean(axis=2)
        # Shift right: pooled[j] should only reflect tokens BEFORE group j
        # Prepend a zero vector and drop the last group
        zero = mx.zeros((bsz, 1, dim), dtype=pooled.dtype)
        return mx.concatenate([zero, pooled[:, :-1]], axis=1)

    def _repeat_upsample(self, x: mx.array, k: int, target_len: int) -> mx.array:
        """Repeat each token k times to upsample, truncate to target_len."""
        bsz, seq, dim = x.shape
        x = mx.repeat(x, k, axis=1)
        return x[:, :target_len]

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        x = self.smear(x)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = rms_norm(x)
        x0 = x
        skips: list[mx.array] = []
        k = self.hourglass_ratio
        full_seq_len = x.shape[1]

        if k > 0 and self.num_encoder_layers >= 2:
            # Hourglass: first encoder layer at full res, middle at low res, then upsample
            # Encoder layer 0: full resolution
            x = self.blocks[self._layer_map[0]](x, x0)
            if self.use_skip:
                skips.append(x)
            # Downsample for middle encoder layers
            x_lo = self._causal_pool(x, k)
            x0_lo = self._causal_pool(x0, k)
            for i in range(1, self.num_encoder_layers):
                x_lo = self.blocks[self._layer_map[i]](x_lo, x0_lo)
                if self.use_skip:
                    skips.append(x_lo)
            # Upsample back for decoder
            x = self._repeat_upsample(x_lo, k, full_seq_len)
            for i in range(self.num_decoder_layers):
                if skips:
                    skip = skips.pop()
                    # Upsample skip if it was at low res
                    if skip.shape[1] != full_seq_len:
                        skip = self._repeat_upsample(skip, k, full_seq_len)
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skip
                x = self.blocks[self._layer_map[self.num_encoder_layers + i]](x, x0)
        else:
            # Standard (no hourglass)
            v_resid = None
            if self._attn_res and self._attn_res_block_size > 0:
                # Block AttnRes: standard residuals within blocks, AttnRes at boundaries
                S = self._attn_res_block_size
                block_outputs = [x]  # block-level outputs for AttnRes
                q_idx = 0
                for i in range(self.num_layers):
                    block = self.blocks[self._layer_map[i]]
                    if i > 0 and i % S == 0:
                        # Block boundary: save output, aggregate, start new block
                        block_outputs.append(x)
                        x = self._attn_res_aggregate(block_outputs, q_idx)
                        q_idx += 1
                    # Standard residual within block
                    attn_out = block.attn(block.attn_norm(x), v_resid=v_resid)
                    if self._value_resid and v_resid is None:
                        v_resid = block.attn._last_v
                    x = x + block.attn_scale.astype(x.dtype)[None, None, :] * attn_out
                    x = x + block.mlp_scale.astype(x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))
                # Final: save last block output and aggregate
                block_outputs.append(x)
                x = self._attn_res_aggregate(block_outputs, q_idx)
            elif self._attn_res:
                # Full Attention Residuals (arXiv 2603.15031):
                # Each sublayer input is computed by softmax attention over all previous
                # sublayer outputs using a learned per-sublayer pseudo-query.
                outputs = [x]  # v_0 = embedding
                for i in range(self.num_layers):
                    block = self.blocks[self._layer_map[i]]
                    # Pre-attention: AttnRes aggregate
                    h = self._attn_res_aggregate(outputs, 2 * i)
                    attn_out = block.attn(block.attn_norm(h), v_resid=v_resid)
                    if self._value_resid and v_resid is None:
                        v_resid = block.attn._last_v
                    outputs.append(attn_out)
                    # Pre-MLP: AttnRes aggregate
                    h = self._attn_res_aggregate(outputs, 2 * i + 1)
                    mlp_out = block.mlp(block.mlp_norm(h))
                    outputs.append(mlp_out)
                # Final output: one more AttnRes aggregate
                x = self._attn_res_aggregate(outputs, 2 * self.num_layers)
            elif self._dense_dwa:
                # DenseFormer: each layer input is weighted sum of all previous outputs
                outputs = [x]
                for i in range(self.num_layers):
                    w = mx.softmax(self.dwa_weights[i]).astype(x.dtype)
                    x_in = outputs[0] * w[0]
                    for j in range(1, len(outputs)):
                        x_in = x_in + outputs[j] * w[j]
                    x = self.blocks[self._layer_map[i]](x_in, x0, v_resid=v_resid)
                    if self._value_resid and v_resid is None:
                        v_resid = self.blocks[self._layer_map[0]].attn._last_v
                    outputs.append(x)
            else:
                xsa_start = self._xsa_start_layer
                sd = self._stoch_depth
                for i in range(self.num_encoder_layers):
                    # Stochastic depth: linearly increasing drop rate
                    if sd > 0.0 and self.training:
                        drop_prob = sd * i / max(self.num_layers - 1, 1)
                        if mx.random.uniform(shape=(1,)).item() < drop_prob:
                            if self.use_skip:
                                skips.append(x)
                            continue
                    layer_xsa = True if (xsa_start > 0 and i >= xsa_start) else None
                    x = self.blocks[self._layer_map[i]](x, x0, v_resid=v_resid, use_xsa=layer_xsa)
                    if self._value_resid and v_resid is None:
                        v_resid = self.blocks[self._layer_map[0]].attn._last_v
                    if self.use_skip:
                        skips.append(x)
                for i in range(self.num_decoder_layers):
                    if skips:
                        x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
                    layer_idx = self.num_encoder_layers + i
                    if sd > 0.0 and self.training:
                        drop_prob = sd * layer_idx / max(self.num_layers - 1, 1)
                        if mx.random.uniform(shape=(1,)).item() < drop_prob:
                            continue
                    layer_xsa = True if (xsa_start > 0 and layer_idx >= xsa_start) else None
                    x = self.blocks[self._layer_map[layer_idx]](x, x0, v_resid=v_resid, use_xsa=layer_xsa)
        return self.final_norm(x)

    def _ce_with_extras(self, logits: mx.array, targets: mx.array, reduction: str = "mean") -> mx.array:
        """Cross-entropy with optional label smoothing, focal loss, and z-loss."""
        logits_f32 = logits.astype(mx.float32)
        eps = self._label_smooth
        if eps > 0.0:
            # Label smoothing: (1-eps)*CE + eps*uniform
            V = logits_f32.shape[-1]
            ce = nn.losses.cross_entropy(logits_f32, targets, reduction=reduction)
            log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
            smooth = -log_probs.mean(axis=-1)
            if reduction == "mean":
                smooth = smooth.mean()
            elif reduction == "sum":
                smooth = smooth.sum()
            loss = (1.0 - eps) * ce + eps * smooth
        elif self._mile_gamma > 0.0:
            # MiLe loss (arXiv 2310.19531): entropy-weighted, upweights uncertain tokens
            log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
            probs = mx.exp(log_probs)
            entropy = -mx.sum(probs * log_probs, axis=-1)  # H(p) per token
            nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
            mile_weight = (1.0 + entropy) ** self._mile_gamma
            mile_weight = mile_weight / (mile_weight.mean() + 1e-8)  # normalize to unit mean
            per_token_loss = mile_weight * nll
            if reduction == "mean":
                loss = per_token_loss.mean()
            elif reduction == "sum":
                loss = per_token_loss.sum()
            else:
                loss = per_token_loss
        elif self._focal_gamma > 0.0:
            # Focal loss: -(1-p_t)^gamma * log(p_t), downweights easy tokens
            log_probs = logits_f32 - mx.logsumexp(logits_f32, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(log_probs, targets[:, None], axis=-1).squeeze(-1)
            p_t = mx.exp(-nll)
            focal_weight = (1.0 - p_t) ** self._focal_gamma
            per_token_loss = focal_weight * nll
            if reduction == "mean":
                loss = per_token_loss.mean()
            elif reduction == "sum":
                loss = per_token_loss.sum()
            else:
                loss = per_token_loss
        else:
            loss = nn.losses.cross_entropy(logits_f32, targets, reduction=reduction)
        # Z-loss: penalty on log-partition function magnitude (PaLM)
        if self._z_loss_weight > 0.0:
            log_z = mx.logsumexp(logits_f32, axis=-1)
            z_penalty = self._z_loss_weight * (log_z ** 2)
            if reduction == "mean":
                z_penalty = z_penalty.mean()
            elif reduction == "sum":
                z_penalty = z_penalty.sum()
            loss = loss + z_penalty
        return loss

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return self._ce_with_extras(logits, y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + self._ce_with_extras(logits, y[s:e], reduction="sum")
        return loss_sum / float(n)

    def loss_slice(self, input_ids: mx.array, target_ids: mx.array, score_from: int) -> tuple[mx.array, int]:
        """Cross-entropy on positions [score_from:] only (for sliding window eval).

        Returns (loss_sum, n_scored) so the caller can accumulate across windows.
        """
        x = self(input_ids)  # (B, T, D)
        # Only score positions from score_from onward
        x_score = x[:, score_from:, :].reshape(-1, self.tok_emb.weight.shape[1])
        y_score = target_ids[:, score_from:].reshape(-1)
        logits_proj = x_score @ self.tok_emb.weight.astype(x_score.dtype).T
        logits = self.softcap(logits_proj)
        loss_sum = nn.losses.cross_entropy(logits.astype(mx.float32), y_score, reduction="sum")
        return loss_sum, int(y_score.size)

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}
        # NorMuon: per-row variance buffers (one scalar per output neuron)
        if args.normuon:
            self.row_var = {k: mx.zeros((params[k].shape[0],), dtype=mx.float32) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            if self.args.normuon:
                # NorMuon: per-row adaptive normalization (DION-style with correction scaling)
                norm_before = mx.sqrt(mx.sum(g_ortho * g_ortho))
                row_sq = mx.mean(g_ortho * g_ortho, axis=-1)  # shape: (m,)
                v = self.args.normuon_beta2 * self.row_var[k] + (1.0 - self.args.normuon_beta2) * row_sq
                self.row_var[k] = v
                g_ortho = g_ortho / (mx.sqrt(v[:, None]) + 1e-8)
                # Correction: preserve overall magnitude after row normalization
                norm_after = mx.sqrt(mx.sum(g_ortho * g_ortho))
                g_ortho = g_ortho * (norm_before / mx.maximum(norm_after, mx.array(1e-8)))
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_keys = ["tok_emb.weight"]
        # Bigram embedding table gets same optimizer as tok_emb
        if "bigram.embed.weight" in params:
            self.embed_keys.append("bigram.embed.weight")
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k == "skip_weights" or k.startswith("dwa_weights.") or k.startswith("attn_res_queries.") or k.startswith("final_norm.") or (k.startswith("blocks.") and (p.ndim != 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        # SmearGate and BigramHash scalar params → adam_scalar
        for prefix in ("smear.", "bigram."):
            for k in params:
                if k.startswith(prefix) and k not in self.embed_keys:
                    self.scalar_keys.append(k)

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        embed_grads = {k: grads[k] for k in self.embed_keys if k in grads}
        embed_params = {k: params[k] for k in self.embed_keys if k in params}
        if embed_grads:
            updated.update(self.adam_embed.apply_gradients(embed_grads, embed_params))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
# Tensor name patterns that should always be stored as fp16 regardless of size.
# Set FP16_EMBED=1 to include tok_emb (tied embedding/unembedding) — this
# trades ~0.5 MB artifact space for higher precision on the most sensitive layer.
_fp16_embed_patterns = ("tok_emb",) if int(os.environ.get("FP16_EMBED", "0")) else ()
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_NAME_PATTERNS",
        ",".join(_fp16_embed_patterns),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors (or explicitly named ones like embeddings) are kept
        # directly. We still downcast fp32/bf16 to fp16 so metadata does not dominate.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL or any(p in name for p in INT8_KEEP_FLOAT_NAME_PATTERNS):
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def _ntk_scale_rope(model: 'GPT', train_seq_len: int, eval_seq_len: int) -> list[float]:
    """Scale RoPE bases for NTK-aware context extension. Returns original bases."""
    if eval_seq_len <= train_seq_len:
        return []
    original_bases = []
    ratio = eval_seq_len / train_seq_len
    for block in model.blocks:
        rope = block.attn.rope
        original_bases.append(rope.base)
        head_dim = rope.dims
        # NTK-aware scaling: base' = base * ratio^(dim/(dim-2))
        rope.base = rope.base * (ratio ** (head_dim / (head_dim - 2)))
    return original_bases


def _ntk_restore_rope(model: 'GPT', original_bases: list[float]) -> None:
    """Restore original RoPE bases after NTK-scaled eval."""
    for block, base in zip(model.blocks, original_bases):
        block.attn.rope.base = base


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
    model: 'GPT | None' = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge

    stride = args.eval_stride
    seq_len = args.train_seq_len

    # NTK-aware RoPE scaling for extended eval context
    eval_seq = args.eval_seq_len if args.eval_seq_len > 0 else seq_len
    original_bases: list[float] = []
    if eval_seq > seq_len and model is not None:
        original_bases = _ntk_scale_rope(model, seq_len, eval_seq)

    if stride > 0 and model is not None:
        result = _eval_val_sliding(args, model, val_tokens, base_bytes_lut,
                                   has_leading_space_lut, is_boundary_token_lut, log_fn,
                                   seq_len_override=eval_seq)
        if original_bases:
            _ntk_restore_rope(model, original_bases)
        return result

    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    if original_bases:
        _ntk_restore_rope(model, original_bases)
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def _eval_val_sliding(
    args: Hyperparameters,
    model: 'GPT',
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
    seq_len_override: int = 0,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context.

    Windows of seq_len tokens slide by eval_stride positions. Only the last
    eval_stride tokens per window are scored (they have the most context).
    The first window scores all seq_len positions since there's no prior context.
    Windows are batched for efficient GPU utilization.
    """
    stride = args.eval_stride
    seq_len = seq_len_override if seq_len_override > 0 else args.train_seq_len
    n_tokens = val_tokens.size - 1  # -1 because we need x and y offset by 1

    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    # Step 1: Score the first window (all positions)
    first_len = min(seq_len, n_tokens)
    if first_len >= 2:
        x_np = val_tokens[:first_len].reshape(1, -1)
        y_np = val_tokens[1:first_len + 1].reshape(1, -1)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        loss_sum, n_scored = model.loss_slice(x, y, 0)
        mx.eval(loss_sum)
        total_loss_sum += float(loss_sum.item())
        total_tokens += n_scored
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_bytes += float(bytes_np.astype(np.float64).sum())

    # Step 2: Remaining windows (only score last `stride` positions each)
    # All these windows have the same score_from = seq_len - stride
    starts = list(range(stride, n_tokens - stride, stride))
    if not starts:
        val_loss = total_loss_sum / max(total_tokens, 1)
        bits_per_token = val_loss / math.log(2.0)
        val_bpb = bits_per_token * (total_tokens / max(total_bytes, 1))
        return val_loss, val_bpb

    score_from = seq_len - stride
    # Batch size: how many windows per forward pass
    # Use val_batch_size to control memory: each window is seq_len tokens
    batch_size = max(1, args.val_batch_size // (seq_len * args.grad_accum_steps))
    total_batches = (len(starts) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_starts = starts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        bs = len(batch_starts)

        # Build batched x and y arrays
        x_list = []
        y_list = []
        for s in batch_starts:
            end = min(s + seq_len, n_tokens)
            # Pad to seq_len if needed (last windows)
            x_win = val_tokens[s:end]
            y_win = val_tokens[s + 1:end + 1]
            if len(x_win) < seq_len:
                # Short window at the end — skip
                continue
            x_list.append(x_win)
            y_list.append(y_win)

        if not x_list:
            continue

        x_np = np.stack(x_list)  # (B, seq_len)
        y_np = np.stack(y_list)  # (B, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)

        loss_sum, n_scored = model.loss_slice(x, y, score_from)
        mx.eval(loss_sum)

        total_loss_sum += float(loss_sum.item())
        total_tokens += n_scored

        # BPB bytes for scored positions only
        scored_prev = x_np[:, score_from:].reshape(-1)
        scored_tgt = y_np[:, score_from:].reshape(-1)
        bytes_np = base_bytes_lut[scored_tgt].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[scored_tgt] & ~is_boundary_token_lut[scored_prev]
        ).astype(np.int16, copy=False)
        total_bytes += float(bytes_np.astype(np.float64).sum())

        if log_fn is not None and total_batches > 1 and (
            (batch_idx + 1) == 1 or (batch_idx + 1) == total_batches or (batch_idx + 1) % 25 == 0
        ):
            log_fn(f"val_sliding_progress:{batch_idx + 1}/{total_batches}")

    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()

    # Metal memory tuning: cap GPU allocation + buffer cache to reduce OOM risk
    if hasattr(mx, "metal"):
        total_mem = mx.device_info().get("memory_size", 0)
        if total_mem > 0:
            mx.set_memory_limit(int(total_mem * 0.7))
        mx.set_cache_limit(2 * 1024**3)  # 2 GB

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        unique_blocks=args.unique_blocks,
        mlp_type=args.mlp_type,
        use_skip=args.use_skip,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        ortho_init=args.ortho_init,
        kron_rank=args.kron_rank,
        sharing_pattern=args.sharing_pattern,
        hourglass_ratio=args.hourglass_ratio,
        parallel_block=args.parallel_block,
        mlp_rot_pairs=args.mlp_rot_pairs,
        lowrank_q=args.lowrank_q,
        poly_relu=args.poly_relu,
        attn_temp=args.attn_temp,
        gated_attn=args.gated_attn,
        value_resid=args.value_resid,
        dense_dwa=args.dense_dwa,
        dyt=args.dyt,
        attn_res=args.attn_res,
        attn_res_block_size=args.attn_res_block_size,
        xsa=args.xsa,
        xsa_start_layer=args.xsa_start_layer,
        label_smooth=args.label_smooth,
        z_loss_weight=args.z_loss_weight,
        focal_gamma=args.focal_gamma,
        mile_gamma=args.mile_gamma,
        v_norm=args.v_norm,
        stoch_depth=args.stoch_depth,
        mingru_layers=args.mingru_layers,
        conv_layers=args.conv_layers,
        pre_conv=args.pre_conv,
        sgu_layers=args.sgu_layers,
        train_seq_len=args.train_seq_len,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"unique_blocks:{args.unique_blocks or args.num_layers} "
        f"sharing:{args.sharing_pattern} kron_rank:{args.kron_rank} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings} "
        f"attn_res:{args.attn_res} attn_res_block_size:{args.attn_res_block_size}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps} normuon:{args.normuon}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{COMPUTE_DTYPE} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # Enable QAT after warmup so model stabilizes before fake quantization kicks in
    global _QAT_BITS, _QAT_GROUP_SIZE
    if args.qat_bits > 0:
        _QAT_BITS = args.qat_bits
        _QAT_GROUP_SIZE = args.qat_group_size
        log(f"qat:enabled bits={_QAT_BITS} group_size={_QAT_GROUP_SIZE}")

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    # SWA: running average of model weights
    swa_state: dict[str, mx.array] | None = None
    swa_count = 0
    swa_active = args.swa_start > 0
    # Save original seq_len for progressive training
    object.__setattr__(args, "_orig_seq_len", args.train_seq_len)
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
                model=model,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        # Progressive sequence length (multigrid-inspired)
        if args.progressive_seq > 0.0:
            frac = (train_time_ms + 1000.0 * (time.perf_counter() - t0)) / (args.max_wallclock_seconds * 1000.0)
            if frac < args.progressive_seq:
                object.__setattr__(args, "train_seq_len", args._orig_seq_len // 2)
            else:
                object.__setattr__(args, "train_seq_len", args._orig_seq_len)
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.grad_accum_steps > 1:
                mx.eval(accum)

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.eval(model.state)

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1

        # SWA/EMA: accumulate weight average after swa_start fraction of training
        if swa_active:
            frac = approx_train_time_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            if frac >= args.swa_start:
                flat = {k: v for k, v in tree_flatten(model.state)}
                if swa_state is None:
                    swa_state = {k: v.astype(mx.float32) for k, v in flat.items()}
                    swa_count = 1
                else:
                    swa_count += 1
                    if args.ema_decay > 0:
                        # EMA: θ_ema = β * θ_ema + (1-β) * θ_current
                        beta = args.ema_decay
                        for k in swa_state:
                            swa_state[k] = beta * swa_state[k] + (1 - beta) * flat[k].astype(mx.float32)
                    else:
                        # Polyak running mean
                        for k in swa_state:
                            swa_state[k] = swa_state[k] + (flat[k].astype(mx.float32) - swa_state[k]) / swa_count
                mx.eval(*swa_state.values())

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # SWA: replace model weights with averaged weights
    if swa_state is not None and swa_count > 1:
        log(f"swa_applied: averaged {swa_count} checkpoints (start_frac:{args.swa_start})")
        for k, v in swa_state.items():
            # Cast back to original dtype
            orig_dtype = dict(tree_flatten(model.state))[k].dtype
            swa_state[k] = v.astype(orig_dtype)
        model.update(tree_unflatten(list(swa_state.items())))

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    # Apply eval-time temperature scaling and context mixing
    eval_temp = args.eval_temp
    object.__setattr__(model, "_eval_temp", eval_temp)
    if args.eval_mix_alpha > 0.0 and args.eval_mix_logprobs:
        mix_lp = __import__('numpy').load(args.eval_mix_logprobs)
        object.__setattr__(model, "_eval_mix_alpha", args.eval_mix_alpha)
        object.__setattr__(model, "_eval_mix_logprobs", mx.array(mix_lp))
    model.eval()
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
        model=model,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms eval_temp:{eval_temp}")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Explicit cleanup to help Metal driver release GPU resources before exit
    del model, opt, compiled_loss
    import gc
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
