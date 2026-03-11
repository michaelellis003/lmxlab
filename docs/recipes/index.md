# Recipes

Ready-to-run scripts in the `recipes/` directory. Each is self-contained —
it creates its own data, builds models, and prints results. No external
downloads required unless noted.

```bash
# Run any recipe
uv run python recipes/<recipe_name>.py

# Most recipes accept --help for CLI options
uv run python recipes/train_tiny_gpt.py --help
```

## Training Basics

Start here. These scripts train small models on Shakespeare text and
demonstrate the core training loop.

| Recipe | What it teaches |
|--------|----------------|
| [`train_tiny_gpt.py`](#train_tiny_gpt) | End-to-end workflow: build, train, generate |
| [`train_llama_shakespeare.py`](#train_llama_shakespeare) | BPE tokenization, compiled training |
| [`train_curriculum.py`](#train_curriculum) | Length curriculum: short sequences before long |
| [`checkpoint_resume.py`](#checkpoint_resume) | Save/load checkpoints, resume training |
| [`train_with_callbacks.py`](#train_with_callbacks) | Logging, throughput monitoring, early stopping |
| [`train_with_datasets.py`](#train_with_datasets) | TextDataset vs TokenDataset classes |
| [`compare_schedules.py`](#compare_schedules) | LR schedules and optimizer comparison |

### train_tiny_gpt

The "hello world" of lmxlab. Trains a tiny GPT on Shakespeare with
character-level tokenization. Good for verifying your install works.

```bash
uv run python recipes/train_tiny_gpt.py
```

### train_llama_shakespeare

Same task, better architecture. Uses LLaMA config (GQA, SwiGLU, RoPE)
with tiktoken BPE tokenization. Demonstrates compiled training with
`mx.compile`.

```bash
uv run python recipes/train_llama_shakespeare.py --steps 500
```

### train_curriculum

Curriculum learning: start with short sequences (easy), gradually increase
to full length. Often converges faster than training on full length from
the start.

```bash
uv run python recipes/train_curriculum.py --stages 4 --steps 500
```

### checkpoint_resume

Save training state (weights, optimizer, step count) to disk, then resume
from the checkpoint. Demonstrates lmxlab's safetensors-based checkpointing.

```bash
uv run python recipes/checkpoint_resume.py --steps 200
```

### train_with_callbacks

Runs all three callback types simultaneously: `MetricsLogger` prints
loss at intervals, `ThroughputMonitor` reports tokens/sec, and
`EarlyStopping` halts training when eval loss plateaus.

```bash
uv run python recipes/train_with_callbacks.py --patience 5 --max-steps 300
```

### train_with_datasets

Compares `TextDataset` (raw text in, handles tokenization) with
`TokenDataset` (pre-tokenized arrays). Shows that both produce
identical training windows and when to use each.

```bash
uv run python recipes/train_with_datasets.py --seq-len 64
```

### compare_schedules

Trains the same model with different LR schedule and optimizer
combinations. Compares cosine vs linear vs constant decay, and
AdamW vs Lion vs Adafactor. Shows loss curves at checkpoints.

```bash
uv run python recipes/compare_schedules.py --optimizers adamw lion
```

---

## Fine-Tuning

Parameter-efficient methods for adapting pretrained models.

| Recipe | What it teaches |
|--------|----------------|
| [`finetune_lora.py`](#finetune_lora) | LoRA: train ~0.1% of parameters |
| [`finetune_qlora.py`](#finetune_qlora) | QLoRA: 4-bit base + LoRA adapters |
| [`load_pretrained.py`](#load_pretrained) | Load HuggingFace models into lmxlab |

### finetune_lora

Apply low-rank adapters to attention layers. Freezes all base weights and
trains only small A and B matrices. After training, merge adapters back
into the base model for inference.

```bash
uv run python recipes/finetune_lora.py --rank 8 --steps 200
```

### finetune_qlora

Combine quantization and LoRA. Base weights are stored in 4-bit (or 8-bit),
while LoRA adapters train in full precision. Maximum memory efficiency for
fine-tuning large models.

```bash
uv run python recipes/finetune_qlora.py --rank 8 --bits 4
```

### load_pretrained

Download a pretrained model from HuggingFace Hub, convert weights to
lmxlab format, and generate text. Requires `huggingface_hub` and
`transformers` packages.

```bash
uv run python recipes/load_pretrained.py --repo meta-llama/Llama-3.2-1B
```

---

## Advanced Training

Specialized training objectives beyond standard next-token prediction.

| Recipe | What it teaches |
|--------|----------------|
| [`train_dpo.py`](#train_dpo) | Direct Preference Optimization |
| [`train_grpo.py`](#train_grpo) | Group Relative Policy Optimization |
| [`train_mtp.py`](#train_mtp) | Multi-Token Prediction (auxiliary heads) |
| [`train_moe.py`](#train_moe) | Mixture of Experts routing |
| [`train_deltanet.py`](#train_deltanet) | Hybrid linear + softmax attention |
| [`distill_model.py`](#distill_model) | Knowledge distillation (teacher → student) |

### train_dpo

Two-phase training: supervised fine-tuning (SFT) first, then DPO with
preference pairs. Learns "A is better than B" without a reward model.

```bash
uv run python recipes/train_dpo.py --dpo-steps 50
```

### train_grpo

SFT followed by GRPO. Generates multiple completions per prompt, scores
them, and normalizes rewards within each group. Closer to classic policy
gradient than DPO.

```bash
uv run python recipes/train_grpo.py --grpo-steps 50
```

### train_mtp

Multi-Token Prediction: each position predicts the next 2-4 tokens via
lightweight auxiliary heads. Provides richer gradients and enables
speculative decoding at inference time.

```bash
uv run python recipes/train_mtp.py --n-predict 2
```

### train_moe

Mixture of Experts: sparse routing where each token uses only top-k of N
expert FFNs. Compares dense vs MoE at matched compute budgets.

```bash
uv run python recipes/train_moe.py --experts 4 --top-k 2
```

### train_deltanet

Hybrid attention from Qwen 3.5: interleaves Gated DeltaNet (linear
attention with delta rule) and standard GQA layers. DeltaNet uses
fixed-size state — O(d^2) per token regardless of sequence length.

```bash
uv run python recipes/train_deltanet.py --steps 300
```

### distill_model

Knowledge distillation (Hinton et al., 2015): train a larger teacher
(LLaMA-tiny), then transfer its knowledge to a smaller student
(GPT-tiny) via temperature-scaled soft targets. Compares distilled
student against a baseline student trained without distillation.

```bash
uv run python recipes/distill_model.py --temperature 4 --alpha 0.7
```

---

## Inference & Sampling

Different strategies for generating text from trained models.

| Recipe | What it teaches |
|--------|----------------|
| [`interactive_generate.py`](#interactive_generate) | Streaming token-by-token generation |
| [`advanced_sampling.py`](#advanced_sampling) | Best-of-N and majority vote |
| [`speculative_decoding.py`](#speculative_decoding) | Draft-then-verify for faster generation |

### interactive_generate

Streaming generation with `stream_generate`. Tokens appear one at a time
as they are produced. Demonstrates repetition penalty and temperature
control.

```bash
uv run python recipes/interactive_generate.py --temperature 0.8
```

### advanced_sampling

Inference-time compute scaling: generate N completions and pick the best
(by log-probability), or generate N answers and take the majority vote.

```bash
uv run python recipes/advanced_sampling.py --n 8
```

### speculative_decoding

A small draft model proposes K tokens, then a larger target model verifies
them in a single forward pass. Mathematically lossless — the output
distribution is identical to the target model's. Especially natural on
unified memory where both models share the same memory pool.

```bash
uv run python recipes/speculative_decoding.py --draft-tokens 4
```

---

## Architecture Comparison

Scripts for understanding the differences between transformer architectures.

| Recipe | What it teaches |
|--------|----------------|
| [`compare_architectures.py`](#compare_architectures) | Parameter counts, KV cache sizes for all 8 architectures |
| [`compare_training.py`](#compare_training) | Training dynamics: loss curves across architectures |
| [`ablation_gpt_to_llama.py`](#ablation_gpt_to_llama) | Feature ablation: which LLaMA innovation matters most? |

### compare_architectures

Instantiates all 8 architectures at matched dimensions and compares
parameter counts, KV cache sizes, and structural differences. No training
— just model construction and analysis.

```bash
uv run python recipes/compare_architectures.py
```

### compare_training

Trains GPT, LLaMA, and DeepSeek on the same data with the same seed.
Compares loss curves and convergence behavior.

```bash
uv run python recipes/compare_training.py --steps 300
```

### ablation_gpt_to_llama

The pre-registered Experiment 1 from the [devlog](../devlog/index.md).
Starts from GPT baseline and adds LLaMA features one at a time
(RMSNorm, RoPE, SwiGLU, GQA, no-bias) to measure individual
contributions.

```bash
uv run python recipes/ablation_gpt_to_llama.py --steps 200
```

---

## Benchmarks & Profiling

Measure performance on your hardware.

| Recipe | What it teaches |
|--------|----------------|
| [`benchmark_compile.py`](#benchmark_compile) | `mx.compile` speedup measurement |
| [`profile_models.py`](#profile_models) | Memory, throughput, and generation speed |
| [`evaluate_model.py`](#evaluate_model) | Perplexity and bits-per-byte metrics |
| [`quantize_and_generate.py`](#quantize_and_generate) | 4-bit/8-bit quantization and memory comparison |

### benchmark_compile

Measures training step time with and without `mx.compile` at multiple
model sizes. Quantifies the speedup from kernel fusion on your specific
Apple Silicon chip.

```bash
uv run python recipes/benchmark_compile.py
```

### profile_models

Benchmarks forward pass, backward pass, and generation across five
architectures. Reports memory usage estimates and tokens/second.

```bash
uv run python recipes/profile_models.py
```

### evaluate_model

Trains on a train/val split and reports perplexity and bits-per-byte
(BPB). Compares GPT and LLaMA on the same evaluation set.

```bash
uv run python recipes/evaluate_model.py
```

### quantize_and_generate

Train a model, then quantize to 4-bit and 8-bit. Compares memory
usage and generation quality across precisions. Also demonstrates
`dequantize_model` for converting back to float (useful before
fine-tuning).

```bash
uv run python recipes/quantize_and_generate.py --bits 4 8
```

---

## Experiments

Structured experiment infrastructure for reproducible research.

| Recipe | What it teaches |
|--------|----------------|
| [`run_experiment.py`](#run_experiment) | Time-budgeted experiments with logging |
| [`sweep_learning_rate.py`](#sweep_learning_rate) | Grid and random hyperparameter sweeps |
| [`analyze_experiments.py`](#analyze_experiments) | Statistical analysis: CI, Cohen's d, simplicity score |

### run_experiment

Uses `ExperimentRunner` for time-budgeted experiments following the
[autoresearch methodology](../experiments/methodology.md). Logs results to
`results.jsonl`, supports multi-seed runs, and tracks keep/discard status.

```bash
uv run python recipes/run_experiment.py --arch llama --seeds 3
```

### sweep_learning_rate

Grid sweep and random sweep over learning rate ranges. Reports best
configuration and trial-by-trial results table.

```bash
uv run python recipes/sweep_learning_rate.py --min-lr 1e-4 --max-lr 1e-2
```

### analyze_experiments

Demonstrates the experiment analysis toolkit on synthetic data. Covers
`compare_experiments`, `compute_statistics`, `confidence_interval`,
`cohens_d`, and `simplicity_score`. No training required — runs instantly.

```bash
uv run python recipes/analyze_experiments.py
```
