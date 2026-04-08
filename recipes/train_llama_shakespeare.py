"""Train a tiny LLaMA on Shakespeare with BPE tokenization.

LLaMA config (RMSNorm, GQA, SwiGLU, RoPE) + tiktoken BPE +
compiled training + generation with temperature/top-k.

Usage:
    uv run python recipes/train_llama_shakespeare.py
"""

import urllib.request
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from lmxlab.data.batching import batch_iterator
from lmxlab.data.dataset import TextDataset
from lmxlab.data.tokenizer import TiktokenTokenizer
from lmxlab.models.base import LanguageModel
from lmxlab.models.generate import generate
from lmxlab.models.llama import llama_tiny
from lmxlab.training.config import TrainConfig
from lmxlab.training.trainer import Trainer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/shakespeare.txt")


def download_data() -> str:
    """Download Shakespeare text if not cached."""
    if DATA_PATH.exists():
        return DATA_PATH.read_text()

    print("Downloading Shakespeare text...")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text()


def main() -> None:
    """Train a tiny LLaMA on Shakespeare."""
    mx.random.seed(42)

    # --- Data ---
    text = download_data()
    print(f"Text: {len(text):,} characters")

    tokenizer = TiktokenTokenizer("gpt2")
    print(f"Tokenizer: GPT-2 BPE, {tokenizer.vocab_size:,} tokens")

    # Use first 80% for training
    split = int(len(text) * 0.8)
    train_text = text[:split]
    val_text = text[split:]

    seq_len = 64
    train_dataset = TextDataset(train_text, tokenizer, seq_len=seq_len)
    val_dataset = TextDataset(val_text, tokenizer, seq_len=seq_len)
    print(
        f"Train: {len(train_dataset)} sequences, "
        f"Val: {len(val_dataset)} sequences"
    )

    # --- Model ---
    config = llama_tiny()
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = LanguageModel(config)
    mx.eval(model.parameters())
    print(f"Model: {model.count_parameters():,} parameters")
    print(
        f"  d_model={config.block.d_model}, "
        f"n_heads={config.block.n_heads}, "
        f"n_layers={config.n_layers}"
    )

    # --- Training ---
    train_config = TrainConfig(
        learning_rate=3e-4,
        max_steps=500,
        batch_size=8,
        eval_interval=100,
        log_interval=50,
        compile_step=True,
        warmup_steps=20,
        max_grad_norm=1.0,
    )

    trainer = Trainer(model, train_config)

    def train_iter():
        yield from batch_iterator(
            train_dataset.tokens,
            batch_size=8,
            seq_len=seq_len,
            shuffle=True,
        )

    def val_iter():
        yield from batch_iterator(
            val_dataset.tokens,
            batch_size=8,
            seq_len=seq_len,
            shuffle=False,
        )

    print("\nTraining...")
    history = trainer.train(train_iter(), eval_data=val_iter())

    # Print loss curve
    for i, m in enumerate(history):
        if i % 50 == 0 or i == len(history) - 1:
            msg = f"  Step {i + 1}: loss={m['loss']:.4f}"
            if "eval_loss" in m:
                msg += f", eval_loss={m['eval_loss']:.4f}"
            print(msg)

    # --- Generation ---
    print("\nGenerating text:")
    prompts = ["To be, or", "ROMEO:", "What is"]
    for prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text)
        prompt_arr = mx.array([prompt_ids])

        generated = generate(
            model,
            prompt_arr,
            max_tokens=80,
            temperature=0.8,
            top_k=40,
        )
        output = tokenizer.decode(generated[0].tolist())
        print(f'  "{output}"')
        print()

    print("Done!")


if __name__ == "__main__":
    main()
