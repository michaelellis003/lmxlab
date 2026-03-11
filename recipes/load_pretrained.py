"""Load a pretrained HuggingFace model and generate text.

Demonstrates the weight conversion pipeline:
1. Download model from HuggingFace Hub
2. Convert weight names from HF → lmt-metal format
3. Load into LanguageModel
4. Generate text

Requires: pip install huggingface_hub

Usage:
    uv run python recipes/load_pretrained.py
    uv run python recipes/load_pretrained.py --repo meta-llama/Llama-3.2-1B
    uv run python recipes/load_pretrained.py --repo Qwen/Qwen2-0.5B
"""

import argparse

import mlx.core as mx

from lmt_metal.models.convert import load_from_hf
from lmt_metal.models.generate import generate


def main() -> None:
    """Load a pretrained model and generate text."""
    parser = argparse.ArgumentParser(
        description="Load pretrained HF model and generate text"
    )
    parser.add_argument(
        "--repo",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--prompt",
        default="The meaning of life is",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.repo}")
    print("(This may take a while on first download...)")

    model, config = load_from_hf(args.repo)
    mx.eval(model.parameters())

    params = model.count_parameters()
    print(f"Loaded: {params:,} parameters")
    print(f"  d_model={config.block.d_model}")
    print(f"  n_heads={config.block.n_heads}")
    print(f"  n_layers={config.n_layers}")
    print(f"  vocab_size={config.vocab_size}")

    # Tokenize prompt
    # For real use, load the tokenizer from the HF repo.
    # Here we use a simple encoding for demonstration.
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        prompt_tokens = enc.encode(args.prompt)
        decode_fn = enc.decode
    except ImportError:
        print(
            "\nNote: tiktoken not installed. "
            "Using character-level tokenization."
        )
        prompt_tokens = [ord(c) % config.vocab_size for c in args.prompt]

        def decode_fn(tokens):
            return "".join(chr(t) if 32 <= t < 127 else "?" for t in tokens)

    prompt = mx.array([prompt_tokens])

    print(f"\nPrompt: {args.prompt}")
    print("Generating...\n")

    output_tokens = generate(
        model,
        prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    mx.eval(output_tokens)

    # Decode all tokens (prompt + generated)
    all_ids = output_tokens[0].tolist()
    output = decode_fn(all_ids)
    n_generated = len(all_ids) - len(prompt_tokens)
    print(output)
    print(f"\n({n_generated} tokens generated)")


if __name__ == "__main__":
    main()
