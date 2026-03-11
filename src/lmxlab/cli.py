"""Command-line interface for lmxlab.

Usage:
    lmxlab info <arch>     Show architecture config details
    lmxlab list             List available architectures
    lmxlab count <arch>     Count parameters for an architecture

Examples:
    lmxlab list
    lmxlab info gpt
    lmxlab info llama --tiny
    lmxlab count deepseek --tiny
"""

import argparse
import sys

import mlx.core as mx
import mlx.utils

from lmxlab.models.base import LanguageModel
from lmxlab.models.deepseek import deepseek_config, deepseek_tiny
from lmxlab.models.gemma import gemma_config, gemma_tiny
from lmxlab.models.gemma3 import gemma3_config, gemma3_tiny
from lmxlab.models.gpt import gpt_config, gpt_tiny
from lmxlab.models.llama import llama_config, llama_tiny
from lmxlab.models.mixtral import mixtral_config, mixtral_tiny
from lmxlab.models.qwen import qwen_config, qwen_tiny
from lmxlab.models.qwen35 import qwen35_config, qwen35_tiny

ARCHITECTURES = {
    "gpt": (gpt_config, gpt_tiny),
    "llama": (llama_config, llama_tiny),
    "gemma": (gemma_config, gemma_tiny),
    "gemma3": (gemma3_config, gemma3_tiny),
    "qwen": (qwen_config, qwen_tiny),
    "qwen35": (qwen35_config, qwen35_tiny),
    "mixtral": (mixtral_config, mixtral_tiny),
    "deepseek": (deepseek_config, deepseek_tiny),
}


def cmd_list(args: argparse.Namespace) -> None:
    """List available architectures."""
    print("Available architectures:")
    for name, (full_fn, _) in ARCHITECTURES.items():
        config = full_fn()
        block = config.block
        print(
            f"  {name:10s}  "
            f"attention={block.attention}, "
            f"norm={block.norm}, "
            f"ffn={block.ffn}"
        )


def cmd_info(args: argparse.Namespace) -> None:
    """Show architecture config details."""
    name = args.arch.lower()
    if name not in ARCHITECTURES:
        print(f"Unknown architecture: {name}")
        print(f"Available: {', '.join(ARCHITECTURES)}")
        sys.exit(1)

    full_fn, tiny_fn = ARCHITECTURES[name]
    config = tiny_fn() if args.tiny else full_fn()
    label = f"{name} (tiny)" if args.tiny else name
    block = config.block

    print(f"Architecture: {label}")
    print(f"  vocab_size:    {config.vocab_size:,}")
    print(f"  n_layers:      {config.n_layers}")
    print(f"  d_model:       {block.d_model}")
    print(f"  n_heads:       {block.n_heads}")
    print(f"  n_kv_heads:    {block.effective_n_kv_heads}")
    print(f"  d_ff:          {block.d_ff}")
    print(f"  head_dim:      {block.head_dim}")
    print(f"  attention:     {block.attention}")
    print(f"  ffn:           {block.ffn}")
    print(f"  norm:          {block.norm}")
    print(f"  position:      {block.position}")
    print(f"  bias:          {block.bias}")
    print(f"  pre_norm:      {block.pre_norm}")
    print(f"  tie_embeddings:{config.tie_embeddings}")

    if block.window_size is not None:
        print(f"  window_size:   {block.window_size}")
    if block.kv_lora_rank is not None:
        print(f"  kv_lora_rank:  {block.kv_lora_rank}")
    if block.q_lora_rank is not None:
        print(f"  q_lora_rank:   {block.q_lora_rank}")
    if block.rope_dim is not None:
        print(f"  rope_dim:      {block.rope_dim}")
    if config.block_configs is not None:
        print(f"  block_configs: {len(config.block_configs)} layers")


def cmd_count(args: argparse.Namespace) -> None:
    """Count parameters for an architecture."""
    name = args.arch.lower()
    if name not in ARCHITECTURES:
        print(f"Unknown architecture: {name}")
        print(f"Available: {', '.join(ARCHITECTURES)}")
        sys.exit(1)

    full_fn, tiny_fn = ARCHITECTURES[name]
    config = tiny_fn() if args.tiny else full_fn()
    label = f"{name} (tiny)" if args.tiny else name

    model = LanguageModel(config)
    mx.eval(model.parameters())

    leaves = mlx.utils.tree_flatten(model.parameters())
    total = sum(p.size for _, p in leaves)

    print(f"{label}: {total:,} parameters")

    if args.detail:
        # Group by component
        groups: dict[str, int] = {}
        for path, p in leaves:
            top = path.split(".")[0]
            groups[top] = groups.get(top, 0) + p.size
        for comp, count in sorted(groups.items(), key=lambda x: -x[1]):
            pct = 100 * count / total
            print(f"  {comp:20s} {count:>12,}  ({pct:.1f}%)")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lmxlab",
        description="Educational MLX library for transformer LMs",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List available architectures")

    # info
    info_p = sub.add_parser("info", help="Show architecture details")
    info_p.add_argument("arch", help="Architecture name")
    info_p.add_argument("--tiny", action="store_true", help="Use tiny config")

    # count
    count_p = sub.add_parser("count", help="Count parameters")
    count_p.add_argument("arch", help="Architecture name")
    count_p.add_argument("--tiny", action="store_true", help="Use tiny config")
    count_p.add_argument(
        "--detail",
        action="store_true",
        help="Show per-component breakdown",
    )

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "count":
        cmd_count(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
