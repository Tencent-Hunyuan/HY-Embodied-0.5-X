"""Command-line entry point for single-sample / batch inference.

Usage::

    # Single-image (thinking disabled by default)
    python -m hy_embodied.cli.infer \\
        --model ckpts/HY-Embodied-0.5-X --image demo.jpg --prompt "Describe this image"

    # Single-image with thinking enabled
    python -m hy_embodied.cli.infer \\
        --model ckpts/HY-Embodied-0.5-X --image demo.jpg --prompt "Describe this image" --enable-thinking

    # Batch demo (two samples, one with image, one text-only)
    python -m hy_embodied.cli.infer --model ckpts/HY-Embodied-0.5-X --image demo.jpg --batch
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from hy_embodied.inference import GenerationConfig, HyEmbodiedPipeline


DEFAULT_MODEL = "tencent/HY-Embodied-0.5-X"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HY-Embodied-0.5-X inference CLI")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Model directory or Hugging Face Hub repo id. "
            f"Defaults to '{DEFAULT_MODEL}' (auto-downloaded from HF Hub). "
            "A local checkpoint directory such as ckpts/HY-Embodied-0.5-X is also accepted."
        ),
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking (CoT) mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch", action="store_true", help="Run the batch inference demo")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = _parse_args()

    image_path = args.image or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "Results-All-benchmarks.png"
    )

    pipe = HyEmbodiedPipeline.from_pretrained(
        args.model,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )

    if args.batch:
        print("\n=== Batch Inference ===")
        messages_batch = [
            pipe.build_messages(args.prompt, image=image_path),
            pipe.build_messages("How to open a fridge?"),
        ]
        results = pipe.generate_batch(messages_batch, generation_config=gen_cfg)
        for i, text in enumerate(results):
            print(f"\n--- Sample {i} ---")
            print(text)
    else:
        print("\n=== Single Inference ===")
        text = pipe.generate(args.prompt, image=image_path, generation_config=gen_cfg)
        print(f"\n{text}")


if __name__ == "__main__":
    main()
