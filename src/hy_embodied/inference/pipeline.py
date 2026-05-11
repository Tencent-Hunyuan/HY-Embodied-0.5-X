"""Programmatic inference pipeline for HY-Embodied-0.5-X.

Minimal wrapper over ``transformers.AutoProcessor`` and
``transformers.AutoModelForImageTextToText`` that tries to load a local
``chat_template.jinja`` / ``chat_template.json`` alongside the ckpt, and
provides :meth:`generate` / :meth:`generate_batch` methods for typical
text / image / video inputs.

Example::

    from hy_embodied.inference import HyEmbodiedPipeline

    pipe = HyEmbodiedPipeline.from_pretrained("ckpts/HY-Embodied-0.5-X")
    out = pipe.generate(
        image="demo.jpg",
        prompt="Describe this image.",
        enable_thinking=False,
    )
    print(out)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def _load_local_chat_template(model_path: str) -> str | None:
    local_dir = Path(model_path)
    if not local_dir.is_dir():
        return None

    jinja_path = local_dir / "chat_template.jinja"
    json_path = local_dir / "chat_template.json"
    if jinja_path.exists():
        logger.info("Loaded chat template from: %s", jinja_path)
        return jinja_path.read_text()
    if json_path.exists():
        logger.info("Loaded chat template from: %s", json_path)
        return json.loads(json_path.read_text()).get("chat_template")
    return None


@dataclass
class GenerationConfig:
    """Decoding knobs exposed on the pipeline."""

    max_new_tokens: int = 32768
    temperature: float = 0.05
    use_cache: bool = True
    enable_thinking: bool = False


class HyEmbodiedPipeline:
    """High-level inference pipeline for HY-Embodied-0.5-X."""

    def __init__(self, model, processor, device: str = "cuda") -> None:
        self.model = model
        self.processor = processor
        self.device = device

    # ------------------------------------------------------------------ init
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
    ) -> "HyEmbodiedPipeline":
        """Load processor + model from a local directory or HF Hub repo id."""
        logger.info("Loading processor from %s", model_path)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)

        # If a locally bundled chat template exists we honour it; otherwise
        # fall back to the template shipped with the processor.
        template = _load_local_chat_template(model_path)
        if template is not None:
            processor.chat_template = template

        logger.info("Loading model from %s", model_path)
        model_kwargs: dict = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        model = model.to(device).eval()
        logger.info("Model loaded on %s", device)
        return cls(model=model, processor=processor, device=device)

    # --------------------------------------------------------- message utils
    @staticmethod
    def build_messages(
        prompt: str,
        *,
        image: str | None = None,
        images: Iterable[str] | None = None,
        video: str | None = None,
    ) -> list[dict]:
        """Build a single-turn user message from the given inputs.

        Only one of ``image`` / ``images`` / ``video`` should be provided.
        """
        content: list[dict] = []
        if images is not None:
            for img in images:
                content.append({"type": "image", "image": img})
        elif image is not None:
            content.append({"type": "image", "image": image})
        elif video is not None:
            content.append({"type": "video", "video": video})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ---------------------------------------------------------- single-shot
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        image: str | None = None,
        images: Iterable[str] | None = None,
        video: str | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        """Run single-sample inference and return the decoded completion."""
        cfg = generation_config or GenerationConfig()
        messages = self.build_messages(prompt, image=image, images=images, video=video)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=cfg.enable_thinking,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            use_cache=cfg.use_cache,
            temperature=cfg.temperature,
            do_sample=cfg.temperature > 0,
        )
        output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # ------------------------------------------------------------ batch API
    @torch.no_grad()
    def generate_batch(
        self,
        messages_batch: list[list[dict]],
        *,
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Run left-padded batch inference.

        Each element of ``messages_batch`` is a list of chat messages (same
        format accepted by :func:`build_messages`).
        """
        cfg = generation_config or GenerationConfig()

        all_inputs = [
            self.processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=cfg.enable_thinking,
            )
            for msgs in messages_batch
        ]
        batch = self.processor.pad(all_inputs, padding=True, padding_side="left").to(self.model.device)

        generated_ids = self.model.generate(
            **batch,
            max_new_tokens=cfg.max_new_tokens,
            use_cache=cfg.use_cache,
            temperature=cfg.temperature,
            do_sample=cfg.temperature > 0,
        )
        padded_input_len = batch["input_ids"].shape[1]
        results: list[str] = []
        for i in range(len(messages_batch)):
            out_ids = generated_ids[i][padded_input_len:]
            results.append(self.processor.decode(out_ids, skip_special_tokens=True))
        return results


__all__ = ["GenerationConfig", "HyEmbodiedPipeline"]
