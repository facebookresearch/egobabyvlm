# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper around a Diffusers text-to-image pipeline (e.g. FLUX.2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers import DiffusionPipeline

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_style_prefixes: dict[str, str] | None = None


def _load_style_prefixes() -> dict[str, str]:
    global _style_prefixes  # noqa: PLW0603
    if _style_prefixes is None:
        from apps.benchmark_creation.paths import get_style_prefixes

        _style_prefixes = get_style_prefixes()
    return _style_prefixes


class FluxPipeline:
    """Wrapper around a Diffusers text-to-image pipeline."""

    DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-9B"

    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 7.5
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 512

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        *,
        compile_model: bool = False,
    ) -> None:
        """Load the diffusion pipeline.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Device to run on; ``None`` auto-selects CUDA if available.
            dtype: Model dtype; ``None`` uses bfloat16 on CUDA, float32 on CPU.
            compile_model: If True, ``torch.compile`` the transformer.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if "cuda" in self.device else torch.float32)

        logger.info("Loading model: %s  (device=%s, dtype=%s)", self.model_id, self.device, self.dtype)
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)

        if compile_model and "cuda" in self.device:
            logger.info("Compiling transformer with torch.compile...")
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="reduce-overhead",
                fullgraph=True,
            )

        logger.info("Model loaded.")

    @staticmethod
    def build_prompt(description: str, style: str = "realistic") -> str:
        """Prepend the style prefix to a scene description."""
        prefixes = _load_style_prefixes()
        if style not in prefixes:
            msg = f"Unknown style '{style}'. Available: {list(prefixes.keys())}"
            raise ValueError(msg)
        return f"{prefixes[style]}{description}."

    def generate(  # noqa: PLR0913
        self,
        description: str,
        style: str = "realistic",
        *,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate a single image from a text description."""
        prompt = self.build_prompt(description, style)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            return self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps or self.DEFAULT_STEPS,
                guidance_scale=guidance_scale or self.DEFAULT_GUIDANCE,
                height=height or self.DEFAULT_HEIGHT,
                width=width or self.DEFAULT_WIDTH,
                generator=generator,
            ).images[0]

    def generate_batch(  # noqa: PLR0913
        self,
        descriptions: list[str],
        style: str = "realistic",
        *,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate multiple images in a single batched forward pass."""
        if not descriptions:
            return []

        prompts = [self.build_prompt(d, style) for d in descriptions]

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            return self.pipe(
                prompt=prompts,
                num_inference_steps=num_inference_steps or self.DEFAULT_STEPS,
                guidance_scale=guidance_scale or self.DEFAULT_GUIDANCE,
                height=height or self.DEFAULT_HEIGHT,
                width=width or self.DEFAULT_WIDTH,
                generator=generator,
            ).images

    def generate_and_save(
        self,
        description: str,
        output_path: str | Path,
        style: str = "realistic",
        **kwargs: Any,  # noqa: ANN401
    ) -> Path:
        """Generate an image and save it to disk; return the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.generate(description, style, **kwargs)
        image.save(output_path)
        return output_path
