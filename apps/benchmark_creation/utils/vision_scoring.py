# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified vision-language scoring engine.

Supports multiple backends for image-text alignment scoring:
  - **Perception Encoder** (``facebook/PE-Core-*``): default; loaded via
    Meta's ``core.vision_encoder.pe`` package (CLIP-style image+text encoders).
    Scored using raw cosine similarity.
  - **SigLIP2** (``google/siglip2-*``): HuggingFace AutoModel/AutoProcessor,
    returns ``logits_per_image`` directly.
  - **CLIP** (``openai/clip-*``): OpenAI CLIP via HuggingFace CLIPModel/CLIPProcessor,
    scored using raw cosine similarity.

Key API:
  - ``load_model(model_name, device)`` -> ``ScoringEngine``
  - ``score_image_text_pairs(engine, images, texts, device)`` -> list[float]
  - ``score_image_text_matrix(engine, images, texts, device)`` -> Tensor
  - ``build_caption(word, category, task)`` -> str  (model-agnostic)
"""

import logging
import time
from typing import Any, NamedTuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring engine container
# ---------------------------------------------------------------------------


class ScoringEngine(NamedTuple):
    """Opaque handle returned by :func:`load_model`."""

    #: Backend-specific model object (HF model, PE CLIP, etc.).
    model: Any
    #: Backend-specific processor: HF processor, or (image_transform, tokenizer) for PE.
    processor: Any
    #: Backend identifier: "pe", "siglip2", or "clip".
    backend: str


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_backend(model_name: str) -> str:
    """Infer the backend from the model name / path."""
    name_lower = model_name.lower()
    if "pe-core" in name_lower or "pe_core" in name_lower or "perception" in name_lower:
        return "pe"
    if "clip" in name_lower and "siglip" not in name_lower:
        return "clip"
    # Default to SigLIP2 for anything else (including google/siglip2-*)
    return "siglip2"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _pe_config_name(model_name: str) -> str:
    """Map a HuggingFace-style PE id (e.g. ``facebook/PE-Core-L14-336``) to the
    config name expected by ``core.vision_encoder.pe.CLIP.from_config``."""
    # Accept either a HF repo ("facebook/PE-Core-L14-336") or a bare config
    # name ("PE-Core-L14-336").
    return model_name.split("/")[-1]


def _import_meta_pe() -> tuple[Any, Any]:
    """Import Meta's Perception Encoder modules from the in-repo subset.

    A copy of the upstream perception_models PE-Core code lives at
    ``apps/alignment_scoring/third_party/perception_models/`` (see that
    directory's README for why it is copied rather than installed). Its
    upstream top-level package is ``core``, which collides with our own
    top-level ``core/`` package; the import-rewriter in ``refresh.py``
    rewrites every ``from core.X``/``import core.X`` to the namespaced
    path used here.
    """
    import importlib

    pe_mod = importlib.import_module(
        "apps.alignment_scoring.third_party.perception_models.core.vision_encoder.pe",
    )
    transforms_mod = importlib.import_module(
        "apps.alignment_scoring.third_party.perception_models.core.vision_encoder.transforms",
    )
    return pe_mod, transforms_mod


def _load_pe(model_name: str, device: str) -> ScoringEngine:
    pe_mod, transforms_mod = _import_meta_pe()

    config_name = _pe_config_name(model_name)
    logger.info("Loading Perception Encoder model: %s (config=%s)", model_name, config_name)
    t0 = time.time()
    model = pe_mod.CLIP.from_config(config_name, pretrained=True).to(device).eval()
    image_transform = transforms_mod.get_image_transform(model.image_size)
    tokenizer = transforms_mod.get_text_tokenizer(model.context_length)
    logger.info("Model loaded in %.1fs", time.time() - t0)
    return ScoringEngine(
        model=model,
        processor=(image_transform, tokenizer),
        backend="pe",
    )


def _load_siglip2(model_name: str, device: str) -> ScoringEngine:
    from transformers import AutoModel, AutoProcessor

    logger.info("Loading SigLIP2 model: %s", model_name)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)
    return ScoringEngine(model=model, processor=processor, backend="siglip2")


def _load_clip(model_name: str, device: str) -> ScoringEngine:
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP model: %s", model_name)
    t0 = time.time()
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    logger.info("Model loaded in %.1fs", time.time() - t0)
    return ScoringEngine(model=model, processor=processor, backend="clip")


def load_model(model_name: str, device: str) -> ScoringEngine:
    """Load a vision-language model and return a :class:`ScoringEngine`.

    The backend is auto-detected from *model_name*:
      - Names containing ``PE-Core`` / ``perception`` -> Perception Encoder backend.
      - Names containing ``clip`` (but not ``siglip``) -> CLIP backend.
      - Everything else -> SigLIP2 backend.
    """
    backend = _detect_backend(model_name)
    if backend == "pe":
        return _load_pe(model_name, device)
    if backend == "clip":
        return _load_clip(model_name, device)
    return _load_siglip2(model_name, device)


# Backward-compatible alias
load_siglip2 = load_model


# ---------------------------------------------------------------------------
# Caption building (model-agnostic)
# ---------------------------------------------------------------------------


#: Singular nouns that happen to end in 's' and should not be treated as plural.
_SINGULAR_S: frozenset[str] = frozenset(
    {
        "amaryllis",
        "apparatus",
        "billiards",
        "bus",
        "cactus",
        "campus",
        "cannabis",
        "canvas",
        "christmas",
        "circus",
        "collins",
        "corpus",
        "cosmos",
        "crocus",
        "curious",
        "discus",
        "dress",
        "exodus",
        "focus",
        "fungus",
        "gas",
        "genus",
        "gladiolus",
        "hibiscus",
        "hippopotamus",
        "ibis",
        "iris",
        "lens",
        "lotus",
        "mantis",
        "minibus",
        "mrs",
        "mucus",
        "nautilus",
        "nexus",
        "oasis",
        "octopus",
        "omnibus",
        "papyrus",
        "pelvis",
        "pharos",
        "platypus",
        "plus",
        "radius",
        "rhinoceros",
        "status",
        "surplus",
        "syllabus",
        "tennis",
        "terminus",
        "tetanus",
        "tis",
        "trellis",
        "venus",
        "virus",
        "walrus",
    }
)


def _add_article(word: str) -> str:
    """Prepend 'a' or 'an' to singular nouns, leave plurals as-is.

    Uses a simple heuristic: words ending in 's' (but not 'ss') are assumed
    plural unless they appear in an explicit singular-exceptions set.
    """
    w = word.strip().lower()

    is_plural = w.endswith("s") and not w.endswith("ss") and w not in _SINGULAR_S
    if is_plural:
        return word

    # Pick "an" before vowel sounds
    if w[0] in "aeiou":
        return f"an {word}"
    return f"a {word}"


def build_caption(word: str, category: str | None = None, task: str = "nouns") -> str:  # noqa: ARG001
    """Build a caption for scoring an image against a word.

    - **Nouns**: ``"This is a photo of {a/an} {word}."`` (HF-recommended template).
    - **Adjectives**: returns *word* as-is (the caller passes in the
      full positive or negative phrase from the word list).

    The ``category`` argument is currently unused but kept for API compatibility.
    """
    if task == "nouns":
        return f"This is a photo of {_add_article(word)}."
    # Adjectives already carry full phrases
    return word


# ---------------------------------------------------------------------------
# Scoring -- SigLIP2 backend
# ---------------------------------------------------------------------------


def _score_pairs_siglip2(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> list[float]:
    inputs = engine.processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = engine.model(**inputs)
    scores = torch.sigmoid(outputs.logits_per_image).cpu()
    return scores.diag().tolist()


def _score_matrix_siglip2(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> torch.Tensor:
    inputs = engine.processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = engine.model(**inputs)
    return torch.sigmoid(outputs.logits_per_image).cpu()


# ---------------------------------------------------------------------------
# Scoring -- CLIP backend
# ---------------------------------------------------------------------------


def _score_pairs_clip(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> list[float]:
    inputs = engine.processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = engine.model(**inputs)
    # Use raw cosine similarity (not logit-scaled) for pair comparisons.
    # CLIP's logits_per_image = logit_scale * cosine_sim, and the learned
    # logit_scale can saturate fine-grained differences under sigmoid.
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    cosine_sims = (image_embeds * text_embeds).sum(dim=-1)
    return cosine_sims.cpu().tolist()


def _score_matrix_clip(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> torch.Tensor:
    inputs = engine.processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = engine.model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return (image_embeds @ text_embeds.T).cpu()


# ---------------------------------------------------------------------------
# Scoring -- Perception Encoder backend
# ---------------------------------------------------------------------------


def _pe_encode(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode images and texts into L2-normalised feature tensors."""
    image_transform, tokenizer = engine.processor
    image_batch = torch.stack([image_transform(img) for img in images]).to(device)
    text_batch = tokenizer(texts).to(device)
    image_features, text_features, _ = engine.model(image_batch, text_batch)
    # CLIP.forward already L2-normalises features.
    return image_features, text_features


def _score_pairs_pe(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> list[float]:
    image_features, text_features = _pe_encode(engine, images, texts, device)
    cosine_sims = (image_features * text_features).sum(dim=-1)
    return cosine_sims.cpu().tolist()


def _score_matrix_pe(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> torch.Tensor:
    image_features, text_features = _pe_encode(engine, images, texts, device)
    return (image_features @ text_features.T).cpu()


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_image_text_pairs(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> list[float]:
    """Score N aligned image-text pairs.

    Returns a list of N scores (one per pair).  The score range depends on
    the backend:

    * **pe** -- raw cosine similarities in [-1, 1].
    * **siglip2** -- sigmoid-scaled scores in [0, 1].
    * **clip** -- raw cosine similarities in [-1, 1].

    *engine* must be a :class:`ScoringEngine` returned by :func:`load_model`.
    """
    if not isinstance(engine, ScoringEngine):
        # Legacy call: score_image_text_pairs(model, processor, images, texts, device)
        #   engine=model, images=processor, texts=real_images, device=real_texts
        #   and the real device is the 5th positional arg -- but we only have 4 here.
        # This path is handled by the shim in siglip.py.
        raise TypeError(
            "score_image_text_pairs() expects a ScoringEngine as first arg. "
            "Use load_model() instead of load_siglip2() to get one."
        )

    if engine.backend == "pe":
        return _score_pairs_pe(engine, images, texts, device)
    if engine.backend == "clip":
        return _score_pairs_clip(engine, images, texts, device)
    return _score_pairs_siglip2(engine, images, texts, device)


@torch.no_grad()
def score_image_text_matrix(
    engine: ScoringEngine,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> torch.Tensor:
    """Score all image-text combinations.

    Returns the full ``(N_images x N_texts)`` score matrix.  The score
    range depends on the backend:

    * **pe** -- raw cosine similarities in [-1, 1].
    * **siglip2** -- sigmoid-scaled scores in [0, 1].
    * **clip** -- raw cosine similarities in [-1, 1].
    """
    if not isinstance(engine, ScoringEngine):
        raise TypeError(
            "score_image_text_matrix() expects a ScoringEngine as first arg. "
            "Use load_model() instead of load_siglip2() to get one."
        )

    if engine.backend == "pe":
        return _score_matrix_pe(engine, images, texts, device)
    if engine.backend == "clip":
        return _score_matrix_clip(engine, images, texts, device)
    return _score_matrix_siglip2(engine, images, texts, device)
