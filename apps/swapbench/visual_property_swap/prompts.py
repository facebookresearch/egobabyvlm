# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-property prompt templates for VP-Swap (Visual Property Swap).

Four property axes covered (matching the EgoBabyVLM paper, App. methods:vp-swap):

* ``color``         - dominant surface color
* ``material``      - the substance an object is typically made of
* ``relative_size`` - typical real-world size relative to a human
* ``shape``         - canonical geometric shape

Each property gets two prompt templates:

* ``GENERATION_PROMPTS[prop]`` - asks an LLM to write a pair of short
  sentences, one per word in ``(w1, w2)``, that each describe the word's
  value for that property. The model must encapsulate both sentences in a
  single ``[...]`` block; the runner's ``_split_two_sentences`` parser
  consumes that shape.
* ``FILTER_PROMPTS[prop]`` - given a pair of sentences A and B, asks the
  LLM which is more physically accurate. After we swap the property
  values between the two sentences, the original assignment ought to
  remain the more plausible one.

The "is this word a physical object?" gate (stage 1 of the pipeline) is
property-agnostic and lives directly in the runner.
"""

from __future__ import annotations

#: Allowed values for ``processor.property``.
SUPPORTED_PROPERTIES: tuple[str, ...] = ("color", "material", "relative_size", "shape")


PHYSICAL_OBJECT_PROMPT_TEMPLATE = (
    "Is the word '{word}' representing something physical? Answer only by yes or no, in brackets."
)


_GENERATION_TEMPLATES: dict[str, str] = {
    "color": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and state "
        "the most typical real-world color of that word. Encapsulate both "
        "sentences together within brackets. Do not relate the two "
        "sentences together."
    ),
    "material": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and state "
        "the most typical real-world material that word is made of. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
    "relative_size": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and "
        "describe its typical real-world size relative to an adult human. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
    "shape": (
        "Using the two words '{w1}' and '{w2}', write a pair of short "
        "sentences. Each sentence should use one of these words and "
        "describe its canonical geometric shape in the real world. "
        "Encapsulate both sentences together within brackets. Do not "
        "relate the two sentences together."
    ),
}


#: All four properties share the same filter prompt: after we swap the
#: property descriptions between the two words, the original (un-swapped)
#: sentence should still be the more physically accurate one.
_SHARED_FILTER_TEMPLATE = (
    "Given the two sentences A and B: "
    "<start of sentence A> {s1} <end of sentence A> "
    "<start of sentence B> {s2} <end of sentence B> "
    "Which of the two sentences, A or B, is more physically accurate? "
    "Write your answer (A or B) in the brackets."
)

_FILTER_TEMPLATES: dict[str, str] = dict.fromkeys(SUPPORTED_PROPERTIES, _SHARED_FILTER_TEMPLATE)


def generation_prompt(prop: str, w1: str, w2: str) -> str:
    """Return the per-property sentence-generation prompt for ``(w1, w2)``."""
    if prop not in _GENERATION_TEMPLATES:
        msg = f"Unsupported property {prop!r}; expected one of {SUPPORTED_PROPERTIES}"
        raise ValueError(msg)
    return _GENERATION_TEMPLATES[prop].format(w1=w1, w2=w2)


def filter_prompt(prop: str, s1: str, s2: str) -> str:
    """Return the per-property A/B filter prompt for sentences ``(s1, s2)``."""
    if prop not in _FILTER_TEMPLATES:
        msg = f"Unsupported property {prop!r}; expected one of {SUPPORTED_PROPERTIES}"
        raise ValueError(msg)
    return _FILTER_TEMPLATES[prop].format(s1=s1, s2=s2)


def physical_object_prompt(word: str) -> str:
    """Return the gating prompt that asks if ``word`` is a physical object."""
    return PHYSICAL_OBJECT_PROMPT_TEMPLATE.format(word=word)
