# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Blocked-word sets, supplemental adjective list, and deterministic negation.

Extracted from ``build_benchmark.py`` — word-level filtering logic and
programmatic negative-caption generation for deterministic categories.
"""

import re

from apps.benchmark_creation.pipeline.grammatical.constants import (
    invert_order_matters,
    pluralize,
    singularize,
)

# ===========================================================================
#  Deterministic negative generation — regex patterns
# ===========================================================================

_NUMERALS = ("one", "two", "three", "four", "five")

# Pattern: "{numeral} {subject(s)} is/are {verb-ing}"
_COUNTING_RE = re.compile(
    r"^(one|two|three|four|five)\s+(.+?)\s+(is|are)\s+(.+)$",
    re.IGNORECASE,
)

# Pattern: "the {A} is {comparative} than the {B}"
_COMPARATIVE_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(much\s+)?(more\s+)?(\S+)\s+than\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)


# ===========================================================================
#  Word filter sets
# ===========================================================================

# Adjectives that are too subjective / abstract / non-physical for comparatives.
# If the selected word is in this set for the comparatives category, reject it.
_SUBJECTIVE_ADJECTIVES = {
    # Emotional / personality
    "happy",
    "sad",
    "angry",
    "brave",
    "clever",
    "smart",
    "kind",
    "nice",
    "gentle",
    "calm",
    "shy",
    "proud",
    "lazy",
    "stubborn",
    "curious",
    "cheerful",
    "grumpy",
    "friendly",
    "polite",
    "rude",
    "mean",
    "patient",
    "generous",
    "selfish",
    "honest",
    "loyal",
    "jealous",
    "nervous",
    "anxious",
    "confident",
    "humble",
    "fierce",
    "bold",
    "playful",
    "creepy",
    "adorable",
    "uncomfortable",
    "impressive",
    "funny",
    "nasty",
    "awesome",
    "successful",
    "scary",
    "silly",
    "crazy",
    "goofy",
    "weird",
    "wacky",
    "moody",
    "cranky",
    "loving",
    # Subjective / aesthetic
    "beautiful",
    "pretty",
    "ugly",
    "cute",
    "lovely",
    "gorgeous",
    "handsome",
    "attractive",
    "elegant",
    "fancy",
    "glamorous",
    "stylish",
    "graceful",
    "charming",
    "delightful",
    "pleasant",
    "wonderful",
    # Abstract / mental
    "interesting",
    "important",
    "popular",
    "famous",
    "expensive",
    "valuable",
    "useful",
    "boring",
    "exciting",
    "amazing",
    "strange",
    "normal",
    "special",
    "perfect",
    "terrible",
    "great",
    "good",
    "bad",
    "better",
    "worse",
    "best",
    "worst",
    "fine",
    "historical",
    "traditional",
    "unusual",
    "everyday",
    "modern",
    "classical",
    "contemporary",
    "typical",
    "common",
    "rare",
    "necessary",
    "final",
    "initial",
    "primary",
    "secondary",
    # Hard to see in a still photo
    "fast",
    "slow",
    "quick",
    "loud",
    "quiet",
    "noisy",
    "silent",
    "strong",
    "weak",
    "powerful",
    "fit",
    "athletic",
    "agile",
    "tasty",
    "delicious",
    "sweet",
    "sour",
    "bitter",
    "spicy",
    "fragrant",
    "smelly",
    "stinky",
    # Non-gradable
    "dead",
    "alive",
    "pregnant",
    "unique",
    "impossible",
    "infinite",
    "correct",
    "wrong",
    "true",
    "false",
    "real",
    "fake",
    # Nationality / identity (not appropriate for visual comparison)
    "mexican",
    "irish",
    "chinese",
    "greek",
    "french",
    "italian",
    "american",
    "british",
    "german",
    "spanish",
    "japanese",
    "korean",
    # Hard for image generation models
    "empty",
    "full",
    "old",
    "young",
    # Non-visual or context-dependent states
    "mechanical",
    "occupied",
    "dotted",
    "numerous",
    "watery",
    "loose",
    "mild",
    "open",
    "fresh",
    "ginger",
    "bust",
    "interested",
}

# Adjectives that don't make sense for negation — non-visual, nonsensical
# when applied to common objects/animals, or not clearly photographable.
_BAD_NEGATION_ADJECTIVES = _SUBJECTIVE_ADJECTIVES | {
    # Non-visual / taste / smell
    "tasty",
    "delicious",
    "sweet",
    "sour",
    "bitter",
    "spicy",
    "fragrant",
    "smelly",
    "stinky",
    "bland",
    # Don't apply to animals/common objects
    "open",
    "closed",
    "shallow",
    "deep",
    "steep",
    "gentle",
    "stocked",
    "tropical",
    "plastic",
    "uniform",
    "nuclear",
    "solar",
    "electric",
    "digital",
    "manual",
    # Too abstract or hard to photograph as negation
    "sunny",
    "cloudy",
    "windy",
    "rainy",
    "stormy",
    "stretched",
    "rolled",
    "compressed",
    "suburban",
    "urban",
    "rural",
    "indoor",
    "outdoor",
}

# Words that are too abstract, non-physical, or not clearly photographable.
_ABSTRACT_WORDS = {
    "visible",
    "invisible",
    "transparent",
    "opaque",
    "fuzzy",
    "vague",
    "obvious",
    "apparent",
    "subtle",
    "distinct",
    "clear",
    "unclear",
    "dim",
    "faint",
    "obscure",
    "blurry",
    "hazy",
    "muted",
    "abstract",
    "conceptual",
    "theoretical",
    "hypothetical",
    "virtual",
    "imaginary",
    "mental",
    "emotional",
    "spiritual",
    "symbolic",
    "figurative",
    "metaphorical",
    "philosophical",
    "psychological",
}

# NSFW or inappropriate words not suitable for a visual benchmark.
_INAPPROPRIATE_WORDS = {
    "naked",
    "nude",
    "topless",
    "shirtless",
    "undressed",
    "bare",
    "sexy",
    "erotic",
    "sensual",
    "provocative",
    "seductive",
    "vulgar",
    "obscene",
    "indecent",
    "lewd",
    "pornographic",
    "explicit",
    "risque",
    "lustful",
    "aroused",
    "bloody",
    "gory",
    "gruesome",
    "mutilated",
    "decapitated",
    "tortured",
    "strangled",
    "murdered",
    "killed",
    "slaughtered",
}

# Union of all globally blocked words — applied to every category's pool.
_GLOBAL_BLOCKED_WORDS = _ABSTRACT_WORDS | _INAPPROPRIATE_WORDS

# Curated visual adjectives that work well for comparatives and negation.
# Supplements COCO vocab which has few adjectives.
_SUPPLEMENTAL_ADJECTIVES: list[str] = [
    # Size
    "big",
    "small",
    "tall",
    "short",
    "long",
    "wide",
    "narrow",
    "thick",
    "thin",
    "fat",
    "slim",
    "tiny",
    "huge",
    # Color / brightness
    "dark",
    "bright",
    "light",
    "pale",
    "shiny",
    "dull",
    "glossy",
    # Texture / surface
    "rough",
    "smooth",
    "bumpy",
    "fuzzy",
    "fluffy",
    "hairy",
    "spiky",
    "wrinkled",
    "flat",
    "curly",
    "wavy",
    "straight",
    # Physical state
    "wet",
    "dry",
    "dirty",
    "clean",
    "dusty",
    "muddy",
    "rusty",
    "cracked",
    "chipped",
    "scratched",
    "dented",
    "bent",
    "twisted",
    "torn",
    "broken",
    "crushed",
    "melted",
    "frozen",
    "burnt",
    # Temperature appearance
    "hot",
    "cold",
    "warm",
    "icy",
    "steamy",
    # Weight / density
    "heavy",
    "dense",
    "hollow",
    "solid",
    # Age / condition
    "new",
    "worn",
    "faded",
    "fresh",
    "stale",
    "rotten",
    "moldy",
    # Pattern
    "striped",
    "spotted",
    "checkered",
    "plain",
    "dotted",
    # Shape
    "round",
    "square",
    "pointed",
    "curved",
    "crooked",
]


# ===========================================================================
#  Deterministic negative generation
# ===========================================================================


def derive_deterministic_negative(  # noqa: PLR0911 -- pipeline orchestration: complexity matches the spec it implements
    sentence: str,
    category: str,
    item_index: int = 0,
) -> str | None:
    """Compute the negative caption programmatically for deterministic tasks.

    Returns the negative sentence or None if parsing fails.
    """
    if category == "negation":
        # "the X is adj" -> "the X is not adj"
        m = re.match(r"^(the\s+.+?\s+is\s+)(.+)$", sentence, re.IGNORECASE)
        if m:
            return m.group(1) + "not " + m.group(2)
        return None

    if category == "order_matters":
        return invert_order_matters(sentence)

    if category == "comparatives":
        # "the A is adj-er than the B" -> "the B is adj-er than the A"
        m = _COMPARATIVE_RE.match(sentence.strip())
        if m:
            noun_a = m.group(1)
            much = m.group(2) or ""
            more = m.group(3) or ""
            comp_adj = m.group(4)
            noun_b = m.group(5)
            return f"the {noun_b} is {much}{more}{comp_adj} than the {noun_a}"
        return None

    if category == "counting":
        m = _COUNTING_RE.match(sentence.strip())
        if not m:
            return None
        orig_numeral = m.group(1).lower()
        subject_word = m.group(2)
        verb_rest = m.group(4)

        # Pick a different numeral
        available = [n for n in _NUMERALS if n != orig_numeral]
        new_numeral = available[item_index % len(available)]

        # Fix grammar: is/are and singular/plural
        if new_numeral == "one":
            new_verb_form = "is"
            words = subject_word.split()
            words[-1] = singularize(words[-1])
            new_subject = " ".join(words)
        else:
            new_verb_form = "are"
            # Pluralize if currently singular (orig was "one")
            if orig_numeral == "one":
                words = subject_word.split()
                words[-1] = pluralize(words[-1])
                new_subject = " ".join(words)
            else:
                new_subject = subject_word

        return f"{new_numeral} {new_subject} {new_verb_form} {verb_rest}"

    return None
