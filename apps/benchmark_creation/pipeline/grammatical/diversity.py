# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Noun/verb diversity tracking and vocabulary coverage checking."""

import re

#: Minimum content-word length.
_MIN_WORD_LEN = 2
#: Length above which words ending in -er are likely comparatives, not nouns.
_MIN_COMPARATIVE_LEN = 4
#: Length above which words ending in -est are likely superlatives, not nouns.
_MIN_SUPERLATIVE_LEN = 5
#: Length above which -ing words are likely gerunds.
_MIN_GERUND_LEN = 4
#: Minimum word length for grouped extraction.
_MIN_GROUPED_WORD_LEN = 2

# ===========================================================================
#  Skip words — function words & common adjectives excluded from extraction
# ===========================================================================

_SKIP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "but",
    "not",
    "on",
    "in",
    "under",
    "behind",
    "above",
    "below",
    "beside",
    "next",
    "to",
    "of",
    "from",
    "with",
    "by",
    "for",
    "at",
    "than",
    "that",
    "also",
    "only",
    "both",
    "neither",
    "nor",
    "one",
    "two",
    "three",
    "four",
    "five",
    "more",
    "less",
    "much",
    "very",
    "being",
    "been",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "front",
    # Common adjectives that the heuristic noun extractor would miscount
    "big",
    "small",
    "tall",
    "short",
    "long",
    "old",
    "new",
    "young",
    "red",
    "blue",
    "green",
    "yellow",
    "white",
    "black",
    "brown",
    "pink",
    "orange",
    "purple",
    "gray",
    "grey",
    "wet",
    "dry",
    "hot",
    "cold",
    "warm",
    "cool",
    "clean",
    "dirty",
    "open",
    "closed",
    "full",
    "empty",
    "flat",
    "round",
    "soft",
    "hard",
    "dark",
    "light",
    "bright",
    "shiny",
    "dull",
    "rough",
    "smooth",
    "thick",
    "thin",
    "wide",
    "narrow",
    "deep",
    "shallow",
    "happy",
    "sad",
    "broken",
    "bent",
    "torn",
    "frozen",
    "cooked",
    "raw",
    "striped",
    "spotted",
    "fluffy",
    "fuzzy",
    "curly",
    "straight",
    "wooden",
    "metal",
    "plastic",
    "glass",
    "plain",
    "fancy",
    "heavy",
    "rusty",
    "moldy",
}

# Maximum allowed uses of a single noun within a category before we reject.
_MAX_NOUN_USES_PER_CATEGORY = 48

# Maximum allowed uses of a single verb (gerund) within a category.
_MAX_VERB_USES_PER_CATEGORY = 48

# Maximum allowed uses of a noun *pair* within a category.
_MAX_NOUN_PAIR_USES_PER_CATEGORY = 16


# ===========================================================================
#  Noun extraction & diversity
# ===========================================================================

_SINGULAR_ENDING_IN_S = {
    "bus",
    "focus",
    "cactus",
    "octopus",
    "corpus",
    "campus",
    "genus",
    "radius",
    "virus",
    "fungus",
    "stimulus",
    "syllabus",
    "apparatus",
    "citrus",
    "census",
    "consensus",
    "canvas",
    "atlas",
    "lens",
    "gas",
    "bias",
    "chaos",
    "iris",
}


def _normalize_noun(w: str) -> str:  # noqa: PLR0911 -- early returns per ending rule are clearer than nesting.
    """Best-effort singularization so 'rabbits' and 'rabbit' share a count."""
    if w in _SINGULAR_ENDING_IN_S:
        return w  # bus → bus (already singular)
    if w.endswith("ies") and len(w) > _MIN_COMPARATIVE_LEN:
        return w[:-3] + "y"  # puppies → puppy
    if w.endswith("ives") and len(w) > _MIN_SUPERLATIVE_LEN:
        return w[:-3] + "fe"  # knives → knife, wives → wife
    if w.endswith("ves") and len(w) > _MIN_COMPARATIVE_LEN:
        return w[:-3] + "f"  # wolves → wolf
    if w.endswith(("sses", "xes", "ches", "shes", "zzes")):
        return w[:-2]  # grasses → grass, boxes → box
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]  # cats → cat
    return w


def extract_nouns_from_caption(caption: str) -> list[str]:
    """Extract candidate nouns from a caption (simple heuristic).

    Returns lowercased content words that are likely nouns — we skip known
    function words, verbs in -ing form, and adjectives.  Nouns are normalized
    to their singular form so that plural/singular variants share diversity
    counts.
    """
    words = re.findall(r"[a-z]+", caption.lower())
    nouns = []
    for w in words:
        if w in _SKIP_WORDS:
            continue
        if w.endswith("ing") and len(w) > _MIN_GERUND_LEN:
            continue  # likely a gerund verb
        if w.endswith("er") and len(w) > _MIN_COMPARATIVE_LEN:
            continue  # likely a comparative adjective
        if w.endswith("est") and len(w) > _MIN_SUPERLATIVE_LEN:
            continue  # likely a superlative adjective
        if len(w) < _MIN_WORD_LEN:
            continue
        nouns.append(_normalize_noun(w))
    return nouns


def check_noun_diversity(
    caption_a: str,
    caption_b: str,
    noun_counts: dict[str, int],
    max_uses: int = _MAX_NOUN_USES_PER_CATEGORY,
) -> str | None:
    """Check that captions don't overuse nouns already common in this category.

    Returns None if OK, or a rejection reason string.
    """
    nouns = set(extract_nouns_from_caption(caption_a))
    nouns.update(extract_nouns_from_caption(caption_b))
    for noun in nouns:
        current = noun_counts.get(noun, 0)
        if current >= max_uses:
            return f"noun '{noun}' already used {current} times in this category"
    return None


def update_noun_counts(
    caption_a: str,
    caption_b: str,
    noun_counts: dict[str, int],
) -> None:
    """Increment noun usage counts for an accepted caption pair."""
    nouns = set(extract_nouns_from_caption(caption_a))
    nouns.update(extract_nouns_from_caption(caption_b))
    for noun in nouns:
        noun_counts[noun] = noun_counts.get(noun, 0) + 1


# ===========================================================================
#  Noun-pair diversity
# ===========================================================================


def _noun_pair_key(caption_a: str, caption_b: str) -> str:
    """Return a canonical, order-independent key for the noun pair in two captions."""
    nouns = sorted(set(extract_nouns_from_caption(caption_a)) | set(extract_nouns_from_caption(caption_b)))
    return ",".join(nouns)


def check_noun_pair_diversity(
    caption_a: str,
    caption_b: str,
    noun_pair_counts: dict[str, int],
    max_uses: int = _MAX_NOUN_PAIR_USES_PER_CATEGORY,
) -> str | None:
    """Reject caption pairs that reuse the same set of nouns too often.

    Returns None if OK, or a rejection reason string.
    """
    key = _noun_pair_key(caption_a, caption_b)
    if not key:
        return None
    current = noun_pair_counts.get(key, 0)
    if current >= max_uses:
        return f"noun combination '{key}' already used {current} times in this category"
    return None


def update_noun_pair_counts(
    caption_a: str,
    caption_b: str,
    noun_pair_counts: dict[str, int],
) -> None:
    """Increment noun-pair usage count for an accepted caption pair."""
    key = _noun_pair_key(caption_a, caption_b)
    if key:
        noun_pair_counts[key] = noun_pair_counts.get(key, 0) + 1


# ===========================================================================
#  Verb extraction & diversity
# ===========================================================================


def extract_verbs_from_caption(caption: str) -> list[str]:
    """Extract gerund verbs (-ing words) from a caption."""
    words = re.findall(r"[a-z]+", caption.lower())
    verbs = []
    for w in words:
        if w in _SKIP_WORDS:
            continue
        if w.endswith("ing") and len(w) > _MIN_GERUND_LEN:
            verbs.append(w)
    return verbs


def check_verb_diversity(
    caption_a: str,
    caption_b: str,
    verb_counts: dict[str, int],
    max_uses: int = _MAX_VERB_USES_PER_CATEGORY,
) -> str | None:
    """Check that captions don't overuse verbs already common in this category.

    Returns None if OK, or a rejection reason string.
    """
    verbs = set(extract_verbs_from_caption(caption_a))
    verbs.update(extract_verbs_from_caption(caption_b))
    for verb in verbs:
        current = verb_counts.get(verb, 0)
        if current >= max_uses:
            return f"verb '{verb}' already used {current} times in this category"
    return None


def update_verb_counts(
    caption_a: str,
    caption_b: str,
    verb_counts: dict[str, int],
) -> None:
    """Increment verb usage counts for an accepted caption pair."""
    verbs = set(extract_verbs_from_caption(caption_a))
    verbs.update(extract_verbs_from_caption(caption_b))
    for verb in verbs:
        verb_counts[verb] = verb_counts.get(verb, 0) + 1


# ===========================================================================
#  Vocabulary coverage
# ===========================================================================


def check_vocab_coverage(
    caption_a: str,
    caption_b: str,
    vocab_set: frozenset[str],
    function_words: set[str],
) -> str | None:
    """Check that all content words in both captions are in the vocabulary.

    Returns None if OK, or a rejection reason string listing OOV words.
    """
    oov: set[str] = set()
    for caption in (caption_a, caption_b):
        tokens = re.findall(r"[a-z]+", caption.lower())
        for w in tokens:
            if w in function_words:
                continue
            if len(w) < _MIN_GROUPED_WORD_LEN:
                continue
            if w not in vocab_set:
                oov.add(w)
    if oov:
        return f"out-of-vocabulary words: {', '.join(sorted(oov))}"
    return None
