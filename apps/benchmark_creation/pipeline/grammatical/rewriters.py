# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Category-specific prompt rewriting for contrastive image generation."""

import logging
import re

#: Captions this short are wrapped in an explicit "A clear depiction of:" prefix.
_SHORT_CAPTION_WORDS = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparative rewriting (ported from old Trog pipeline)
# ---------------------------------------------------------------------------

# Comparative adjective → base (positive) form.
# Used to rewrite "the A is taller than the B" → "a tall A next to a short B".
_COMPARATIVE_TO_BASE: dict[str, str] = {
    "taller": "tall",
    "shorter": "short",
    "bigger": "big",
    "smaller": "small",
    "larger": "large",
    "longer": "long",
    "wider": "wide",
    "narrower": "narrow",
    "thicker": "thick",
    "thinner": "thin",
    "heavier": "heavy",
    "lighter": "light",
    "darker": "dark",
    "brighter": "bright",
    "older": "old",
    "younger": "young",
    "newer": "new",
    "faster": "fast",
    "slower": "slow",
    "louder": "loud",
    "quieter": "quiet",
    "cleaner": "clean",
    "dirtier": "dirty",
    "wetter": "wet",
    "drier": "dry",
    "hotter": "hot",
    "colder": "cold",
    "warmer": "warm",
    "cooler": "cool",
    "fuller": "full",
    "emptier": "empty",
    "rounder": "round",
    "flatter": "flat",
    "sharper": "sharp",
    "duller": "dull",
    "softer": "soft",
    "harder": "hard",
    "rougher": "rough",
    "smoother": "smooth",
    "shinier": "shiny",
    "fatter": "fat",
    "skinnier": "skinny",
    "curvier": "curvy",
    "straighter": "straight",
}

# Base adjective → its antonym for the "losing" noun.
_ADJ_ANTONYMS: dict[str, str] = {
    "tall": "short",
    "short": "tall",
    "big": "small",
    "small": "big",
    "large": "small",
    "long": "short",
    "wide": "narrow",
    "narrow": "wide",
    "thick": "thin",
    "thin": "thick",
    "heavy": "light",
    "light": "heavy",
    "dark": "bright",
    "bright": "dim",
    "old": "young",
    "young": "old",
    "new": "old",
    "fast": "slow",
    "slow": "fast",
    "loud": "quiet",
    "quiet": "loud",
    "clean": "dirty",
    "dirty": "clean",
    "wet": "dry",
    "dry": "wet",
    "hot": "cold",
    "cold": "hot",
    "warm": "cool",
    "cool": "warm",
    "full": "empty",
    "empty": "full",
    "round": "flat",
    "flat": "round",
    "sharp": "dull",
    "dull": "sharp",
    "soft": "hard",
    "hard": "soft",
    "rough": "smooth",
    "smooth": "rough",
    "shiny": "dull",
    "fat": "skinny",
    "skinny": "fat",
    "curvy": "straight",
    "straight": "curvy",
    # Extended coverage for "more X" adjectives
    "colorful": "dull",
    "beautiful": "ugly",
    "clear": "cloudy",
    "cloudy": "clear",
    "strong": "weak",
    "weak": "strong",
    "deep": "shallow",
    "shallow": "deep",
    "rich": "plain",
    "plain": "rich",
    "dense": "sparse",
    "sparse": "dense",
    "fancy": "plain",
    "modern": "ancient",
    "ancient": "modern",
    "elegant": "crude",
    "crude": "elegant",
    "glossy": "matte",
    "matte": "glossy",
    "opaque": "transparent",
    "transparent": "opaque",
    "sturdy": "fragile",
    "fragile": "sturdy",
    "ornate": "simple",
    "simple": "ornate",
    "vibrant": "faded",
    "faded": "vibrant",
    "pale": "vivid",
    "vivid": "pale",
    "bulky": "slim",
    "slim": "bulky",
    "steep": "gentle",
    "gentle": "steep",
    "coarse": "fine",
    "fine": "coarse",
    "crisp": "soggy",
    "soggy": "crisp",
    "tight": "loose",
    "loose": "tight",
    "dim": "bright",
    # Negation-relevant adjectives (visual states, textures, conditions)
    "frozen": "melted",
    "melted": "frozen",
    "tasty": "bland",
    "bland": "tasty",
    "muddy": "clean",
    "messy": "tidy",
    "tidy": "messy",
    "sandy": "clean",
    "dusty": "clean",
    "rusty": "shiny",
    "cute": "ugly",
    "ugly": "cute",
    "happy": "sad",
    "sad": "happy",
    "bald": "hairy",
    "hairy": "bald",
    "neat": "messy",
    "fuzzy": "smooth",
    "fluffy": "sleek",
    "sleek": "fluffy",
    "shaggy": "sleek",
    "furry": "bare",
    "bare": "furry",
    "stocked": "empty",
    "rotten": "fresh",
    "fresh": "rotten",
    "ripe": "unripe",
    "unripe": "ripe",
    "whole": "broken",
    "broken": "intact",
    "intact": "broken",
    "stretched": "compressed",
    "rolled": "flat",
    "grassy": "barren",
    "barren": "grassy",
    "wooden": "metallic",
    "metallic": "wooden",
    "circular": "square",
    "square": "circular",
    "fair": "dark",
    # Colors — use a contrasting color so the two entities look visually distinct
    "red": "blue",
    "blue": "red",
    "green": "brown",
    "brown": "green",
    "yellow": "purple",
    "purple": "yellow",
    "orange": "blue",
    "pink": "green",
    "white": "black",
    "black": "white",
    "gray": "brown",
    "grey": "brown",
    "golden": "silver",
    "silver": "golden",
    "striped": "plain",
    "spotted": "plain",
    "checkered": "plain",
}

# Pattern: "the <A> is <comparative> than the <B>"
_COMPARATIVE_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(much\s+)?(more\s+)?(\w+(?:er)?)\s+than\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)


def _rewrite_comparative_prompt(sentence: str) -> str:
    """Rewrite a comparative sentence into a descriptive image prompt.

    ``"the lamp is brighter than the candle"``
    → ``"a bright lamp next to a dim candle"``

    Falls back to the original sentence if parsing fails.
    """
    m = _COMPARATIVE_RE.match(sentence.strip())
    if not m:
        return sentence

    noun_a = m.group(1)  # e.g. "lamp"
    # m.group(2) = "much " or None
    more = m.group(3)  # "more " or None
    comp_adj = m.group(4)  # e.g. "brighter" or "colorful"
    noun_b = m.group(5)  # e.g. "candle"

    # Determine the base form of the adjective
    if more:
        # "more colorful" → base is the word itself
        base_adj = comp_adj
    else:
        base_adj = _COMPARATIVE_TO_BASE.get(comp_adj.lower())
        if base_adj is None:
            # Best-effort: strip trailing -er/-r
            if comp_adj.endswith("ier"):
                base_adj = comp_adj[:-3] + "y"
            elif comp_adj.endswith("er"):
                base_adj = comp_adj[:-2]
            else:
                base_adj = comp_adj

    antonym = _ADJ_ANTONYMS.get(base_adj.lower())
    if antonym:
        return (
            f"a {base_adj} {noun_a} next to a {antonym} {noun_b}. "
            f"The {base_adj}/{antonym} difference must be dramatic and obvious."
        )
    # No known antonym — describe noun_a with the adjective and noun_b as
    # ordinary/plain, avoiding negation words like "not".
    return (
        f"a very {base_adj} {noun_a} next to an ordinary plain {noun_b}. "
        f"The {noun_a} must look extremely {base_adj} compared to the {noun_b}."
    )


# ---------------------------------------------------------------------------
# Comparative feasibility check
# ---------------------------------------------------------------------------

# Rough size tiers for common nouns (1=tiny → 5=huge).
# Used to detect comparisons that are too one-sided to produce useful images.
_NOUN_SIZE_TIER: dict[str, int] = {
    # tiny
    "ant": 1,
    "bee": 1,
    "butterfly": 1,
    "spider": 1,
    "bug": 1,
    "fly": 1,
    "worm": 1,
    "snail": 1,
    # small
    "mouse": 2,
    "rat": 2,
    "hamster": 2,
    "frog": 2,
    "crab": 2,
    "squirrel": 2,
    "bird": 2,
    "fish": 2,
    "kitten": 2,
    "puppy": 2,
    "parrot": 2,
    "lizard": 2,
    "goldfish": 2,
    "hedgehog": 2,
    # medium
    "cat": 3,
    "dog": 3,
    "rabbit": 3,
    "duck": 3,
    "chicken": 3,
    "fox": 3,
    "monkey": 3,
    "penguin": 3,
    "turtle": 3,
    "lamb": 3,
    "owl": 3,
    "eagle": 3,
    "hawk": 3,
    "goose": 3,
    "baby": 3,
    "child": 3,
    "kid": 3,
    "toddler": 3,
    "rooster": 3,
    "swan": 3,
    "raccoon": 3,
    # large
    "boy": 4,
    "girl": 4,
    "man": 4,
    "woman": 4,
    "person": 4,
    "pig": 4,
    "goat": 4,
    "sheep": 4,
    "deer": 4,
    "wolf": 4,
    "seal": 4,
    "donkey": 4,
    "pony": 4,
    "dolphin": 4,
    "zebra": 4,
    "lion": 4,
    "tiger": 4,
    "leopard": 4,
    "gorilla": 4,
    "kangaroo": 4,
    # huge
    "horse": 5,
    "cow": 5,
    "bear": 5,
    "giraffe": 5,
    "elephant": 6,
    "whale": 6,
    "hippo": 5,
    "rhino": 5,
    "moose": 5,
    "buffalo": 5,
}

# Adjectives where real-world physical size strongly biases the image model.
_SIZE_ADJECTIVES: set[str] = {
    "big",
    "small",
    "large",
    "tall",
    "short",
    "long",
    "wide",
    "thick",
    "thin",
    "heavy",
    "light",
    "fat",
    "skinny",
    "slim",
    "bulky",
    "tiny",
    "huge",
    "massive",
}


def _check_comparative_feasibility(
    noun_a: str,
    noun_b: str,
    base_adj: str,
) -> bool:
    """Check whether a comparative between two nouns is ambiguous enough.

    Returns ``True`` if the comparison could plausibly go either way (good
    for the benchmark), ``False`` if one direction is obviously always true
    (too easy — the image model will depict reality regardless of the prompt).

    Only applies to size-related adjectives; non-size adjectives always pass.
    """
    if base_adj.lower() not in _SIZE_ADJECTIVES:
        return True

    # Use the last word of each noun phrase as the head noun
    head_a = noun_a.lower().split()[-1]
    head_b = noun_b.lower().split()[-1]

    tier_a = _NOUN_SIZE_TIER.get(head_a)
    tier_b = _NOUN_SIZE_TIER.get(head_b)

    if tier_a is None or tier_b is None:
        return True  # unknown nouns — can't judge, assume OK

    # If tiers differ by more than 1, the comparison is too one-sided
    return abs(tier_a - tier_b) <= 1


# ---------------------------------------------------------------------------
# Negation prompt rewriting
# ---------------------------------------------------------------------------

# Pattern: "the <noun> is (not) <adj>"
_NEGATION_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(not\s+)?(.+?)\.?$",
    re.IGNORECASE,
)


def _rewrite_negation_prompt(sentence: str) -> str:
    """Rewrite a negation sentence into a positive image-friendly description.

    Positive captions are simplified::

        ``"the duck is old"``   → ``"an old duck"``

    Negative captions replace the adjective with its antonym::

        ``"the duck is not old"`` → ``"a young duck"``

    Falls back to the original sentence if parsing fails.
    """
    m = _NEGATION_RE.match(sentence.strip())
    if not m:
        return sentence

    noun = m.group(1)  # e.g. "duck"
    is_negated = m.group(2) is not None
    adj = m.group(3)  # e.g. "old"

    if is_negated:
        # Use the antonym to produce a positive description
        antonym = _ADJ_ANTONYMS.get(adj.lower())
        if antonym:
            article = "an" if antonym[0] in "aeiou" else "a"
            return f"{article} {antonym} {noun}"
        # No known antonym — use a plain/ordinary description to contrast
        # with the positive version (avoids negation words entirely)
        article = "an" if noun[0] in "aeiou" else "a"
        return f"{article} ordinary plain {noun}"
    # Positive: "the duck is old" → "an old duck"
    article = "an" if adj[0] in "aeiou" else "a"
    return f"{article} {adj} {noun}"


# ---------------------------------------------------------------------------
# Counting prompt helper
# ---------------------------------------------------------------------------

# Pattern to extract a numeral/number word followed by a noun phrase
_COUNTING_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b\s+(.+?)(?:\s+(?:is|are|has|have)\b\s+(.+))?$",
    re.IGNORECASE,
)


def _rewrite_counting_prompt(sentence: str) -> str:
    """Emphasize the exact count for image generation.

    ``"three dogs are running"``
    → ``"Exactly three dogs are running. Show precisely 3 dogs."``
    """
    m = _COUNTING_RE.search(sentence.strip())
    if not m:
        return sentence
    numeral = m.group(1)
    return (
        f"Exactly {numeral} {m.group(2)}"
        + (f" {m.group(3)}" if m.group(3) else "")
        + f". Show precisely {numeral} {m.group(2).split()[0]}."
    )


# ---------------------------------------------------------------------------
# subject_adjective prompt rewriting
# ---------------------------------------------------------------------------

# Pattern: "a [A1] [S1] and a [A2] [S2]"
_SUBJ_ADJ_RE = re.compile(
    r"^a\s+(.+?)\s+(\S+)\s+and\s+a\s+(.+?)\s+(\S+)\.?$",
    re.IGNORECASE,
)


def _rewrite_subject_adjective_prompt(sentence: str) -> str:
    """Rewrite subject_adjective captions to emphasize adj-noun bindings.

    ``"a blue bird and a red truck"``
    → ``"a blue bird next to a red truck. The bird is blue and the truck is red."``

    This makes the adjective-noun associations explicit and unambiguous for the
    image model, preventing it from mixing up which object has which property.
    """
    m = _SUBJ_ADJ_RE.match(sentence.strip())
    if not m:
        return sentence
    adj1, noun1, adj2, noun2 = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"a {adj1} {noun1} next to a {adj2} {noun2}. The {noun1} is {adj1} and the {noun2} is {adj2}."


# ---------------------------------------------------------------------------
# subject_verb prompt rewriting
# ---------------------------------------------------------------------------

# Pattern: "the [S1] is [V1-ing] and the [S2] is [V2-ing]"
_SUBJ_VERB_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(\S+ing)\s+and\s+the\s+(.+?)\s+is\s+(\S+ing)\.?$",
    re.IGNORECASE,
)


def _rewrite_subject_verb_prompt(sentence: str) -> str:
    """Rewrite subject_verb captions to emphasize subject-action bindings.

    ``"the boy is running and the girl is jumping"``
    → ``"a running boy next to a jumping girl. The boy is running and the girl is jumping."``
    """
    m = _SUBJ_VERB_RE.match(sentence.strip())
    if not m:
        return sentence
    s1, v1, s2, v2 = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"a {v1} {s1} next to a {v2} {s2}. The {s1} is {v1} and the {s2} is {v2}."


# ---------------------------------------------------------------------------
# embedded_relative prompt rewriting
# ---------------------------------------------------------------------------

# Pattern: "the [subject] [verb_s] the [object] that is [adj]"
_EMBED_REL_OBJECT_RE = re.compile(
    r"^the\s+(.+?)\s+(\S+s)\s+the\s+(.+?)\s+that\s+is\s+(.+?)\.?$",
    re.IGNORECASE,
)

# Pattern: "the [subject] that is [adj] [verb_s] the [object]"
_EMBED_REL_SUBJECT_RE = re.compile(
    r"^the\s+(.+?)\s+that\s+is\s+(.+?)\s+(\S+s)\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)


def _rewrite_embedded_relative_prompt(sentence: str, antonym: str | None = None) -> str:
    """Rewrite embedded_relative captions into disambiguated image prompts.

    Uses the LLM-generated *antonym* if available, falls back to the static
    ``_ADJ_ANTONYMS`` dict, and finally to explicit negation language.
    """
    # Try "the [subject] [verb_s] the [object] that is [adj]"
    m = _EMBED_REL_OBJECT_RE.match(sentence.strip())
    if m:
        subject, verb, obj, adj = m.group(1), m.group(2), m.group(3), m.group(4)
        ant = antonym or _ADJ_ANTONYMS.get(adj.lower())
        if ant:
            return (
                f"a {ant} {subject} {verb} a {adj} {obj}. "
                f"The {subject} is clearly {ant} and NOT {adj}. "
                f"The {obj} is clearly {adj}. "
                f"The two must look visually very different from each other."
            )
        return (
            f"a plain {subject} {verb} a {adj} {obj}. "
            f"ONLY the {obj} is {adj}. The {subject} is definitely NOT {adj}. "
            f"The two must look visually very different from each other."
        )

    # Try "the [subject] that is [adj] [verb_s] the [object]"
    m = _EMBED_REL_SUBJECT_RE.match(sentence.strip())
    if m:
        subject, adj, verb, obj = m.group(1), m.group(2), m.group(3), m.group(4)
        ant = antonym or _ADJ_ANTONYMS.get(adj.lower())
        if ant:
            return (
                f"a {adj} {subject} {verb} a {ant} {obj}. "
                f"The {subject} is clearly {adj}. "
                f"The {obj} is clearly {ant} and NOT {adj}. "
                f"The two must look visually very different from each other."
            )
        return (
            f"a {adj} {subject} {verb} a plain {obj}. "
            f"ONLY the {subject} is {adj}. The {obj} is definitely NOT {adj}. "
            f"The two must look visually very different from each other."
        )

    return sentence


# ---------------------------------------------------------------------------
# prepositions prompt rewriting
# ---------------------------------------------------------------------------

# Pattern: "the [object] is [preposition] the [landmark]"
_PREPOSITION_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(on|under|below|above|behind|beside|"
    r"in\s+front\s+of|next\s+to|in|near|over)\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)

# Per-preposition spatial descriptions to help image models
_PREP_SPATIAL_HINTS: dict[str, str] = {
    "on": (
        "resting on top of the surface of the {landmark}. "
        "The {obj} is touching the top of the {landmark}, supported by it."
    ),
    "under": (
        "on the ground directly beneath the {landmark}. "
        "The {landmark} is above and the {obj} is below it, "
        "visible underneath. Shot from a low angle to clearly show "
        "the {obj} tucked under the {landmark}."
    ),
    "below": (
        "positioned lower than the {landmark}, directly beneath it. "
        "The {landmark} is higher up and the {obj} is down below. "
        "Shot from a side angle to clearly show the vertical separation."
    ),
    "above": (
        "floating or positioned higher than the {landmark}, directly over it. "
        "The {obj} is up in the air above the {landmark}."
    ),
    "behind": (
        "positioned behind the {landmark}, partially hidden. "
        "The {landmark} is in the foreground and the {obj} peeks out "
        "from behind it."
    ),
    "in front of": (
        "positioned in front of the {landmark}, closer to the camera. "
        "The {obj} is in the foreground and the {landmark} is behind it."
    ),
    "beside": (
        "placed right next to the {landmark}, side by side on the same surface. "
        "The {obj} and the {landmark} are at the same height, sitting together."
    ),
    "next to": (
        "placed right next to the {landmark}, side by side on the same surface. "
        "The {obj} and the {landmark} are at the same height, sitting together."
    ),
    "in": ("placed inside the {landmark}. The {obj} is contained within the {landmark}."),
    "near": ("placed close to the {landmark} but not touching it. The {obj} is nearby the {landmark}."),
    "over": ("positioned directly over the {landmark}, higher up. The {obj} is above the {landmark}."),
}


def _rewrite_prepositions_prompt(sentence: str, avoid: str) -> str:
    """Rewrite prepositions captions into spatially explicit image prompts.

    Image models struggle with spatial relationships, especially 'under' and
    'below'.  This rewriter adds detailed spatial hints, camera angle
    suggestions, and explicit contrast with the avoided arrangement.
    """
    m = _PREPOSITION_RE.match(sentence.strip())
    if not m:
        return f"{sentence}. Important: depict ONLY this exact spatial arrangement. The image must NOT show: {avoid}"

    obj, prep, landmark = m.group(1), m.group(2).lower(), m.group(3)

    hint_template = _PREP_SPATIAL_HINTS.get(prep)
    hint = hint_template.format(obj=obj, landmark=landmark) if hint_template else f"{prep} the {landmark}"

    return f"a {obj} {hint} The image must clearly show the {obj} {prep} the {landmark}. Do NOT show: {avoid}"


# Pattern: "the <subject> is <verb>ing the <object>"
_ORDER_MATTERS_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(\S+ing)\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)


def _rewrite_order_matters_prompt(sentence: str, avoid: str) -> str:
    """Rewrite order_matters captions into disambiguated image prompts.

    The sentence follows the pattern "the X is Ving the Y".  The rewritten
    prompt clarifies *who is doing the action* so the generated image is
    unambiguous and clearly distinct from the inverted sentence.
    """
    m = _ORDER_MATTERS_RE.match(sentence.strip())
    if m:
        subject, verb, obj = m.group(1), m.group(2), m.group(3)
        return (
            f"a {subject} {verb} a {obj}. "
            f"The {subject} is clearly performing the action on the {obj}, "
            f"not the other way around. "
            f"The image must NOT depict: {avoid}"
        )
    # Fallback: return the sentence with avoid guidance
    return f"{sentence}. The image must NOT depict: {avoid}"


# ---------------------------------------------------------------------------
# Contrastive prompt construction
# ---------------------------------------------------------------------------


def build_contrastive_prompt(  # noqa: C901, PLR0911, PLR0912 -- pipeline orchestration: complexity matches the spec it implements
    caption_a: str,
    caption_b: str,
    image_index: int,
    category: str = "",
    antonym: str | None = None,
) -> str:
    """Build a prompt that clearly depicts the target and avoids the other.

    Parameters
    ----------
    caption_a : str
        The first caption (caption_a).
    caption_b : str
        The second caption (caption_b).
    image_index : int
        0 for image depicting caption_a, 1 for image depicting caption_b.
    category : str
        Grammatical category name.
    antonym : str, optional
        LLM-generated antonym for the adjective (used by embedded_relative).

    Returns
    -------
    str
        An enhanced prompt for the image generation model.
    """
    if image_index == 0:
        target = caption_a
        avoid = caption_b
    else:
        target = caption_b
        avoid = caption_a

    # --- Category-specific strategies ---

    if category == "prepositions":
        return _rewrite_prepositions_prompt(target, avoid)

    if category == "subject_adjective":
        return _rewrite_subject_adjective_prompt(target)

    if category == "subject_verb":
        return _rewrite_subject_verb_prompt(target)

    if category == "comparatives":
        rewritten = _rewrite_comparative_prompt(target)

        # Verify feasibility: check if the comparison is ambiguous enough
        # that both directions can produce meaningful, differentiable images.
        m = _COMPARATIVE_RE.match(target.strip())
        if m:
            noun_a, noun_b = m.group(1), m.group(5)
            more = m.group(3)
            comp_adj = m.group(4)
            base_adj = comp_adj if more else _COMPARATIVE_TO_BASE.get(comp_adj.lower(), comp_adj)

            if not _check_comparative_feasibility(noun_a, noun_b, base_adj):
                # The comparison is too one-sided (e.g. elephant vs mouse).
                # Add explicit override language so the image model doesn't
                # just fall back to real-world proportions.
                logger.debug(
                    "Comparative not ambiguous — adding override: %s",
                    target,
                )
                return (
                    f"A fantasy illustration where {rewritten} "
                    f"Exaggerate the difference so it is obvious and unambiguous."
                )

        return rewritten

    if category == "negation":
        # Rewrite both positive and negative captions into image-friendly
        # descriptions without negation words.  "the duck is not old" becomes
        # "a young duck" (antonym), while "the duck is old" becomes "an old duck".
        return _rewrite_negation_prompt(target)

    if category == "counting":
        return _rewrite_counting_prompt(target)

    if category == "embedded_relative":
        return _rewrite_embedded_relative_prompt(target, antonym=antonym)

    if category == "order_matters":
        return _rewrite_order_matters_prompt(target, avoid)

    # --- Default: generic contrastive strategy ---

    # Short captions (e.g., "a red ball and a blue cup") — wrap in clear instruction
    if len(target.split()) <= _SHORT_CAPTION_WORDS:
        return f"A clear depiction of: {target}"

    return f"{target} Important: depict ONLY this exact scene. The image must NOT show: {avoid}"
