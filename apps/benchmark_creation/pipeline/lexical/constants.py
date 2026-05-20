# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Constants for the lexical pipeline.

Centralizes word lists, category mappings, prompt templates, antonym look-ups,
and POS-related helpers that were previously scattered across the pipeline
scripts.
"""

# ---------------------------------------------------------------------------
#  WordNet noun super-categories
# ---------------------------------------------------------------------------

WORDNET_SUPER_CATEGORIES: dict[str, str] = {
    "animal.n.01": "animals",
    "food.n.01": "food_drink",
    "food.n.02": "food_drink",
    "beverage.n.01": "food_drink",
    "body_part.n.01": "body_parts",
    "clothing.n.01": "clothing",
    "garment.n.01": "clothing",
    "container.n.01": "household",
    "utensil.n.01": "household",
    "furniture.n.01": "furniture_rooms",
    "room.n.01": "furniture_rooms",
    "vehicle.n.01": "vehicles_outside",
    "conveyance.n.03": "vehicles_outside",
    "plant.n.02": "nature",
    "natural_object.n.01": "nature",
    "geological_formation.n.01": "nature",
    "tool.n.01": "tools",
    "implement.n.01": "tools",
    "plaything.n.01": "toys_play",
    "game.n.01": "toys_play",
    "person.n.01": "people",
    "building.n.01": "places",
    "structure.n.01": "places",
    "establishment.n.04": "places",
}

VALID_CATEGORIES: list[str] = sorted(set(WORDNET_SUPER_CATEGORIES.values()) | {"miscellaneous"})

# ---------------------------------------------------------------------------
#  WordNet hypernym sets for noun filtering (visualizability)
# ---------------------------------------------------------------------------

# Synset *names* -- resolved to actual wn.Synset objects lazily so the module
# can be imported even when the WordNet corpus is not downloaded yet.
ABSTRACT_HYPERNYM_NAMES: list[str] = [
    "abstraction.n.06",
    "attribute.n.02",
    "communication.n.02",
    "event.n.01",
    "state.n.02",
    "act.n.02",
    "feeling.n.01",
    "cognition.n.01",
    "process.n.06",
    "relation.n.01",
    "measure.n.02",
    "time_period.n.01",
    "group.n.01",
]

PERSON_HYPERNYM_NAMES: list[str] = [
    "person.n.01",
]

# ---------------------------------------------------------------------------
#  POS helpers
# ---------------------------------------------------------------------------

try:
    from nltk.corpus import wordnet as wn

    WN_POS: dict[str, str] = {"ADJ": wn.ADJ}
except Exception:  # noqa: BLE001 -- nltk corpus missing at import time falls back to literal POS code.
    WN_POS = {"ADJ": "a"}

POS_ARTICLES: dict[str, str] = {"ADJ": "an"}
POS_NAMES: dict[str, str] = {"ADJ": "adjective"}
POS_LABELS: dict[str, str] = {"ADJ": "Adjectives"}

# ---------------------------------------------------------------------------
#  Antonym lookup tables
# ---------------------------------------------------------------------------

ADJ_ANTONYMS: dict[str, str] = {
    "big": "small",
    "small": "big",
    "tall": "short",
    "short": "tall",
    "long": "short",
    "wide": "narrow",
    "narrow": "wide",
    "hot": "cold",
    "cold": "hot",
    "warm": "cool",
    "cool": "warm",
    "fast": "slow",
    "slow": "fast",
    "hard": "soft",
    "soft": "hard",
    "light": "dark",
    "dark": "light",
    "heavy": "light",
    "old": "new",
    "new": "old",
    "young": "old",
    "wet": "dry",
    "dry": "wet",
    "clean": "dirty",
    "dirty": "clean",
    "thick": "thin",
    "thin": "thick",
    "rough": "smooth",
    "smooth": "rough",
    "loud": "quiet",
    "quiet": "loud",
    "bright": "dim",
    "dim": "bright",
    "open": "closed",
    "closed": "open",
    "full": "empty",
    "empty": "full",
    "sharp": "dull",
    "dull": "sharp",
    "tight": "loose",
    "loose": "tight",
    "strong": "weak",
    "weak": "strong",
    "rich": "poor",
    "poor": "rich",
    "happy": "sad",
    "sad": "happy",
    "alive": "dead",
    "dead": "alive",
    "deep": "shallow",
    "shallow": "deep",
    "flat": "bumpy",
    "straight": "curved",
    "curved": "straight",
}

# ---------------------------------------------------------------------------
#  LLM prompt templates -- noun filtering
# ---------------------------------------------------------------------------

LLM_FILTER_PROMPT: str = """\
You are reviewing words for a child language-development benchmark.
Word: "{word}"
Assigned category: "{category}"
Available categories: {categories}

Answer these two questions:
1. Is this word safe and appropriate for a child benchmark? (YES/NO)
2. Is the category "{category}" correct for this word? (YES/NO/RECATEGORIZE: <new_category>)

Be conservative with recategorization: only recategorize if the word is clearly and \
unambiguously a better fit for another category. When in doubt, keep the current category.

Respond in exactly this format:
SAFE: YES or NO
CATEGORY: CORRECT or WRONG or RECATEGORIZE: <category_name>"""

# ---------------------------------------------------------------------------
#  LLM prompt templates -- adjective visual filtering
# ---------------------------------------------------------------------------

LLM_VISUALIZABLE_PROMPT: str = """\
You are reviewing words for a visual benchmark.

Word: "{word}"
Part of speech: {pos}

{pos_guidance}

IMPORTANT: If the word is primarily used as a DIFFERENT part of speech \
(e.g. a noun being tested as a verb, or a verb being tested as an adjective), \
answer NO. Only answer YES if the word is commonly and naturally used as \
a {pos} AND can be clearly depicted in a photograph.

Respond in exactly this format:
VISUALIZABLE: YES or NO
REASON: <brief explanation>"""

POS_GUIDANCE: dict[str, str] = {
    "ADJ": (
        "We want adjectives whose meaning can be CONVEYED VISUALLY -- either "
        "as a direct physical attribute of an object, or as a scene/environment "
        "property that a photograph can depict.\n"
        "Accepted categories (with examples):\n"
        "  - Size / shape: big, small, round, flat, narrow, thick\n"
        "  - Color / brightness: red, dark, shiny, translucent, vibrant, dull\n"
        "  - Texture / material: rough, smooth, wooden, metallic, fluffy\n"
        "  - Age / condition: old, new, rusty, broken, ripe, stale, fragile\n"
        "  - Temperature cues: frozen, melted, steaming, hot, cold, warm, icy\n"
        "  - Wetness / cleanliness: wet, dry, dirty, clean, muddy, dusty\n"
        "  - Weight / density (when visible): heavy, hollow, inflated\n"
        "  - Scene / environment: sunny, rainy, foggy, snowy, urban, tropical, "
        "indoor, outdoor, underwater, underground, wild, busy, chaotic\n"
        "  - Style / appearance: modern, classic, fancy, messy, neat, plain, "
        "handmade, striped, spotted, curly, straight\n"
        "  - Food properties: fresh, ripe, raw, cooked, spicy, sour\n"
        "  - Body state (visible): hungry, sleepy, strong, weak\n"
        "  - Speed / motion (depictable): fast, slow\n"
        "For example, 'red' -> a red object (YES), 'tall' -> a tall object (YES), "
        "'sunny' -> a sunny day (YES), 'messy' -> a messy room (YES), "
        "'vibrant' -> a vibrant painting (YES).\n"
        "Be LENIENT -- accept any adjective whose meaning can be depicted or "
        "strongly implied in a photograph.\n"
        "Reject ONLY if:\n"
        "  - The property is emotional, psychological, or describes a mood "
        "or personality trait (e.g. 'amusing', 'boring', 'annoyed', 'calm', "
        "'helpful', 'cheerful', 'nervous', 'generous', 'shy').\n"
        "  - The property is a subjective judgment or opinion with no visual "
        "correlate (e.g. 'interesting', 'important', 'useful', "
        "'wonderful', 'terrible').\n"
        "  - The property is purely abstract or conceptual with no physical "
        "manifestation (e.g. 'theoretical', 'political', 'cultural', "
        "'economic', 'legal', 'possible').\n"
        "  - The word is primarily a NOUN or VERB even if it has a rare "
        "adjective sense.\n"
        "  - The word is a nationality or ethnicity (e.g. 'american', "
        "'chinese', 'african')."
    ),
}

POS_CHECK_PROMPT: str = """\
Is the word "{word}" primarily used as {pos_article} {pos_name} in everyday English?

Consider how the word is MOST COMMONLY used.  For example:
- "run" -> primarily a VERB (YES if testing as verb)
- "light" -> ambiguous (noun, verb, and adjective uses are all common)
- "cow" -> primarily a NOUN (NO if testing as verb)

Answer in exactly this format:
PRIMARY_{pos_tag}: YES or NO
REASON: <brief explanation>"""

# ---------------------------------------------------------------------------
#  LLM prompt templates -- phrase generation
# ---------------------------------------------------------------------------

PHRASE_PROMPT: str = """\
Write a short phrase (2 to 4 words) that clearly shows the property "{adjective}".
The phrase must describe a simple, concrete object with this property that can \
be depicted in a single photograph.

Rules:
- You MUST use the exact adjective "{adjective}" in the phrase
- Do NOT replace it with a synonym or related adjective
- Use the pattern "a {adjective} [noun]" (e.g., "a red ball")
- The noun must be a common, concrete, physical object
- No abstract concepts, no metaphors
- Maximum 4 words

Examples:
- "red" -> "a red ball"
- "tall" -> "a tall building"
- "wet" -> "a wet dog"
- "broken" -> "a broken window"

Write ONLY the phrase, nothing else."""

NEG_PHRASE_PROMPT: str = """\
Given the adjective "{adjective}" and the positive phrase "{pos_phrase}", write a \
short phrase (2 to 4 words) that shows the OPPOSITE property with the SAME noun.

Rules:
- Keep the same noun from the positive phrase
- Use the pattern "a [opposite-adjective] [noun]"
- Maximum 4 words

Examples:
- "a tall building" -> "a short building"
- "a wet dog" -> "a dry dog"
- "a broken window" -> "a fixed window"

Write ONLY the phrase, nothing else."""

# ---------------------------------------------------------------------------
#  Default image styles
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_STYLES: list[str] = ["realistic", "cartoon"]
