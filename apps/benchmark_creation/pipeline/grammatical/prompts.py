# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt templates, category definitions, and prompt-building helpers for grammatical 2-AFC."""

import random
from typing import NamedTuple

#: Minimum noun pool size required to sample a 2-noun pair.
_MIN_NOUN_PAIR_POOL = 2

# ===========================================================================
#  Category definition
# ===========================================================================


class GrammaticalCategory(NamedTuple):
    """Descriptor for a single grammatical category."""

    pos: str
    template: str
    pair_mode: str  # "llm" or "deterministic"


# ===========================================================================
#  Shared guidance
# ===========================================================================

_TROG_SHARED_GUIDANCE = (
    " All words in the sentence must be common, everyday words."
    " Keep the sentence short — at most 8-10 words."
    " Every noun must be a concrete, physical object, animal, or person — "
    "never use abstract concepts, emotions, or invisible things."
    " The entire scene must be easy to depict in a single photograph — "
    "all mentioned objects and actions must be visible together in one image."
    " Choose nouns, verbs, and adjectives that are easy to swap or contrast "
    "with alternatives — avoid overly specific or unusual combinations that "
    "would make it hard to generate visually distinct distractor scenes."
    " IMPORTANT — VARIETY: do NOT always use 'dog' and 'cat'. Use a WIDE "
    "variety of familiar agents and objects that a young child would recognize. "
    "Good examples: boy, girl, man, woman, rabbit, horse, frog, bear, "
    "elephant, monkey, turtle, fish, duck, squirrel. "
    "For people use ONLY boy, girl, man, woman — no professions or family roles. "
    "Mix humans and animals across different sentences."
    " REALISM: the sentence must describe a scene that could actually happen "
    "in the real world. Do NOT describe impossible, absurd, or unlikely "
    "situations (e.g., 'the fish is climbing' or 'the chair is running')."
    " Write only the sentence, nothing else."
)

# ---------------------------------------------------------------------------
#  Noun pools used for injecting random noun suggestions into prompts
# ---------------------------------------------------------------------------

# Animate nouns for subject_verb, order_matters, counting, etc.
_ANIMATE_NOUNS = [
    "boy",
    "girl",
    "man",
    "woman",
    "dog",
    "cat",
    "bear",
    "monkey",
    "rabbit",
    "duck",
    "frog",
    "turtle",
    "fox",
    "pig",
    "goat",
    "deer",
    "sheep",
    "cow",
    "horse",
    "squirrel",
    "wolf",
    "owl",
    "penguin",
    "donkey",
    "mouse",
    "hen",
    "rooster",
    "otter",
    "raccoon",
    "pony",
    "puppy",
    "kitten",
    "lamb",
    "parrot",
]

# Concrete nouns for comparatives, grouped by scale so that randomly sampled
# pairs are plausible in both comparison directions.
_COMPARATIVE_NOUN_GROUPS = [
    # People
    ["boy", "girl", "man", "woman"],
    # Small-medium animals
    ["dog", "cat", "rabbit", "duck", "frog", "monkey", "fox"],
    # Large animals
    ["horse", "cow", "bear", "elephant"],
    # Vehicles
    ["truck", "car", "bus", "bicycle", "boat", "train"],
    # Structures
    ["house", "building", "fence", "bridge", "tower", "barn"],
    # Small objects / food
    [
        "cup",
        "plate",
        "bottle",
        "basket",
        "shoe",
        "hat",
        "ball",
        "book",
        "apple",
        "banana",
        "cake",
        "pie",
        "sandwich",
        "box",
        "bag",
    ],
    # Nature (similar-scale items)
    ["tree", "flower", "rock", "pond"],
    # Furniture / household
    ["table", "chair", "coat", "shirt", "umbrella"],
]

# Flat list for backward compatibility
_COMPARATIVE_NOUNS = [noun for group in _COMPARATIVE_NOUN_GROUPS for noun in group]

# ===========================================================================
#  Grammatical templates (9 categories)
# ===========================================================================

GRAMMATICAL_TEMPLATES = {
    "subject_verb": GrammaticalCategory(
        pos="VERB",
        template=(
            "Using the verb '{word}', write a "
            "sentence following EXACTLY this pattern:\n"
            "'the [S1] is [V1-ing] and the [S2] is [V2-ing]'\n"
            "where one of the verbs is '{word}' in its -ing form.\n\n"
            "RULES:\n"
            "1. S1 and S2 must be two DIFFERENT concrete nouns (persons or "
            "animals). Pick ONLY from this list: boy, girl, man, woman, "
            "dog, cat, bear, monkey, rabbit, duck, frog, turtle, fox, "
            "pig, goat, deer, sheep, cow, horse, squirrel, wolf, owl, "
            "penguin, donkey, mouse, hen, rooster, otter, raccoon, pony, "
            "puppy, kitten, lamb, parrot.\n"
            "2. V1 and V2 must be two DIFFERENT intransitive verbs in -ing form.\n"
            "3. Both actions must be visible in a single photograph.\n"
            "4. Both verbs must be common, everyday actions.\n"
            "5. V1 and V2 must NOT be synonyms or near-synonyms — they must "
            "describe visually DISTINCT actions that look clearly different in a "
            "photograph. BAD pairs: running/jogging, yelling/screaming, "
            "watching/looking, sitting/resting, leaping/jumping. "
            "GOOD pairs: running/sitting, dancing/sleeping, swimming/climbing.\n"
            "6. SWAP PLAUSIBILITY: caption_b swaps which subject does which "
            "verb. BOTH subjects must be able to plausibly do BOTH verbs. "
            "If swapping creates an impossible scene, pick different verbs "
            "or subjects.\n"
            "   BAD: 'the rabbit is flying' (rabbits cannot fly)\n"
            "   BAD: 'the frog is skiing' (frogs cannot ski)\n"
            "   BAD: 'the boy is galloping' (boys do not gallop)\n"
            "   GOOD: 'the boy is running and the girl is sitting' → swap → "
            "'the boy is sitting and the girl is running' (both make sense)\n\n"
            "Output EXACTLY two lines:\n"
            "caption_a: the [S1] is [V1-ing] and the [S2] is [V2-ing]\n"
            "caption_b: the [S1] is [V2-ing] and the [S2] is [V1-ing]\n"
            "(caption_b swaps which subject does which verb)" + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="llm",
    ),
    "subject_adjective": GrammaticalCategory(
        pos="ADJ",
        template=(
            "Using the adjective '{word}', write a phrase following EXACTLY "
            "this pattern:\n"
            "'a [A1] [S1] and a [A2] [S2]'\n"
            "where one of the adjectives is '{word}'.\n\n"
            "RULES:\n"
            "1. S1 and S2 must be two DIFFERENT concrete nouns (objects or animals).\n"
            "2. A1 and A2 must be two DIFFERENT visually obvious adjectives.\n"
            "3. Both adjective-noun pairs must be photographable.\n"
            "4. The adjectives must describe visually DISTINCT properties — "
            "do NOT use synonyms or near-synonyms. They must look clearly "
            "different in a photograph. BAD pairs: big/large, small/tiny, "
            "pretty/beautiful, wet/damp, quick/fast, glad/happy. "
            "GOOD pairs: red/tall, wet/round, big/striped, fluffy/broken.\n\n"
            "Output EXACTLY two lines:\n"
            "caption_a: a [A1] [S1] and a [A2] [S2]\n"
            "caption_b: a [A2] [S1] and a [A1] [S2]\n"
            "(caption_b swaps which noun has which adjective)" + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="llm",
    ),
    "negation": GrammaticalCategory(
        pos="ADJ",
        template=(
            "Using the adjective '{word}', write a sentence following EXACTLY "
            "this pattern:\n"
            "'the [subject] is {word}'\n\n"
            "RULES:\n"
            "1. The subject must be a concrete object or animal where '{word}' "
            "is a visually obvious property.\n"
            "2. It must be easy to photograph the subject both WITH and WITHOUT "
            "the property '{word}'. Ask yourself: can I take one photo where "
            "the subject IS {word}, and another where it is NOT? Both must be "
            "natural, realistic photos.\n"
            "3. The subject must be a single common noun.\n"
            "4. The adjective '{word}' must describe a PHYSICAL, VISIBLE state "
            "of the subject — something you can clearly SEE in a photograph.\n"
            "5. The adjective must make COMMON SENSE with the subject. "
            "Do NOT combine adjectives with subjects where the property "
            "doesn't naturally apply. BAD examples:\n"
            "   - 'the bird is open' (birds are not open/closed)\n"
            "   - 'the banana is tasty' (not visible in a photo)\n"
            "   - 'the duck is shallow' (shallow describes water, not ducks)\n"
            "   - 'the duck is plastic' (ducks are not plastic)\n"
            "   - 'the bear is rolled' (bears don't get rolled)\n"
            "   GOOD examples:\n"
            "   - 'the dog is wet' (dogs can be wet or dry)\n"
            "   - 'the shirt is dirty' (shirts can be dirty or clean)\n"
            "   - 'the cat is fluffy' (cats can be fluffy or sleek)\n"
            "   - 'the car is rusty' (cars can be rusty or not)\n\n"
            "Output ONLY the sentence, nothing else." + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="deterministic",
    ),
    "order_matters": GrammaticalCategory(
        pos="VERB",
        template=(
            "Using the verb '{word}', write a "
            "transitive sentence in the pattern "
            "'the [subject] is [verb-ing] the [object]', where the verb is "
            "'{word}' in its -ing form.\n"
            "RULES (you must follow ALL of them):\n"
            "1. Both subject and object MUST be simple, common nouns — "
            "pick ONLY from this list: boy, girl, man, woman, dog, cat, "
            "bear, monkey, rabbit, duck, frog, turtle, fox, "
            "pig, goat, deer, sheep, cow, horse, squirrel, wolf, owl, "
            "penguin, donkey, mouse, hen, rooster, otter, raccoon, pony, "
            "puppy, kitten, lamb, parrot.\n"
            "2. Subject and object must be DIFFERENT nouns.\n"
            "3. Both subject and object must be of SIMILAR SIZE so the action "
            "is physically plausible in both directions. Pair humans with "
            "humans (boy/girl, man/woman) and animals with animals of "
            "similar size (dog/cat, bear/monkey, duck/frog).\n"
            "4. The verb MUST be asymmetric: swapping subject and object MUST "
            "change the meaning.\n"
            "5. The INVERTED sentence (subject and object swapped) must ALSO "
            "be a realistic, everyday scene. Ask yourself: can BOTH do this "
            "action TO the other?\n"
            "6. Use a DIFFERENT pair of nouns for each sentence — do NOT "
            "keep reusing the same pair.\n\n"
            "GOOD examples (inversion works both ways):\n"
            "- 'the boy is chasing the girl' -> 'the girl is chasing the boy'\n"
            "- 'the cat is following the dog' -> 'the dog is following the cat'\n"
            "- 'the man is pushing the woman' -> 'the woman is pushing the man'\n"
            "- 'the dog is licking the cat' -> 'the cat is licking the dog'\n"
            "- 'the bear is scratching the monkey' -> 'the monkey is scratching the bear'\n"
            "- 'the duck is bumping the frog' -> 'the frog is bumping the duck'\n\n"
            "BAD examples (DO NOT USE):\n"
            "- 'the girl is riding the horse' -> 'the horse is riding the girl' WRONG — size mismatch\n"
            "- 'the man is carrying the baby' -> 'the baby is carrying the man' WRONG — size mismatch\n"
            "- 'the elephant is kicking the frog' -> 'the frog is kicking the elephant' WRONG — size mismatch\n"
            "- 'the bird is grabbing the bear' -> 'the bear is grabbing the bird' WRONG — size mismatch\n\n"
            "Output ONLY the sentence, nothing else." + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="deterministic",
    ),
    "prepositions": GrammaticalCategory(
        pos="NOUN",
        template=(
            "Using the noun '{word}', write a pair of sentences that differ "
            "ONLY in the spatial preposition.\n\n"
            "Pattern: 'the [object] is [prep] the {word}'\n"
            "where [object] is a small, movable thing and '{word}' is the "
            "LARGER landmark/reference point.\n\n"
            "RULES:\n"
            "1. Use the SAME object and the SAME noun '{word}' in both sentences.\n"
            "2. The two prepositions must create VISUALLY DISTINCT spatial "
            "arrangements (e.g., 'on' vs 'under', 'behind' vs 'in front of', "
            "'above' vs 'below').\n"
            "3. Both spatial arrangements must be physically possible and "
            "photographable.\n"
            "4. The [object] must be a SMALL, concrete, movable thing (e.g., "
            "cup, ball, book, cat, shoe, hat, toy) — something that can "
            "plausibly be placed on, under, behind, or beside '{word}'.\n"
            "5. SCALE: the [object] must be SMALLER than '{word}'. A lamp can "
            "be on a table, but a table cannot be on a lamp. A cat can be "
            "under a chair, but a chair cannot be under a cat.\n"
            "6. SEMANTIC FIT: both sentences must describe a scene that makes "
            "common sense. Ask yourself: could someone actually place [object] "
            "in that position relative to '{word}'? For example:\n"
            "   GOOD: 'the cup is on the table' / 'the cup is under the table'\n"
            "   BAD:  'the lamp is below the bedroom' (a lamp is INSIDE a "
            "bedroom, not below it)\n"
            "   BAD:  'the sofa is behind the button' (wrong scale)\n"
            "7. TANGIBLE LANDMARK: '{word}' must be a TANGIBLE, household-scale "
            "object (table, chair, box, bench, shelf, bed, basket, bucket). "
            "Do NOT use landscapes, locations, or large-scale features as "
            "landmarks — e.g., river, hill, island, carnival, pathway, ocean, "
            "mountain, forest, field, lake, road, city, park, beach, valley. "
            "You cannot physically place a small object 'under' or 'on' a "
            "river or a hill.\n"
            "8. 'UNDER' TEST: if using 'under' or 'below', the landmark MUST "
            "have a clear underside that a small object can fit beneath "
            "(e.g., table, chair, bridge, bench, bed). A duck cannot be "
            "'under' a carnival. A bird cannot be 'under' a hill.\n\n"
            "Output EXACTLY two lines:\n"
            "caption_a: the [object] is [prep1] the {word}\n"
            "caption_b: the [object] is [prep2] the {word}" + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="llm",
    ),
    "comparatives": GrammaticalCategory(
        pos="ADJ",
        template=(
            "Using the adjective '{word}' in its comparative form, "
            "write a sentence following EXACTLY this pattern:\n"
            "'the [noun_A] is [comparative of {word}] than the [noun_B]'\n\n"
            "CRITICAL RULE — BOTH DIRECTIONS MUST WORK:\n"
            "The two nouns will be SWAPPED to create a second caption. "
            "This means BOTH 'noun_A is more {word} than noun_B' AND "
            "'noun_B is more {word} than noun_A' must be plausible scenes "
            "that could be photographed. Pick two nouns of SIMILAR type "
            "and scale so the comparison could go EITHER way.\n\n"
            "GOOD (both directions photographable — use pairs like these):\n"
            "- 'the dog is bigger than the cat' (some dogs are small, some cats are big)\n"
            "- 'the truck is dirtier than the car' (either could be dirtier)\n"
            "- 'the boy is taller than the girl' (depends on individuals)\n"
            "- 'the cow is muddier than the horse' (either could be muddier)\n"
            "- 'the cup is fuller than the glass' (either could be fuller)\n\n"
            "BAD (only one direction works — DO NOT USE):\n"
            "- 'the elephant is bigger than the mouse' (a mouse can never be bigger)\n"
            "- 'the sun is brighter than the candle' (a candle can never be brighter)\n"
            "- 'the whale is heavier than the ant' (extreme size difference)\n"
            "- 'the building is taller than the cup' (wrong scale entirely)\n"
            "- 'the ice cream is hotter than the soup' (ice cream is always cold)\n\n"
            "RULES:\n"
            "1. Both nouns must be concrete, physical objects, animals, or people.\n"
            "2. Both must appear together in the same photograph.\n"
            "3. They must be VISUALLY DISTINCT — NOT variants of the same thing.\n"
            "4. The two nouns must be of SIMILAR SIZE and TYPE so the "
            "comparison is not predetermined by real-world knowledge.\n"
            "5. The property '{word}' must be visually obvious on both.\n\n"
            "Only use TANGIBLE, PHYSICAL adjectives — properties you can "
            "directly see or measure in a photo:\n"
            "- Size: big, small, tall, short, long, wide, thick, thin\n"
            "- Color intensity: dark, bright, light\n"
            "- Age/wear: old, new, young\n"
            "- Physical state: wet, dry, dirty, clean, full, empty, heavy\n"
            "- Temperature appearance: hot, cold\n\n"
            "Do NOT use subjective or abstract adjectives:\n"
            "- NOT: beautiful, pretty, ugly, cute, fancy, elegant\n"
            "- NOT: happy, sad, brave, clever, smart, kind, nice\n"
            "- NOT: interesting, important, popular, famous, expensive\n"
            "- NOT: fast, slow, loud, quiet (hard to see in a still photo)\n\n"
            "Output ONLY the sentence, nothing else." + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="deterministic",
    ),
    "counting": GrammaticalCategory(
        pos="VERB",
        template=(
            "Using the verb '{word}' in its -ing form, write a "
            "sentence with EXACTLY the numeral '{numeral}'.\n"
            "Pattern: '{numeral} [subject(s)] is/are [verb-ing]'.\n"
            "RULES:\n"
            "1. You MUST use the numeral '{numeral}' — do NOT change it.\n"
            "2. If the numeral is 'one', use a SINGULAR subject and 'is' "
            "(e.g., 'one dog is running').\n"
            "3. If the numeral is two or more, use a PLURAL subject and 'are' "
            "(e.g., 'three cats are running').\n"
            "4. The subject must be a concrete, familiar noun (person or animal). "
            "Use varied subjects — e.g., birds, frogs, rabbits, boys, girls, "
            "horses, monkeys, ducks, elephants. Avoid always using 'dogs' and 'cats'.\n"
            "5. Do NOT add objects, prepositional phrases, or extra details — "
            "keep it to the pattern '[numeral] [subject(s)] is/are [verb-ing]'.\n\n"
            "Output ONLY the sentence, nothing else." + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="deterministic",
    ),
    "embedded_relative": GrammaticalCategory(
        pos="VERB",
        template=(
            "Using the verb '{word}' in the third-person singular present "
            "tense (e.g., 'chase' becomes 'chases'), write a pair of "
            "sentences that differ ONLY in which noun the relative clause "
            "modifies.\n\n"
            "RULES:\n"
            "1. Subject and object must be concrete, familiar things visible in "
            "the same photograph.\n"
            "2. The adjective must describe physical appearance and be visually "
            "obvious (e.g., big, red, wet — NOT happy, clever, tired).\n"
            "3. CRITICAL: Use EXACTLY the same subject, verb, object, AND "
            "adjective in BOTH sentences. The ONLY difference is which noun "
            "the 'that is [adj]' clause is attached to. Do NOT change the "
            "adjective between caption_a and caption_b.\n"
            "4. CRITICAL: The adjective must be plausible for BOTH the subject "
            "AND the object, since the relative clause attaches to each in "
            "turn. When using COLOR adjectives (red, blue, yellow, green, "
            "pink, orange, purple), do NOT use a human subject (boy, girl, "
            "man, woman) — 'the boy that is red' is nonsensical. Instead "
            "pair colors with animals or objects that can naturally be that "
            "color. SIZE adjectives (big, small, tall, short) and STATE "
            "adjectives (wet, dirty, fluffy) work well with any subject.\n\n"
            "Output EXACTLY three lines:\n"
            "caption_a: the [subject] [verb-s] the [object] that is [adj]\n"
            "caption_b: the [subject] that is [adj] [verb-s] the [object]\n"
            "antonym: [a visually contrasting adjective to adj]\n"
            "(caption_a = the adjective applies to the object; "
            "caption_b = the adjective applies to the subject;\n"
            "antonym = a word that looks clearly OPPOSITE in a photo, "
            "e.g., if adj is 'red' the antonym could be 'blue', "
            "if adj is 'big' the antonym could be 'small')\n\n"
            "Example with verb 'hold' and adjective 'green':\n"
            "caption_a: the turtle holds the leaf that is green\n"
            "caption_b: the turtle that is green holds the leaf\n"
            "antonym: brown" + _TROG_SHARED_GUIDANCE
        ),
        pair_mode="llm",
    ),
}


# ===========================================================================
#  Prompt builders
# ===========================================================================


def format_word_pool_for_prompt(
    pool: list,
    selected: set[str],
    bucket_counts: dict[int, int],
    bucket_targets: dict[int, int],
    max_per_bin: int = 30,
) -> str:
    """Format available words grouped by frequency bin with distribution guidance.

    To keep prompts within token limits, each bin is capped at *max_per_bin*
    words (randomly sampled when the bin is larger).
    """
    by_bin: dict[int, list[str]] = {}
    for e in pool:
        if e.word not in selected:
            by_bin.setdefault(e.freq_bin, []).append(e.word)

    lines = []
    for bin_idx in sorted(by_bin.keys()):
        words = by_bin[bin_idx]
        if len(words) > max_per_bin:
            words = random.sample(words, max_per_bin)
        current = bucket_counts.get(bin_idx, 0)
        target = bucket_targets.get(bin_idx, 0)
        needed = max(0, target - current)
        status = f"(have {current}/{target}, needs {needed} more)" if needed > 0 else "(full)"
        lines.append(f"Frequency bin {bin_idx} {status}:")
        lines.append(f"  {', '.join(words)}")

    return "\n".join(lines)


# ===========================================================================
#  Per-category word selection guidance
# ===========================================================================

_WORD_SELECTION_GUIDANCE: dict[str, str] = {
    "subject_verb": (
        "This category needs simple INTRANSITIVE verbs — actions a person or "
        "animal does without an object (e.g., running, sleeping, dancing, "
        "jumping, swimming, eating, climbing, crawling, sitting, standing).\n"
        "CRITICAL: both subjects must be able to do BOTH verbs (since the "
        "verbs get swapped). Pick universal actions that humans AND animals "
        "can all plausibly do.\n"
        "AVOID:\n"
        "- Transitive-only verbs that need an object (e.g., 'carry', 'build', 'throw')\n"
        "- Abstract or invisible actions (e.g., 'think', 'believe', 'know', 'want')\n"
        "- Actions that look the same in a photo (e.g., 'listen' vs 'hear')\n"
        "- Verbs only some subjects can do (e.g., 'fly' — only birds can fly, "
        "'gallop' — only horses gallop)\n"
        "PREFER verbs whose -ing form shows a clearly visible body posture or movement."
    ),
    "subject_adjective": (
        "This category needs VISUALLY OBVIOUS adjectives — properties you can "
        "clearly see in a photograph (e.g., red, big, small, wet, dirty, round, "
        "tall, broken, striped, fluffy, shiny, wooden, spotted).\n"
        "AVOID:\n"
        "- Abstract or emotional adjectives (e.g., 'happy', 'nice', 'clever', "
        "'brave', 'important', 'interesting')\n"
        "- Subjective adjectives that depend on opinion (e.g., 'beautiful', 'ugly')\n"
        "- Adjectives that are hard to photograph (e.g., 'loud', 'fast', 'quiet')\n"
        "PREFER adjectives that describe color, size, shape, texture, or physical state."
    ),
    "negation": (
        "This category needs adjectives where BOTH the presence AND absence "
        "of the property are visually obvious and photographable "
        "(e.g., 'wet' — you can photograph a wet dog AND a dry dog; "
        "'dirty' — you can photograph dirty vs clean; 'broken' — broken vs intact).\n"
        "AVOID:\n"
        "- Adjectives whose absence is hard to photograph (e.g., 'loud', 'fast')\n"
        "- Abstract adjectives (e.g., 'happy', 'clever', 'important')\n"
        "- Non-visual properties (e.g., 'tasty', 'sweet', 'fragrant')\n"
        "- States that don't apply to common objects/animals (e.g., 'open', "
        "'shallow', 'steep', 'stocked', 'tropical', 'plastic', 'uniform')\n"
        "PREFER simple, physical adjectives: wet, dirty, muddy, fluffy, "
        "striped, spotted, old, broken, rusty, round, flat, bent, shiny."
    ),
    "order_matters": (
        "This category needs verbs where 'A is Xing B' and 'B is Xing A' are "
        "BOTH realistic, everyday scenes — the inversion MUST make sense.\n\n"
        "The word pool has already been filtered to safe verbs that pass the "
        "inversion test. Pick any verb from the pool — all of them work in "
        "both directions.\n\n"
        "The KEY test: pick two agents of similar size (boy/girl, cat/dog, "
        "bear/monkey). Can BOTH do the action TO the other? If not, don't pick it."
    ),
    "prepositions": (
        "This category needs NOUNS that are TANGIBLE, household-scale objects "
        "where small things can be placed in different spatial relationships "
        "(on, under, behind, in front of, above, below, beside, in).\n"
        "Good nouns: table, chair, box, basket, shelf, bench, bed, rock, "
        "fence, bridge, wall, bowl, bucket, pillow, blanket, desk, stool, "
        "crate, barrel, wagon, ladder, sofa, cabinet, cart.\n"
        "AVOID:\n"
        "- Abstract nouns (e.g., 'idea', 'love', 'time')\n"
        "- Nouns that are too small for spatial relations (e.g., 'button', 'seed')\n"
        "- Landscapes, locations, and large-scale features (e.g., 'river', "
        "'hill', 'island', 'carnival', 'pathway', 'ocean', 'mountain', "
        "'forest', 'field', 'lake', 'road', 'city', 'park', 'beach', "
        "'valley', 'town', 'village', 'garden', 'street') — you cannot "
        "place a small object 'on' or 'under' a river or a hill\n"
        "- Nouns where spatial prepositions don't make physical sense\n"
        "PREFER nouns that are medium-sized, common objects with clear spatial "
        "affordances (things can be on/under/behind them)."
    ),
    "comparatives": (
        "This category needs TANGIBLE, PHYSICAL adjectives — properties you "
        "can directly see or measure in a photograph.\n\n"
        "GOOD adjective types (pick from these):\n"
        "- Size: big, small, tall, short, long, wide, thick, thin, fat, slim\n"
        "- Color/brightness: dark, bright, light, pale\n"
        "- Age/wear: old, new, young\n"
        "- Physical state: wet, dry, dirty, clean, full, empty, rough, smooth\n"
        "- Weight appearance: heavy, light\n"
        "- Temperature appearance: hot, cold, warm\n\n"
        "AVOID (do NOT pick these):\n"
        "- Subjective: beautiful, pretty, ugly, cute, fancy, elegant, nice\n"
        "- Emotional: happy, sad, brave, clever, smart, kind, gentle\n"
        "- Abstract: interesting, important, popular, famous, expensive, valuable\n"
        "- Hard to photograph: fast, slow, loud, quiet, strong, weak\n"
        "- Non-gradable: dead, alive, pregnant, unique, perfect\n\n"
        "The adjective must produce a comparison that is obviously TRUE for "
        "common objects/animals — e.g., 'big' works because an elephant IS "
        "bigger than a cat, and you can clearly see that in a photo."
    ),
    "counting": (
        "This category needs INTRANSITIVE verbs — actions that multiple agents "
        "can do simultaneously, and that are visually countable in a photograph "
        "(e.g., running, swimming, sleeping, jumping, sitting, flying, eating, "
        "dancing, standing, climbing).\n"
        "AVOID:\n"
        "- Transitive verbs needing an object (e.g., 'throw', 'carry')\n"
        "- Actions that are hard to count visually (e.g., 'think', 'breathe')\n"
        "- Actions where multiple agents would overlap and be hard to count "
        "(e.g., 'hiding')\n"
        "PREFER verbs that produce distinct, countable visual postures — you "
        "should be able to look at a photo and count how many agents are doing it."
    ),
    "embedded_relative": (
        "This category needs TRANSITIVE verbs — actions one agent does TO "
        "another or to an object, used in 3rd-person singular present tense "
        "(e.g., chases, pushes, watches, carries, holds, kicks, pulls, feeds, "
        "lifts, washes, paints, catches).\n"
        "AVOID:\n"
        "- Intransitive verbs that don't take an object (e.g., 'sleep', 'run')\n"
        "- Abstract verbs (e.g., 'think', 'believe', 'know')\n"
        "- Verbs that are hard to photograph (e.g., 'hear', 'smell', 'feel')\n"
        "PREFER verbs that show a clear, visible action between two entities — "
        "the action should be obvious in a photograph."
    ),
}


def build_word_selection_prompt(
    category: str,
    template: str,
    pool_text: str,
    selected: set[str],
) -> str:
    """Call 1: ask LLM to pick a word that fits the category well."""
    exclusion = ""
    if selected:
        exclusion = f"\n\nAlready selected words (DO NOT pick any of these): {', '.join(sorted(selected))}"

    # Category-specific guidance
    cat_guidance = _WORD_SELECTION_GUIDANCE.get(category, "")
    guidance_block = ""
    if cat_guidance:
        guidance_block = f"\n\n=== WHAT MAKES A GOOD WORD FOR THIS CATEGORY ===\n{cat_guidance}\n"

    return (
        f"You are selecting a word for the grammatical category '{category}'.\n\n"
        f"The word you pick will be used to build a sentence following this "
        f"template:\n{template}\n"
        f"{guidance_block}\n"
        f"=== GENERAL REQUIREMENTS ===\n"
        f"The word must lead to a sentence that:\n"
        f"- Describes a REALISTIC scene that could actually happen in real life\n"
        f"- Uses only concrete, everyday, familiar things (a child would know)\n"
        f"- Can be clearly depicted in a single photograph\n"
        f"- Is NOT abstract, metaphorical, or impossible to visualize\n\n"
        f"=== AVAILABLE WORDS ===\n"
        f"Below are available words grouped by frequency bin. Prefer picking "
        f"from bins that still need more words (marked 'needs N more').\n\n"
        f"{pool_text}"
        f"{exclusion}\n\n"
        f"Pick ONE word from the list above that best fits this category. "
        f"The word must appear in the list above.\n\n"
        f"Respond with ONLY the chosen word, nothing else."
    )


def build_pair_generation_prompt(  # noqa: C901, PLR0913 -- pipeline-level orchestration: many parallel context fields
    category: str,
    template: str,
    word: str,
    pos: str,  # noqa: ARG001 -- `pos` kept for parallel-worker signature uniformity
    item_index: int = 0,
    overused_nouns: list[str] | None = None,
    overused_verbs: list[str] | None = None,
    overused_noun_pairs: list[str] | None = None,
    suggested_noun_pool: list[str] | None = None,
) -> str:
    """Call 2: generate caption pair (or single sentence for deterministic tasks).

    For ``pair_mode="llm"`` tasks, the template already asks for two lines.
    For ``pair_mode="deterministic"`` tasks, the template asks for a single
    sentence; the negative is computed in code.

    If *overused_nouns* or *overused_verbs* is provided, avoidance hints are
    appended to encourage the LLM to use different words.

    If *suggested_noun_pool* is provided (pre-filtered against the dataset
    vocabulary), two nouns are randomly sampled and injected as a suggestion
    to increase pair diversity.
    """
    fmt_kwargs = {"word": word}
    if category == "counting":
        numerals = ("one", "two", "three", "four", "five")
        numeral = numerals[item_index % len(numerals)]
        fmt_kwargs["numeral"] = numeral

    prompt = template.format(**fmt_kwargs)

    # --- Positive steering: suggest random nouns to use ---
    if suggested_noun_pool is not None:
        pool = list(suggested_noun_pool)
        # Remove overused nouns from the suggestion pool
        if overused_nouns:
            exclude = set(overused_nouns)
            pool = [n for n in pool if n not in exclude]
        # Categories with a single subject noun need only one suggestion
        single_noun_categories = {"counting", "negation", "prepositions"}
        if category in single_noun_categories:
            if pool:
                noun = random.choice(pool)
                prompt += f"\n\nFor THIS sentence, use '{noun}' as the subject."
        elif category == "comparatives" and len(pool) >= _MIN_NOUN_PAIR_POOL:
            # For comparatives, prefer scale-matched noun pairs so both
            # comparison directions are plausible.
            pool_set = set(pool)
            eligible_groups = [[n for n in g if n in pool_set] for g in _COMPARATIVE_NOUN_GROUPS]
            eligible_groups = [g for g in eligible_groups if len(g) >= _MIN_NOUN_PAIR_POOL]
            if eligible_groups:
                group = random.choice(eligible_groups)
                pair = random.sample(group, 2)
            else:
                pair = random.sample(pool, 2)
            prompt += f"\n\nFor THIS sentence, use '{pair[0]}' and '{pair[1]}' as your two nouns."
        elif len(pool) >= _MIN_NOUN_PAIR_POOL:
            pair = random.sample(pool, 2)
            label = "subjects" if category in ("subject_verb", "order_matters") else "nouns"
            prompt += f"\n\nFor THIS sentence, use '{pair[0]}' and '{pair[1]}' as your two {label}."

    # --- Negative steering: avoid overused words ---
    if overused_nouns:
        avoid_list = ", ".join(overused_nouns[:10])
        prompt += (
            f"\n\nIMPORTANT — for VARIETY, do NOT use any of these nouns "
            f"(they have been used too many times already): {avoid_list}. "
            f"Pick different, fresh nouns instead."
        )

    if overused_verbs:
        avoid_list = ", ".join(overused_verbs[:10])
        prompt += (
            f"\n\nIMPORTANT — for VARIETY, do NOT use any of these verbs "
            f"(they have been used too many times already): {avoid_list}. "
            f"Pick different, fresh verbs instead."
        )

    if overused_noun_pairs:
        pair_list = "; ".join(overused_noun_pairs[:8])
        prompt += (
            f"\n\nIMPORTANT — for VARIETY, do NOT reuse these noun combinations "
            f"(they have appeared too many times already): {pair_list}. "
            f"Pick a DIFFERENT combination of nouns."
        )

    return prompt


def build_validation_prompt(
    caption_a: str,
    caption_b: str,
    category: str,
    word: str,
) -> str:
    """Call 3: ask LLM to validate both captions and their contrastive quality."""
    base = (
        f"You are validating a caption pair generated for the grammatical "
        f"category '{category}', using the word '{word}'.\n\n"
        f'Caption A: "{caption_a}"\n'
        f'Caption B: "{caption_b}"\n\n'
        f"Check ALL of the following:\n"
        f"1. Are both captions grammatically correct?\n"
        f"2. Do both describe concrete, visually representable scenes?\n"
        f"3. Are all nouns/objects concrete, specific, and familiar "
        f"(things a young child would recognize)?\n"
        f"4. Could each scene be depicted in a single, simple photograph?\n"
        f"5. Are both captions simple and easy to understand?\n"
        f"6. Do the two captions form a meaningful CONTRAST — i.e., someone "
        f"who understands the grammar would clearly pick different images "
        f"for each caption?\n"
        f"7. Are the two captions DIFFERENT from each other (not identical)?\n"
    )

    if category in ("subject_verb", "counting"):
        base += (
            "8. REALISM — Are BOTH scenes physically POSSIBLE? Answer NO only "
            "if a scene is truly IMPOSSIBLE given the animal's anatomy "
            "(e.g., 'a fish is climbing', 'a turtle is flying'). "
            "Do NOT reject scenes that are unusual but possible — most animals "
            "CAN run, sit, eat, jump, climb, swim, sleep, walk, stretch, roll, "
            "slide, wave, scratch, dance, spin, play, dig, lean, splash, "
            "bounce, hide, crawl, yawn, and crouch.\n"
        )
    else:
        base += (
            "8. REALISM — Are BOTH scenes physically possible and realistic? "
            "Would each scene actually happen in real life? Answer NO if either "
            "caption describes something impossible, absurd, or extremely unlikely "
            "(e.g., 'a fish is climbing', 'a chair is running', "
            "'the table is swimming').\n"
        )

    if category == "order_matters":
        from apps.benchmark_creation.pipeline.grammatical.constants import invert_order_matters

        inverted = invert_order_matters(caption_a)
        if inverted:
            base += (
                f"9. Are BOTH the subject and the object humans or animals?\n"
                f'10. INVERSION TEST — The inverted sentence is: "{inverted}"\n'
                f"   Would this inverted scene happen in everyday life?\n"
                f"   Answer NO if the inverted sentence is physically impossible "
                f"or absurd.\n"
            )

    if category == "subject_verb":
        base += (
            "9. SYNONYM CHECK — Are the two verbs synonyms or near-synonyms? "
            "They must describe visually DISTINCT actions that look clearly "
            "different in a photograph. Answer NO if the two verbs are "
            "synonyms (e.g., running/jogging, yelling/screaming, "
            "watching/looking, sitting/resting, leaping/jumping).\n"
            "10. SWAP PLAUSIBILITY — Caption_b swaps which subject does "
            "which verb. Can BOTH subjects plausibly do BOTH verbs? Answer "
            "NO ONLY if the swap is truly PHYSICALLY IMPOSSIBLE. Examples:\n"
            "   - 'the rabbit is flying' → NO (rabbits cannot fly)\n"
            "   - 'the frog is skiing' → NO (frogs cannot ski)\n"
            "   GOOD (answer YES — these are plausible):\n"
            "   - 'the monkey is dancing' → YES (monkeys can dance)\n"
            "   - 'the dog is jumping' → YES (dogs jump)\n"
            "   - 'the cat is climbing' → YES (cats climb)\n"
            "   - 'the duck is swimming' → YES (ducks swim)\n"
            "   - 'the bear is scratching' → YES (bears scratch)\n"
            "   Be LENIENT — most common animals CAN do most basic physical "
            "actions (running, sitting, eating, jumping, climbing, swimming, "
            "sleeping, walking, stretching, rolling, sliding, waving, "
            "scratching, dancing, spinning, playing, digging, leaning, "
            "splashing, bouncing, hiding, crawling). Only reject actions that "
            "are truly impossible for the animal's anatomy (flying for "
            "non-birds, skiing, frying, typing, driving).\n"
        )

    if category == "subject_adjective":
        base += (
            "9. SYNONYM CHECK — Are the two adjectives synonyms or "
            "near-synonyms? They must describe visually DISTINCT properties "
            "that look clearly different in a photograph. Answer NO if the "
            "two adjectives are synonyms (e.g., big/large, small/tiny, "
            "pretty/beautiful, wet/damp, quick/fast).\n"
        )

    if category == "comparatives":
        base += (
            "9. BOTH DIRECTIONS — Caption B swaps the nouns. Could BOTH "
            "Caption A AND Caption B be depicted in a real photograph? "
            "The two nouns must be similar enough that the comparison "
            "could plausibly go either way. Answer NO if one direction is "
            "physically impossible (e.g., 'the mouse is bigger than the "
            "elephant' — a mouse can never be bigger).\n"
            "   GOOD: 'the dog is bigger than the cat' (both ways possible)\n"
            "   BAD: 'the elephant is bigger than the ant' (only one way)\n"
            "10. Is the adjective TANGIBLE and PHYSICAL (size, color, age, "
            "weight, texture, temperature)? Answer NO if it is subjective "
            "(beautiful, cute, nice), emotional (happy, brave, clever), "
            "abstract (interesting, important), or hard to see in a still "
            "photo (fast, slow, loud, quiet).\n"
        )

    if category == "prepositions":
        base += (
            f"9. SCALE — Is the [object] (the subject) SMALLER than the "
            f"landmark noun '{word}'? The object should be something that "
            f"can be placed on, under, or beside the landmark. Answer NO if "
            f"the object is larger than or the same scale as the landmark.\n"
            f"10. SEMANTIC FIT — Do both spatial relationships make common "
            f"sense? Ask: could someone actually place the object in that "
            f"position relative to '{word}'? For example:\n"
            f"   - 'the cup is on the table' → YES (makes sense)\n"
            f"   - 'the lamp is below the bedroom' → NO (a lamp is INSIDE "
            f"a bedroom, not below it)\n"
            f"   - 'the sofa is behind the button' → NO (wrong scale)\n"
            f"   Answer NO if either caption describes a spatially nonsensical "
            f"arrangement.\n"
            f"11. TANGIBLE LANDMARK — Is '{word}' a tangible, household-scale "
            f"object? Answer NO if it is a landscape, location, or large-scale "
            f"feature (river, hill, island, carnival, pathway, ocean, mountain, "
            f"forest, field, lake, road, city, park, beach, valley). You cannot "
            f"place a small object 'on' or 'under' a river or a hill.\n"
            f"12. UNDER/BELOW TEST — If either caption uses 'under' or 'below', "
            f"does the landmark '{word}' have a clear underside that a small "
            f"object can fit beneath? Answer NO if the landmark is a landscape "
            f"or open area with no underside (e.g., 'under the hill', "
            f"'below the river', 'under the pathway').\n"
        )

    # Universal check for all categories: self-contained meaning
    base += (
        "\nSELF-CONTAINED MEANING — Does each sentence make complete sense on "
        "its own, without needing extra context? Answer NO if any word implies "
        "an unfinished process, a relative position, or a comparison that isn't "
        "specified. Every property or state must be fully understandable from "
        "the sentence alone.\n"
        "   BAD examples (reject these):\n"
        "   - 'the horse is halfway' → NO (halfway through what?)\n"
        "   - 'the dog is due' → NO (due for what?)\n"
        "   - 'the cat is ahead' → NO (ahead of what?)\n"
        "   - 'the bird is pending' → NO (pending is not a visual state)\n"
        "   - 'the boy is former' → NO (former what?)\n"
    )

    base += "\nRespond with ONLY 'YES' if the pair passes all checks, or 'NO: [brief reason]' if it fails any check."
    return base


def build_post_verification_prompt(
    caption_a: str,
    caption_b: str,
    category: str,
    word: str,
) -> str:
    """Build a strict common-sense verification prompt for accepted pairs.

    This is a second-pass filter that runs after the initial validation.
    It focuses on catching subtle common-sense violations that the first
    validator might miss, such as implausible comparisons, physically
    impossible spatial arrangements, or scenes that cannot be photographed.
    """
    # Categories where caption_b is a deliberate inversion of caption_a
    inverted_b_categories = {"comparatives", "order_matters", "counting", "negation"}

    inversion_note = ""
    if category in inverted_b_categories:
        inversion_note = (
            "NOTE: In this category, Caption B is a DELIBERATE transformation "
            "of Caption A (e.g., swapped nouns, added negation, changed number). "
            "BOTH captions must describe scenes that COULD plausibly happen "
            "and be photographed. If Caption B is physically impossible or "
            "absurd (e.g., 'the mouse is bigger than the elephant'), then the "
            "pair is BAD — reject it. Good pairs have BOTH directions be "
            "plausible.\n\n"
        )

    base = (
        f"You are a STRICT quality checker for a visual language benchmark. "
        f"Your job is to REJECT any caption pair that has even a SMALL problem. "
        f"When in doubt, REJECT.\n\n"
        f"Category: '{category}'\n"
        f"Word: '{word}'\n"
        f'Caption A: "{caption_a}"\n'
        f'Caption B: "{caption_b}"\n\n'
        f"{inversion_note}"
        f"Answer NO if ANY of the following is true:\n\n"
        f"1. COMMON SENSE — Does either caption describe something that "
        f"contradicts basic real-world knowledge so strongly that it could "
        f"NEVER be depicted in a photograph?\n"
        f"   Examples of common-sense violations:\n"
        f"   - 'the ice cream is hotter than the soup' → NO (ice cream is cold)\n"
        f"   - 'the pillow is heavier than the car' → NO (pillows are light)\n"
        f"   - 'the snail is faster than the cheetah' → NO (snails are slow)\n"
        f"   - 'the fire is wetter than the towel' → NO (fire is not wet)\n"
        f"   - 'the stone is softer than the cotton' → NO (stones are hard)\n\n"
        f"2. PHYSICAL POSSIBILITY — Can BOTH scenes ACTUALLY exist in the "
        f"real world? Could you take a photograph of EACH scene?\n"
        f"   Examples of physical impossibilities:\n"
        f"   - 'the ball is under the floor' → NO (objects can't be under a floor in a photo)\n"
        f"   - 'the car is inside the cup' → NO (a car doesn't fit in a cup)\n"
        f"   - 'the fish is climbing the tree' → NO (fish can't climb)\n"
        f"   - 'the house is on the table' → NO (wrong scale)\n\n"
        f"3. VISUAL CLARITY — Can you CLEARLY distinguish caption A from "
        f"caption B by looking at two different photographs? If the two "
        f"scenes would look the same or nearly the same in photos, reject.\n\n"
        f"4. AMBIGUITY — Is either caption confusing, ambiguous, or open to "
        f"multiple very different interpretations?\n\n"
        f"5. REALISM — Would each scene naturally occur? Reject contrived, "
        f"forced, or extremely unusual combinations.\n"
        f"   - 'the elephant is standing on the skateboard' → NO (unlikely)\n"
        f"   - 'the baby is driving the car' → NO (impossible)\n\n"
        f"6. COMPLETENESS — Does each caption form a complete, meaningful thought? "
        f"Reject if any word requires missing context to make sense. Every "
        f"property or state must be fully understandable from the sentence alone.\n"
        f"   - 'the horse is halfway' → NO (halfway through what?)\n"
        f"   - 'the dog is due' → NO (due for what?)\n"
        f"   - 'the cake is partial' → NO (partial what?)\n"
        f"   - 'the cat is ahead' → NO (ahead of what?)\n"
        f"   - 'the boy is former' → NO (former what?)\n\n"
    )

    # Category-specific additional checks
    if category == "comparatives":
        base += (
            f"IMPORTANT — For comparatives, Caption B swaps the nouns from "
            f"Caption A. BOTH directions must be plausible photos. This means "
            f"the two nouns must be SIMILAR enough in '{word}' that either "
            f"could plausibly be more {word} than the other.\n\n"
            f"7. COMPARATIVE FEASIBILITY — Could BOTH Caption A AND Caption B "
            f"be depicted in a photograph? The nouns must be similar enough "
            f"that the comparison could go EITHER way depending on the "
            f"specific instances shown.\n"
            f"   GOOD (both directions photographable):\n"
            f"   - 'the dog is bigger than the cat' → YES (some dogs are bigger, "
            f"some cats are bigger than small dogs)\n"
            f"   - 'the truck is dustier than the car' → YES (either could be dustier)\n"
            f"   - 'the boy is taller than the girl' → YES (depends on the individuals)\n"
            f"   BAD (only one direction makes sense — REJECT THESE):\n"
            f"   - 'the elephant is bigger than the mouse' → NO (a mouse can NEVER "
            f"be bigger than an elephant)\n"
            f"   - 'the sun is brighter than the candle' → NO (a candle can NEVER "
            f"be brighter than the sun)\n"
            f"   - 'the ice cream is hotter than the soup' → NO (ice cream is cold)\n"
            f"   - 'the whale is heavier than the ant' → NO (extreme size difference)\n"
            f"   The key test: can you imagine a REAL photo where the comparison "
            f"goes the OTHER way? If not, reject.\n\n"
            f"8. Is the adjective '{word}' a TANGIBLE, PHYSICAL property that "
            f"is clearly visible in a photograph? Answer NO if it is subjective, "
            f"emotional, abstract, or hard to see in a still photo.\n\n"
        )

    if category == "prepositions":
        base += (
            f"7. SPATIAL SENSE — Do BOTH spatial arrangements make physical sense?\n"
            f"   - Can the object actually be placed in that position relative "
            f"to '{word}'?\n"
            f"   - Is the scale correct? (a ball can be ON a table, but not a "
            f"table ON a ball)\n"
            f"   - 'the ball is under the floor' → NO (nothing is visibly under a floor)\n"
            f"   - 'the lamp is behind the wall' → NO (you can't see behind a wall)\n\n"
            f"8. TANGIBLE LANDMARK — Is '{word}' a tangible, household-scale "
            f"object (table, chair, box, bench, shelf)? Answer NO if it is a "
            f"landscape, location, or large-scale feature (river, hill, island, "
            f"carnival, pathway, ocean, mountain, forest, field). You cannot "
            f"place small objects 'on' or 'under' a landscape.\n\n"
            f"9. UNDER/BELOW TEST — If either caption uses 'under' or 'below', "
            f"does '{word}' have a clear physical underside? A table, chair, "
            f"bridge, or bench has an underside. A hill, river, pathway, or "
            f"island does NOT. Answer NO if the landmark has no underside.\n\n"
        )

    if category == "negation":
        base += (
            f"7. NEGATION CLARITY — Is the property '{word}' visually obvious?\n"
            f"   Can you clearly see the DIFFERENCE between having and not having "
            f"the property in a photo?\n"
            f"   - 'the dog is not happy' → NO ('happy' is not clearly visible)\n"
            f"   - 'the shirt is not wet' → YES ('wet' is clearly visible)\n\n"
        )

    if category in ("subject_verb", "counting"):
        base += (
            f"7. ACTION VISIBILITY — Is the action '{word}' clearly visible in "
            f"a still photograph? Can you tell what the subject is doing just by "
            f"looking at a single image?\n"
            f"   - 'thinking' → NO (not visible)\n"
            f"   - 'running' → YES (visible posture)\n\n"
        )

    if category == "subject_verb":
        base += (
            "8. SWAP PLAUSIBILITY — Caption_b swaps the verbs between the "
            "two subjects. Can BOTH subjects plausibly perform BOTH actions? "
            "Answer NO ONLY if a subject-verb combination is truly PHYSICALLY "
            "IMPOSSIBLE given anatomy (e.g., 'the fish is climbing', "
            "'the turtle is flying'). Most animals CAN do most basic "
            "physical actions — be LENIENT.\n\n"
        )

    base += (
        "Respond with ONLY 'YES' or 'NO: [brief reason]'. "
        "Do NOT explain your reasoning. Do NOT write anything before YES or NO. "
        "Be STRICT — reject anything questionable."
    )
    return base


def build_word_suitability_prompt(word: str, pos: str, category: str) -> str:
    """Lightweight LLM check: is a selected word concrete, visual, and appropriate?

    Returns a prompt that asks the LLM for a single YES/NO answer.
    """
    if category == "comparatives":
        return (
            f"You are a filter for a visual benchmark. "
            f"An adjective '{word}' was selected for the 'comparatives' category, "
            f"where two objects are compared (e.g., 'the dog is bigger than the cat').\n\n"
            f"Answer YES if BOTH are true:\n"
            f"1. The property described by '{word}' can be VISUALLY COMPARED between "
            f"two objects in a photograph (one can look more '{word}' than the other)\n"
            f"2. The word is appropriate for all audiences (no profanity, violence, "
            f"or sexual content)\n\n"
            f"Be LENIENT — most physical adjectives work fine for comparatives. "
            f"Words like 'messy', 'clean', 'chubby', 'plain', 'colorful', 'flowery', "
            f"'polished', 'rusty' are all GOOD because you can visually compare them.\n"
            f"Only reject words that are truly INVISIBLE in a photo (e.g., 'possible', "
            f"'lucky', 'necessary', 'interested') or that describe nationality/identity.\n\n"
            f"Answer with EXACTLY one line: YES or NO: <brief reason>\n"
        )
    if category == "subject_adjective":
        return (
            f"You are a filter for a visual benchmark. "
            f"An adjective '{word}' was selected for the 'subject_adjective' category, "
            f"where two objects each have a different adjective "
            f"(e.g., 'a red ball and a tall tree').\n\n"
            f"Answer YES if BOTH are true:\n"
            f"1. The property '{word}' is VISUALLY APPARENT on a concrete object in a "
            f"photograph — you can see it or strongly infer it from the image\n"
            f"2. The word is appropriate for all audiences\n\n"
            f"Be LENIENT — most physical/visual adjectives work. "
            f"Words like 'messy', 'clean', 'wet', 'dry', 'rusty', 'fluffy', 'striped', "
            f"'spotted', 'plain', 'colorful', 'faded', 'shiny' are all GOOD.\n"
            f"Only reject words that are truly INVISIBLE in a photo (e.g., 'possible', "
            f"'lucky', 'interesting', 'important'), describe nationality/identity "
            f"(e.g., 'chinese', 'greek'), or are primarily nouns/verbs.\n\n"
            f"Answer with EXACTLY one line: YES or NO: <brief reason>\n"
        )
    if category == "counting":
        return (
            f"You are a filter for a visual benchmark. "
            f"A verb '{word}' was selected for the 'counting' category, "
            f"where subjects perform an action (e.g., 'three dogs are running').\n\n"
            f"Answer YES if BOTH are true:\n"
            f"1. The action '{word}' is PHYSICALLY VISIBLE in a still photograph — "
            f"you can tell what the subject is doing just by looking\n"
            f"2. The word is appropriate for all audiences\n\n"
            f"Be LENIENT — most physical actions work. Words like 'carving', "
            f"'climbing', 'digging', 'sliding', 'spinning', 'grooming' are all GOOD.\n"
            f"Only reject words that are truly INVISIBLE (e.g., 'thinking', 'wanting', "
            f"'appreciating') or are nouns/adjectives, not verbs.\n\n"
            f"Answer with EXACTLY one line: YES or NO: <brief reason>\n"
        )
    return (
        f"You are a strict filter for a visual benchmark. "
        f"A {pos.lower()} '{word}' was selected for the '{category}' category.\n\n"
        f"Answer YES only if ALL of these are true:\n"
        f"1. The word describes something CONCRETE and PHYSICALLY VISIBLE in a photograph\n"
        f"2. The word is appropriate for all audiences (no nudity, violence, profanity, or sexual content)\n"
        f"3. The word is NOT abstract, metaphorical, or subjective\n"
        f"4. The word does NOT describe an incomplete, relative, or context-dependent "
        f"state — i.e., a word that REQUIRES additional context to form a meaningful "
        f"sentence. Answer NO for words like: halfway (halfway through what?), "
        f"partial (partial what?), former (former what?), pending (pending what?), "
        f"due (due for what?), overdue, ready (ready for what?), ahead (ahead of "
        f"what?), behind (behind what/whom?), remaining, upcoming, ongoing.\n\n"
        f"Answer with EXACTLY one line: YES or NO: <brief reason>\n"
    )
