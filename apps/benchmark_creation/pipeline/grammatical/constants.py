# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linguistic constants and morphology helpers for the TROG pipeline."""

import re

#: Minimum length for CVC consonant-doubling rule.
_MIN_CVC_LEN = 3
#: Words shorter than this are short adjectives (-er suffix only).
_SHORT_ADJ_LEN = 5
#: Words longer than this always take "more X" instead of -er.
_LONG_ADJ_LEN = 7
#: Minimum singularization length to safely strip "ies" -> "y".
_MIN_IES_LEN = 4

#: Adjective suffixes that always take "more X" rather than -er.
_MORE_SUFFIXES = ("ful", "less", "ous", "ive", "ing", "al", "ent", "ant", "ible", "able")

# ===========================================================================
#  Verb morphology data
# ===========================================================================

# Words where CVC-doubling should NOT apply (unstressed final syllable).
_NO_DOUBLE_VERBS = {
    "abandon",
    "answer",
    "benefit",
    "bother",
    "broaden",
    "brighten",
    "budget",
    "button",
    "cancel",
    "color",
    "comfort",
    "consider",
    "cover",
    "dampen",
    "deliver",
    "deposit",
    "develop",
    "differ",
    "discover",
    "edit",
    "encounter",
    "enter",
    "envelop",
    "exhibit",
    "fasten",
    "flatten",
    "flicker",
    "focus",
    "follow",
    "foster",
    "gather",
    "glisten",
    "gallop",
    "gossip",
    "happen",
    "harden",
    "hasten",
    "herald",
    "honor",
    "imagine",
    "imprison",
    "inhabit",
    "inherit",
    "interpret",
    "iron",
    "labor",
    "lessen",
    "limit",
    "listen",
    "loosen",
    "lower",
    "market",
    "master",
    "matter",
    "mention",
    "merit",
    "model",
    "monitor",
    "murder",
    "murmur",
    "offer",
    "open",
    "order",
    "orbit",
    "pardon",
    "pedal",
    "pilot",
    "plaster",
    "poison",
    "polish",
    "ponder",
    "powder",
    "power",
    "prison",
    "profit",
    "prohibit",
    "prosper",
    "publish",
    "punish",
    "quiet",
    "quicken",
    "reason",
    "reckon",
    "recover",
    "remember",
    "render",
    "ripen",
    "rivet",
    "roar",
    "ruin",
    "rumor",
    "season",
    "shelter",
    "shorten",
    "shower",
    "shiver",
    "signal",
    "slacken",
    "slaughter",
    "soften",
    "solder",
    "suffer",
    "summon",
    "surrender",
    "swallow",
    "sweeten",
    "thunder",
    "tower",
    "travel",
    "treasure",
    "trigger",
    "trumpet",
    "uncover",
    "visit",
    "volunteer",
    "wander",
    "water",
    "weaken",
    "weather",
    "whisper",
    "widen",
    "wonder",
    "worship",
}

# Irregular past participles for common verbs.
_IRREGULAR_PAST_PARTICIPLES = {
    "arise": "arisen",
    "awake": "awoken",
    "be": "been",
    "bear": "borne",
    "beat": "beaten",
    "become": "become",
    "begin": "begun",
    "bend": "bent",
    "bet": "bet",
    "bid": "bid",
    "bind": "bound",
    "bite": "bitten",
    "bleed": "bled",
    "blow": "blown",
    "break": "broken",
    "breed": "bred",
    "bring": "brought",
    "build": "built",
    "burn": "burnt",
    "burst": "burst",
    "buy": "bought",
    "catch": "caught",
    "choose": "chosen",
    "cling": "clung",
    "come": "come",
    "cost": "cost",
    "creep": "crept",
    "cut": "cut",
    "deal": "dealt",
    "dig": "dug",
    "do": "done",
    "draw": "drawn",
    "drink": "drunk",
    "drive": "driven",
    "eat": "eaten",
    "fall": "fallen",
    "feed": "fed",
    "feel": "felt",
    "fight": "fought",
    "find": "found",
    "flee": "fled",
    "fling": "flung",
    "fly": "flown",
    "forbid": "forbidden",
    "forget": "forgotten",
    "forgive": "forgiven",
    "freeze": "frozen",
    "get": "got",
    "give": "given",
    "go": "gone",
    "grind": "ground",
    "grow": "grown",
    "hang": "hung",
    "have": "had",
    "hear": "heard",
    "hide": "hidden",
    "hit": "hit",
    "hold": "held",
    "hurt": "hurt",
    "keep": "kept",
    "kneel": "knelt",
    "knit": "knit",
    "know": "known",
    "lay": "laid",
    "lead": "led",
    "leave": "left",
    "lend": "lent",
    "let": "let",
    "lie": "lain",
    "light": "lit",
    "lose": "lost",
    "make": "made",
    "mean": "meant",
    "meet": "met",
    "mow": "mown",
    "overcome": "overcome",
    "pay": "paid",
    "prove": "proven",
    "put": "put",
    "quit": "quit",
    "read": "read",
    "ride": "ridden",
    "ring": "rung",
    "rise": "risen",
    "run": "run",
    "say": "said",
    "see": "seen",
    "seek": "sought",
    "sell": "sold",
    "send": "sent",
    "set": "set",
    "sew": "sewn",
    "shake": "shaken",
    "shed": "shed",
    "shine": "shone",
    "shoot": "shot",
    "show": "shown",
    "shrink": "shrunk",
    "shut": "shut",
    "sing": "sung",
    "sink": "sunk",
    "sit": "sat",
    "slay": "slain",
    "sleep": "slept",
    "slide": "slid",
    "sling": "slung",
    "slit": "slit",
    "smell": "smelt",
    "sow": "sown",
    "speak": "spoken",
    "speed": "sped",
    "spend": "spent",
    "spill": "spilt",
    "spin": "spun",
    "spit": "spat",
    "split": "split",
    "spread": "spread",
    "spring": "sprung",
    "stand": "stood",
    "steal": "stolen",
    "stick": "stuck",
    "sting": "stung",
    "stink": "stunk",
    "stride": "stridden",
    "strike": "struck",
    "string": "strung",
    "strive": "striven",
    "swear": "sworn",
    "sweep": "swept",
    "swim": "swum",
    "swing": "swung",
    "take": "taken",
    "teach": "taught",
    "tear": "torn",
    "tell": "told",
    "think": "thought",
    "throw": "thrown",
    "tread": "trodden",
    "understand": "understood",
    "wake": "woken",
    "wear": "worn",
    "weave": "woven",
    "weep": "wept",
    "win": "won",
    "wind": "wound",
    "withdraw": "withdrawn",
    "wring": "wrung",
    "write": "written",
    # Common transitive verbs used in TROG
    "carry": "carried",
    "chase": "chased",
    "drag": "dragged",
    "grab": "grabbed",
    "hug": "hugged",
    "kick": "kicked",
    "kiss": "kissed",
    "lick": "licked",
    "lift": "lifted",
    "pull": "pulled",
    "push": "pushed",
    "wash": "washed",
    "watch": "watched",
}


# ===========================================================================
#  Noun morphology data
# ===========================================================================

_PLURAL_IRREGULARS = {
    "man": "men",
    "woman": "women",
    "child": "children",
    "baby": "babies",
    "mouse": "mice",
    "goose": "geese",
    "fish": "fish",
    "sheep": "sheep",
    "deer": "deer",
    "foot": "feet",
    "tooth": "teeth",
    "person": "people",
}

# Words ending in -f/-fe that form plurals with -ves (not just -s).
_F_TO_VES = {
    "calf",
    "elf",
    "half",
    "knife",
    "leaf",
    "life",
    "loaf",
    "self",
    "shelf",
    "thief",
    "wife",
    "wolf",
}


# ===========================================================================
#  Morphology helper functions
# ===========================================================================

# Irregular gerund forms that the heuristic rule gets wrong (e.g. "qu" words
# where 'u' is part of the consonant cluster, not a true vowel).
_IRREGULAR_GERUNDS = {
    "squat": "squatting",
    "quit": "quitting",
    "quiz": "quizzing",
    "equip": "equipping",
}


def get_gerund(word: str) -> str:
    """Best-effort -ing form of a verb."""
    if word in _IRREGULAR_GERUNDS:
        return _IRREGULAR_GERUNDS[word]
    if word.endswith("ing"):
        return word
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee") and not word.endswith("oe"):
        return word[:-1] + "ing"
    if (
        len(word) >= _MIN_CVC_LEN
        and word[-1] not in "aeiouywx"
        and word[-2] in "aeiou"
        and word[-3] not in "aeiou"
        and word not in _NO_DOUBLE_VERBS
    ):
        return word + word[-1] + "ing"
    return word + "ing"


# Irregular comparative forms where the heuristic fails.
_IRREGULAR_COMPARATIVES = {
    "gooey": "gooier",
    "good": "better",
    "bad": "worse",
    "far": "farther",
    "little": "littler",
}


def _short_comparative(word: str) -> str:
    """Comparative for short words (length <= _SHORT_ADJ_LEN)."""
    if word.endswith("e"):
        return word + "r"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ier"
    if len(word) >= _MIN_CVC_LEN and word[-1] not in "aeiouywx" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        return word + word[-1] + "er"
    return word + "er"


def get_comparative(word: str) -> str:
    """Best-effort comparative form of an adjective.

    Short monosyllabic adjectives get -er; multi-syllable adjectives use
    'more X'.  A length-based heuristic with suffix checks handles most
    cases.
    """
    if word in _IRREGULAR_COMPARATIVES:
        return _IRREGULAR_COMPARATIVES[word]
    if len(word) <= _SHORT_ADJ_LEN:
        return _short_comparative(word)

    # Longer words: suffixes that always require "more"
    if len(word) >= _LONG_ADJ_LEN or word.endswith(_MORE_SUFFIXES):
        return f"more {word}"
    if word.endswith("e"):
        return word + "r"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ier"
    # Don't double consonants for multi-syllable words (6 chars)
    return word + "er"


def get_comparative_forms(word: str) -> list[str]:
    """Return ALL plausible comparative forms for vocabulary expansion.

    Many adjectives accept both '-er' and 'more X' forms.  The LLM may
    generate either, so we expand both to avoid OOV rejections.
    """
    forms = set()
    # Primary form from heuristic
    forms.add(get_comparative(word))
    # Always add "more X"
    forms.add(f"more {word}")
    # Always try the -er / -ier suffixed form
    if word.endswith("e"):
        forms.add(word + "r")
    elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        forms.add(word[:-1] + "ier")
    else:
        forms.add(word + "er")
        # Also try with consonant doubling
        if len(word) >= _MIN_CVC_LEN and word[-1] not in "aeiouywx" and word[-2] in "aeiou":
            forms.add(word + word[-1] + "er")
    return list(forms)


def get_past_participle(word: str) -> str:
    """Best-effort past participle of a verb."""
    if word in _IRREGULAR_PAST_PARTICIPLES:
        return _IRREGULAR_PAST_PARTICIPLES[word]
    # Regular rules
    if word.endswith("e"):
        return word + "d"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ied"
    if (
        len(word) >= _MIN_CVC_LEN
        and word[-1] not in "aeiouywx"
        and word[-2] in "aeiou"
        and word[-3] not in "aeiou"
        and word not in _NO_DOUBLE_VERBS
    ):
        return word + word[-1] + "ed"
    return word + "ed"


def pluralize(word: str) -> str:  # noqa: PLR0911 -- early returns per pluralization rule are clearer than nesting.
    """Naive English pluralization."""
    if word in _PLURAL_IRREGULARS:
        return _PLURAL_IRREGULARS[word]
    if len(word) <= 1:
        return word + "s"
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word in _F_TO_VES:
        if word.endswith("fe"):
            return word[:-2] + "ves"
        return word[:-1] + "ves"
    return word + "s"


# Reverse mapping: plural -> singular for irregular nouns.
_SINGULAR_IRREGULARS = {v: k for k, v in _PLURAL_IRREGULARS.items()}
# Reverse mapping: plural -ves -> singular -f/-fe.
_VES_TO_F = {pluralize(w): w for w in _F_TO_VES}


def singularize(word: str) -> str:  # noqa: PLR0911 -- early returns per singularization rule are clearer than nesting.
    """Best-effort English singularization (inverse of pluralize).

    Handles irregular plurals, -ves -> -f/-fe, -ies -> -y, -es, and -s.
    """
    if word in _SINGULAR_IRREGULARS:
        return _SINGULAR_IRREGULARS[word]
    if word in _VES_TO_F:
        return _VES_TO_F[word]
    if word.endswith("ies") and len(word) > _MIN_IES_LEN:
        # "puppies" -> "puppy", but not "series" -> "sery"
        return word[:-3] + "y"
    if word.endswith("ves"):
        # Generic -ves -> -f fallback (e.g., "scarves" -> "scarf")
        return word[:-3] + "f"
    if word.endswith(("shes", "ches", "xes", "zes")):
        # "churches" -> "church", "boxes" -> "box", "foxes" -> "fox"
        return word[:-2]
    if word.endswith("sses"):
        # "grasses" -> "grass"
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def get_third_person_singular(word: str) -> str:
    """Best-effort third-person singular present form of a verb (he/she chases)."""
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


# ===========================================================================
#  order_matters constants and helpers
# ===========================================================================

# Pattern: "the <subject> is <verb>ing the <object>" (with optional trailing punctuation)
ORDER_MATTERS_RE = re.compile(
    r"^the\s+(.+?)\s+is\s+(\S+ing)\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)

# Allowlist of animate nouns — subject and object must each be one of these.
ANIMATE_NOUNS: set[str] = {
    # humans
    "boy",
    "girl",
    "man",
    "woman",
    "baby",
    "child",
    # domestic animals
    "dog",
    "cat",
    "horse",
    "cow",
    "pig",
    "sheep",
    "goat",
    "duck",
    "chicken",
    "rabbit",
    "mouse",
    "kitten",
    "puppy",
    "lamb",
    "donkey",
    "pony",
    "hen",
    "rooster",
    "hamster",
    "parrot",
    # wild animals
    "bird",
    "fish",
    "frog",
    "bear",
    "lion",
    "tiger",
    "elephant",
    "monkey",
    "fox",
    "wolf",
    "deer",
    "owl",
    "eagle",
    "penguin",
    "turtle",
    "snake",
    "dolphin",
    "whale",
    "squirrel",
    "zebra",
    "giraffe",
    "bee",
    "butterfly",
    "ant",
    "spider",
    "crab",
    "seal",
    "otter",
    "beaver",
    "piglet",
    "crow",
    "swan",
    "goose",
    "raccoon",
    "gorilla",
    "kangaroo",
    "lizard",
    "goldfish",
    # extra
    "chick",
    "duckling",
    "calf",
    "panda",
    "koala",
    "flamingo",
    "peacock",
    "cheetah",
    "camel",
}


# Verbs are sampled directly from this list (no vocabulary matching needed).
# Only simple, easy-to-depict verbs that clearly show who does what.
ORDER_MATTERS_SAFE_VERBS: list[str] = [
    "chase",
    "push",
    "pull",
    "lick",
    "kick",
    "bite",
    "scratch",
    "grab",
    "follow",
    "pursue",
    "hug",
    "poke",
    "carry",
    "hold",
    "tap",
]

# Intransitive verbs for subject_verb — actions any person or animal can do.
# Every verb must be (1) clearly visible in a photo, (2) plausible for both
# humans and common animals, and (3) unambiguous without an object.
SUBJECT_VERB_SAFE_VERBS: list[str] = [
    "run",
    "sleep",
    "sit",
    "stand",
    "jump",
    "swim",
    "climb",
    "crawl",
    "eat",
    "drink",
    "dance",
    "cry",
    "laugh",
    "hide",
    "walk",
    "dig",
    "sing",
    "stretch",
    "yawn",
    "roll",
    "slide",
    "fall",
    "spin",
    "wave",
    "crouch",
    "splash",
    "hang",
    "lean",
    "kneel",
    "bow",
    "skip",
    "hop",
    "rest",
    "play",
    "bend",
    "balance",
    "squat",
    "tiptoe",
    "march",
    "gallop",
    "shake",
    "sniff",
    "scratch",
    "lick",
    "shiver",
    "peek",
    "twirl",
    "sneeze",
    "wiggle",
    "stomp",
    "bounce",
]


# Intransitive verbs for counting — actions multiple agents can visibly do
# at the same time.  Every verb must be (1) intransitive (no object needed),
# (2) clearly visible in a still photo, (3) plausible for BOTH humans AND
# common animals (since the prompt pairs verbs with random animate nouns),
# and (4) easy to count individual agents performing the action.
#
# EXCLUDED (and why):
#   fly      — only birds/insects; rejected for dogs, boys, turtles, etc.
#   gallop   — only equines (horses, ponies, donkeys)
#   kneel    — human-only posture
#   sing     — invisible in a still photo
#   cry/laugh — subtle facial expression, hard to photograph for animals
#   shiver   — subtle body movement, barely visible in a photo
#   tiptoe   — human-only (requires feet anatomy)
#   sneeze   — instantaneous, hard to capture
COUNTING_SAFE_VERBS: list[str] = [
    "run",
    "sleep",
    "sit",
    "stand",
    "jump",
    "swim",
    "climb",
    "crawl",
    "eat",
    "drink",
    "dance",
    "hide",
    "walk",
    "dig",
    "stretch",
    "yawn",
    "roll",
    "slide",
    "fall",
    "spin",
    "wave",
    "crouch",
    "splash",
    "lean",
    "skip",
    "hop",
    "rest",
    "hang",
    "play",
    "shake",
    "bow",
    "bend",
    "balance",
    "squat",
    "wiggle",
    "stomp",
    "bounce",
    "peek",
]

# Transitive verbs for embedded_relative — actions one agent does TO another
# or to an object.  Every verb must be (1) clearly transitive, (2) visible in
# a still photo, (3) plausible between animate nouns (the subject/object are
# drawn from _ANIMATE_NOUNS, so both humans and animals must work), and
# (4) work as 3rd-person singular (e.g., "chases", "holds").
#
# EXCLUDED (and why):
#   paint  — human-only; "the cat paints the dog" is nonsensical
#   tap    — too subtle to distinguish from "touch" in a photo
#   nudge  — too subtle, hard to photograph
#   sniff  — invisible action in a still photo
EMBEDDED_RELATIVE_SAFE_VERBS: list[str] = [
    # core actions (from ORDER_MATTERS_SAFE_VERBS — already validated)
    "chase",
    "push",
    "pull",
    "lick",
    "kick",
    "bite",
    "scratch",
    "grab",
    "follow",
    "hug",
    "carry",
    "hold",
    "poke",
    "pursue",
    # clear transitive actions visible in a photo
    "watch",
    "wash",
    "feed",
    "lift",
    "catch",
    "drag",
    "touch",
    "splash",
    "bump",
    "guard",
    "block",
    # additional transitive verbs that work between animate nouns
    "shake",
    "spin",
    "lead",
    "cover",
    "wrap",
    "pet",
    "groom",
    "ride",
    "climb",
]


def invert_order_matters(sentence: str) -> str | None:
    """Swap subject and object in 'the X is verbing the Y'.

    Returns the inverted sentence, or None if the pattern doesn't match.
    """
    m = ORDER_MATTERS_RE.match(sentence.strip())
    if not m:
        return None
    subj, verb, obj = m.group(1), m.group(2), m.group(3)
    return f"the {obj} is {verb} the {subj}"


def extract_animate_head(noun_phrase: str) -> str | None:
    """Return the head noun of a noun phrase if it is in ANIMATE_NOUNS.

    Handles optional determiners/adjectives before the head noun,
    e.g. "little boy" -> "boy", "old gray cat" -> "cat".
    We check each word right-to-left and return the first animate match,
    falling back to the last word.
    """
    words = noun_phrase.lower().split()
    for w in reversed(words):
        if w in ANIMATE_NOUNS:
            return w
    return None


def check_order_matters_sanity(sentence: str) -> str | None:
    """Hard-coded sanity checks for order_matters sentences.

    Returns None if OK, or a rejection reason string.
    """
    m = ORDER_MATTERS_RE.match(sentence.strip())
    if not m:
        return "does not match 'the X is verbing the Y' pattern"
    subj, _verb, obj = m.group(1), m.group(2), m.group(3)

    subj_head = extract_animate_head(subj)
    obj_head = extract_animate_head(obj)
    if subj_head is None:
        return f"subject '{subj}' is not a recognized human or animal"
    if obj_head is None:
        return f"object '{obj}' is not a recognized human or animal"
    return None


# ===========================================================================
#  subject_adjective constants and helpers
# ===========================================================================

# Pattern: "a <A1> <S1> and a <A2> <S2>"
_SUBJ_ADJ_RE = re.compile(
    r"^a\s+(.+?)\s+(\S+)\s+and\s+a\s+(.+?)\s+(\S+)\.?$",
    re.IGNORECASE,
)


def parse_subject_adjective(caption: str) -> tuple[str, str, str, str] | None:
    """Parse 'a <adj1> <noun1> and a <adj2> <noun2>' from a caption.

    Returns ``(adj1, noun1, adj2, noun2)`` or ``None`` if parsing fails.
    """
    m = _SUBJ_ADJ_RE.match(caption.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def check_subject_adjective_sanity(
    caption_a: str,
    caption_b: str,
    word: str,
) -> str | None:
    """Verify that a subject_adjective pair correctly swaps adjectives.

    Expected pattern:
      caption_a: ``a <A1> <S1> and a <A2> <S2>``
      caption_b: ``a <A2> <S1> and a <A1> <S2>``

    The nouns must stay the same and the adjectives must swap across them.

    Returns ``None`` if the pair is valid, or a rejection reason string.
    """
    parts_a = parse_subject_adjective(caption_a)
    if parts_a is None:
        return f"caption_a does not match 'a A1 S1 and a A2 S2': {caption_a[:60]}"

    parts_b = parse_subject_adjective(caption_b)
    if parts_b is None:
        return f"caption_b does not match 'a A1 S1 and a A2 S2': {caption_b[:60]}"

    a1, s1, a2, s2 = parts_a
    b1, bs1, b2, bs2 = parts_b

    # Nouns must be the same in both captions (same order)
    if (s1, s2) != (bs1, bs2):
        return (
            f"nouns differ between captions: ({s1}, {s2}) vs ({bs1}, {bs2}). "
            f"caption_b must keep the same nouns in the same positions"
        )

    # Adjectives must be swapped: A1↔A2
    if not (b1 == a2 and b2 == a1):
        return (
            f"adjectives not swapped: caption_a has ({a1}, {a2}), caption_b has ({b1}, {b2}) — expected ({a2}, {a1})"
        )

    # The target word must appear as one of the adjectives
    if word not in (a1, a2):
        return f"target word '{word}' not used as an adjective in the captions (found: {a1}, {a2})"

    return None


# ===========================================================================
#  embedded_relative constants and helpers
# ===========================================================================

# Pattern: "the [subject] [verb_s (+ optional prep)] the [object] that is/are [adj]"
# Supports phrasal verbs like "sits on", "laughs at", "looks at".
_EMBED_REL_OBJECT_RE = re.compile(
    r"^the\s+(.+?)\s+(\S+s(?:\s+\S+)?)\s+the\s+(.+?)\s+that\s+(?:is|are)\s+(.+?)\.?$",
    re.IGNORECASE,
)

# Pattern: "the [subject] that is/are [adj] [verb_s (+ optional prep)] the [object]"
_EMBED_REL_SUBJECT_RE = re.compile(
    r"^the\s+(.+?)\s+that\s+(?:is|are)\s+(.+?)\s+(\S+s(?:\s+\S+)?)\s+the\s+(.+?)\.?$",
    re.IGNORECASE,
)


def check_embedded_relative_sanity(  # noqa: PLR0911 -- one return per validation rule is clearest.
    caption_a: str,
    caption_b: str,
) -> str | None:
    """Verify that an embedded_relative pair uses the same words.

    Expected patterns:
      caption_a: ``the [subject] [verb-s] the [object] that is [adj]``
      caption_b: ``the [subject] that is [adj] [verb-s] the [object]``

    Both must use the same subject, verb, object, AND adjective.
    Returns ``None`` if valid, or a rejection reason string.
    """
    m_a = _EMBED_REL_OBJECT_RE.match(caption_a.strip())
    if m_a is None:
        return f"caption_a does not match embedded_relative pattern: {caption_a[:60]}"
    subj_a, verb_a, obj_a, adj_a = m_a.group(1), m_a.group(2), m_a.group(3), m_a.group(4)

    m_b = _EMBED_REL_SUBJECT_RE.match(caption_b.strip())
    if m_b is None:
        return f"caption_b does not match embedded_relative pattern: {caption_b[:60]}"
    subj_b, adj_b, verb_b, obj_b = m_b.group(1), m_b.group(2), m_b.group(3), m_b.group(4)

    if subj_a != subj_b:
        return f"subjects differ: '{subj_a}' vs '{subj_b}'"
    if verb_a != verb_b:
        return f"verbs differ: '{verb_a}' vs '{verb_b}'"
    if obj_a != obj_b:
        return f"objects differ: '{obj_a}' vs '{obj_b}'"
    if adj_a != adj_b:
        return f"adjectives differ: '{adj_a}' vs '{adj_b}' — must be the same"

    return None
