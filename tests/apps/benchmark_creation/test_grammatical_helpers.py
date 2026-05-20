# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for grammatical pipeline parsers and morphology helpers."""

from __future__ import annotations

import pytest

from apps.benchmark_creation.pipeline.grammatical.constants import (
    get_comparative,
    get_gerund,
    get_third_person_singular,
    invert_order_matters,
    pluralize,
    singularize,
)
from apps.benchmark_creation.pipeline.grammatical.parsers import (
    clean_text,
    parse_pair_response,
    parse_validation_response,
    validate_response_text,
)

# ---- parsers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"hello"', "hello"),
        ("'world'", "world"),
        ("[bracketed]", "bracketed"),
        ("Hello World", "hello world"),
        ("  spaced  ", "spaced"),
        ("first line\nsecond line", "first line"),
        ("", ""),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_validate_response_text_rejects_too_short() -> None:
    assert validate_response_text("ok") is None


def test_validate_response_text_rejects_too_long() -> None:
    assert validate_response_text("x" * 200) is None


def test_validate_response_text_rejects_refusal_markers() -> None:
    # Markers are matched case-sensitively against the lowercase form.
    assert validate_response_text("sorry, i can't help with that") is None


def test_validate_response_text_accepts_normal() -> None:
    assert validate_response_text("a brown dog") == "a brown dog"


def test_parse_validation_response_yes_at_start() -> None:
    assert parse_validation_response("YES") == (True, "")
    assert parse_validation_response("yes, this is fine") == (True, "")


def test_parse_validation_response_no_with_reason() -> None:
    ok, reason = parse_validation_response("NO: subject does not match")
    assert ok is False
    assert "subject does not match" in reason


def test_parse_validation_response_no_in_final_line() -> None:
    text = "Let me think...\nThe sentence has issues.\nFINAL ANSWER: NO: bad grammar"
    ok, reason = parse_validation_response(text)
    assert ok is False
    # The slow-path scan upper-cases the line before extracting the reason.
    assert reason.lower() == "bad grammar"


def test_parse_validation_response_unparsable_returns_false() -> None:
    ok, reason = parse_validation_response("definitely maybe perhaps")
    assert ok is False
    assert "unparsable" in reason


def test_parse_pair_response_keyed_format() -> None:
    response = "caption_a: the dog runs\ncaption_b: the cat runs"
    a, b = parse_pair_response(response)
    assert a == "the dog runs"
    assert b == "the cat runs"


def test_parse_pair_response_two_plain_lines() -> None:
    a, b = parse_pair_response("the dog runs\nthe cat runs")
    assert a == "the dog runs"
    assert b == "the cat runs"


def test_parse_pair_response_single_line_returns_none() -> None:
    a, b = parse_pair_response("only one line")
    assert (a, b) == (None, None)


# ---- morphology -------------------------------------------------------


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("dog", "dogs"),
        ("cat", "cats"),
        ("bus", "buses"),  # ends in -s
        ("box", "boxes"),  # ends in -x
        ("baby", "babies"),  # consonant + y
        ("toy", "toys"),  # vowel + y
        ("knife", "knives"),  # -fe → -ves
        ("leaf", "leaves"),  # -f → -ves
    ],
)
def test_pluralize(singular: str, plural: str) -> None:
    assert pluralize(singular) == plural


@pytest.mark.parametrize(
    ("plural", "singular"),
    [
        ("dogs", "dog"),
        ("babies", "baby"),
        ("knives", "knife"),
        ("leaves", "leaf"),
    ],
)
def test_singularize(plural: str, singular: str) -> None:
    assert singularize(plural) == singular


def test_pluralize_singularize_roundtrip() -> None:
    """Plural-then-singular is the identity for the regular cases this pipeline cares about."""
    for word in ("dog", "cat", "baby", "toy"):
        assert singularize(pluralize(word)) == word


@pytest.mark.parametrize(
    ("base", "gerund"),
    [
        ("run", "running"),  # short vowel + consonant → double
        ("jump", "jumping"),
        ("write", "writing"),  # silent -e dropped
        ("swim", "swimming"),
        ("play", "playing"),
        ("die", "dying"),  # -ie → -ying
    ],
)
def test_get_gerund(base: str, gerund: str) -> None:
    assert get_gerund(base) == gerund


@pytest.mark.parametrize(
    ("base", "comparative"),
    [
        ("big", "bigger"),  # short vowel + consonant → double
        ("small", "smaller"),
        ("happy", "happier"),  # consonant + y → -ier
        ("nice", "nicer"),  # -e → -er
        ("good", "better"),  # irregular
        ("bad", "worse"),  # irregular
    ],
)
def test_get_comparative(base: str, comparative: str) -> None:
    assert get_comparative(base) == comparative


@pytest.mark.parametrize(
    ("base", "third_person"),
    [
        ("run", "runs"),
        ("watch", "watches"),  # ends in -ch
        ("fly", "flies"),  # consonant + y
        ("play", "plays"),  # vowel + y
    ],
)
def test_get_third_person_singular(base: str, third_person: str) -> None:
    assert get_third_person_singular(base) == third_person


def test_invert_order_matters_returns_none_for_unmatched() -> None:
    """Sentences that don't fit the expected pattern return None."""
    assert invert_order_matters("running fast") is None
