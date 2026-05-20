# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Response parsing functions for the grammatical pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from apps.benchmark_creation.utils.vocabulary import VocabEntry

#: Minimum length for a wrapped/quoted token to retain meaningful content.
_MIN_WRAPPED_LEN = 2
#: Minimum length for a validated response.
_MIN_RESPONSE_LEN = 3
#: Maximum length for a validated response.
_MAX_RESPONSE_LEN = 100
#: Minimum captions required to consider a paired-caption response valid.
_MIN_CAPTIONS = 2


def clean_text(text: str) -> str:
    """Strip whitespace, remove wrapping quotes/brackets, deduplicate lines, lowercase."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else text.strip()
    if len(text) >= _MIN_WRAPPED_LEN and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    if len(text) >= _MIN_WRAPPED_LEN and text[0] == "[" and text[-1] == "]":
        text = text[1:-1].strip()
    return text.lower()


def validate_response_text(text: str) -> str | None:
    """Validate a response string. Returns cleaned text or None if invalid."""
    if not text or len(text) < _MIN_RESPONSE_LEN or len(text) > _MAX_RESPONSE_LEN:
        return None
    error_markers = ["sorry", "i can't", "i cannot", "as an ai", "error"]
    if any(marker in text for marker in error_markers):
        return None
    return text


def parse_word_selection_response(
    response: str,
    pool: Iterable[VocabEntry],
    selected: set[str],
) -> VocabEntry | None:
    """Parse word selection response. Returns VocabEntry or None."""
    word = response.strip().strip("'\"").lower()
    for e in pool:
        if e.word == word and word not in selected:
            return e
    return None


def _strip_verdict_prefixes(line: str, prefixes: tuple[str, ...]) -> str:
    """Strip any of *prefixes* from the start of *line* (case-insensitive)."""
    for prefix in prefixes:
        if line.upper().startswith(prefix.upper()):
            return line[len(prefix) :].strip()
    return line


def _extract_no_reason(line: str) -> str:
    """Extract the rationale text following a NO verdict on a single line."""
    reason_line = _strip_verdict_prefixes(
        line.strip(),
        (
            "ANSWER:",
            "Answer:",
            "VERDICT:",
            "Verdict:",
            "FINAL ANSWER:",
            "Final answer:",
            "Final Answer:",
            "CONCLUSION:",
            "Conclusion:",
            "**",
        ),
    ).strip("* ")
    if ":" in reason_line:
        return reason_line.split(":", 1)[1].strip()
    return reason_line[2:].strip() if len(reason_line) > _MIN_WRAPPED_LEN else ""


def parse_validation_response(response: str) -> tuple[bool, str]:
    """Parse YES/NO validation response.

    Checks the start of the response first, then searches the full text
    for a YES/NO verdict in case the model produced reasoning before the
    answer.
    """
    text = response.strip()
    upper = text.upper()

    # Fast path: response starts with YES/NO
    if upper.startswith("YES"):
        return True, ""
    if upper.startswith("NO"):
        if ":" in text:
            reason = text.split(":", 1)[1].strip()
        elif len(text) > _MIN_WRAPPED_LEN:
            reason = text[2:].strip()
        else:
            reason = ""
        return False, reason

    # Slow path: search for YES/NO anywhere (model may have reasoned first)
    for line in reversed(text.splitlines()):
        stripped = _strip_verdict_prefixes(
            line.strip().upper(),
            ("ANSWER:", "VERDICT:", "FINAL ANSWER:", "CONCLUSION:", "**"),
        ).strip("* ")
        if stripped.startswith("YES"):
            return True, ""
        if stripped.startswith("NO"):
            return False, _extract_no_reason(line)

    return False, f"unparsable response: {text[:80]}"


def parse_pair_response(response: str) -> tuple[str | None, str | None]:
    """Parse a two-line caption_a/caption_b response from the LLM.

    Accepts formats like:
      caption_a: the dog is running and the cat is sleeping
      caption_b: the dog is sleeping and the cat is running
    Or just two plain lines.
    """
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    captions: list[str] = []
    for line in lines:
        # Strip "caption_a:" or "caption_b:" prefix (case-insensitive)
        cleaned = re.sub(r"^caption_[ab]\s*:\s*", "", line, flags=re.IGNORECASE)
        # Strip leading number + punctuation
        cleaned = re.sub(r"^\d+[.):\-]\s*", "", cleaned)
        # Strip wrapping quotes
        if len(cleaned) >= _MIN_WRAPPED_LEN and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'"):
            cleaned = cleaned[1:-1].strip()
        if cleaned and len(cleaned) >= _MIN_RESPONSE_LEN:
            captions.append(cleaned.lower())

    if len(captions) >= _MIN_CAPTIONS:
        return captions[0], captions[1]
    return None, None


def parse_embedded_relative_response(
    response: str,
) -> tuple[str | None, str | None, str | None]:
    """Parse embedded_relative response: caption_a, caption_b, and antonym.

    Expected format::

        caption_a: the turtle holds the leaf that is green
        caption_b: the turtle that is green holds the leaf
        antonym: brown
    """
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

    antonym: str | None = None

    for line in lines:
        low = line.lower().strip()
        if low.startswith("antonym"):
            # Extract antonym value after "antonym:" or "antonym ="
            val = re.sub(r"^antonym\s*[:=]\s*", "", line, flags=re.IGNORECASE).strip()
            if val and len(val) >= _MIN_WRAPPED_LEN:
                # Strip wrapping quotes
                if len(val) >= _MIN_WRAPPED_LEN and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1].strip()
                antonym = val.lower()

    # Parse captions using existing logic
    cap_a, cap_b = parse_pair_response(response)
    return cap_a, cap_b, antonym
