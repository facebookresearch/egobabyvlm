# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Constants for the native Flamingo baseline.

Mirrors ``apps/baselines/llava/constants.py`` so the two generative VLMs share
the same masking / media-token conventions.
"""

#: Label value ignored by the cross-entropy loss (prompt / padding positions).
IGNORE_INDEX = -100

#: The media placeholder token. ``media_locations = input_ids == media_token_id``
#: drives the gated cross-attention masking (Flamingo's ``<image>`` wiring).
#: Bracketed token string, not a credential — silence ruff's secret heuristic.
DEFAULT_MEDIA_TOKEN = "<image>"  # noqa: S105

#: Default caption prompt. ``{text}`` MUST come last so the label mask
#: ("count target tokens from the end") isolates the caption tokens.
DEFAULT_PROMPT_TEMPLATE = "<image> {text}"
