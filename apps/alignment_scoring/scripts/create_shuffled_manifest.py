# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shuffle captions in a manifest, producing a (caption_a_for_image_X,
image_X) → (caption_a_for_image_Y, image_X) misalignment.

Used to build the "shuffled" side of the alignment-scoring pipelines (matched
vs shuffled JSD aggregation). The shuffle is deterministic via ``--random-seed``.

Supports four manifest formats:

- ``json``: standard COCO captions (top-level ``images`` + ``annotations``).
  Image-level captions are gathered, the per-image caption *lists* are
  shuffled across images, then re-flattened into a new annotations list.
- ``karpathy_json``: Karpathy split (``images[*].sentences``). The per-image
  ``sentences`` lists are shuffled across images.
- ``karpathy_json_with_permutation``: applies an externally-provided
  permutation map JSON instead of random shuffling.
- ``csv``: CSV with ``clip_filename`` + ``utterance`` columns. The
  ``utterance`` column is shuffled in place.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def shuffle_json_manifest(manifest_path: str, output_path: str) -> None:
    with Path(manifest_path).open() as f:
        data = json.load(f)
    logger.info(
        "Loaded %d images and %d annotations",
        len(data["images"]),
        len(data["annotations"]),
    )
    image_captions: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        image_captions[ann["image_id"]].append(ann["caption"])

    image_ids = list(image_captions.keys())
    captions_list = [captions[:] for captions in image_captions.values()]
    random.shuffle(captions_list)
    shuffled_image_captions = dict(zip(image_ids, captions_list, strict=False))

    data["annotations"] = [
        {"image_id": image_id, "caption": caption}
        for image_id, captions in shuffled_image_captions.items()
        for caption in captions
    ]
    with Path(output_path).open("w") as f:
        json.dump(data, f)


def shuffle_karpathy_json(manifest_path: str, output_path: str) -> None:
    with Path(manifest_path).open() as f:
        data = json.load(f)
    logger.info("Loaded %d images with captions", len(data["images"]))
    image_captions = [image["sentences"][:] for image in data["images"]]
    random.shuffle(image_captions)
    for image, captions in zip(data["images"], image_captions, strict=True):
        image["sentences"] = captions
    with Path(output_path).open("w") as f:
        json.dump(data, f)


def apply_permutation_karpathy_json(manifest_path: str, permutation_metadata_path: str, output_path: str) -> None:
    with Path(manifest_path).open() as f:
        data = json.load(f)
    with Path(permutation_metadata_path).open() as f:
        metadata = json.load(f)
    permutation_map = {int(k): v for k, v in metadata["permutation_map"].items()}

    counter: Counter[str] = Counter()
    image_captions = []
    for idx, image in enumerate(data["images"]):
        counter["total"] += 1
        if idx in permutation_map:
            image_captions.append(data["images"][permutation_map[idx]]["sentences"][:])
            counter["permuted"] += 1
        else:
            image_captions.append(image["sentences"][:])

    for image, captions in zip(data["images"], image_captions, strict=True):
        image["sentences"] = captions
    with Path(output_path).open("w") as f:
        json.dump(data, f)
    logger.info(
        "Applied permutations to %d out of %d images (%.1f%%)",
        counter["permuted"],
        counter["total"],
        100 * counter["permuted"] / counter["total"] if counter["total"] else 0.0,
    )


def shuffle_csv_manifest(manifest_path: str, output_path: str) -> None:
    rows = pd.read_csv(manifest_path)
    logger.info("Loaded %d rows from CSV manifest", len(rows))
    captions = rows["utterance"].tolist()
    random.shuffle(captions)
    rows["utterance"] = captions
    rows.to_csv(output_path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--manifest-path", required=True, help="Source manifest path.")
    parser.add_argument("--output-path", required=True, help="Where to write the shuffled manifest.")
    parser.add_argument(
        "--type",
        required=True,
        choices=("json", "karpathy_json", "karpathy_json_with_permutation", "csv"),
    )
    parser.add_argument(
        "--permutation-metadata-path",
        default=None,
        help="JSON file with a `permutation_map` (used only for karpathy_json_with_permutation).",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.random_seed)

    if args.type == "json":
        shuffle_json_manifest(args.manifest_path, args.output_path)
    elif args.type == "karpathy_json":
        shuffle_karpathy_json(args.manifest_path, args.output_path)
    elif args.type == "karpathy_json_with_permutation":
        if args.permutation_metadata_path is None:
            raise ValueError(
                "--permutation-metadata-path required for karpathy_json_with_permutation",
            )
        apply_permutation_karpathy_json(
            args.manifest_path,
            args.permutation_metadata_path,
            args.output_path,
        )
    elif args.type == "csv":
        shuffle_csv_manifest(args.manifest_path, args.output_path)


if __name__ == "__main__":
    main()
