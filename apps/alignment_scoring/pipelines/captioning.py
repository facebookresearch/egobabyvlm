# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PLM-based captioning pipeline.

Re-captions an image or video manifest using Perception-LM. Schedules a single
:class:`PLMGenerationModule` job array via Stopes, then writes the new captions
back into a copy of the manifest at ``output_manifest_path``. Both COCO-format
JSON and CSV manifests are supported (file format is inferred by extension).

Run with::

    alignment-captioning --config-path apps/alignment_scoring/configs \\
        --config-name pipeline/captioning \\
        ++generation.dataset.manifest_path=/data/coco/captions_train2017.json \\
        output_manifest_path=/tmp/coco_recaptioned.json
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
from stopes.core import Launcher

from apps.alignment_scoring.configs import CaptioningPipelineConfig, PLMGenerationConfig
from apps.alignment_scoring.modeling.plm import PLMGenerationModule
from apps.alignment_scoring.utils import flattened
from core.utils import resolve_and_print_config, setup_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def aggregate_and_save_manifest(
    results: list[dict],
    captioning_config: PLMGenerationConfig,
    output_manifest_path: str,
) -> None:
    """Write the new captions back into a copy of the source manifest.

    JSON manifests are interpreted as either standard COCO-format (top-level
    ``images`` + ``annotations`` arrays) or Karpathy split (``images[*].sentences``).
    CSV manifests must have a ``clip_filename`` column.
    """
    logger.info("Processing %d captioned examples", len(results))
    original_manifest_path = str(captioning_config.dataset.manifest_path)
    caption_map = {item["index"]: item["generated_caption"] for item in results}

    manifest_ext = Path(original_manifest_path).suffix.lower()

    if manifest_ext == ".json":
        with Path(original_manifest_path).open() as f:
            original_data = json.load(f)

        if isinstance(original_data, dict) and "images" in original_data and "annotations" in original_data:
            # COCO format.
            original_data["annotations"] = []
            for idx, (image_id, caption) in enumerate(caption_map.items()):
                original_data["annotations"].append(
                    {
                        "image_id": image_id,
                        "caption": caption,
                        "id": idx,
                    }
                )
        elif (
            isinstance(original_data, dict) and "images" in original_data and "sentences" in original_data["images"][0]
        ):
            # Karpathy format.
            for image in original_data["images"]:
                image_id = image["imgid"]
                if image_id in caption_map:
                    image["sentences"] = [{"raw": caption_map[image_id], "imgid": image_id}]

        with Path(output_manifest_path).open("w") as f:
            json.dump(original_data, f, indent=2)

    elif manifest_ext == ".csv":
        with Path(original_manifest_path).open() as f:
            reader = csv.DictReader(f)
            original_rows = list(reader)

        for row in original_rows:
            if "clip_filename" not in row:
                raise ValueError("Expected 'clip_filename' column in CSV manifest file")
            if row["clip_filename"] in caption_map:
                row["utterance"] = caption_map[row["clip_filename"]]

        if original_rows:
            fieldnames = list(original_rows[0].keys())
            with Path(output_manifest_path).open("w") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(original_rows)
        else:
            logger.warning("No rows to write")
    else:
        raise ValueError(f"Unsupported manifest file format: {manifest_ext}")

    logger.info("Saved updated manifest to %s", output_manifest_path)


async def _run_pipeline(config: CaptioningPipelineConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    launcher: Launcher = hydra.utils.instantiate(config.launcher)
    raw = await launcher.schedule(PLMGenerationModule(config.generation))
    all_results = sorted(flattened(raw), key=lambda x: x["index"])
    aggregate_and_save_manifest(all_results, config.generation, config.output_manifest_path)


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/captioning")
def main(config: DictConfig) -> None:
    """CLI entrypoint."""
    import apps.alignment_scoring  # noqa: F401

    setup_logging()
    resolve_and_print_config(config)
    try:
        asyncio.run(_run_pipeline(cast("CaptioningPipelineConfig", config)))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
