# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BabyView image-text dataset for EgoVLP fine-tuning.

Reads ``[{"utterance", "frame_filenames" | "frames", ...}]`` manifests, i.e. the
output of ``egobabyvlm-build-clip-manifest`` (``apps/data_preprocessing``). This
subclass of the CLIP baseline's ``HowToCaptionsDataset`` only overrides record
loading (to accept the ``frames`` key used by our internal manifests as well)
and ``__getitem__`` (to skip unreadable frames); frame sampling, image loading
and the transform reuse the base class.

Default locations come from ``$BABYVIEW_ROOT``, laid out as
``{train,val}.json`` plus frames under ``frames``. Override
per file with ``--babyview_{train,val}_json`` / ``--babyview_image_root``.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

from PIL import Image

from apps.baselines.clip.data.captions import HowToCaptionsDataset

logger = logging.getLogger(__name__)

BABYVIEW_ROOT = os.environ.get("BABYVIEW_ROOT")
BABYVIEW_TRAIN_JSON = f"{BABYVIEW_ROOT}/train.json" if BABYVIEW_ROOT else None
BABYVIEW_VAL_JSON = f"{BABYVIEW_ROOT}/val.json" if BABYVIEW_ROOT else None
BABYVIEW_IMAGE_ROOT = f"{BABYVIEW_ROOT}/frames" if BABYVIEW_ROOT else None


class BabyViewCaptionsDataset(HowToCaptionsDataset):
    """BabyView utterance/frames dataset (``frames`` schema).

    Emits ``(image_tensor, caption, sample_id)``; one frame is sampled per
    ``__getitem__`` from ``frames`` when ``multiple_frames=True`` (default).
    """

    def _load_records(self) -> list[dict[str, Any]]:
        with self.manifest_path.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data)
        records: list[dict[str, Any]] = []
        for i, entry in enumerate(data):
            # Tolerate both BabyView (``frames``) and HowTo (``frame_filenames``) schemas.
            frames = entry.get("frames", entry.get("frame_filenames"))
            if not frames:
                continue
            records.append(
                {
                    "image_path": None,
                    "frame_filenames": frames,
                    "caption": entry["utterance"],
                    "id": f"{entry.get('video_id', i)}_{entry.get('utterance_num', i)}",
                },
            )
        return records

    def __getitem__(self, idx: int) -> tuple[Any, str, str]:
        # Some BabyView manifests reference frames that are absent on disk;
        # tolerate these instead of crashing the DataLoader. Try the record's frames in random order, then fall back to
        # subsequent records until a readable frame is found.
        n = len(self.records)
        for offset in range(n):
            record = self.records[(idx + offset) % n]
            frames = list(record["frame_filenames"])
            if self.multiple_frames and len(frames) > 1:
                random.shuffle(frames)
            for frame in frames:
                try:
                    image = Image.open(self._resolve_image_path(frame)).convert("RGB")
                except (FileNotFoundError, OSError) as exc:
                    # Same skip-and-fall-back behaviour as the paper runs, but
                    # visible: a wrong image_root shows up as a flood of warnings.
                    logger.warning("BabyView: skipping unreadable frame %s (%s)", frame, exc)
                    continue
                if self.transform is not None:
                    image = self.transform(image)
                return image, record["caption"], str(record["id"])
        msg = "BabyViewCaptionsDataset: no readable frames found in any record"
        raise RuntimeError(msg)
