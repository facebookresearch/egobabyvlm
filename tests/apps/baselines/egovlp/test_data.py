# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the EgoVLP data pipeline (collate, COCO-MC permutation, BabyView)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from apps.baselines.egovlp.data import (
    BabyViewCaptionsDataset,
    CocoMetaclipCaptionsDataset,
    build_egovlp_transform,
    egovlp_collate,
    permutation_metadata_path,
)
from apps.baselines.egovlp.data.coco_metaclip import _Pair, build_pair_index, load_permutation_map


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _is_color(image: Image.Image, color: tuple[int, int, int]) -> bool:
    """JPEG is lossy, so compare the corner pixel with a small tolerance."""
    return all(abs(a - b) <= 8 for a, b in zip(image.getpixel((0, 0)), color, strict=True))


def test_collate_makes_single_frame_clips() -> None:
    batch = [(torch.zeros(3, 8, 8), "a", "0"), (torch.ones(3, 8, 8), "b", "1")]
    out = egovlp_collate(batch)
    assert out["clip"].shape == (2, 1, 3, 8, 8)
    assert out["captions"] == ["a", "b"]
    assert torch.equal(out["clip"][1, 0], torch.ones(3, 8, 8))


def test_transform_output_shape() -> None:
    tf = build_egovlp_transform(image_size=224)
    assert tf(Image.new("RGB", (300, 200))).shape == (3, 224, 224)


def test_permutation_path_p0_is_none_and_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert permutation_metadata_path(0) is None
    monkeypatch.delenv("COCO_MC_PERMUTATION_DIR", raising=False)
    with pytest.raises(RuntimeError, match="COCO_MC_PERMUTATION_DIR"):
        permutation_metadata_path(25)
    monkeypatch.setenv("COCO_MC_PERMUTATION_DIR", "/perm")
    assert permutation_metadata_path(25) == "/perm/permutation_metadata_cocomc_train_p25_seed42.json"
    with pytest.raises(ValueError, match="percent"):
        permutation_metadata_path(30)


def test_pair_index_routes_by_caption_split(tmp_path: Path) -> None:
    manifest = tmp_path / "coco_to_metaclip.jsonl"
    rows = [
        {"caption": "a dog", "coco_id": 1, "matches": [{"uuid": "u1", "tar_path": "t.tar"}]},
        {"caption": "a cat", "coco_id": 2, "matches": [{"uuid": "u2", "tar_path": "t.tar"}]},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "coco_metaclip_train.txt").write_text("a dog\n")
    (tmp_path / "coco_metaclip_val.txt").write_text("a cat\n")

    train, val = build_pair_index(manifest_path=str(manifest), split_dir=str(tmp_path))
    assert [(p.kind, p.locator) for p in train] == [("coco", "1"), ("metaclip", "u1")]
    assert [(p.kind, p.locator) for p in val] == [("coco", "2"), ("metaclip", "u2")]


def test_permutation_swaps_caption_but_keeps_image(tmp_path: Path) -> None:
    _write_image(tmp_path / f"{1:012d}.jpg", (255, 0, 0))
    _write_image(tmp_path / f"{2:012d}.jpg", (0, 0, 255))
    pairs = [_Pair("red caption", "coco", "1"), _Pair("blue caption", "coco", "2")]
    ds = CocoMetaclipCaptionsDataset(pairs, transform=None, coco_image_dir=str(tmp_path), permutation_map={0: 1})

    image0, caption0, _ = ds[0]
    image1, caption1, _ = ds[1]
    assert caption0 == "blue caption"  # donor caption
    assert _is_color(image0, (255, 0, 0))  # own image
    assert caption1 == "blue caption"  # not in map -> unchanged
    assert _is_color(image1, (0, 0, 255))


def test_load_permutation_map_validates_pair_count(tmp_path: Path) -> None:
    path = tmp_path / "perm.json"
    path.write_text(
        json.dumps({"split": "train", "num_pairs": 4, "percent_permuted": 50, "permutation_map": {"0": 1}})
    )
    assert load_permutation_map(str(path), 4) == {0: 1}
    with pytest.raises(ValueError, match="num_pairs"):
        load_permutation_map(str(path), 5)


def test_babyview_skips_missing_frames(tmp_path: Path) -> None:
    _write_image(tmp_path / "frames" / "v1_2.jpg", (0, 255, 0))
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            [
                {"video_id": "v0", "utterance": "missing", "frames": ["v0_1.jpg"], "utterance_num": 1},
                {"video_id": "v1", "utterance": "look", "frame_filenames": ["v1_2.jpg"], "utterance_num": 2},
            ],
        ),
    )
    ds = BabyViewCaptionsDataset(manifest, tmp_path / "frames", transform=None)
    image, caption, sample_id = ds[0]  # record 0 has no readable frame -> falls back to record 1
    assert caption == "look"
    assert sample_id == "v1_2"
    assert _is_color(image, (0, 255, 0))


@pytest.mark.parametrize("member_name", ["./u1.jpeg", "u1.jpeg"])
def test_metaclip_reader_accepts_both_tar_member_names(tmp_path: Path, member_name: str) -> None:
    import io
    import tarfile

    from apps.baselines.egovlp.data.coco_metaclip import _MetaclipImageReader

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 30)).save(buf, format="JPEG")
    tar_path = tmp_path / "shard.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(buf.getvalue())
        tar.addfile(info, io.BytesIO(buf.getvalue()))

    reader = _MetaclipImageReader(extracted_root=None)
    assert _is_color(reader.read("u1", str(tar_path)), (10, 200, 30))
    with pytest.raises(FileNotFoundError):
        reader.read("missing", str(tar_path))


def test_stage_marker_only_written_on_success(tmp_path: Path) -> None:
    from apps.baselines.egovlp.data import stage_metaclip

    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "coco_id": 1,
                "caption": "c",
                "matches": [{"uuid": "u1", "shard_id": 0, "tar_path": str(tmp_path / "missing.tar")}],
            }
        )
    )
    out = stage_metaclip.stage_all(str(tmp_path / "scratch"), str(manifest), num_workers=1)
    assert not (Path(out) / "_done").exists()  # the tar is missing -> failure -> no marker


def test_babyview_logs_skipped_frames(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_image(tmp_path / "frames" / "ok.jpg", (0, 0, 0))
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([{"utterance": "u", "frames": ["gone.jpg", "ok.jpg"]}]))
    ds = BabyViewCaptionsDataset(manifest, tmp_path / "frames", transform=None, multiple_frames=False)
    with caplog.at_level("WARNING"):
        ds[0]
    assert "gone.jpg" in caplog.text
