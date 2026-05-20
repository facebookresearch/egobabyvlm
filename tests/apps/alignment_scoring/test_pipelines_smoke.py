# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end smoke tests for the alignment-scoring pipelines.

These spin up the full Hydra+Stopes stack against tiny fixture data and a
small model, then assert metrics are finite and in plausible range. Gated
behind ``pytest -m gpu`` because they require CUDA + a real model download.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]


def _build_coco_fixture(root: Path, *, n_images: int = 16) -> tuple[Path, Path, Path]:
    """Write a tiny COCO-format dataset with discriminative color-keyed images.

    Each image is a solid color and matched with a caption naming that color.
    Shuffled gets a different color caption — easy enough for CLIP to score
    higher on matched than shuffled.
    """
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    colors = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 0, 255),  # blue
        (255, 255, 0),  # yellow
    ]
    color_names = ["a red square", "a green square", "a blue square", "a yellow square"]
    captions_matched: list[dict] = []
    captions_shuffled: list[dict] = []
    images: list[dict] = []
    for image_id in range(n_images):
        color_idx = image_id % len(colors)
        Image.new("RGB", (224, 224), colors[color_idx]).save(image_dir / f"img_{image_id}.jpg")
        images.append({"id": image_id, "file_name": f"img_{image_id}.jpg"})
        captions_matched.append({"image_id": image_id, "caption": color_names[color_idx]})
        # Pick a different color caption — guaranteed mismatch.
        captions_shuffled.append(
            {
                "image_id": image_id,
                "caption": color_names[(color_idx + 2) % len(colors)],
            }
        )

    matched_path = root / "matched.json"
    matched_path.write_text(json.dumps({"images": images, "annotations": captions_matched}))
    shuffled_path = root / "shuffled.json"
    shuffled_path.write_text(json.dumps({"images": images, "annotations": captions_shuffled}))
    return image_dir, matched_path, shuffled_path


@pytest.mark.gpu
def test_clip_scoring_smoke(tmp_path: Path) -> None:
    """Run clip_scoring with ViT-B/16 against 16 COCO-fixture images.

    Asserts the matched mean cosine sim is higher than the shuffled mean
    (real CLIP should always reward the right caption) and that JS divergence
    is positive and finite.
    """
    image_dir, matched, shuffled = _build_coco_fixture(tmp_path)
    output_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "-m",
        "apps.alignment_scoring.pipelines.clip_scoring",
        f"--config-path={REPO_ROOT / 'apps/alignment_scoring/configs'}",
        "--config-name=pipeline/clip_scoring",
        "name=smoke",
        f"output_dir={output_dir}",
        f"matched_processor.data.dataset.manifest_path={matched}",
        f"matched_processor.data.dataset.dataset_dir={image_dir}",
        f"shuffled_processor.data.dataset.manifest_path={shuffled}",
        f"shuffled_processor.data.dataset.dataset_dir={image_dir}",
        "model@matched_processor.model=vit_b16_openai",
        "model@shuffled_processor.model=vit_b16_openai",
        "matched_processor.data.batch_size=4",
        "matched_processor.data.num_workers=0",
        "shuffled_processor.data.batch_size=4",
        "shuffled_processor.data.num_workers=0",
        "matched_processor.num_items_per_chunk=8",
        "shuffled_processor.num_items_per_chunk=8",
        "launcher.cluster=local",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    results = yaml.safe_load((output_dir / "results.yaml").read_text())
    assert results["mean_cos_sim_matched"] > results["mean_cos_sim_shuffled"], (
        f"matched={results['mean_cos_sim_matched']} shuffled={results['mean_cos_sim_shuffled']}"
    )
    assert results["bootstrap_js_mean"] > 0
    assert results["bootstrap_js_mean"] < 1


@pytest.mark.gpu
def test_finetune_lora_smoke(tmp_path: Path) -> None:
    """Run finetune_lora for 1 epoch on a 16-image fixture; assert checkpoint emitted.

    The model is ViT-B/16 with LoRA wrapping. We don't assert anything about
    the loss going down (16 samples in a single epoch isn't enough to expect
    learning), only that the pipeline runs end-to-end and saves a checkpoint
    in open_clip format at the expected path.
    """
    image_dir, matched, _ = _build_coco_fixture(tmp_path)
    output_dir = tmp_path / "lora_out"

    cmd = [
        sys.executable,
        "-m",
        "apps.alignment_scoring.pipelines.finetune_lora",
        f"--config-path={REPO_ROOT / 'apps/alignment_scoring/configs'}",
        "--config-name=pipeline/finetune_lora",
        "name=lora_smoke",
        f"output_dir={output_dir}",
        f"data_train.dataset.manifest_path={matched}",
        f"data_train.dataset.dataset_dir={image_dir}",
        f"data_val.dataset.manifest_path={matched}",
        f"data_val.dataset.dataset_dir={image_dir}",
        "data_train.batch_size=4",
        "data_train.num_workers=0",
        "data_val.batch_size=4",
        "data_val.num_workers=0",
        "optim.epochs=1",
        "optim.warmup_epochs=0",
        "optim.lr=1e-4",
        "log_interval=1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # The "best" alias always gets written.
    best_dir = output_dir / "openclip_checkpoint_best"
    assert best_dir.exists(), f"missing {best_dir}"
    assert (best_dir / "open_clip_model.safetensors").exists()
    assert (best_dir / "open_clip_config.json").exists()


@pytest.mark.gpu
def test_sts_scoring_smoke(tmp_path: Path) -> None:
    """Run sts_scoring with SONAR against tiny matched + shuffled COCO fixtures.

    Matched processor: caption A vs caption A (perfect identity, cos_sim ~1).
    Shuffled processor: caption A vs different-color caption B (low cos_sim).
    Asserts matched mean is much higher than shuffled mean and JS > 0.
    """
    image_dir, matched, shuffled = _build_coco_fixture(tmp_path)
    output_dir = tmp_path / "sts_out"

    cmd = [
        sys.executable,
        "-m",
        "apps.alignment_scoring.pipelines.sts_scoring",
        f"--config-path={REPO_ROOT / 'apps/alignment_scoring/configs'}",
        "--config-name=pipeline/sts_scoring",
        "name=sts_smoke",
        f"output_dir={output_dir}",
        f"matched_processor.dataset_a.manifest_path={matched}",
        f"matched_processor.dataset_a.dataset_dir={image_dir}",
        f"matched_processor.dataset_b.manifest_path={matched}",
        f"matched_processor.dataset_b.dataset_dir={image_dir}",
        f"shuffled_processor.dataset_a.manifest_path={matched}",
        f"shuffled_processor.dataset_a.dataset_dir={image_dir}",
        f"shuffled_processor.dataset_b.manifest_path={shuffled}",
        f"shuffled_processor.dataset_b.dataset_dir={image_dir}",
        "matched_processor.batch_size=4",
        "matched_processor.num_items_per_chunk=8",
        "shuffled_processor.batch_size=4",
        "shuffled_processor.num_items_per_chunk=8",
        "launcher.cluster=local",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    results = yaml.safe_load((output_dir / "results.yaml").read_text())
    assert results["mean_cos_sim_matched"] > results["mean_cos_sim_shuffled"], (
        f"matched={results['mean_cos_sim_matched']} shuffled={results['mean_cos_sim_shuffled']}"
    )
    assert results["bootstrap_js_mean"] > 0
    assert results["bootstrap_js_mean"] < 1


@pytest.mark.gpu
def test_captioning_smoke(tmp_path: Path) -> None:
    """Run captioning with PLM-8B against a 16-image color-keyed fixture.

    Asserts every image gets a non-empty caption and that at least one
    caption mentions the right color word — proves PLM is actually
    conditioning on the image, not just emitting boilerplate.
    """
    image_dir, matched, _ = _build_coco_fixture(tmp_path)
    output_dir = tmp_path / "cap_out"
    output_manifest = output_dir / "recaptioned.json"

    cmd = [
        sys.executable,
        "-m",
        "apps.alignment_scoring.pipelines.captioning",
        f"--config-path={REPO_ROOT / 'apps/alignment_scoring/configs'}",
        "--config-name=pipeline/captioning",
        "name=cap_smoke",
        f"output_dir={output_dir}",
        f"output_manifest_path={output_manifest}",
        f"generation.dataset.manifest_path={matched}",
        f"generation.dataset.dataset_dir={image_dir}",
        "generation.num_items_per_chunk=8",
        "generation.max_gen_len=32",
        "launcher.cluster=local",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    out = json.loads(output_manifest.read_text())
    assert len(out["images"]) == 16
    assert len(out["annotations"]) == 16
    captions = [a["caption"].strip() for a in out["annotations"]]
    assert all(captions), f"empty captions present: {captions}"
    color_words = ("red", "green", "blue", "yellow")
    assert any(any(w in cap.lower() for w in color_words) for cap in captions), (
        f"PLM produced no color words on color-keyed fixture: {captions}"
    )


@pytest.mark.gpu
def test_vqa_smoke(tmp_path: Path) -> None:
    """Run vqa_scoring with PLM-8B against matched + shuffled COCO fixtures.

    Matched processor: image vs its own color caption (P(Yes) high).
    Shuffled processor: image vs a different-color caption (P(Yes) lower).
    Asserts matched mean > shuffled mean and JS divergence is finite, in (0, 1).
    """
    image_dir, matched, shuffled = _build_coco_fixture(tmp_path)
    output_dir = tmp_path / "vqa_out"

    cmd = [
        sys.executable,
        "-m",
        "apps.alignment_scoring.pipelines.vqa_scoring",
        f"--config-path={REPO_ROOT / 'apps/alignment_scoring/configs'}",
        "--config-name=pipeline/vqa_scoring",
        "name=vqa_smoke",
        f"output_dir={output_dir}",
        f"matched_processor.dataset.manifest_path={matched}",
        f"matched_processor.dataset.dataset_dir={image_dir}",
        f"shuffled_processor.dataset.manifest_path={shuffled}",
        f"shuffled_processor.dataset.dataset_dir={image_dir}",
        "matched_processor.num_items_per_chunk=8",
        "shuffled_processor.num_items_per_chunk=8",
        "launcher.cluster=local",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    results = yaml.safe_load((output_dir / "results.yaml").read_text())
    assert results["mean_vqa_score_matched"] > results["mean_vqa_score_shuffled"], (
        f"matched={results['mean_vqa_score_matched']} shuffled={results['mean_vqa_score_shuffled']}"
    )
    assert 0 < results["bootstrap_js_mean"] < 1
    assert math.isfinite(results["bootstrap_js_error"])
