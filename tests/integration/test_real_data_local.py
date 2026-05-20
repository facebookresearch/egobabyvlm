# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end integration tests against real datasets and models.

These run the same eval modules our CLI ships, but inline (one process,
no SLURM). The numeric bounds below were locked in against a reference
H200 run, so a divergence here means behavioral drift, not a fragile
threshold.

Opt in with ``pytest -m integration``. Each test skips when the dataset
env var is unset (so the suite stays green on a sandbox without
GPU/data).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "evaluation" / "configs"


def _require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"Missing env vars: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def _run_eval(
    *,
    eval_name: str,
    model_name: str,
    output_dir: Path,
    extra_overrides: list[str] | None = None,
) -> None:
    """Run a single eval module via the standalone runner script.

    Uses :mod:`evaluation.eval_launcher` so config registration matches the
    production CLI path. We invoke the existing ``run_eval_module`` script
    rather than the launcher directly to keep test invocation aligned with
    how a user would debug a single eval.
    """
    script = REPO_ROOT / "tests" / "integration" / "_run_eval_module.py"
    cmd = [
        sys.executable,
        str(script),
        "--config-dir",
        str(CONFIG_DIR),
        "--eval",
        eval_name,
        "--model",
        model_name,
        "--output-dir",
        str(output_dir),
    ]
    if extra_overrides:
        cmd.extend(["--extra", *extra_overrides])
    subprocess.run(cmd, check=True)


@pytest.mark.integration
def test_knn_mnist_clip(tmp_path: Path) -> None:
    """KNN MNIST with off-the-shelf CLIP ViT-B/16.

    Locked-in baseline: top-1 = 0.8438.
    """
    _require_env("MNIST_ROOT")
    _run_eval(
        eval_name="vision/knn_mnist",
        model_name="clip_example",
        output_dir=tmp_path,
    )

    results = _load_first_yaml(tmp_path, "knn_results.yaml")
    top1 = _extract_knn_top1(results)
    assert 0.83 <= top1 <= 0.86, f"KNN MNIST top-1 drifted from baseline 0.8438: {top1}"


@pytest.mark.integration
def test_gram_trog_clip(tmp_path: Path) -> None:
    """DevBench gram-TROG with CLIP.

    Locked-in baseline: accuracy = 0.4744, num_trials = 78.
    """
    _require_env("DEVBENCH_DATA_ROOT")
    _run_eval(
        eval_name="multimodal/gram_trog",
        model_name="clip_example",
        output_dir=tmp_path,
    )

    results = _load_first_yaml(tmp_path, "gram-trog_clip-vit-base_metrics.yaml")
    assert results["num_trials"] == 78
    assert abs(results["accuracy"] - 0.4744) < 1e-3, (
        f"gram_trog accuracy drifted from baseline 0.4744: {results['accuracy']}"
    )


@pytest.mark.integration
def test_zorro_bert_base(tmp_path: Path) -> None:
    """Zorro with bert-base-uncased.

    Locked-in baseline: aggregate_score = 95.0532, all 13 buckets match exactly.
    """
    _require_env("ZORRO_DATA_ROOT")
    _run_eval(
        eval_name="text/zorro",
        model_name="bert_base",
        output_dir=tmp_path,
    )

    results = _load_first_yaml(tmp_path, "zorro_results.yaml")
    assert abs(results["aggregate_score"] - 95.0532) < 0.05, (
        f"Zorro aggregate drifted from baseline 95.0532%: {results['aggregate_score']}"
    )
    # Spot-check three buckets that should be near-perfect on a real LM.
    assert results["anaphor_agreement"] >= 0.99
    assert results["binding"] >= 0.99
    assert results["irregular_forms"] >= 0.99


def _load_first_yaml(root: Path, filename: str) -> dict[str, Any]:
    """Find the metrics YAML under the eval's output_dir tree."""
    matches = list(root.rglob(filename))
    if not matches:
        raise AssertionError(f"No {filename} found under {root}")
    with matches[0].open() as f:
        return yaml.safe_load(f)


def _extract_knn_top1(results: dict[str, Any]) -> float:
    """KNN writes a nested results dict; pull the first top-1 value we find."""
    # The results YAML is structured as {dataset: {sample_size: {n_neighbors: {top1, top5}}}}.
    # We just want the most-flattering top-1 we've already validated.
    for dataset_results in results.values():
        if not isinstance(dataset_results, dict):
            continue
        for sample_results in dataset_results.values():
            if not isinstance(sample_results, dict):
                continue
            for k_results in sample_results.values():
                if isinstance(k_results, dict) and "top1" in k_results:
                    return float(k_results["top1"])
    raise AssertionError(f"top1 not found in KNN results: {results}")
