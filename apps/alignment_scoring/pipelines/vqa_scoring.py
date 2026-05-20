# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VQA-style alignment scoring with PLM, matched + shuffled JSD aggregation.

Each processor prompts PLM with a Yes/No question per (media, caption) pair
("Does this figure show '<caption>'? Please answer 'Yes' or 'No'.") and
records the probability mass on the Yes token. The pipeline runs matched and
shuffled processors in parallel via Stopes and bootstrap-estimates JS
divergence between the two P(Yes) distributions, mirroring CLIP and STS
scoring.

Run with::

    alignment-vqa-scoring --config-path apps/alignment_scoring/configs \\
        --config-name pipeline/vqa_scoring \\
        name=coco_v1 \\
        ++matched_processor.dataset.manifest_path=/data/coco/captions_orig.json \\
        ++shuffled_processor.dataset.manifest_path=/data/coco/captions_orig_shuffled.json
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stopes.core import Launcher

from apps.alignment_scoring.configs import VQAScoringPipelineConfig
from apps.alignment_scoring.modeling.plm import PLMGenerationModule
from apps.alignment_scoring.utils import bootstrap_js, calculate_kl_divergence, flattened
from core.utils import resolve_and_print_config, setup_logging, to_yaml

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _plot_vqa_distributions(matched: list[float], shuffled: list[float]) -> bytes:
    """KDE plot of matched-vs-shuffled VQA P(Yes) distributions."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(matched, label="Matched Pairs", color="blue", fill=True, alpha=0.5)
    sns.kdeplot(shuffled, label="Shuffled Pairs", color="red", fill=True, alpha=0.5)
    plt.axvline(float(np.mean(matched)), color="blue", linestyle="--", label="Mean Matched")
    plt.axvline(float(np.mean(shuffled)), color="red", linestyle="--", label="Mean Shuffled")
    plt.title("VQA P(Yes) Distributions")
    plt.xlabel("P(Yes)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(visible=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return buf.getvalue()


def _save_outputs(
    matched_results: list[dict],
    shuffled_results: list[dict],
    output_dir: str,
) -> None:
    """Aggregate matched + shuffled VQA scores; write CSVs, JSD/KL stats, KDE PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matched_df = pd.DataFrame(matched_results)
    shuffled_df = pd.DataFrame(shuffled_results)
    matched_df.to_csv(out / "vqa_results_matched.csv", index=False)
    shuffled_df.to_csv(out / "vqa_results_shuffled.csv", index=False)

    matched_scores = matched_df["vqa_score"].astype(float).tolist()
    shuffled_scores = shuffled_df["vqa_score"].astype(float).tolist()

    bootstrap_results = bootstrap_js(
        matched_scores,
        shuffled_scores,
        max_samples=1_000_000,
    )
    np.save(out / "js_bootstrap_distribution.npy", bootstrap_results["bootstrap_distribution"])

    results = {
        **calculate_kl_divergence(matched_scores, shuffled_scores),
        "bootstrap_js_mean": float(bootstrap_results["bootstrap_distribution"].mean()),
        "bootstrap_js_error": float(bootstrap_results["standard_error"]),
        "bootstrap_js_ci_lower": float(bootstrap_results["ci"][0]),
        "bootstrap_js_ci_upper": float(bootstrap_results["ci"][1]),
        "mean_vqa_score_matched": float(np.mean(matched_scores)),
        "std_vqa_score_matched": float(np.std(matched_scores)),
        "mean_vqa_score_shuffled": float(np.mean(shuffled_scores)),
        "std_vqa_score_shuffled": float(np.std(shuffled_scores)),
    }
    results_yaml = to_yaml(results)
    logger.info("VQA scoring results:\n%s", results_yaml)
    (out / "results.yaml").write_text(results_yaml)
    (out / "vqa_score_distribution.png").write_bytes(_plot_vqa_distributions(matched_scores, shuffled_scores))


async def _run_pipeline(config: VQAScoringPipelineConfig) -> None:
    if not config.matched_processor.vqa_scoring or not config.shuffled_processor.vqa_scoring:
        raise ValueError("Both processors must have vqa_scoring=True for VQA scoring pipeline")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    launcher: Launcher = hydra.utils.instantiate(config.launcher)

    modules = {
        "matched": PLMGenerationModule(config.matched_processor),
        "shuffled": PLMGenerationModule(config.shuffled_processor),
    }
    futures = []
    for name, module in modules.items():
        logger.info("Scheduling VQA processor: %s", name)
        futures.append((name, launcher.schedule(module)))

    logger.info("Waiting for %d VQA processors...", len(futures))
    names = [n for n, _ in futures]
    fs = [f for _, f in futures]
    raw_results = await asyncio.gather(*fs, return_exceptions=True)

    results: dict[str, list] = {}
    for name, result in zip(names, raw_results, strict=True):
        if isinstance(result, BaseException):
            logger.error("VQA processor %s failed: %s", name, result)
            raise result
        logger.info("VQA processor %s completed", name)
        results[name] = list(flattened(result))

    _save_outputs(results["matched"], results["shuffled"], config.output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/vqa_scoring")
def main(config: DictConfig) -> None:
    """CLI entrypoint."""
    import apps.alignment_scoring  # noqa: F401

    setup_logging()
    resolve_and_print_config(config)
    try:
        asyncio.run(_run_pipeline(cast("VQAScoringPipelineConfig", config)))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
