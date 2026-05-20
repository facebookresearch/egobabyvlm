# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLIP-style alignment scoring pipeline.

Schedules two :class:`CLIPProcessorModule` jobs in parallel via Stopes — one
over a "matched" caption manifest, one over a "shuffled" manifest — then
aggregates the per-pair cosine similarities into JS-divergence + KL stats and
writes a YAML, NPY, CSV, and PNG plot to ``output_dir``.

Run with::

    alignment-clip-scoring --config-path apps/alignment_scoring/configs/pipeline \\
        --config-name clip_scoring \\
        ++matched_processor.data.dataset.manifest_path=/path/to/matched.json \\
        ++shuffled_processor.data.dataset.manifest_path=/path/to/shuffled.json
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from stopes.core import Launcher, Requirements, StopesModule
from tqdm import tqdm

from apps.alignment_scoring.configs import CLIPProcessorConfig, CLIPScoringPipelineConfig
from apps.alignment_scoring.utils import (
    bootstrap_js,
    calculate_kl_divergence,
    clip_forward,
    create_alignment_dataloader,
    create_model,
    flattened,
    post_collate_fn,
)
from core.utils import resolve_and_print_config, setup_logging, to_yaml

if TYPE_CHECKING:
    import open_clip
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CLIPProcessorModule(StopesModule):
    """Stopes job-array module: scores one (manifest, model) pair into per-item cosines."""

    def __init__(self, config: CLIPProcessorConfig) -> None:
        super().__init__(config, CLIPProcessorConfig)
        self.num_items = len(self._build_dataloader(None, None)[0].dataset)

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=140,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.data.num_workers + 2,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        return f"processor_{self.config.name}"

    @property
    def num_chunks(self) -> int:
        return math.ceil(self.num_items / self.config.num_items_per_chunk)

    def array(self) -> list[tuple[int, int]]:
        return [
            (
                self.config.num_items_per_chunk * i,
                self.config.num_items_per_chunk * (i + 1),
            )
            for i in range(self.num_chunks)
        ]

    def _build_model(self, device: torch.device | str = "cuda") -> tuple:
        return create_model(config=self.config.model, device=device)[:3]

    def _build_dataloader(
        self,
        indices: tuple[int, int] | None,
        preprocess: open_clip.SimpleTokenizer | None,
    ) -> tuple:
        return create_alignment_dataloader(
            self.config.data,
            indices=indices,
            preprocessor=preprocess,
            mode="eval",
        )

    def run(self, iteration_value: tuple[int, int], iteration_index: int) -> list[float]:
        logger.info(
            "Processing chunk %d (indices %d-%d)",
            iteration_index,
            iteration_value[0],
            iteration_value[1],
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device %s", device)

        model, preprocess, tokenizer = self._build_model(device=device)
        dataloader, is_video_dataset = self._build_dataloader(
            indices=iteration_value,
            preprocess=preprocess,
        )

        chunk_cos_sims: list[float] = []
        for batch in tqdm(dataloader, desc="Processing batches"):
            try:
                model_inputs = post_collate_fn(batch, tokenizer, device=device)
            except Exception:
                logger.exception("Error processing batch, skipping...")
                continue

            with torch.inference_mode():
                image_features, text_features = clip_forward(
                    model,
                    model_inputs,
                    is_video_dataset=is_video_dataset,
                    is_video_model=self.config.model.is_video_model,
                )
            cos_sims_diag = (image_features * text_features).sum(dim=-1).cpu().tolist()
            chunk_cos_sims.extend(cos_sims_diag)

        return chunk_cos_sims


def _plot_distributions(
    matched: list[float],
    shuffled: list[float],
    matched_mean: float,
    shuffled_mean: float,
) -> bytes:
    """KDE plot of matched-vs-shuffled cosine sim distributions."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(matched, label="Matched Pairs", color="blue", fill=True, alpha=0.5)
    sns.kdeplot(shuffled, label="Shuffled Pairs", color="red", fill=True, alpha=0.5)
    plt.axvline(matched_mean, color="blue", linestyle="--", label="Mean Matched")
    plt.axvline(shuffled_mean, color="red", linestyle="--", label="Mean Shuffled")
    plt.title("Cosine Similarity Distributions")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(visible=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return buf.getvalue()


def _save_outputs(
    cos_sims: list[float],
    cos_sims_shuffled: list[float],
    output_dir: str,
) -> None:
    """Write metrics YAML, bootstrap NPY, per-pair CSV, and KDE PNG to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bootstrap_results = bootstrap_js(cos_sims, cos_sims_shuffled, max_samples=1_000_000)
    np.save(out / "js_bootstrap_distribution.npy", bootstrap_results["bootstrap_distribution"])

    results = {
        **calculate_kl_divergence(cos_sims, cos_sims_shuffled),
        "bootstrap_js_mean": float(bootstrap_results["bootstrap_distribution"].mean()),
        "bootstrap_js_error": float(bootstrap_results["standard_error"]),
        "bootstrap_js_ci_lower": float(bootstrap_results["ci"][0]),
        "bootstrap_js_ci_upper": float(bootstrap_results["ci"][1]),
        "mean_cos_sim_matched": float(np.mean(cos_sims)),
        "std_cos_sim_matched": float(np.std(cos_sims)),
        "mean_cos_sim_shuffled": float(np.mean(cos_sims_shuffled)),
        "std_cos_sim_shuffled": float(np.std(cos_sims_shuffled)),
    }
    results_yaml = to_yaml(results)
    logger.info("Results:\n%s", results_yaml)
    (out / "results.yaml").write_text(results_yaml)

    plot_bytes = _plot_distributions(
        cos_sims,
        cos_sims_shuffled,
        matched_mean=results["mean_cos_sim_matched"],
        shuffled_mean=results["mean_cos_sim_shuffled"],
    )
    (out / "similarity_histogram.png").write_bytes(plot_bytes)

    cos_sim_df = pd.DataFrame(
        {
            "cosine_similarity_matched": cos_sims,
            "cosine_similarity_shuffled": cos_sims_shuffled,
        }
    )
    cos_sim_df.to_csv(out / "cosine_similarities.csv", index=False)


async def _run_pipeline(config: CLIPScoringPipelineConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    launcher: Launcher = hydra.utils.instantiate(config.launcher)

    modules = {
        "matched": CLIPProcessorModule(config.matched_processor),
        "shuffled": CLIPProcessorModule(config.shuffled_processor),
    }
    futures = []
    for name, module in modules.items():
        logger.info("Scheduling processor: %s", name)
        futures.append((name, launcher.schedule(module)))

    logger.info("Waiting for %d processors...", len(futures))
    names = [n for n, _ in futures]
    fs = [f for _, f in futures]
    raw_results = await asyncio.gather(*fs, return_exceptions=True)

    results: dict[str, list] = {}
    for name, result in zip(names, raw_results, strict=True):
        if isinstance(result, BaseException):
            logger.error("Processor %s failed: %s", name, result)
            raise result
        logger.info("Processor %s completed", name)
        results[name] = result

    cos_sims = list(flattened(results["matched"]))
    cos_sims_shuffled = list(flattened(results["shuffled"]))
    _save_outputs(cos_sims, cos_sims_shuffled, config.output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/clip_scoring")
def main(config: DictConfig) -> None:
    """CLI entrypoint — wires Hydra config into the async pipeline."""
    # Importing the package registers ConfigStore nodes the YAML refers to.
    import apps.alignment_scoring  # noqa: F401

    setup_logging()
    resolve_and_print_config(config)
    try:
        asyncio.run(_run_pipeline(cast("CLIPScoringPipelineConfig", config)))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
