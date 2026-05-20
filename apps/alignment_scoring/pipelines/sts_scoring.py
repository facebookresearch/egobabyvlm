# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Semantic Textual Similarity (STS) scoring with SONAR text embeddings.

Mirrors the CLIP-scoring pipeline layout: schedule a matched + shuffled
processor in parallel via Stopes, then bootstrap-estimate JS divergence and
KL between the two cosine-sim distributions. Each processor joins two text
manifests by ``media_id`` and encodes both sides with a SONAR text encoder.

Run with::

    alignment-sts-scoring --config-path apps/alignment_scoring/configs \\
        --config-name pipeline/sts_scoring \\
        name=coco_v1 \\
        ++matched_processor.dataset_a.manifest_path=/data/coco/captions_orig.json \\
        ++matched_processor.dataset_b.manifest_path=/data/coco/captions_recap.json \\
        ++shuffled_processor.dataset_a.manifest_path=/data/coco/captions_orig.json \\
        ++shuffled_processor.dataset_b.manifest_path=/data/coco/captions_shuffled.json
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from stopes.core import Launcher, Requirements, StopesModule
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from apps.alignment_scoring.configs import SonarSTSProcessorConfig, SonarSTSScoringPipelineConfig
from apps.alignment_scoring.data import CaptionsPathDataset, TextPairDataset
from apps.alignment_scoring.utils import bootstrap_js, calculate_kl_divergence, flattened
from core.utils import resolve_and_print_config, setup_logging, to_yaml

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class SonarSTSProcessorModule(StopesModule):
    """Stopes job-array module: encodes paired captions and emits diag cosine sims."""

    def __init__(self, config: SonarSTSProcessorConfig) -> None:
        super().__init__(config, SonarSTSProcessorConfig)
        self.num_items = len(self._build_dataloader().dataset)  # type: ignore[arg-type]

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=12,
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

    def _build_model(self, device: torch.device | str = "cuda") -> TextToEmbeddingModelPipeline:
        logger.info("Loading SONAR text encoder %s on %s", self.config.encoder, device)
        return TextToEmbeddingModelPipeline(
            encoder=self.config.encoder,
            tokenizer=self.config.encoder,
            device=torch.device(device),
            dtype=torch.float16,
        )

    def _build_dataloader(self, indices: tuple[int, int] | None = None) -> DataLoader:
        dataset_a: CaptionsPathDataset | Subset = hydra.utils.instantiate(self.config.dataset_a)
        dataset_b: CaptionsPathDataset | Subset = hydra.utils.instantiate(self.config.dataset_b)
        if indices is not None:
            dataset_a = Subset(dataset_a, range(indices[0], min(indices[1], len(dataset_a))))
            dataset_b = Subset(dataset_b, range(indices[0], min(indices[1], len(dataset_b))))
        dataset = TextPairDataset(dataset_a, dataset_b)  # type: ignore[arg-type]
        logger.info("Built TextPairDataset with %d joined samples", len(dataset))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # SONAR encoder isn't fork-safe.
            persistent_workers=False,
        )

    def run(
        self,
        iteration_value: tuple[int, int],
        iteration_index: int,
    ) -> dict[str, list[str | int | float]]:
        logger.info(
            "Processing chunk %d (indices %d-%d)",
            iteration_index,
            iteration_value[0],
            iteration_value[1],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self._build_model(device=device)
        dataloader = self._build_dataloader(indices=iteration_value)

        results: dict[str, list] = defaultdict(list)
        for batch in tqdm(dataloader, desc="Encoding pairs"):
            sentences_a, sentences_b, media_ids = batch
            embeddings_a = F.normalize(
                model.predict(list(sentences_a), source_lang=self.config.source_lang),
                dim=-1,
            )
            embeddings_b = F.normalize(
                model.predict(list(sentences_b), source_lang=self.config.source_lang),
                dim=-1,
            )
            cos_sims = (embeddings_a * embeddings_b).sum(dim=-1).cpu().tolist()
            results["media_id"].extend(media_ids)
            results["cos_sim"].extend(cos_sims)
        return dict(results)


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
    plt.title("STS Cosine Similarity Distributions")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(visible=True)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return buf.getvalue()


def _aggregate_chunks(per_chunk: list[dict[str, list]]) -> tuple[list[float], pd.DataFrame]:
    """Concatenate per-chunk results dicts into one DataFrame; return (cos_sims, df)."""
    chunks_df = pd.concat([pd.DataFrame(r) for r in per_chunk], ignore_index=True)
    return chunks_df["cos_sim"].astype(float).tolist(), chunks_df


def _save_outputs(
    matched: list[dict[str, list]],
    shuffled: list[dict[str, list]],
    output_dir: str,
) -> None:
    """Aggregate matched + shuffled chunks, write CSVs, JSD/KL stats YAML, KDE PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    matched_cos_sims, matched_df = _aggregate_chunks(matched)
    shuffled_cos_sims, shuffled_df = _aggregate_chunks(shuffled)
    matched_df.to_csv(out / "sts_results_matched.csv", index=False)
    shuffled_df.to_csv(out / "sts_results_shuffled.csv", index=False)

    bootstrap_results = bootstrap_js(
        matched_cos_sims,
        shuffled_cos_sims,
        max_samples=1_000_000,
    )
    np.save(out / "js_bootstrap_distribution.npy", bootstrap_results["bootstrap_distribution"])

    results = {
        **calculate_kl_divergence(matched_cos_sims, shuffled_cos_sims),
        "bootstrap_js_mean": float(bootstrap_results["bootstrap_distribution"].mean()),
        "bootstrap_js_error": float(bootstrap_results["standard_error"]),
        "bootstrap_js_ci_lower": float(bootstrap_results["ci"][0]),
        "bootstrap_js_ci_upper": float(bootstrap_results["ci"][1]),
        "mean_cos_sim_matched": float(np.mean(matched_cos_sims)),
        "std_cos_sim_matched": float(np.std(matched_cos_sims)),
        "mean_cos_sim_shuffled": float(np.mean(shuffled_cos_sims)),
        "std_cos_sim_shuffled": float(np.std(shuffled_cos_sims)),
    }
    results_yaml = to_yaml(results)
    logger.info("STS scoring results:\n%s", results_yaml)
    (out / "results.yaml").write_text(results_yaml)
    (out / "similarity_histogram.png").write_bytes(
        _plot_distributions(
            matched_cos_sims,
            shuffled_cos_sims,
            matched_mean=results["mean_cos_sim_matched"],
            shuffled_mean=results["mean_cos_sim_shuffled"],
        ),
    )


async def _run_pipeline(config: SonarSTSScoringPipelineConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    # Pre-download the SONAR encoder once on the main process to avoid two
    # parallel workers racing on the asset cache (which fairseq2 doesn't
    # synchronize — both try to write the same .download.tmp file).
    logger.info("Pre-fetching SONAR encoder %s on CPU...", config.matched_processor.encoder)
    TextToEmbeddingModelPipeline(
        encoder=config.matched_processor.encoder,
        tokenizer=config.matched_processor.encoder,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    launcher: Launcher = hydra.utils.instantiate(config.launcher)

    modules = {
        "matched": SonarSTSProcessorModule(config.matched_processor),
        "shuffled": SonarSTSProcessorModule(config.shuffled_processor),
    }
    futures = []
    for name, module in modules.items():
        logger.info("Scheduling STS processor: %s", name)
        futures.append((name, launcher.schedule(module)))

    logger.info("Waiting for %d STS processors...", len(futures))
    names = [n for n, _ in futures]
    fs = [f for _, f in futures]
    raw_results = await asyncio.gather(*fs, return_exceptions=True)

    results: dict[str, list] = {}
    for name, result in zip(names, raw_results, strict=True):
        if isinstance(result, BaseException):
            logger.error("STS processor %s failed: %s", name, result)
            raise result
        logger.info("STS processor %s completed", name)
        results[name] = result

    matched = list(flattened([[r] for r in results["matched"]]))
    shuffled = list(flattened([[r] for r in results["shuffled"]]))
    _save_outputs(matched, shuffled, config.output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline/sts_scoring")
def main(config: DictConfig) -> None:
    """CLI entrypoint."""
    import apps.alignment_scoring  # noqa: F401

    setup_logging()
    resolve_and_print_config(config)
    try:
        asyncio.run(_run_pipeline(cast("SonarSTSScoringPipelineConfig", config)))
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
