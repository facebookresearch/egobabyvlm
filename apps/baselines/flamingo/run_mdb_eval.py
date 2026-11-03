# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone MachineDevBench (MDB) eval for the native Flamingo adapter.

Eval-glue that avoids the stopes/asyncio nested-event-loop issue hit when
running the multi-style ``machine_devbench_pipeline`` under a ``debug``
launcher (the pipeline module calls ``asyncio.run`` internally). It reuses the
**exact** repo scoring path (``FlamingoMultiModalAdapter.compute_similarity``,
via the same ``instantiate({"_target_": ...}, **kwargs)`` construction as
``machine_devbench/base.py``) and the **exact** repo aggregation
(``ResultAggregator`` + ``merge_style_results`` + ``build_summary``), so the
reported ``overall_agg`` matches the pipeline's pooled "Overall Agg" metric.

In addition to accuracy it records the log-likelihood **grounding delta** —
mean ``|ΔLL|`` of the (correct) caption between the correct vs. foil image —
which directly probes whether the model uses the image (a delta near 0 means
the caption likelihood is image-independent).

Decision rules mirror ``machine_devbench/base.py::_predict_batch`` (target 0 =
correct):
  * lexical (2 img, 1 txt): pred 0 iff ``LL(img_pos) > LL(img_neg)``.
  * grammatical (2 img, 2 txt): pred 0 iff
    ``LL(i0,ca)+LL(i1,cb) > LL(i0,cb)+LL(i1,ca)``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.utils import set_seed, setup_logging
from evaluation.data.machine_devbench import BenchmarkData
from evaluation.multimodal.machine_devbench.metrics import (
    ResultAggregator,
    build_summary,
    merge_style_results,
)

if TYPE_CHECKING:
    from apps.baselines.flamingo.multimodal_feature_extractor import FlamingoMultiModalAdapter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_CFG = Path(__file__).resolve().parents[3] / "evaluation" / "configs" / "model" / "flamingo.yaml"


def build_model(model_path: str, model_cfg_path: str, device: str = "cuda") -> FlamingoMultiModalAdapter:
    """Instantiate the FlamingoMultiModalAdapter exactly like the eval module."""
    cfg = OmegaConf.load(model_cfg_path)
    kwargs = OmegaConf.to_container(cfg.kwargs, resolve=True)
    kwargs["model_path"] = model_path
    model = instantiate({"_target_": cfg._target_}, **kwargs)
    model.eval()
    model.to(device)
    return model


def _sim_for_trial(model: FlamingoMultiModalAdapter, item: dict[str, Any]) -> np.ndarray:
    """Return the (N_img, N_txt) log-likelihood matrix for one trial."""
    features = model.extract_features({"image": item["image"], "text": item["text"]})
    sim = model.compute_similarity(features["image_features"], features["text_features"], normalize=True)
    return sim.detach().cpu().numpy()


def _predict_and_delta(task_name: str, sim: np.ndarray) -> tuple[int, float]:
    """Return (prediction, |ΔLL| correct-vs-foil-image) for one trial."""
    if task_name.startswith("lex_"):
        score_pos = float(sim[0, 0])
        score_neg = float(sim[1, 0])
        pred = 0 if score_pos > score_neg else 1
        delta = abs(score_pos - score_neg)
        return pred, delta
    # grammatical (2 img, 2 txt); correct pairing is the diagonal.
    matched = float(sim[0, 0] + sim[1, 1])
    mismatched = float(sim[0, 1] + sim[1, 0])
    pred = 0 if matched > mismatched else 1
    # image-dependence: hold each correct caption fixed, swap correct<->foil image.
    d_a = abs(float(sim[0, 0]) - float(sim[1, 0]))  # cap_a: img0 (correct) vs img1 (foil)
    d_b = abs(float(sim[1, 1]) - float(sim[0, 1]))  # cap_b: img1 (correct) vs img0 (foil)
    delta = 0.5 * (d_a + d_b)
    return pred, delta


def run(
    model_path: str,
    styles: list[str],
    data_root: str,
    model_cfg_path: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one checkpoint over all tasks/styles; return a results dict."""
    setup_logging()
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_path, model_cfg_path, device=device)

    per_style_results: dict[str, Any] = {}
    total_agg = ResultAggregator()

    deltas_all: list[float] = []
    deltas_lex: list[float] = []
    deltas_gram: list[float] = []
    n_trials_total = 0

    for style in styles:
        bench = BenchmarkData(data_root, style=style)
        tasks = bench.get_tasks()
        logger.info("style=%s tasks(%d)=%s", style, len(tasks), tasks)
        style_agg = ResultAggregator()
        for task_name in tasks:
            dataset = bench.build_dataset(task_name)
            with torch.no_grad():
                for idx in range(len(dataset)):
                    item = dataset[idx]
                    sim = _sim_for_trial(model, item)
                    pred, delta = _predict_and_delta(task_name, sim)
                    meta = item["metadata"]
                    style_agg.add(task_name, pred, 0, meta)
                    total_agg.add(task_name, pred, 0, meta)
                    deltas_all.append(delta)
                    (deltas_lex if task_name.startswith("lex_") else deltas_gram).append(delta)
                    n_trials_total += 1
            logger.info("  done task=%s style=%s", task_name, style)
        per_style_results[style] = style_agg.compute()

    total_results = total_agg.compute()
    merged = merge_style_results(per_style_results, total_results)
    summary = build_summary(merged)

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    grounding = {
        "mean_abs_delta_ll": _mean(deltas_all),
        "lexical": _mean(deltas_lex),
        "grammatical": _mean(deltas_gram),
        "n": len(deltas_all),
    }

    return {
        "model_path": model_path,
        "seed": seed,
        "styles": styles,
        "n_trials_total": n_trials_total,
        "overall_agg": summary["overall"],
        "total": summary["total"],
        "per_style": summary["per_style"],
        "grounding": grounding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Flamingo MDB eval.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Final-epoch .pt (paper: epoch_0009.pt) or a run checkpoint dir (picks latest.pt).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="MachineDevBench root (contains Lexical/ and Grammatical/).",
    )
    parser.add_argument("--styles", nargs="+", default=["realistic", "cartoon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cfg", default=str(_DEFAULT_MODEL_CFG))
    parser.add_argument("--out", required=True, help="Output JSON path.")
    args = parser.parse_args()

    results = run(
        model_path=args.model_path,
        styles=args.styles,
        data_root=args.data_root,
        model_cfg_path=args.model_cfg,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
