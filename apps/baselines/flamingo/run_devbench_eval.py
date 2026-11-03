# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone (human-norm) DevBench eval for the native Flamingo adapter.

Companion to ``run_mdb_eval.py`` (which targets MachineDevBench). This scores a
**native** :class:`apps.baselines.flamingo.multimodal_feature_extractor.FlamingoMultiModalAdapter`
checkpoint on the four image-text-matching DevBench tasks
(``lex-lwl``, ``lex-viz_vocab``, ``gram-trog``, ``gram-winoground``).

It reuses the **exact** repo scoring path (``FlamingoMultiModalAdapter.
extract_features`` -> ``compute_similarity(normalize=True)``, built via the same
``instantiate({"_target_": ...}, **kwargs)`` as
``evaluation/multimodal/devbench/base.py``), the **exact** DevBench dataset
loader (:class:`evaluation.data.devbench.DevBenchDataset`), and the **exact**
accuracy definitions from ``evaluation/multimodal/devbench``:

  * ``lex-*`` / ``gram-trog`` (N_img, 1 txt): ``compute_accuracy(argmax==0)``
    (correct image is column 0), matching ``DevBench{LexLWL,LexVizVocab,
    GramTROG}EvalModule.compute_metrics``.
  * ``gram-winoground`` (2 img, 2 txt): the text-score accuracy from
    ``DevBenchGramWinogroundEvalModule._compute_winoground_accuracy``.

Running the full stopes ``DevBenchPipeline`` under a ``debug`` launcher hits the
nested-asyncio issue (``pipeline.py`` calls ``asyncio.run`` internally); this
glue side-steps that exactly like ``run_mdb_eval.py`` does for MDB, and loads
the (expensive) model once per checkpoint for all four tasks.

DevBench chance = mean(50, 25, 25, 50) = 37.5% (lwl/wg are 2-way, viz_vocab/trog
are 4-way). KL-divergence-to-human is intentionally *not* computed here (the
diagnostic target is accuracy; human-eval files are only partially present in
the assembled data root).
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
from evaluation.data.devbench import DevBenchDataset
from evaluation.multimodal.devbench.metrics import compute_accuracy

if TYPE_CHECKING:
    from apps.baselines.flamingo.multimodal_feature_extractor import FlamingoMultiModalAdapter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_CFG = Path(__file__).resolve().parents[3] / "evaluation" / "configs" / "model" / "flamingo.yaml"

# task_name -> (subfolder, scoring kind). Column 0 is the correct image for the
# argmax0 tasks (matches the manifests + the eval modules' correct_index=0).
_TASKS: dict[str, tuple[str, str]] = {
    "lex-lwl": ("lex-lwl", "argmax0"),
    "lex-viz_vocab": ("lex-viz_vocab", "argmax0"),
    "gram-trog": ("gram-trog", "argmax0"),
    "gram-winoground": ("gram-winoground", "winoground"),
}

# Per-task chance accuracy (n-way forced choice).
_CHANCE: dict[str, float] = {
    "lex-lwl": 50.0,
    "lex-viz_vocab": 25.0,
    "gram-trog": 25.0,
    "gram-winoground": 50.0,
}


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


def _winoground_accuracy(scores: np.ndarray) -> float:
    """Text-score accuracy, mirroring the Winoground eval module.

    ``scores`` has shape (n_trials, 2, 2) indexed ``[trial, image, text]``.
    """
    correct = 0
    for i in range(len(scores)):
        if scores[i, 0, 0] > scores[i, 0, 1]:
            correct += 1
        if scores[i, 1, 0] < scores[i, 1, 1]:
            correct += 1
    return correct / (len(scores) * 2)


def _eval_task(model: FlamingoMultiModalAdapter, task_name: str, task_dir: Path) -> dict[str, Any]:
    """Evaluate one DevBench task; return {accuracy, num_trials}."""
    _, kind = _TASKS[task_name]
    dataset = DevBenchDataset(task_dir)
    logger.info("task=%s trials=%d kind=%s", task_name, len(dataset), kind)

    with torch.no_grad():
        sims = [_sim_for_trial(model, dataset[idx]) for idx in range(len(dataset))]

    if kind == "winoground":
        scores = np.stack(sims, axis=0)  # (n, 2, 2)
        accuracy = _winoground_accuracy(scores)
    else:
        # (n_img, 1) -> (n_img,); stack -> (n_trials, n_img); argmax == 0.
        stacked = np.stack([s.squeeze(-1) if s.ndim == 2 and s.shape[1] == 1 else s for s in sims], axis=0)  # noqa: PLR2004
        accuracy = compute_accuracy(stacked, correct_index=0)

    return {"accuracy": float(accuracy), "num_trials": len(dataset)}


def run(
    model_path: str,
    data_root: str,
    model_cfg_path: str,
    seed: int,
    tasks: list[str],
) -> dict[str, Any]:
    """Evaluate one checkpoint over the DevBench tasks; return a results dict."""
    setup_logging()
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_path, model_cfg_path, device=device)

    root = Path(data_root)
    per_task: dict[str, Any] = {}
    for task_name in tasks:
        subfolder, _ = _TASKS[task_name]
        task_dir = root / subfolder
        if not (task_dir / "manifest.csv").exists():
            raise FileNotFoundError(f"Missing manifest for {task_name}: {task_dir / 'manifest.csv'}")
        per_task[task_name] = _eval_task(model, task_name, task_dir)
        logger.info("  done task=%s acc=%.4f", task_name, per_task[task_name]["accuracy"])

    accs = [per_task[t]["accuracy"] for t in tasks]
    overall = float(np.mean(accs)) if accs else None
    chance_overall = float(np.mean([_CHANCE[t] for t in tasks])) / 100.0

    return {
        "model_path": model_path,
        "seed": seed,
        "tasks": tasks,
        "per_task": per_task,
        "overall_accuracy": overall,
        "chance_per_task": {t: _CHANCE[t] / 100.0 for t in tasks},
        "chance_overall": chance_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Flamingo DevBench eval (accuracy).")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Final-epoch .pt (paper: epoch_0009.pt) or a run checkpoint dir (picks latest.pt).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="DevBench root (contains lex-lwl/, lex-viz_vocab/, gram-trog/, gram-winoground/).",
    )
    parser.add_argument("--tasks", nargs="+", default=list(_TASKS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_cfg", default=str(_DEFAULT_MODEL_CFG))
    parser.add_argument("--out", required=True, help="Output JSON path.")
    args = parser.parse_args()

    results = run(
        model_path=args.model_path,
        data_root=args.data_root,
        model_cfg_path=args.model_cfg,
        seed=args.seed,
        tasks=args.tasks,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
