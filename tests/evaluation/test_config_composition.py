# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Hydra config composition and override behavior in :mod:`evaluation`.

These tests verify that the eval entry point composes configs correctly:
- Top-level ``model=`` selects from the model config group (no ``+`` prefix needed).
- ``${model}`` interpolation flows into single-task ``eval.model``.
- Pipeline ``eval=`` configs propagate ``model``, ``seed``, ``output_dir`` into all child tasks.
- Per-task overrides (e.g. ``eval.tasks.knn_imagenet.seed=99``) take effect without affecting siblings.

Tests use :func:`hydra.compose` against the real ``evaluation/configs`` directory and
the ConfigStore registrations from :mod:`evaluation.eval_launcher`. Pipeline configs
contain ``${hydra:sweep.dir}`` interpolations in launcher fields that require an active
HydraConfig (set by ``@hydra.main``); helpers below access fields without a full resolve
to keep tests self-contained.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Importing the launcher registers all eval/model groups with Hydra's ConfigStore.
import evaluation.eval_launcher  # noqa: F401

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "evaluation" / "configs")

# Dataset-path env vars our YAMLs interpolate. We set placeholder values so config
# composition succeeds even when the operator hasn't downloaded the actual data.
DUMMY_ENV_VARS = {
    "MNIST_ROOT": "/tmp/dummy/mnist",
    "IMAGENET_ROOT": "/tmp/dummy/imagenet",
    "IMAGENET_EXTRA": "/tmp/dummy/imagenet_extra",
    "COUNTBENCH_ROOT": "/tmp/dummy/countbench",
    "COCOSTUFF_ROOT": "/tmp/dummy/cocostuff",
    "NYU_ROOT": "/tmp/dummy/nyu",
    "DEVBENCH_DATA_ROOT": "/tmp/dummy/devbench",
    "MACHINE_DEVBENCH_DATA_ROOT": "/tmp/dummy/machine_devbench",
    "ZORRO_DATA_ROOT": "/tmp/dummy/zorro",
    "LTSWAP_DATA_ROOT": "/tmp/dummy/ltswap",
}


@pytest.fixture(autouse=True)
def _stub_dataset_env() -> Iterator[None]:
    """Set placeholder dataset paths so ``${oc.env:...}`` interpolations resolve."""
    saved = {k: os.environ.get(k) for k in DUMMY_ENV_VARS}
    os.environ.update(DUMMY_ENV_VARS)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _compose(*overrides: str) -> DictConfig:
    """Compose the launcher config with the given Hydra overrides."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=list(overrides))


def _resolve_node(cfg: DictConfig, *path: str) -> Any:  # noqa: ANN401
    """Resolve a single node by path, isolating it from unrelated unresolved interpolations.

    Pipeline configs contain ``${hydra:sweep.dir}`` interpolations under ``launcher`` that
    require an active ``HydraConfig`` (only set by ``@hydra.main``). To verify per-task
    fields without initializing Hydra fully, we extract the subtree and resolve only it.
    """
    node = cfg
    for key in path:
        node = node[key]
    return OmegaConf.to_container(node, resolve=True)


# ---- model group selection -------------------------------------------------


def test_model_selection_without_plus_prefix() -> None:
    """``model=bert_base`` resolves the model group via the new defaults entry."""
    cfg = _compose("eval=text/zorro", "model=bert_base")
    assert cfg.model._target_ == "lm_eval.models.huggingface.AutoMaskedLM"
    assert cfg.model.name == "bert-base-uncased"


def test_legacy_plus_model_is_rejected() -> None:
    """``+model=`` is rejected after the defaults entry is added.

    The new ``- model: null`` defaults entry makes ``model`` part of the defaults
    list, so the ``+model=`` *append* syntax fails with a "Multiple values for
    model" error. Users must drop the ``+`` and use plain ``model=foo``.
    """
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException, match="Multiple values for model"):
        _compose("eval=text/zorro", "+model=gpt2")


def test_model_defaults_to_null_when_omitted() -> None:
    """No ``model=`` selection leaves the top-level model field as ``None``."""
    cfg = _compose("eval=vision/knn_imagenet")
    assert cfg.model is None


@pytest.mark.parametrize(
    ("model_name", "expected_target"),
    [
        ("bert_base", "lm_eval.models.huggingface.AutoMaskedLM"),
        ("gpt2", "apps.baselines.lm_training.eval.FastAutoCausalLM"),
        ("roberta-base", "lm_eval.models.huggingface.AutoMaskedLM"),
        ("dino", "apps.baselines.dinov2.extractor.DINOv2FeatureExtractor"),
        ("dino_reg", "apps.baselines.dinov2.extractor.DINOv2FeatureExtractor"),
        ("clip_image", "apps.baselines.clip.openclip_extractor.CLIPImageFeatureExtractor"),
        ("clip_text", "apps.baselines.clip.openclip_extractor.CLIPTextFeatureExtractor"),
        ("clip_example", "apps.baselines.clip.openclip_extractor.CLIPFeatureExtractor"),
        ("pe", "apps.baselines.clip.openclip_extractor.CLIPFeatureExtractor"),
    ],
)
def test_all_shipped_model_yamls_resolve(model_name: str, expected_target: str) -> None:
    """Every YAML in ``configs/model/`` selects a recognized ``_target_``."""
    cfg = _compose("eval=text/zorro", f"model={model_name}")
    assert cfg.model._target_ == expected_target


# ---- single-eval interpolation --------------------------------------------


def test_single_eval_model_flows_into_eval_model() -> None:
    """``${model}`` in zorro.yaml resolves to the top-level ``model`` selection."""
    cfg = _compose("eval=text/zorro", "model=bert_base")
    eval_node = _resolve_node(cfg, "eval")
    assert eval_node["model"]["_target_"] == "lm_eval.models.huggingface.AutoMaskedLM"
    assert eval_node["model"]["name"] == "bert-base-uncased"


def test_single_eval_seed_and_output_dir_overrides() -> None:
    """Top-level ``eval.seed=`` and ``eval.output_dir=`` overrides take effect."""
    cfg = _compose(
        "eval=text/zorro",
        "model=bert_base",
        "eval.seed=7",
        "eval.output_dir=/tmp/eval-out",
    )
    eval_node = _resolve_node(cfg, "eval")
    assert eval_node["seed"] == 7
    assert eval_node["output_dir"] == "/tmp/eval-out"


# ---- pipeline propagation -------------------------------------------------


VISION_CHILD_TASKS = (
    "knn_imagenet",
    "linear_imagenet",
    "abx_imagenet",
    "linear_countbench",
    "abx_countbench",
    "linear_mnist",
    "abx_mnist",
    "semantic_seg",
    "depth",
)

DEVBENCH_CHILD_TASKS = (
    "gram_trog",
    "gram_winoground",
    "lex_lwl",
    "lex_viz_vocab",
    "sem_things",
    "sem_viz_obj_cat",
)

TEXT_CHILD_TASKS = ("zorro", "ltswap")


@pytest.mark.parametrize(
    ("eval_name", "child_tasks"),
    [
        ("vision/vision_pipeline", VISION_CHILD_TASKS),
        ("multimodal/devbench_pipeline", DEVBENCH_CHILD_TASKS),
        ("text/text_pipeline", TEXT_CHILD_TASKS),
    ],
)
def test_pipeline_propagates_seed_and_output_dir(eval_name: str, child_tasks: tuple[str, ...]) -> None:
    """Pipeline-level ``eval.seed`` and ``eval.output_dir`` reach every child task."""
    cfg = _compose(
        f"eval={eval_name}",
        "model=bert_base",
        "eval.seed=7",
        "eval.output_dir=/tmp/eval-out",
    )
    tasks = _resolve_node(cfg, "eval", "tasks")
    for task in child_tasks:
        node = tasks[task]
        assert node["seed"] == 7, f"{eval_name}.{task} seed not propagated"
        assert node["output_dir"] == "/tmp/eval-out", f"{eval_name}.{task} output_dir not propagated"


@pytest.mark.parametrize(
    ("eval_name", "child_tasks", "model_field"),
    [
        ("vision/vision_pipeline", VISION_CHILD_TASKS[:-2], "model"),  # knn/linear/abx use `model`
        ("multimodal/devbench_pipeline", DEVBENCH_CHILD_TASKS, "model"),
        ("text/text_pipeline", TEXT_CHILD_TASKS, "model"),
    ],
)
def test_pipeline_propagates_model(eval_name: str, child_tasks: tuple[str, ...], model_field: str) -> None:
    """Top-level ``model=`` selection reaches every child task that uses ``${model}``."""
    cfg = _compose(f"eval={eval_name}", "model=bert_base")
    tasks = _resolve_node(cfg, "eval", "tasks")
    for task in child_tasks:
        node = tasks[task]
        assert node[model_field]["_target_"] == "lm_eval.models.huggingface.AutoMaskedLM", (
            f"{eval_name}.{task}.{model_field} not bound to top-level model"
        )


def test_vision_pipeline_depth_uses_backbone_field() -> None:
    """The depth task references the model via ``backbone:`` (not ``model:``)."""
    cfg = _compose("eval=vision/vision_pipeline", "model=bert_base")
    depth = _resolve_node(cfg, "eval", "tasks", "depth")
    assert depth["backbone"]["_target_"] == "lm_eval.models.huggingface.AutoMaskedLM"


# ---- per-task overrides ---------------------------------------------------


def test_per_task_seed_override_does_not_affect_siblings() -> None:
    """Overriding one child task's seed leaves all other children at the pipeline default."""
    cfg = _compose(
        "eval=vision/vision_pipeline",
        "model=bert_base",
        "eval.seed=42",
        "eval.tasks.knn_imagenet.seed=99",
    )
    tasks = _resolve_node(cfg, "eval", "tasks")
    assert tasks["knn_imagenet"]["seed"] == 99
    for task in [t for t in VISION_CHILD_TASKS if t != "knn_imagenet"]:
        assert tasks[task]["seed"] == 42, f"sibling {task} accidentally inherited the overridden seed"


def test_per_task_output_dir_override_does_not_affect_siblings() -> None:
    """Same isolation as above for ``output_dir``."""
    cfg = _compose(
        "eval=vision/vision_pipeline",
        "model=bert_base",
        "eval.output_dir=/tmp/all",
        "eval.tasks.linear_imagenet.output_dir=/tmp/just-linear",
    )
    tasks = _resolve_node(cfg, "eval", "tasks")
    assert tasks["linear_imagenet"]["output_dir"] == "/tmp/just-linear"
    for task in [t for t in VISION_CHILD_TASKS if t != "linear_imagenet"]:
        assert tasks[task]["output_dir"] == "/tmp/all"


# ---- MachineDevBench ------------------------------------------------------


def test_machine_devbench_single_style_composes() -> None:
    """Single-style ``machine_devbench`` config resolves with the model interpolation."""
    cfg = _compose("eval=multimodal/machine_devbench", "model=clip_image")
    eval_node = _resolve_node(cfg, "eval")
    assert eval_node["_target_"] == "evaluation.multimodal.machine_devbench.base.MachineDevBenchEvalModule"
    assert eval_node["style"] == "realistic"
    assert eval_node["data_root"] == "/tmp/dummy/machine_devbench"
    assert eval_node["model"]["_target_"] == "apps.baselines.clip.openclip_extractor.CLIPImageFeatureExtractor"


def test_machine_devbench_pipeline_composes_and_propagates_model() -> None:
    """Pipeline config resolves and the top-level model flows into ``task_eval_config.model``.

    Note: MachineDevBench pipeline doesn't use Hydra's per-task ``defaults`` like
    DevBench. Instead the per-style task config is built at runtime from a single
    ``task_eval_config`` template, so we only need to verify the template + styles.
    """
    cfg = _compose("eval=multimodal/machine_devbench_pipeline", "model=clip_image")
    # Resolve sub-nodes individually to skip the launcher's ``${hydra:sweep.dir}``
    # interpolation (only set by ``@hydra.main``).
    assert cfg.eval._target_ == "evaluation.multimodal.machine_devbench.pipeline.MachineDevBenchPipeline"
    styles = _resolve_node(cfg, "eval", "styles")
    assert styles == ["realistic", "cartoon"]
    template = _resolve_node(cfg, "eval", "task_eval_config")
    assert template["_target_"] == "evaluation.multimodal.machine_devbench.base.MachineDevBenchEvalModule"
    model = _resolve_node(cfg, "eval", "model")
    assert model["_target_"] == "apps.baselines.clip.openclip_extractor.CLIPImageFeatureExtractor"
