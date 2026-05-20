#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download all six DevBench tasks our pipeline runs.

Output layout matches what :class:`evaluation.data.devbench.DevBenchDataset`
expects: ``<cache_dir>/<task>/`` containing ``manifest.csv``, the image assets
referenced by the manifest, and any human-eval file the task uses.

Sources:

- ``sem-things``: OSF (THINGS images + spose_similarity.mat).
- ``gram-trog``: GitHub levante-framework/core-tasks (images via API listing).
- ``gram-winoground``: HuggingFace facebook/winoground (images.zip + human.jsonl).
  Requires ``huggingface-cli login`` because the dataset is gated (auto-approved).
- ``lex-lwl``, ``sem-viz_obj_cat``: bundled in the alvinwmtan/dev-bench repo.
- ``lex-viz_vocab``: reuses ``sem-things`` THINGS images via symlink.

Manifests and human-eval files for all tasks come from the alvinwmtan/dev-bench
repo, which is cloned into ``<cache_dir>/_repo/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from scripts.eval_data._common import (
    announce,
    cache_argparser,
    is_already_downloaded,
    mark_downloaded,
    setup_logging,
)

logger = logging.getLogger(__name__)

DEVBENCH_REPO_URL = "https://github.com/alvinwmtan/dev-bench.git"
THINGS_IMAGES_URL = "https://osf.io/download/wb36u/"
THINGS_SPOSE_URL = "https://osf.io/download/w75eu/"
TROG_API_URL = (
    "https://api.github.com/repos/levante-framework/core-tasks/contents/"
    "assets/TROG/original?ref=1dba11a50621186daad2ccc4dc2943f1536fb6db"
)
WINOGROUND_HF_REPO = "facebook/winoground"


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``."""
    logger.info("Downloading %s -> %s", url, dest)
    if shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
    else:
        urllib.request.urlretrieve(url, dest)  # noqa: S310


def _unzip(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Unzipping %s -> %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _clone_repo(repo_dir: Path) -> None:
    logger.info("Cloning %s -> %s", DEVBENCH_REPO_URL, repo_dir)
    subprocess.run(["git", "clone", "--depth=1", DEVBENCH_REPO_URL, str(repo_dir)], check=True)


def _setup_sem_things(repo_dir: Path, task_dir: Path) -> None:
    """Download THINGS images + spose_similarity.mat."""
    task_dir.mkdir(parents=True, exist_ok=True)
    images_zip = task_dir / "_things_assets.zip"
    _download(THINGS_IMAGES_URL, images_zip)
    _unzip(images_zip, task_dir)
    images_zip.unlink()

    _download(THINGS_SPOSE_URL, task_dir / "spose_similarity.mat")
    shutil.copy(repo_dir / "assets" / "sem-things" / "manifest.csv", task_dir / "manifest.csv")


def _setup_gram_trog(repo_dir: Path, task_dir: Path) -> None:
    """Download TROG images via the levante-framework GitHub API listing.

    The API at the pinned commit returns 416 of the 424 images the manifest
    references; the remaining 8 (``67-*``, ``68-*``) only ship with the
    devbench repo bundle. We copy the bundled set after the API listing to
    cover those.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    images_dir = task_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Listing TROG images from GitHub API")
    with urllib.request.urlopen(TROG_API_URL) as r:  # noqa: S310
        listing = json.load(r)

    for entry in listing:
        url = entry["download_url"]
        _download(url, images_dir / entry["name"])

    bundled_images = repo_dir / "assets" / "gram-trog" / "images"
    if bundled_images.is_dir():
        for img in bundled_images.iterdir():
            shutil.copy(img, images_dir / img.name)
        logger.info("Copied %d bundled TROG images", sum(1 for _ in bundled_images.iterdir()))

    shutil.copy(repo_dir / "assets" / "gram-trog" / "manifest.csv", task_dir / "manifest.csv")
    shutil.copy(repo_dir / "evals" / "gram-trog" / "human.csv", task_dir / "human.csv")


def _setup_gram_winoground(repo_dir: Path, task_dir: Path) -> None:
    """Download Winoground via HF (gated; requires ``huggingface-cli login``)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is required for Winoground") from e

    task_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Winoground from HF (must be `huggingface-cli login`'d)")
    try:
        images_zip = hf_hub_download(
            repo_id=WINOGROUND_HF_REPO,
            filename="data/images.zip",
            repo_type="dataset",
        )
        human_jsonl = hf_hub_download(
            repo_id=WINOGROUND_HF_REPO,
            filename="statistics/model_scores/human.jsonl",
            repo_type="dataset",
        )
    except Exception as e:
        logger.error(
            "Winoground download failed: %s. The dataset is gated; run `huggingface-cli login` "
            "and accept the access terms at https://huggingface.co/datasets/facebook/winoground.",
            e,
        )
        raise

    _unzip(Path(images_zip), task_dir)
    shutil.copy(human_jsonl, task_dir / "winoground-human.jsonl")
    shutil.copy(repo_dir / "assets" / "gram-winoground" / "manifest.csv", task_dir / "manifest.csv")


def _setup_lex_viz_vocab(repo_dir: Path, task_dir: Path, sem_things_dir: Path) -> None:
    """Reuse the THINGS images from sem-things via a symlink."""
    task_dir.mkdir(parents=True, exist_ok=True)
    things_images = sem_things_dir / "object_images_CC0"
    if not things_images.is_dir():
        raise RuntimeError(f"lex-viz_vocab needs sem-things images at {things_images}; download sem-things first.")

    images_link = task_dir / "images"
    if images_link.is_symlink() or images_link.exists():
        images_link.unlink()
    os.symlink(things_images.resolve(), images_link)
    shutil.copy(repo_dir / "assets" / "lex-viz_vocab" / "manifest.csv", task_dir / "manifest.csv")
    shutil.copy(repo_dir / "evals" / "lex-viz_vocab" / "human.csv", task_dir / "human.csv")


def _setup_lex_lwl(repo_dir: Path, task_dir: Path) -> None:
    """Copy the bundled adams/donnelly/frank image dirs + manifest + human file."""
    task_dir.mkdir(parents=True, exist_ok=True)
    src = repo_dir / "assets" / "lex-lwl"
    for sub in ("images_adams", "images_donnelly", "images_frank"):
        if (task_dir / sub).exists():
            shutil.rmtree(task_dir / sub)
        shutil.copytree(src / sub, task_dir / sub)
    shutil.copy(src / "manifest.csv", task_dir / "manifest.csv")
    shutil.copy(repo_dir / "evals" / "lex-lwl" / "human.csv", task_dir / "human.csv")


def _setup_sem_viz_obj_cat(repo_dir: Path, task_dir: Path) -> None:
    """Copy the bundled images dir + manifest + human.rds."""
    task_dir.mkdir(parents=True, exist_ok=True)
    src = repo_dir / "assets" / "sem-viz_obj_cat"
    if (task_dir / "images").exists():
        shutil.rmtree(task_dir / "images")
    shutil.copytree(src / "images", task_dir / "images")
    shutil.copy(src / "manifest.csv", task_dir / "manifest.csv")
    shutil.copy(repo_dir / "evals" / "sem-viz_obj_cat" / "human.rds", task_dir / "human.rds")


TASKS = ("sem-things", "gram-trog", "gram-winoground", "lex-viz_vocab", "lex-lwl", "sem-viz_obj_cat")


def main() -> None:
    parser: argparse.ArgumentParser = cache_argparser("devbench")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=TASKS,
        help="Subset of DevBench tasks to download (default: all).",
    )
    args = parser.parse_args()
    setup_logging()

    if is_already_downloaded(args.cache_dir) and not args.force:
        logger.info("DevBench already present at %s; pass --force to redownload.", args.cache_dir)
        announce(args.cache_dir, "DevBench", env_var="DEVBENCH_DATA_ROOT")
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = args.cache_dir / "_repo"
    if repo_dir.exists() and args.force:
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        _clone_repo(repo_dir)

    sem_things_dir = args.cache_dir / "sem-things"

    # Order matters: lex-viz_vocab depends on sem-things being present first.
    ordered = sorted(args.tasks, key=lambda t: 0 if t == "sem-things" else 1)

    for task in ordered:
        task_dir = args.cache_dir / task
        logger.info("--- Setting up %s ---", task)
        try:
            if task == "sem-things":
                _setup_sem_things(repo_dir, task_dir)
            elif task == "gram-trog":
                _setup_gram_trog(repo_dir, task_dir)
            elif task == "gram-winoground":
                _setup_gram_winoground(repo_dir, task_dir)
            elif task == "lex-viz_vocab":
                _setup_lex_viz_vocab(repo_dir, task_dir, sem_things_dir)
            elif task == "lex-lwl":
                _setup_lex_lwl(repo_dir, task_dir)
            elif task == "sem-viz_obj_cat":
                _setup_sem_viz_obj_cat(repo_dir, task_dir)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to set up %s: %s", task, e)
            sys.exit(1)

    if set(args.tasks) == set(TASKS):
        mark_downloaded(args.cache_dir)
    announce(args.cache_dir, "DevBench", env_var="DEVBENCH_DATA_ROOT")


if __name__ == "__main__":
    main()
