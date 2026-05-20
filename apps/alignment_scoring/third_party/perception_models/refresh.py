#!/usr/bin/env python
"""Refresh the in-tree snapshot of facebookresearch/perception_models.

Copies the closure of files transitively reachable from ``apps/plm/{tokenizer,
transformer}.py`` and from the PE-Core scoring loader into this directory,
rewriting all ``from core.X`` and ``from apps.X`` imports to point at the
bundled namespace.

Usage::

    python apps/alignment_scoring/third_party/perception_models/refresh.py \\
        --src /path/to/checked-out/perception_models

Run after bumping the SHA in the local README.md. The script writes stubs
for ``core/distributed.py`` (only ``get_is_master`` needed) and
``core/probe.py`` (only ``log_stats`` needed) instead of copying the heavy
originals.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

BUNDLED_FILES = (
    "apps/plm/__init__.py",
    "apps/plm/tokenizer.py",
    "apps/plm/transformer.py",
    "core/__init__.py",
    "core/args.py",
    "core/checkpoint.py",
    "core/tokenizer.py",
    "core/transformer.py",
    "core/utils.py",
    "core/data/__init__.py",
    "core/data/conversation.py",
    "core/transforms/__init__.py",
    "core/transforms/image_transform.py",
    "core/transforms/video_transform.py",
    "core/vision_encoder/__init__.py",
    "core/vision_encoder/config.py",
    "core/vision_encoder/pe.py",
    "core/vision_encoder/rope.py",
    "core/vision_encoder/tokenizer.py",
    "core/vision_encoder/transforms.py",
    "core/vision_projector/__init__.py",
    "core/vision_projector/base.py",
    "core/vision_projector/mlp.py",
)

# Non-Python data files loaded at runtime via __file__ (no import-rewrite).
BUNDLED_DATA_FILES = ("core/vision_encoder/bpe_simple_vocab_16e6.txt.gz",)

BUNDLED_PREFIX = "apps.alignment_scoring.third_party.perception_models"

DISTRIBUTED_STUB = '''"""Stub for upstream `core/distributed.py` — only `get_is_master` is needed."""

from __future__ import annotations

import torch.distributed as dist


def get_is_master() -> bool:
    """Return True if the current process is rank 0 (or no distributed init yet)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0
'''

PROBE_STUB = '''"""Stub for upstream `core/probe.py` — `log_stats` is a no-op for inference."""

from __future__ import annotations

import torch


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:  # noqa: ARG001
    """No-op replacement for the upstream tensor-stats logger."""
    return x
'''


def _rewrite_imports(text: str) -> str:
    """Rewrite ``from core.X``, ``from apps.X``, ``from core import X``, etc. into the bundled namespace."""
    # `from <pkg>.X import Y` and `from <pkg> import Y`.
    text = re.sub(
        r"^(from\s+)(core|apps)((?:\.[\w.]+)?\s+import)",
        rf"\1{BUNDLED_PREFIX}.\2\3",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(
        r"^(import\s+)(core|apps)(\.[\w.]+)?",
        rf"\1{BUNDLED_PREFIX}.\2\3",
        text,
        flags=re.MULTILINE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Checkout of perception_models repo.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    for rel in BUNDLED_FILES:
        src = args.src / rel
        dst = here / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"WARN: {src} not found; skipping")  # noqa: T201
            continue
        rewritten = _rewrite_imports(src.read_text())
        dst.write_text(rewritten)
        print(f"copied {rel}")  # noqa: T201

    for rel in BUNDLED_DATA_FILES:
        src = args.src / rel
        dst = here / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"WARN: data file {src} not found; skipping")  # noqa: T201
            continue
        shutil.copy(src, dst)
        print(f"copied data {rel}")  # noqa: T201

    # Stubbed files (originals are too heavy and pull in unused training-only deps).
    (here / "core/distributed.py").write_text(DISTRIBUTED_STUB)
    (here / "core/probe.py").write_text(PROBE_STUB)
    print("wrote stubs for core/distributed.py and core/probe.py")  # noqa: T201

    # Copy licenses too.
    for license_file in ("LICENSE.PE", "LICENSE.PLM"):
        src_license = args.src / license_file
        if src_license.exists():
            shutil.copy(src_license, here / license_file)


if __name__ == "__main__":
    main()
