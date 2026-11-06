# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-stage the COCO-MC MetaCLIP jpegs from the source tars to a scratch tree.

Why: ``coco_metaclip._MetaclipImageReader`` reads MetaCLIP jpegs via
``tarfile.extractfile()``. The 483,208 images needed by the COCO settings are
scattered across ~143k tars (~3.4 imgs/tar), so on-the-fly reads open/scan huge
tars for a handful of members each — the GPU sits at ~0% util (>10s/step). Once
the jpegs are extracted to ``{extracted_root}/{uuid[:2]}/{uuid}.jpeg`` the reader
opens them directly and training becomes GPU-bound.

All 15 COCO jobs (5 permutation settings x 3 seeds) reuse the SAME image set —
permutations only shuffle *captions*, never images — so this is a one-time,
shared stage. Stage once to a persistent root and point every COCO job's
``--extracted_root`` at ``{scratch_root}/extracted``.

Reads the ``coco_to_metaclip.jsonl`` manifest directly (one row per coco_id with
a ``matches`` array). If ``$COCO_MC_TAR_INDEX_ROOT`` points at a precomputed tar
offset index (``{root}/{shard_id%100}/{shard_id}.idx``, JSON
``{uuid: [offset, size]}``), wanted members are read with seek+read; otherwise
each tar is scanned sequentially.

Output layout (matches ``_MetaclipImageReader.read``)::

    {scratch_root}/extracted/{uuid[:2]}/{uuid}.jpeg

Usage::

    # Full stage to shared persistent scratch (one-time; ~40 GB, 483k jpegs):
    python -m apps.baselines.egovlp.data.stage_metaclip \\
        --manifest $COCO_MC_MANIFEST --scratch_root /path/to/scratch/coco_metaclip

    # Stage a small subset for a throughput smoke test:
    python -m apps.baselines.egovlp.data.stage_metaclip \\
        --manifest $COCO_MC_MANIFEST --scratch_root /tmp/mc_subset --limit 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = os.environ.get("COCO_MC_MANIFEST")
#: Optional precomputed tar offset index (see module docstring).
TAR_INDEX_ROOT = os.environ.get("COCO_MC_TAR_INDEX_ROOT")


def _write_atomic(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` via a temp file so readers never see partial jpegs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jpeg.tmp")
    with tmp.open("wb") as out:
        out.write(data)
    tmp.replace(dest)


def _extract_with_index(tar_path: str, idx_path: Path, todo: list[tuple[str, Path]]) -> tuple[int, int]:
    """Seek+read the wanted members using a ``{uuid: [offset, size]}`` index."""
    n_extracted = n_failed = 0
    with idx_path.open() as f:
        offsets = json.load(f)
    with Path(tar_path).open("rb") as raw:
        for uuid, dest in todo:
            info = offsets.get(uuid)
            if info is None:
                n_failed += 1
                continue
            off, sz = info
            raw.seek(off)
            _write_atomic(dest, raw.read(sz))
            n_extracted += 1
    return n_extracted, n_failed


def _extract_by_scan(tar_path: str, todo: list[tuple[str, Path]]) -> tuple[int, int]:
    """Scan the tar sequentially (member names may be ``uuid.jpeg`` or ``./uuid.jpeg``)."""
    n_extracted = n_failed = 0
    wanted = {f"{u}.jpeg": dest for u, dest in todo}
    wanted.update({f"./{u}.jpeg": dest for u, dest in todo})
    with tarfile.open(tar_path, mode="r") as tar:
        for member in tar:
            dest = wanted.get(member.name)
            if dest is None:
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    n_failed += 1
                    continue
                _write_atomic(dest, f.read())
                n_extracted += 1
            except (OSError, tarfile.TarError):
                n_failed += 1
    return n_extracted, n_failed


def _shard_extract_worker(args: tuple[str, int, list[str], str]) -> tuple[int, int, int]:
    """Extract one tar's needed uuids into the flat hashed scratch tree.

    Uses ``{TAR_INDEX_ROOT}/{shard_id%100}/{shard_id}.idx`` when available to
    seek+read only the wanted bytes; otherwise falls back to a sequential
    ``tarfile`` scan. Already-staged files are counted as extracted.
    Returns ``(n_extracted, n_failed, n_tar_failed)``.
    """
    tar_path, shard_id, uuids, out_root = args
    root = Path(out_root)
    todo: list[tuple[str, Path]] = []
    n_present = 0
    for uuid in uuids:
        dest = root / uuid[:2] / f"{uuid}.jpeg"
        if dest.exists():
            n_present += 1
        else:
            todo.append((uuid, dest))
    if not todo:
        return (n_present, 0, 0)

    idx_path = Path(TAR_INDEX_ROOT) / f"{shard_id % 100}" / f"{shard_id}.idx" if TAR_INDEX_ROOT else None
    try:
        if idx_path is not None and idx_path.exists():
            n_extracted, n_failed = _extract_with_index(tar_path, idx_path, todo)
        else:
            n_extracted, n_failed = _extract_by_scan(tar_path, todo)
    except (OSError, tarfile.TarError, json.JSONDecodeError):
        return (n_present, len(todo), 1)
    return (n_present + n_extracted, n_failed, 0)


def _by_shard(manifest_path: str, limit: int | None) -> dict[tuple[str, int], list[str]]:
    """Group the manifest's MetaCLIP matches by (tar_path, shard_id) -> uuids.

    ``limit`` caps the number of *unique* uuids collected (subset staging for
    throughput smoke tests); shards are kept whole so their idx is read once.
    """
    by_shard: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[str] = set()
    with Path(manifest_path).open() as f:
        for line in f:
            row = json.loads(line)
            for m in row["matches"]:
                uuid = m["uuid"]
                if uuid in seen:
                    continue
                seen.add(uuid)
                by_shard[(m["tar_path"], int(m["shard_id"]))].append(uuid)
                if limit is not None and len(seen) >= limit:
                    return by_shard
    return by_shard


def stage_all(
    scratch_root: str,
    manifest_path: str,
    num_workers: int | None = None,
    limit: int | None = None,
) -> str:
    """Extract every needed MetaCLIP jpeg into ``{scratch_root}/extracted``.

    Idempotent: a marker file ``extracted/_done`` (written only when every image
    was staged) short-circuits a re-run for the same ``(manifest, limit)``. Returns the ``extracted`` dir (pass this as
    ``--extracted_root``).
    """
    manifest_path = str(manifest_path)
    out_root = Path(scratch_root) / "extracted"
    out_root.mkdir(parents=True, exist_ok=True)
    marker = out_root / "_done"
    manifest_mtime = Path(manifest_path).stat().st_mtime
    expected = f"{manifest_path}\t{manifest_mtime}\tlimit={limit}"
    if marker.exists():
        with marker.open() as f:
            if f.read().strip() == expected.strip():
                logger.info("[stage] marker found at %s, skipping", marker)
                return str(out_root)

    logger.info("[stage] reading manifest %s (limit=%s)", manifest_path, limit)
    by_shard = _by_shard(manifest_path, limit)
    n_uuids = sum(len(v) for v in by_shard.values())
    logger.info("[stage] %d shards, %d uuids to stage", len(by_shard), n_uuids)

    if num_workers is None:
        num_workers = min(60, max(1, (os.cpu_count() or 8) - 2))
    logger.info("[stage] launching %d worker processes", num_workers)

    tasks = [(tar_path, shard_id, uuids, str(out_root)) for (tar_path, shard_id), uuids in by_shard.items()]
    t0 = time.time()
    n_extracted = n_failed = n_tar_failed = 0
    with mp.get_context("spawn").Pool(num_workers) as pool:
        for i, (e, fc, tf) in enumerate(pool.imap_unordered(_shard_extract_worker, tasks, chunksize=8)):
            n_extracted += e
            n_failed += fc
            n_tar_failed += tf
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / max(rate, 1e-6)
                logger.info(
                    "[stage] shard %d/%d  extracted=%d failed=%d tar_failed=%d  rate=%.1f shard/s  eta=%.1f min",
                    i + 1,
                    len(tasks),
                    n_extracted,
                    n_failed,
                    n_tar_failed,
                    rate,
                    eta / 60,
                )

    elapsed = time.time() - t0
    logger.info(
        "[stage] DONE in %.1f min: extracted=%d failed=%d tar_failed=%d",
        elapsed / 60,
        n_extracted,
        n_failed,
        n_tar_failed,
    )
    if n_failed or n_tar_failed:
        # Leave no marker so a re-run retries the missing images (already-staged
        # files are skipped, so the retry is cheap).
        logger.warning(
            "[stage] %d images / %d tars failed; re-run to retry (no _done marker written)", n_failed, n_tar_failed
        )
        return str(out_root)
    with marker.open("w") as f:
        f.write(expected)
    return str(out_root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-stage COCO-MC MetaCLIP jpegs to scratch.")
    p.add_argument(
        "--scratch_root",
        required=True,
        help="Parent dir; jpegs land in {scratch_root}/extracted/{uuid[:2]}/{uuid}.jpeg. "
        "Pass {scratch_root}/extracted as the trainer's --extracted_root.",
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        required=DEFAULT_MANIFEST is None,
        help="coco_to_metaclip.jsonl path (default: $COCO_MC_MANIFEST).",
    )
    p.add_argument("--num_workers", type=int, default=None, help="Extraction processes (default: min(60, cpus-2)).")
    p.add_argument("--limit", type=int, default=None, help="Cap unique uuids staged (subset smoke tests).")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    out = stage_all(
        args.scratch_root,
        manifest_path=args.manifest,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    logger.info("[stage] pass --extracted_root %s", out)


if __name__ == "__main__":
    main()
