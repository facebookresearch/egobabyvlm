#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare NYUv2 depth-estimation data into the standard monocular-depth layout.

Output (under ``--out``) is the per-scene ``rgb_<NNNNN>.jpg`` +
``sync_depth_<NNNNN>.png`` (uint16 mm) layout consumed by AdaBins, NeWCRFs,
ZoeDepth, DPT, DepthAnything, and DINOv3's ``DATASETS.md``.

The depth-projection math runs through the NYU Depth Toolbox V2's
``project_depth_map`` via Octave; Python handles 16-bit PGM I/O (Octave's
GraphicsMagick is Q8-only) and parallelizes across frames.

User must obtain ``nyu_depth_v2_raw.zip`` (~399 GB) from
http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/ first, OR pre-extract
the scenes locally and pass ``--raw-dir``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

from scripts.eval_data._common import DEFAULT_DATA_ROOT, announce, setup_logging

logger = logging.getLogger(__name__)

NYU_TOOLBOX_URL = "https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip"
NYU_LABELED_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLITS_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"

#: Kinect RGB focal length from the NYU toolbox calibration.
FX_RGB = 518.8579

#: Sample one frame per N raw depth captures.
SAMPLE_STEP = 7

#: Truncate scene-list previews at this length.
SCENE_PREVIEW_LIMIT = 5

#: Frame the toolbox flags as faulty.
FAULTY_DEPTH = "d-1315166703.129542-2466101449.pgm"

#: JPEG quality for written RGB images.
TRAIN_JPEG_QUALITY = 75
TEST_JPEG_QUALITY = 95

OCTAVE_BRIDGE = """% Bridge: load (rgb, depth_raw), project, save (depth_uint16, rgb_undist).
inputs = load(input_mat);
[imgDepthProj, imgRgbUd] = project_depth_map(inputs.imgDepthRaw, inputs.imgRgb);
imgDepthProj = uint16(imgDepthProj * 1000.0);
save('-v7', output_mat, 'imgDepthProj', 'imgRgbUd');
"""


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("[skip] %s already present", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _fetch_toolbox(work_dir: Path) -> Path:
    toolbox_zip = work_dir / "toolbox_nyu_depth_v2.zip"
    _download(NYU_TOOLBOX_URL, toolbox_zip)
    toolbox_dir = work_dir / "nyu_toolbox"
    if not (toolbox_dir / "project_depth_map.m").exists():
        toolbox_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(toolbox_zip) as zf:
            zf.extractall(toolbox_dir)
    return toolbox_dir


def _read_scene_filenames(scene_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each depth frame with its temporally-nearest RGB frame, then subsample."""

    def parse_ts(name: str) -> float:
        return float(name[2:].split("-", 2)[0])

    depths = sorted(p for p in scene_dir.iterdir() if p.name.startswith("d-"))
    rgbs = sorted(p for p in scene_dir.iterdir() if p.name.startswith("r-"))
    if not depths or not rgbs:
        msg = f"no Kinect frames in {scene_dir}"
        raise FileNotFoundError(msg)

    rgb_ts = [parse_ts(p.name) for p in rgbs]
    pairs: list[tuple[Path, Path]] = []
    j = 0
    for d in depths:
        td = parse_ts(d.name)
        while j < len(rgbs) - 1 and abs(rgb_ts[j + 1] - td) <= abs(rgb_ts[j] - td):
            j += 1
        pairs.append((d, rgbs[j]))
    return pairs[::SAMPLE_STEP]


def _process_one_frame(args: tuple[Path, Path, Path, str, int, str]) -> tuple[int, str | None]:
    depth_pgm, rgb_ppm, save_dir, octave_runner, ind, toolbox_dir = args
    if depth_pgm.name == FAULTY_DEPTH:
        return ind, "faulty-skip"

    depth_raw = cv2.imread(str(depth_pgm), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.dtype != np.uint16:
        return ind, f"depth read failed: {depth_pgm}"
    # NYU PGMs are big-endian; cv2 reads them in native (LE on x86) byte order.
    depth_raw = depth_raw.byteswap()

    rgb_bgr = cv2.imread(str(rgb_ppm), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None:
        return ind, f"rgb read failed: {rgb_ppm}"
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    with tempfile.TemporaryDirectory() as td:
        in_mat = Path(td) / "in.mat"
        out_mat = Path(td) / "out.mat"
        savemat(in_mat, {"imgRgb": rgb, "imgDepthRaw": depth_raw})
        cmd = [
            octave_runner,
            "--no-gui",
            "--eval",
            f"input_mat = '{in_mat}'; output_mat = '{out_mat}'; "
            + (Path(__file__).parent / "_nyu_octave_bridge.m").read_text(),
        ]
        env = os.environ.copy()
        env["OCTAVE_PATH"] = str(toolbox_dir) + os.pathsep + env.get("OCTAVE_PATH", "")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            return ind, f"octave failed: {result.stderr[-400:]}"
        if not out_mat.exists():
            return ind, "octave produced no output"
        out = loadmat(out_mat)

    Image.fromarray(out["imgDepthProj"]).save(save_dir / f"sync_depth_{ind:05d}.png")
    Image.fromarray(out["imgRgbUd"]).save(save_dir / f"rgb_{ind:05d}.jpg", quality=TRAIN_JPEG_QUALITY)
    return ind, None


def _process_scene(
    raw_scene_dir: Path,
    out_scene_dir: Path,
    octave_runner: str,
    workers: int,
    toolbox_dir: Path,
) -> int:
    pairs = _read_scene_filenames(raw_scene_dir)
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    work = [(d, r, out_scene_dir, octave_runner, i, str(toolbox_dir)) for i, (d, r) in enumerate(pairs)]

    successes = 0
    if workers <= 1:
        for w in work:
            ind, err = _process_one_frame(w)
            if err is None:
                successes += 1
            elif err != "faulty-skip":
                logger.warning("[%s] frame %d: %s", out_scene_dir.name, ind, err)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one_frame, w) for w in work]
            for f in as_completed(futures):
                ind, err = f.result()
                if err is None:
                    successes += 1
                elif err != "faulty-skip":
                    logger.warning("[%s] frame %d: %s", out_scene_dir.name, ind, err)
    return successes


def _extract_scenes_from_raw_zip(raw_zip: Path, scene_names: Iterable[str], out_root: Path) -> None:
    targets = [f"{name}/" for name in scene_names]
    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as zf:
        members = [m for m in zf.namelist() if any(m.startswith(t) for t in targets)]
        if not members:
            msg = f"no members matching {targets} in {raw_zip}"
            raise RuntimeError(msg)
        logger.info("Extracting %d files into %s", len(members), out_root)
        zf.extractall(out_root, members=members)


def _derive_test_scenes(work_dir: Path) -> set[str]:
    """Return the scene-type names belonging to the official 654-image test split."""
    import h5py

    labeled_mat = work_dir / "nyu_depth_v2_labeled.mat"
    splits_mat = work_dir / "splits.mat"
    _download(NYU_LABELED_MAT_URL, labeled_mat)
    _download(NYU_SPLITS_MAT_URL, splits_mat)

    test_idxs = sorted(int(x) for x in loadmat(splits_mat)["testNdxs"].flatten())
    logger.info("Read %d test indices from splits.mat", len(test_idxs))

    with h5py.File(labeled_mat, "r") as f:
        scene_refs = f["sceneTypes"][0]
        test_scenes: set[str] = set()
        for idx in test_idxs:
            chars = f[scene_refs[idx - 1]][:].flatten()
            test_scenes.add("".join(chr(int(c)) for c in chars))

    logger.info("Derived %d test scene-types", len(test_scenes))
    return test_scenes


def _extract_labeled_test_set(work_dir: Path, out_root: Path) -> int:
    """Write the 654 official labeled test images into <out>/<scene_type>/."""
    import h5py

    labeled_mat = work_dir / "nyu_depth_v2_labeled.mat"
    splits_mat = work_dir / "splits.mat"
    _download(NYU_LABELED_MAT_URL, labeled_mat)
    _download(NYU_SPLITS_MAT_URL, splits_mat)

    test_idxs = sorted(int(x) for x in loadmat(splits_mat)["testNdxs"].flatten())
    written = 0

    with h5py.File(labeled_mat, "r") as f:
        scene_refs = f["sceneTypes"][0]
        images = f["images"]
        depths = f["rawDepths"]

        for idx in test_idxs:
            scene = "".join(chr(int(c)) for c in f[scene_refs[idx - 1]][:].flatten())
            scene_dir = out_root / scene
            scene_dir.mkdir(parents=True, exist_ok=True)

            # HDF5 stores transposed: (channels, W, H) → (H, W, channels).
            img = np.array(images[idx - 1]).T
            depth_mm = (np.array(depths[idx - 1]).T * 1000.0).astype(np.uint16)

            # Black-boundary crop on a 480x640 canvas, matching the toolbox convention.
            img_crop = np.zeros((480, 640, 3), dtype=np.uint8)
            img_crop[7:474, 7:632, :] = img[7:474, 7:632, :]

            file_idx = idx - 1
            Image.fromarray(img_crop).save(scene_dir / f"rgb_{file_idx:05d}.jpg", quality=TEST_JPEG_QUALITY)
            Image.fromarray(depth_mm).save(scene_dir / f"sync_depth_{file_idx:05d}.png")
            written += 1
            if written % 100 == 0:
                logger.info("[test-extract] %d/%d", written, len(test_idxs))

    logger.info("[test-extract] wrote %d test pairs", written)
    return written


def _write_split_files(out_root: Path, test_scenes: set[str]) -> None:
    """Walk out_root and write nyu_train.txt + nyu_test.txt."""
    train_lines: list[str] = []
    test_lines: list[str] = []
    for scene_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        bucket = test_lines if scene in test_scenes else train_lines
        for depth_path in sorted(scene_dir.glob("sync_depth_*.png")):
            idx = depth_path.stem.removeprefix("sync_depth_")
            if not (scene_dir / f"rgb_{idx}.jpg").exists():
                continue
            bucket.append(f"{scene}/rgb_{idx}.jpg {scene}/sync_depth_{idx}.png {FX_RGB}")

    (out_root / "nyu_train.txt").write_text("\n".join(train_lines) + "\n")
    (out_root / "nyu_test.txt").write_text("\n".join(test_lines) + "\n")
    logger.info(
        "Wrote nyu_train.txt (%d lines, %d scenes) + nyu_test.txt (%d lines, %d scenes)",
        len(train_lines),
        len({line.split("/", 1)[0] for line in train_lines}),
        len(test_lines),
        len({line.split("/", 1)[0] for line in test_lines}),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NYUv2 depth data into the standard layout.")
    parser.add_argument("--raw-zip", type=Path, help="Path to nyu_depth_v2_raw.zip.")
    parser.add_argument("--raw-dir", type=Path, help="Already-extracted raw NYU dir.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_ROOT / "nyu",
        help=f"Output root (default: {DEFAULT_DATA_ROOT / 'nyu'}).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Scene names to process. Default: all under --raw-dir.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Per-scene parallel frame workers.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT.parent / "nyu_workspace",
        help="Where to put the downloaded NYU toolbox + (with --write-splits) labeled.mat.",
    )
    parser.add_argument(
        "--octave",
        type=str,
        default="octave",
        help="Path to octave binary (default: from PATH).",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help=(
            "Also extract the labeled test set + write nyu_train.txt / nyu_test.txt. "
            "Triggers a one-time 2.97 GB download of nyu_depth_v2_labeled.mat into --work-dir."
        ),
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    setup_logging()

    if not args.raw_zip and not args.raw_dir:
        parser.error("provide --raw-zip or --raw-dir")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    toolbox_dir = _fetch_toolbox(args.work_dir)

    bridge = Path(__file__).parent / "_nyu_octave_bridge.m"
    bridge.write_text(OCTAVE_BRIDGE)

    if args.raw_dir:
        raw_root = args.raw_dir
    else:
        raw_root = args.work_dir / "raw_extracted"
        if args.scenes is None:
            parser.error("--raw-zip requires --scenes")
        _extract_scenes_from_raw_zip(args.raw_zip, args.scenes, raw_root)

    scenes = list(args.scenes) if args.scenes else sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    preview = scenes[:SCENE_PREVIEW_LIMIT] + (["..."] if len(scenes) > SCENE_PREVIEW_LIMIT else [])
    logger.info("Processing %d scene(s): %s", len(scenes), preview)

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene in scenes:
        raw_scene = raw_root / scene
        if not raw_scene.exists():
            logger.warning("[skip] %s not in raw root %s", scene, raw_root)
            continue
        out_scene = args.out / scene
        n = _process_scene(raw_scene, out_scene, args.octave, args.workers, toolbox_dir)
        logger.info("[scene] %s: wrote %d frame pairs", scene, n)
        total += n

    if args.write_splits:
        test_scenes = _derive_test_scenes(args.work_dir)
        _extract_labeled_test_set(args.work_dir, args.out)
        _write_split_files(args.out, test_scenes)

    logger.info("Wrote %d train frames across %d scene(s)", total, len(scenes))
    announce(args.out, "NYUv2", env_var="NYU_ROOT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
