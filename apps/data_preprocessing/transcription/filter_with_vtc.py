# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Filter WhisperX transcripts to drop key-child speech (KCHI) and low-confidence segments.

Pairs each transcript JSON (one per source video) with a VTC RTTM annotation
file matched by filename stem, then in parallel:

  * removes any transcript segment overlapping a ``KCHI`` annotation,
  * optionally removes segments whose mean per-word confidence is below
    ``min_avg_word_score``,
  * records per-VTC-label confidence statistics for monitoring.

Workers are CPU-bound JSON+RTTM parsing, so this runs as ``joblib.Parallel``
within a single process — no Stopes job array needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm

from core.utils.logging import setup_logging

logger = logging.getLogger(__name__)

#: VTC speaker labels we track confidence statistics for.
VTC_LABELS: tuple[str, ...] = ("FEM", "MAL", "KCHI", "OCH")

#: Synthetic label for transcript segments that overlap no VTC annotation.
NO_OVERLAP_LABEL = "NO_OVERLAP"

#: An RTTM line is space-separated; we read the label from index 7, so any
#: shorter line is malformed and skipped.
_RTTM_MIN_FIELDS = 8


@dataclass(frozen=True)
class VTCAnnotation:
    """A single speaker-diarization annotation parsed from an RTTM line."""

    start_time_s: float
    end_time_s: float
    label: str


@dataclass
class FilterTranscriptsConfig:
    """Configuration for the VTC + word-confidence transcript filter."""

    #: Local directory containing WhisperX transcription JSON files.
    transcripts_dir: str = "???"

    #: Local directory containing VTC annotation RTTM files. Stem-matched
    #: against ``transcripts_dir/*.json``; missing matches are processed with
    #: an empty annotation list (no KCHI removal possible).
    vtc_annotations_dir: str = "???"

    #: Local directory where filtered transcription JSONs + summary will be written.
    output_dir: str = "???"

    #: Number of joblib worker processes.
    num_workers: int = 8

    #: Minimum mean per-word confidence for a segment to be kept; ``None``
    #: disables the score filter (only KCHI removal still applies).
    min_avg_word_score: float | None = None


@dataclass
class FilterTranscriptsPipelineConfig:
    """Top-level Hydra config; processor only — no launcher needed."""

    processor: FilterTranscriptsConfig = field(default_factory=FilterTranscriptsConfig)


cs = ConfigStore.instance()
cs.store(name="filter_transcripts_vtc_pipeline", node=FilterTranscriptsPipelineConfig)


def load_vtc_annotations_for_file(
    vtc_path: str | Path,
    labels: tuple[str, ...] | None = None,
) -> list[VTCAnnotation]:
    """Parse a VTC RTTM file into ``VTCAnnotation`` records.

    RTTM line format (space-separated):
    ``SPEAKER <uid> 1 <start_s> <duration_s> <NA> <NA> <label> <NA> <NA>``

    Args:
        vtc_path: Path to the RTTM file.
        labels: If given, only annotations with ``label`` in this set are kept.

    Returns:
        Parsed annotations in source order.
    """
    annotations: list[VTCAnnotation] = []
    with Path(vtc_path).open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < _RTTM_MIN_FIELDS:
                continue
            label = parts[7]
            if labels is not None and label not in labels:
                continue
            start_time = float(parts[3])
            duration = float(parts[4])
            annotations.append(
                VTCAnnotation(
                    start_time_s=start_time,
                    end_time_s=start_time + duration,
                    label=label,
                )
            )
    return annotations


def get_segment_avg_word_score(segment: dict) -> float:
    """Mean WhisperX word-confidence score for a segment, or NaN if none."""
    words = segment.get("words") or []
    if not words:
        return float("nan")
    scores = [w.get("score", np.nan) for w in words]
    if all(np.isnan(s) for s in scores):
        return float("nan")
    return float(np.nanmean(scores))


def is_valid_segment(segment: dict, min_avg_word_score: float | None) -> bool:
    """True iff a segment has text+words and (if a threshold is set) clears it."""
    if not segment.get("text", "").strip():
        return False
    if not segment.get("words"):
        return False
    if min_avg_word_score is None:
        return True
    avg_word_score = get_segment_avg_word_score(segment)
    if np.isnan(avg_word_score):
        # No usable scores — keep the segment rather than silently dropping it.
        logger.warning("No valid word scores in segment; keeping it.")
        return True
    return avg_word_score >= min_avg_word_score


def get_segment_vtc_labels(
    segment_start: float,
    segment_end: float,
    annotations: list[VTCAnnotation],
) -> set[str]:
    """Set of VTC labels whose interval overlaps ``[segment_start, segment_end)``."""
    return {ann.label for ann in annotations if segment_start < ann.end_time_s and segment_end > ann.start_time_s}


def segment_overlaps_kchi(
    segment_start: float,
    segment_end: float,
    annotations: list[VTCAnnotation],
) -> bool:
    """True iff any KCHI annotation overlaps ``[segment_start, segment_end)``."""
    return any(
        ann.label == "KCHI" and segment_start < ann.end_time_s and segment_end > ann.start_time_s
        for ann in annotations
    )


def compute_word_confidence_by_label(
    transcription: dict,
    annotations: list[VTCAnnotation],
) -> dict[str, list[float]]:
    """Group per-segment mean word confidences by overlapping VTC label.

    Returns a mapping ``label -> [avg_score, ...]``; segments overlapping no
    VTC annotation are bucketed under ``NO_OVERLAP_LABEL``.
    """
    scores_by_label: dict[str, list[float]] = {label: [] for label in VTC_LABELS}
    scores_by_label[NO_OVERLAP_LABEL] = []

    for segment in transcription.get("segments", []):
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", segment_start)

        avg_score = get_segment_avg_word_score(segment)
        if np.isnan(avg_score):
            continue

        labels = get_segment_vtc_labels(segment_start, segment_end, annotations)
        if not labels:
            scores_by_label[NO_OVERLAP_LABEL].append(avg_score)
        else:
            for label in labels:
                if label in scores_by_label:
                    scores_by_label[label].append(avg_score)

    return scores_by_label


def filter_segments(
    transcription: dict,
    annotations: list[VTCAnnotation],
    min_avg_word_score: float | None = None,
) -> tuple[dict, int, int]:
    """Drop KCHI-overlapping and (optionally) low-confidence segments.

    Returns ``(filtered_transcription, kchi_removed_count, low_conf_removed_count)``.
    The returned transcription is a shallow copy with ``segments`` rewritten.
    """
    original_segments = transcription.get("segments", [])
    filtered_segments = []
    kchi_removed_count = 0
    low_confidence_removed_count = 0

    for segment in original_segments:
        segment_start = segment.get("start", 0.0)
        segment_end = segment.get("end", segment_start)

        if annotations and segment_overlaps_kchi(segment_start, segment_end, annotations):
            kchi_removed_count += 1
            continue

        if not is_valid_segment(segment, min_avg_word_score):
            low_confidence_removed_count += 1
            continue

        filtered_segments.append(segment)

    filtered_transcription = transcription.copy()
    filtered_transcription["segments"] = filtered_segments
    filtered_transcription["kchi_filtered"] = True
    filtered_transcription["kchi_segments_removed"] = kchi_removed_count
    filtered_transcription["low_confidence_segments_removed"] = low_confidence_removed_count
    if min_avg_word_score is not None:
        filtered_transcription["min_avg_word_score"] = min_avg_word_score

    return filtered_transcription, kchi_removed_count, low_confidence_removed_count


def process_single_file(
    transcript_path: str | Path,
    vtc_path: str | Path | None,
    output_dir: str | Path,
    min_avg_word_score: float | None = None,
) -> dict:
    """Filter one transcript JSON, write the result, and return per-file stats.

    The returned dict contains per-label score *counts and means* (under
    ``word_confidence_by_label``) but NOT the raw per-segment scores —
    those are kept in a separate ``_raw_scores`` channel so aggregator
    quantile math still works without bloating the per-file JSON entries
    that get serialized into the summary file. At ~50k transcripts and
    ~thousands of segments per file, the raw lists would otherwise be
    hundreds of MB of JSON.
    """
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    filename = transcript_path.name
    uid = transcript_path.stem

    result: dict[str, Any] = {
        "input_path": str(transcript_path),
        "vtc_path": str(vtc_path) if vtc_path is not None else None,
        "uid": uid,
        "status": "success",
        "error": None,
        "original_segments": 0,
        "filtered_segments": 0,
        "kchi_segments_removed": 0,
        "low_confidence_segments_removed": 0,
        "output_path": None,
        "word_confidence_by_label": {},
        "_raw_scores": {},
    }

    try:
        with transcript_path.open() as f:
            transcription = json.load(f)

        result["original_segments"] = len(transcription.get("segments", []))

        annotations = load_vtc_annotations_for_file(vtc_path) if vtc_path else []

        word_confidence_by_label = compute_word_confidence_by_label(transcription, annotations)
        result["word_confidence_by_label"] = {
            label: {
                "count": len(scores),
                "mean": float(np.mean(scores)) if scores else None,
            }
            for label, scores in word_confidence_by_label.items()
        }
        result["_raw_scores"] = word_confidence_by_label

        filtered_transcription, kchi_removed, low_conf_removed = filter_segments(
            transcription, annotations, min_avg_word_score
        )

        result["kchi_segments_removed"] = kchi_removed
        result["low_confidence_segments_removed"] = low_conf_removed
        result["filtered_segments"] = len(filtered_transcription.get("segments", []))

        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(filtered_transcription, f, indent=2, ensure_ascii=False)

        result["output_path"] = str(output_path)
    except Exception as e:
        logger.exception("Error processing %s", transcript_path)
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def _label_stats(scores: list[float], min_avg_word_score: float | None) -> dict:
    """Summary statistics for one label's collected per-segment mean scores."""
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "below_threshold_count": None,
            "below_threshold_pct": None,
        }
    arr = np.array(scores)
    if min_avg_word_score is not None:
        below_count = int(np.sum(arr < min_avg_word_score))
        below_pct = round(below_count / len(scores) * 100, 2)
    else:
        below_count = None
        below_pct = None
    return {
        "count": len(scores),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "below_threshold_count": below_count,
        "below_threshold_pct": below_pct,
    }


def aggregate_results(results: list[dict], min_avg_word_score: float | None = None) -> dict:
    """Aggregate per-file filter results into a summary dict.

    Consumes the per-file ``_raw_scores`` side channel populated by
    ``process_single_file`` to compute corpus-level quantile statistics,
    then strips it from each result so the returned summary stays
    JSON-serializable at modest size.
    """
    total_files = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    total_original_segments = sum(r.get("original_segments", 0) for r in results)
    total_filtered_segments = sum(r.get("filtered_segments", 0) for r in results)
    total_kchi_removed = sum(r.get("kchi_segments_removed", 0) for r in results)
    total_low_confidence_removed = sum(r.get("low_confidence_segments_removed", 0) for r in results)

    files_with_kchi = sum(1 for r in results if r.get("kchi_segments_removed", 0) > 0)
    files_with_low_confidence = sum(1 for r in results if r.get("low_confidence_segments_removed", 0) > 0)

    all_labels = (*VTC_LABELS, NO_OVERLAP_LABEL)
    aggregated_scores: dict[str, list[float]] = {label: [] for label in all_labels}
    for r in results:
        raw_scores = r.pop("_raw_scores", {})
        for label in all_labels:
            scores = raw_scores.get(label, [])
            if scores:
                aggregated_scores[label].extend(scores)

    word_confidence_stats_by_label = {
        label: _label_stats(scores, min_avg_word_score) for label, scores in aggregated_scores.items()
    }

    return {
        "summary": {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / total_files * 100, 2) if total_files else 0,
            "total_original_segments": total_original_segments,
            "total_filtered_segments": total_filtered_segments,
            "total_kchi_segments_removed": total_kchi_removed,
            "total_low_confidence_segments_removed": total_low_confidence_removed,
            "files_with_kchi_overlap": files_with_kchi,
            "files_with_low_confidence": files_with_low_confidence,
            "kchi_removal_rate": round(total_kchi_removed / total_original_segments * 100, 2)
            if total_original_segments
            else 0,
            "low_confidence_removal_rate": round(total_low_confidence_removed / total_original_segments * 100, 2)
            if total_original_segments
            else 0,
            "word_confidence_stats_by_label": word_confidence_stats_by_label,
        },
        "files": results,
    }


def _log_summary(summary: dict) -> None:
    """Pretty-print the aggregated summary at INFO level."""
    logger.info("Done! Processed %d files", summary["total_files"])
    logger.info("Success rate: %.2f%%", summary["success_rate"])
    logger.info(
        "Removed %d KCHI segments (%.2f%%); files with KCHI overlap: %d",
        summary["total_kchi_segments_removed"],
        summary["kchi_removal_rate"],
        summary["files_with_kchi_overlap"],
    )
    logger.info(
        "Removed %d low-confidence segments (%.2f%%); files with low-confidence: %d",
        summary["total_low_confidence_segments_removed"],
        summary["low_confidence_removal_rate"],
        summary["files_with_low_confidence"],
    )

    logger.info("Word confidence statistics by VTC label:")
    wc_stats = summary.get("word_confidence_stats_by_label", {})
    for label in (*VTC_LABELS, NO_OVERLAP_LABEL):
        stats = wc_stats.get(label, {})
        if not stats.get("count"):
            logger.info("  %s: no segments", label)
            continue
        below_info = ""
        if stats.get("below_threshold_pct") is not None:
            below_info = f", below_threshold={stats['below_threshold_count']} ({stats['below_threshold_pct']:.2f}%)"
        logger.info(
            "  %s: count=%d, mean=%.4f, std=%.4f, median=%.4f, min=%.4f, max=%.4f%s",
            label,
            stats["count"],
            stats["mean"],
            stats["std"],
            stats["median"],
            stats["min"],
            stats["max"],
            below_info,
        )


@hydra.main(version_base=None, config_name="filter_transcripts_vtc_pipeline")
def main(config: FilterTranscriptsPipelineConfig) -> None:
    """Hydra entry point."""
    setup_logging()
    cfg = config.processor

    transcripts_dir = Path(cfg.transcripts_dir)
    vtc_dir = Path(cfg.vtc_annotations_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Discovering transcript files in %s", transcripts_dir)
    transcript_paths = sorted(transcripts_dir.rglob("*.json"))
    logger.info("Found %d transcript files", len(transcript_paths))
    if not transcript_paths:
        logger.warning("No transcript files found!")
        return

    logger.info("Discovering VTC annotation files in %s", vtc_dir)
    vtc_paths = sorted(vtc_dir.rglob("*.rttm"))
    logger.info("Found %d VTC annotation files", len(vtc_paths))

    vtc_by_stem: dict[str, Path] = {p.stem: p for p in vtc_paths}

    file_pairs: list[tuple[Path, Path | None]] = [(tp, vtc_by_stem.get(tp.stem)) for tp in transcript_paths]
    matched_count = sum(1 for _, vp in file_pairs if vp is not None)
    logger.info("Matched %d/%d transcripts with VTC annotations", matched_count, len(transcript_paths))

    if cfg.min_avg_word_score is not None:
        logger.info("Filtering segments with min_avg_word_score=%.2f", cfg.min_avg_word_score)

    logger.info("Processing with %d workers", cfg.num_workers)
    results = Parallel(n_jobs=cfg.num_workers)(
        delayed(process_single_file)(tp, vp, output_dir, cfg.min_avg_word_score)
        for tp, vp in tqdm(file_pairs, desc="Filtering transcripts")
    )

    aggregated = aggregate_results(results, cfg.min_avg_word_score)

    summary_path = output_dir / "filter_summary.json"
    logger.info("Writing summary to %s", summary_path)
    with summary_path.open("w") as f:
        json.dump(aggregated, f, indent=2)

    _log_summary(aggregated["summary"])


if __name__ == "__main__":
    main()
