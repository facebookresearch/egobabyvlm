# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the VTC + word-confidence transcript filter helpers."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from apps.data_preprocessing.transcription.filter_with_vtc import (
    NO_OVERLAP_LABEL,
    VTC_LABELS,
    VTCAnnotation,
    aggregate_results,
    compute_word_confidence_by_label,
    filter_segments,
    get_segment_avg_word_score,
    get_segment_vtc_labels,
    is_valid_segment,
    load_vtc_annotations_for_file,
    process_single_file,
    segment_overlaps_kchi,
)

if TYPE_CHECKING:
    from pathlib import Path


def _segment(start: float, end: float, text: str = "hello world", scores: list[float] | None = None) -> dict:
    """Build a minimal WhisperX segment dict for testing."""
    if scores is None:
        scores = [1.0, 1.0]
    words = [{"word": f"w{i}", "score": s} for i, s in enumerate(scores)]
    return {"start": start, "end": end, "text": text, "words": words}


def _rttm_line(start: float, duration: float, label: str) -> str:
    return f"SPEAKER uid 1 {start} {duration} <NA> <NA> {label} <NA> <NA>\n"


# ---------------------------------------------------------------------------
# RTTM parsing
# ---------------------------------------------------------------------------


def test_load_vtc_annotations_parses_rttm_lines(tmp_path: Path) -> None:
    rttm = tmp_path / "uid.rttm"
    rttm.write_text(_rttm_line(0.0, 1.0, "FEM") + _rttm_line(2.0, 0.5, "KCHI"))

    annotations = load_vtc_annotations_for_file(rttm)

    assert len(annotations) == 2
    assert annotations[0] == VTCAnnotation(start_time_s=0.0, end_time_s=1.0, label="FEM")
    assert annotations[1] == VTCAnnotation(start_time_s=2.0, end_time_s=2.5, label="KCHI")


def test_load_vtc_annotations_filters_by_label_and_skips_short_lines(tmp_path: Path) -> None:
    rttm = tmp_path / "uid.rttm"
    rttm.write_text(
        _rttm_line(0.0, 1.0, "FEM") + "TOO SHORT\n" + _rttm_line(2.0, 0.5, "KCHI") + _rttm_line(3.0, 0.5, "MAL"),
    )

    annotations = load_vtc_annotations_for_file(rttm, labels=("FEM", "MAL"))

    assert {a.label for a in annotations} == {"FEM", "MAL"}


# ---------------------------------------------------------------------------
# Per-segment scoring + validity
# ---------------------------------------------------------------------------


def test_get_segment_avg_word_score_returns_mean() -> None:
    segment = _segment(0.0, 1.0, scores=[0.4, 0.6, 0.8])
    assert get_segment_avg_word_score(segment) == pytest.approx(0.6)


def test_get_segment_avg_word_score_returns_nan_for_no_words() -> None:
    assert math.isnan(get_segment_avg_word_score({"text": "hi", "words": []}))
    assert math.isnan(get_segment_avg_word_score({"text": "hi"}))


def test_is_valid_segment_rejects_empty_text_or_words() -> None:
    assert not is_valid_segment({"text": "  ", "words": [{"word": "x", "score": 1.0}]}, None)
    assert not is_valid_segment({"text": "hi", "words": []}, None)


def test_is_valid_segment_threshold_check() -> None:
    segment = _segment(0.0, 1.0, scores=[0.4, 0.6])  # mean = 0.5
    assert is_valid_segment(segment, min_avg_word_score=0.4)
    assert is_valid_segment(segment, min_avg_word_score=0.5)
    assert not is_valid_segment(segment, min_avg_word_score=0.6)


def test_is_valid_segment_keeps_segments_with_no_scores() -> None:
    """A segment with words but no usable scores is kept (warned about, not dropped)."""
    segment = {"text": "hi", "words": [{"word": "x"}, {"word": "y"}]}
    assert is_valid_segment(segment, min_avg_word_score=0.5)


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------


def test_get_segment_vtc_labels_returns_overlapping() -> None:
    annotations = [
        VTCAnnotation(0.0, 2.0, "FEM"),
        VTCAnnotation(1.5, 3.0, "KCHI"),
        VTCAnnotation(5.0, 6.0, "MAL"),
    ]
    assert get_segment_vtc_labels(1.0, 1.6, annotations) == {"FEM", "KCHI"}
    assert get_segment_vtc_labels(4.0, 4.5, annotations) == set()


def test_segment_overlaps_kchi() -> None:
    annotations = [
        VTCAnnotation(0.0, 1.0, "FEM"),
        VTCAnnotation(2.0, 3.0, "KCHI"),
    ]
    assert segment_overlaps_kchi(2.5, 2.8, annotations)
    assert not segment_overlaps_kchi(0.0, 1.0, annotations)
    # KCHI ends at 3.0; an exactly-touching interval does not overlap.
    assert not segment_overlaps_kchi(3.0, 4.0, annotations)


# ---------------------------------------------------------------------------
# Word-confidence-by-label aggregation
# ---------------------------------------------------------------------------


def test_compute_word_confidence_by_label_buckets_segments() -> None:
    transcription = {
        "segments": [
            _segment(0.0, 1.0, scores=[0.5, 0.5]),  # overlaps FEM
            _segment(2.0, 3.0, scores=[0.9, 0.9]),  # overlaps KCHI
            _segment(5.0, 6.0, scores=[0.1, 0.1]),  # NO_OVERLAP
        ],
    }
    annotations = [
        VTCAnnotation(0.0, 2.0, "FEM"),
        VTCAnnotation(2.0, 3.0, "KCHI"),
    ]
    by_label = compute_word_confidence_by_label(transcription, annotations)

    assert by_label["FEM"] == [pytest.approx(0.5)]
    assert by_label["KCHI"] == [pytest.approx(0.9)]
    assert by_label["MAL"] == []
    assert by_label["OCH"] == []
    assert by_label[NO_OVERLAP_LABEL] == [pytest.approx(0.1)]


# ---------------------------------------------------------------------------
# filter_segments
# ---------------------------------------------------------------------------


def test_filter_segments_drops_kchi_and_low_confidence() -> None:
    transcription = {
        "segments": [
            _segment(0.0, 1.0, scores=[0.9, 0.9]),  # kept
            _segment(2.0, 3.0, scores=[0.9, 0.9]),  # KCHI overlap → dropped
            _segment(4.0, 5.0, scores=[0.1, 0.1]),  # low confidence → dropped
        ],
    }
    annotations = [VTCAnnotation(2.0, 3.0, "KCHI")]
    filtered, kchi_removed, low_conf_removed = filter_segments(transcription, annotations, min_avg_word_score=0.5)

    assert kchi_removed == 1
    assert low_conf_removed == 1
    assert len(filtered["segments"]) == 1
    assert filtered["kchi_filtered"] is True
    assert filtered["min_avg_word_score"] == 0.5


def test_filter_segments_without_threshold_keeps_low_confidence() -> None:
    transcription = {
        "segments": [
            _segment(0.0, 1.0, scores=[0.1, 0.1]),
            _segment(2.0, 3.0, scores=[0.9, 0.9]),
        ],
    }
    filtered, kchi_removed, low_conf_removed = filter_segments(transcription, annotations=[], min_avg_word_score=None)

    assert kchi_removed == 0
    assert low_conf_removed == 0
    assert len(filtered["segments"]) == 2
    assert "min_avg_word_score" not in filtered


# ---------------------------------------------------------------------------
# End-to-end process_single_file + aggregate_results
# ---------------------------------------------------------------------------


def test_process_single_file_writes_filtered_json(tmp_path: Path) -> None:
    transcription = {
        "language": "en",
        "segments": [
            _segment(0.0, 1.0, scores=[0.9, 0.9]),
            _segment(2.0, 3.0, scores=[0.9, 0.9]),  # KCHI overlap
        ],
    }
    transcripts_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    transcripts_dir.mkdir()
    transcript_path = transcripts_dir / "uid.json"
    transcript_path.write_text(json.dumps(transcription))

    rttm_path = tmp_path / "uid.rttm"
    rttm_path.write_text(_rttm_line(2.0, 1.0, "KCHI"))

    result = process_single_file(transcript_path, rttm_path, output_dir, min_avg_word_score=0.5)

    assert result["status"] == "success"
    assert result["original_segments"] == 2
    assert result["filtered_segments"] == 1
    assert result["kchi_segments_removed"] == 1
    assert result["low_confidence_segments_removed"] == 0
    # The lightweight word_confidence_by_label payload only carries summary
    # stats (count, mean) — raw per-segment scores live in the side channel.
    assert "scores" not in result["word_confidence_by_label"]["FEM"]
    assert "_raw_scores" in result

    written = json.loads((output_dir / "uid.json").read_text())
    assert len(written["segments"]) == 1
    assert written["kchi_filtered"] is True


def test_aggregate_results_strips_raw_scores(tmp_path: Path) -> None:
    """aggregate_results must drop the _raw_scores side channel from each result."""
    transcription = {
        "language": "en",
        "segments": [_segment(0.0, 1.0, scores=[0.9, 0.9])],
    }
    transcripts_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    transcripts_dir.mkdir()
    transcript_path = transcripts_dir / "uid.json"
    transcript_path.write_text(json.dumps(transcription))

    result = process_single_file(transcript_path, None, output_dir, min_avg_word_score=None)
    assert "_raw_scores" in result

    aggregated = aggregate_results([result], min_avg_word_score=None)
    assert "_raw_scores" not in aggregated["files"][0]


def test_process_single_file_handles_missing_input(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    result = process_single_file(tmp_path / "missing.json", None, output_dir, min_avg_word_score=None)
    assert result["status"] == "failed"
    assert result["error"]


def test_aggregate_results_summary_math() -> None:
    results = [
        {
            "status": "success",
            "original_segments": 4,
            "filtered_segments": 2,
            "kchi_segments_removed": 1,
            "low_confidence_segments_removed": 1,
            "word_confidence_by_label": {
                "FEM": {"count": 1, "mean": 0.6},
                "MAL": {"count": 0, "mean": None},
                "KCHI": {"count": 1, "mean": 0.8},
                "OCH": {"count": 0, "mean": None},
                NO_OVERLAP_LABEL: {"count": 1, "mean": 0.4},
            },
            "_raw_scores": {
                "FEM": [0.6],
                "MAL": [],
                "KCHI": [0.8],
                "OCH": [],
                NO_OVERLAP_LABEL: [0.4],
            },
        },
        {"status": "failed", "error": "boom"},
    ]
    summary = aggregate_results(results, min_avg_word_score=0.5)["summary"]

    assert summary["total_files"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert summary["success_rate"] == 50.0
    assert summary["total_original_segments"] == 4
    assert summary["total_kchi_segments_removed"] == 1
    assert summary["files_with_kchi_overlap"] == 1
    assert summary["files_with_low_confidence"] == 1
    assert summary["kchi_removal_rate"] == 25.0

    fem = summary["word_confidence_stats_by_label"]["FEM"]
    assert fem["count"] == 1
    assert fem["mean"] == pytest.approx(0.6)
    assert fem["below_threshold_count"] == 0
    no_overlap = summary["word_confidence_stats_by_label"][NO_OVERLAP_LABEL]
    assert no_overlap["below_threshold_count"] == 1


def test_vtc_labels_constant_unchanged() -> None:
    assert VTC_LABELS == ("FEM", "MAL", "KCHI", "OCH")
