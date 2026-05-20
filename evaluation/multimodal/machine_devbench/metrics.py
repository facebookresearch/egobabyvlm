# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Python metrics + aggregation helpers for MachineDevBench."""

from collections import defaultdict
from typing import Any

_LOW_FREQ_BINS_TO_MERGE = ("[1,2)", "[2,4)")
_MERGED_LOW_FREQ_BIN = "[1,4)"


def _merge_low_freq_bin(bin_label: str) -> str:
    """Map the two lowest-frequency bins into a single combined bin.

    ``[1,2)`` and ``[2,4)`` are merged into ``[1,4)``. All other labels are
    returned unchanged. Applied to both lexical (``frequency_bin``) and
    grammatical (``freq_bin``) metadata before aggregating accuracy.
    """
    if bin_label in _LOW_FREQ_BINS_TO_MERGE:
        return _MERGED_LOW_FREQ_BIN
    return bin_label


def accuracy(predictions: list[int], targets: list[int]) -> float:
    """Compute the fraction of correct predictions."""
    if not predictions:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, targets, strict=True))
    return correct / len(predictions)


def accuracy_per_group(
    predictions: list[int],
    targets: list[int],
    groups: list[str],
) -> dict[str, float]:
    """Compute accuracy broken down by a group key.

    Args:
        predictions: Model predictions.
        targets: Ground-truth targets.
        groups: Group label for each trial (e.g. frequency bin or category).

    Returns:
        Mapping from group name to accuracy within that group, sorted by key.
    """
    group_preds: dict[str, list[int]] = defaultdict(list)
    group_tgts: dict[str, list[int]] = defaultdict(list)
    for pred, tgt, grp in zip(predictions, targets, groups, strict=True):
        group_preds[grp].append(pred)
        group_tgts[grp].append(tgt)
    return {grp: accuracy(group_preds[grp], group_tgts[grp]) for grp in sorted(group_preds)}


class ResultAggregator:
    """Accumulates per-trial predictions and computes structured metrics.

    Aggregation strategy (based on :class:`custom_devbench_eval.aggregator.ResultAggregator`,
    with one intentional deviation — see "Frequency-bin merge" below):

    * Each lexical task's accuracy is the mean of its per-frequency-bin accuracies.
    * Lexical overall = mean of per-task (bin-averaged) accuracies.
    * Grammatical overall = mean of per-task accuracies.
    * Overall = mean of lexical and grammatical overall accuracies.

    Frequency-bin merge
    -------------------
    Trials whose ``frequency_bin`` (lexical) or ``freq_bin`` (grammatical) is
    ``"[1,2)"`` or ``"[2,4)"`` are pooled into a single ``"[1,4)"`` bin via
    :func:`_merge_low_freq_bin` *before* computing per-bin accuracies. This is
    equivalent to the ``weighted`` rebin strategy used by
    ``scripts/20_evals/rebin_results.py`` in the standalone CustomDevBench
    harness — it is **not** part of the original
    :class:`custom_devbench_eval.aggregator.ResultAggregator`. As a result,
    lexical task accuracies (which are means over per-bin accuracies) will
    differ from the standalone harness whenever a task has trials in either of
    the two merged bins.
    """

    def __init__(self) -> None:
        # task_name -> list of (prediction, target, metadata)
        self._records: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)

    def add(self, task_name: str, prediction: int, target: int, metadata: dict) -> None:
        """Append one trial result to the aggregator."""
        self._records[task_name].append((prediction, target, metadata))

    def compute(self) -> dict[str, Any]:
        """Compute the full structured results dict."""
        by_task: dict[str, dict[str, Any]] = {}
        lex_task_accs: list[float] = []
        gram_task_accs: list[float] = []

        for task_name in sorted(self._records):
            records = self._records[task_name]
            preds = [r[0] for r in records]
            tgts = [r[1] for r in records]
            metas = [r[2] for r in records]

            is_lexical = task_name.startswith("lex_")

            task_result: dict[str, Any] = {"n_trials": len(preds)}

            if is_lexical:
                # Per-frequency-bin breakdown.
                freq_bins = [_merge_low_freq_bin(str(m.get("frequency_bin", "unknown"))) for m in metas]
                bin_accs = accuracy_per_group(preds, tgts, freq_bins)
                task_result["by_freq_bin"] = bin_accs
                # Task accuracy = mean of per-bin accuracies.
                task_acc = sum(bin_accs.values()) / len(bin_accs) if bin_accs else 0.0
                task_result["accuracy"] = task_acc
                lex_task_accs.append(task_acc)

                # Per-category breakdown is only meaningful for nouns.
                if task_name == "lex_nouns":
                    categories = [m.get("category", "unknown") for m in metas]
                    task_result["by_category"] = accuracy_per_group(preds, tgts, categories)
            else:
                freq_bins = [_merge_low_freq_bin(str(m.get("freq_bin", "unknown"))) for m in metas]
                task_result["by_freq_bin"] = accuracy_per_group(preds, tgts, freq_bins)
                task_acc = accuracy(preds, tgts)
                task_result["accuracy"] = task_acc
                gram_task_accs.append(task_acc)

            by_task[task_name] = task_result

        lex_acc = sum(lex_task_accs) / len(lex_task_accs) if lex_task_accs else None
        gram_acc = sum(gram_task_accs) / len(gram_task_accs) if gram_task_accs else None

        type_accs = [a for a in (lex_acc, gram_acc) if a is not None]
        overall_acc = sum(type_accs) / len(type_accs) if type_accs else 0.0

        results: dict[str, Any] = {
            "overall": {"accuracy": overall_acc},
            "by_task_type": {},
            "by_task": by_task,
        }
        if lex_acc is not None:
            results["by_task_type"]["lexical"] = {"accuracy": lex_acc}
        if gram_acc is not None:
            results["by_task_type"]["grammatical"] = {"accuracy": gram_acc}
        return results


def merge_style_results(per_style: dict[str, Any], total_results: dict[str, Any]) -> dict[str, Any]:
    """Merge per-style results into a hierarchical structure.

    The overall accuracy comes from the pooled total aggregator (all trials
    from all styles combined), which is more principled and lower-variance
    than averaging per-style accuracies.

    Returns::

        {
            "overall": {"accuracy": <pooled accuracy>},
            "total":   {<full pooled results>},
            "by_style": {
                "realistic": {<full single-style results>},
                ...
            },
        }
    """
    return {
        "overall": total_results.get("overall", {"accuracy": 0.0}),
        "total": total_results,
        "by_style": per_style,
    }


def build_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Build a compact accuracy-only summary from the merged results.

    Returns::

        {
            "overall": <float>,
            "total": {"overall": ..., "lexical": ..., "grammatical": ..., <task>: ...},
            "per_style": {<style>: {"overall": ..., "lexical": ..., <task>: ...}, ...},
        }
    """
    summary: dict[str, Any] = {"overall": results["overall"]["accuracy"]}

    total_results = results.get("total", {})
    total_summary: dict[str, Any] = {
        "overall": total_results.get("overall", {}).get("accuracy"),
    }
    for type_key, type_info in total_results.get("by_task_type", {}).items():
        total_summary[type_key] = type_info.get("accuracy")
    for task_name, task_info in sorted(total_results.get("by_task", {}).items()):
        total_summary[task_name] = task_info.get("accuracy")
    summary["total"] = total_summary

    by_style = results.get("by_style", {})
    per_style: dict[str, Any] = {}
    for style, style_results in sorted(by_style.items()):
        style_summary: dict[str, Any] = {
            "overall": style_results.get("overall", {}).get("accuracy"),
        }
        for type_key, type_info in style_results.get("by_task_type", {}).items():
            style_summary[type_key] = type_info.get("accuracy")
        for task_name, task_info in sorted(style_results.get("by_task", {}).items()):
            style_summary[task_name] = task_info.get("accuracy")
        per_style[style] = style_summary

    summary["per_style"] = per_style
    return summary
