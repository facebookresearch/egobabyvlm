# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import MISSING
from stopes.core import Requirements

from core.utils import setup_logging, to_yaml
from evaluation.base import to_path
from evaluation.base.eval_module import EvalConfig, EvalModule

logger = logging.getLogger(__name__)

FREQ_BINS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, np.inf])


TASK_FILE_PATTERNS = {
    "wordswap": "wordswap_pairs.txt",
    "agrswap": "agrswap_pairs.txt",
    "inflswap": "inflswap_pairs.txt",
    # "visual" auto-merges all vp_swap_<property>_pairs.txt files in data_dir;
    # see _get_pair_files. Set pair_files["visual"] explicitly to override.
    "visual": "vp_swap_*_pairs.txt",
}


@dataclass
class LTSwapEvalConfig(EvalConfig):
    """Configuration for LT-Swap evaluation."""

    _target_: str = "evaluation.text.ltswap.LTSwapEvalModule"

    #: As dict to support interpolation from shared_model.
    model: dict[str, Any] = MISSING

    task_types: list[str] = field(default_factory=lambda: ["wordswap", "visual", "agrswap", "inflswap"])

    #: Path to directory containing pair files.
    data_dir: str | None = None

    #: Maps task_type to pair-file path or list of paths (lists are concatenated).
    pair_files: dict[str, Any] = field(default_factory=dict)

    #: Use prefix method (only for WordSwap).
    use_prefix: bool = False

    batch_size: int = 200


class LTSwapEvalModule(EvalModule):
    """Evaluation module for LT-Swap vocabulary benchmark.

    Uses lm_eval's AutoMaskedLM for scoring, consistent with Zorro.
    """

    def __init__(self, config: LTSwapEvalConfig) -> None:
        super().__init__(config, LTSwapEvalConfig)
        self.output_dir = to_path(self.config.output_dir) / "ltswap" / self.config.model.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def requirements(self) -> Requirements:
        return Requirements(
            nodes=1,
            mem_gb=64,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=8,
            timeout_min=60 * 72,
        )

    def name(self) -> str:
        return f"{self.config.name}_{self.config.model.name}_seed{self.config.seed}"

    def _create_model(self) -> Any:
        """Create the lm_eval model via hydra instantiation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Creating model %s on %s", self.config.model.name, device)

        return instantiate({"_target_": self.config.model._target_}, **self.config.model.kwargs, device=device)

    def _swap_words(self, w1: str, ig1: int, g1: str, w2: str, ig2: int, g2: str) -> tuple[str, str]:
        """Swap words between two sentences."""
        try:
            assert g1[ig1 : ig1 + len(w1)].lower() == w1
            assert g2[ig2 : ig2 + len(w2)].lower() == w2
            gg1 = g1[:ig1] + w2 + g1[ig1 + len(w1) :]
            gg2 = g2[:ig2] + w1 + g2[ig2 + len(w2) :]
        except (AssertionError, IndexError):
            gg1_parts, gg2_parts = g1.split(" "), g2.split(" ")
            try:
                assert gg1_parts[ig1].lower() == w1
                assert gg2_parts[ig2].lower() == w2
            except (AssertionError, IndexError):
                logger.warning("Issue with words %s and %s", w1, w2)
            gg1_parts[ig1], gg2_parts[ig2] = w2, w1
            gg1, gg2 = " ".join(gg1_parts), " ".join(gg2_parts)

        gg1 = gg1[0].upper() + gg1[1:]
        gg2 = gg2[0].upper() + gg2[1:]
        return gg1, gg2

    def _get_log_probs(self, model: Any, sentences: list[str]) -> list[float]:
        """Compute log probabilities using original LT-Swap scoring.

        Uses the underlying HuggingFace model from lm_eval's wrapper:
        - BERT (AutoMaskedLM): Pseudo-log-likelihood with original masking logic
        - GPT (AutoCausalLM): Standard next-token prediction loss

        Args:
            model: The lm_eval model (AutoMaskedLM or AutoCausalLM).
            sentences: List of sentences to score.

        Returns:
            List of negative log probability scores (lower loss = higher prob).
        """
        hf_model = model.model
        tokenizer = model.tokenizer
        device = model.device

        is_causal = not hasattr(tokenizer, "mask_token_id") or tokenizer.mask_token_id is None

        scores = []
        for sentence in sentences:
            encoded = tokenizer(sentence, return_tensors="pt", padding=True, add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            nb_tokens = attention_mask.sum().int().item()
            if nb_tokens <= 1:
                scores.append(0.0)
                continue

            if is_causal:
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                if tokenizer.eos_token_id is not None:
                    labels[:, -1] = tokenizer.eos_token_id
                else:
                    labels[:, -1] = input_ids[:, -1]

                with torch.inference_mode():
                    outputs = hf_model(input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                )
                loss = (loss.view(input_ids.size()) * attention_mask).sum()
            else:
                repeat_input = input_ids.repeat(nb_tokens - 1, 1)
                attention_repeated = attention_mask.repeat(nb_tokens - 1, 1)

                mask = torch.ones(input_ids.size(-1) - 1).diag(1)[: nb_tokens - 1].to(device)
                masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
                labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

                with torch.inference_mode():
                    outputs = hf_model(masked_input, attention_mask=attention_repeated)

                logits = outputs.logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

            scores.append(-loss.item())

        return scores

    def _parse_pair_line(self, pair: str, task_type: str) -> dict | None:
        """Parse a single pair line based on task type."""
        pair = pair.rstrip()
        fields = pair.split("|")

        try:
            if task_type == "wordswap":
                bin_val, rule, w1, s1, i1, g1, ig1, w2, s2, i2, g2, ig2 = fields
                rule = rule.split("_")[0]
            elif task_type == "visual":
                bin_val, rule, w1, g1, ig1, w2, g2, ig2 = fields
                s1, s2 = "", ""
                if not ((w1.strip() in g1) and (w2.strip() in g2)):
                    return None
                rule = rule.split("_")[0]
            elif task_type == "inflswap":
                bin_val, rule, w1, g1, ig1, w2, g2, ig2 = fields
                s1, s2 = "", ""
            elif task_type == "agrswap":
                bin_val, rule, w1, g1, ig1, w2, g2, ig2 = fields
                s1, s2 = "", ""
                if "SHORT" in rule or "DET" in rule:
                    rule = "SHORT"
                elif "LONG" in rule:
                    rule = "LONG"
            else:
                return None

            return {
                "bin": int(bin_val),
                "rule": rule,
                "w1": w1,
                "g1": g1,
                "ig1": int(ig1),
                "w2": w2,
                "g2": g2,
                "ig2": int(ig2),
                "s1": s1 if task_type == "wordswap" else "",
                "s2": s2 if task_type == "wordswap" else "",
            }
        except (ValueError, IndexError) as e:
            logger.debug("Failed to parse line: %s", e)
            return None

    def _evaluate_task(self, task_type: str, pair_file: str | list[str], model: Any) -> dict[str, Any]:
        """Evaluate a single LT-Swap task.

        ``pair_file`` may be a single path or a list of paths. When a list is
        passed, lines from all files are concatenated before scoring — used
        for ``visual`` to merge per-property VP-Swap outputs into one score.
        """
        if isinstance(pair_file, str):
            pair_files = [pair_file]
        else:
            pair_files = list(pair_file)
        logger.info("Evaluating task: %s from %s", task_type, pair_files)

        pairs: list[str] = []
        for path in pair_files:
            with Path(path).open() as f:
                pairs.extend(f.readlines())

        parsed_pairs = []
        issues = 0

        for pair in pairs:
            parsed = self._parse_pair_line(pair, task_type)
            if parsed is None:
                issues += 1
                continue
            parsed_pairs.append(parsed)

        logger.info("%d pairs with alignment issues skipped.", issues)

        success = dict.fromkeys(range(len(FREQ_BINS) - 1), 0)
        all_pairs = dict.fromkeys(range(len(FREQ_BINS) - 1), 0)
        pos_success: dict[int, dict[str, int]] = {bin_idx: {} for bin_idx in range(len(FREQ_BINS) - 1)}
        pos_all_pairs: dict[int, dict[str, int]] = {bin_idx: {} for bin_idx in range(len(FREQ_BINS) - 1)}

        for bin_idx in range(len(FREQ_BINS) - 1):
            for pos_tag in ["VERB", "NOUN", "LONG", "SHORT", "VISUAL"]:
                pos_success[bin_idx][pos_tag] = 0
                pos_all_pairs[bin_idx][pos_tag] = 0

        batch_size = self.config.batch_size

        for i in tqdm.tqdm(range(0, len(parsed_pairs), batch_size)):
            batch_parsed = parsed_pairs[i : i + batch_size]
            if len(batch_parsed) == 0:
                continue

            sentences = []
            bin_info = []

            for parsed in batch_parsed:
                gg1, gg2 = self._swap_words(
                    parsed["w1"],
                    parsed["ig1"],
                    parsed["g1"],
                    parsed["w2"],
                    parsed["ig2"],
                    parsed["g2"],
                )

                sentences.extend([parsed["g1"], parsed["g2"], gg1, gg2])
                bin_info.append((parsed["bin"], parsed["rule"]))

            log_probs = self._get_log_probs(model, sentences)

            for j, (bin_val, pos) in enumerate(bin_info):
                prob_g1 = log_probs[j * 4]
                prob_g2 = log_probs[j * 4 + 1]
                prob_b1 = log_probs[j * 4 + 2]
                prob_b2 = log_probs[j * 4 + 3]

                if prob_g1 > prob_b1:
                    success[bin_val] = success.get(bin_val, 0) + 1
                    pos_success[bin_val][pos] = pos_success[bin_val].get(pos, 0) + 1
                if prob_g2 > prob_b2:
                    success[bin_val] = success.get(bin_val, 0) + 1
                    pos_success[bin_val][pos] = pos_success[bin_val].get(pos, 0) + 1
                all_pairs[bin_val] = all_pairs.get(bin_val, 0) + 2
                pos_all_pairs[bin_val][pos] = pos_all_pairs[bin_val].get(pos, 0) + 2

        matrix: dict[int, dict[str, Any]] = {}
        for bin_val in success:
            for pos in pos_success.get(bin_val, {}):
                if pos_all_pairs[bin_val].get(pos, 0) > 0:
                    if bin_val not in matrix:
                        matrix[bin_val] = {}
                    if pos_success[bin_val][pos] > 0:
                        tmp_res = float(pos_success[bin_val][pos]) / float(pos_all_pairs[bin_val][pos])
                        matrix[bin_val][pos] = np.around(tmp_res, 4)

        bins_list = sorted(list(matrix.keys()))
        if bins_list:
            pos_tags = sorted(list(matrix[bins_list[0]].keys()))
            sorted_matrix_rows: list[list[float]] = []
            for bin_val in bins_list:
                row = [matrix[bin_val].get(pos, 0) for pos in pos_tags]
                sorted_matrix_rows.append(row)

            sorted_matrix = np.array(sorted_matrix_rows, dtype=float)
            sorted_matrix[sorted_matrix == 0] = np.nan
            avg_per_bin: Any = np.around(np.nanmean(sorted_matrix, axis=1), 3)
            avg_per_pos: Any = np.around(np.nanmean(sorted_matrix, axis=0), 3)
            avg_accuracy = float(np.around(np.nanmean(avg_per_bin), 3))
        else:
            avg_per_bin = []
            avg_per_pos = []
            avg_accuracy = 0.0
            pos_tags = []

        return {
            "avg_accuracy": avg_accuracy,
            "avg_per_bin": list(avg_per_bin) if isinstance(avg_per_bin, np.ndarray) else avg_per_bin,
            "avg_per_subtask": list(avg_per_pos) if isinstance(avg_per_pos, np.ndarray) else avg_per_pos,
            "subtasks": pos_tags,
            "bins": bins_list,
            "matrix": {int(k): {pos: float(v) for pos, v in vals.items()} for k, vals in matrix.items()},
        }

    def _get_pair_files(self) -> dict[str, str | list[str]]:
        """Resolve pair-file paths for each requested task type.

        Patterns containing a ``*`` glob (e.g. ``visual``'s
        ``vp_swap_*_pairs.txt``) expand to a sorted list of every match
        in ``data_dir``; fixed-name patterns resolve to a single path.
        Paths set explicitly in ``pair_files`` always win.
        """
        pair_files: dict[str, str | list[str]] = dict(self.config.pair_files) if self.config.pair_files else {}

        if self.config.data_dir:
            data_dir = to_path(self.config.data_dir)
            for task_type in self.config.task_types:
                if task_type in pair_files:
                    continue
                file_pattern = TASK_FILE_PATTERNS.get(task_type, f"{task_type}_pairs.txt")
                if "*" in file_pattern:
                    matches = sorted(str(p) for p in data_dir.glob(file_pattern))
                    if matches:
                        pair_files[task_type] = matches
                        logger.info("Auto-discovered %d pair files for %s: %s", len(matches), task_type, matches)
                    else:
                        logger.warning("No files matched %s for %s in %s", file_pattern, task_type, data_dir)
                else:
                    candidate_path = data_dir / file_pattern
                    if candidate_path.exists():
                        pair_files[task_type] = str(candidate_path)
                        logger.info("Auto-discovered pair file for %s: %s", task_type, candidate_path)
                    else:
                        logger.warning("Could not find pair file for %s at %s", task_type, candidate_path)

        return pair_files

    def run(self, iteration_value: Any = None, iteration_index: int = 0) -> dict[str, Any]:
        """Run the LT-Swap evaluation pipeline."""
        setup_logging()

        logger.info("Running LT-Swap evaluation for model: %s", self.config.model.name)

        model = self._create_model()

        pair_files = self._get_pair_files()
        results = {}
        for task_type in self.config.task_types:
            pair_file = pair_files.get(task_type)
            if not pair_file:
                logger.warning("No pair file configured for task type: %s", task_type)
                continue

            paths = [pair_file] if isinstance(pair_file, str) else list(pair_file)
            missing = [p for p in paths if not Path(p).exists()]
            if missing:
                logger.error("Pair file(s) not found for %s: %s", task_type, missing)
                continue

            try:
                task_results = self._evaluate_task(task_type, pair_file, model)
                results[task_type] = task_results
                logger.info("%s avg_accuracy: %.3f", task_type, task_results["avg_accuracy"])

                task_output_dir = self.output_dir / task_type
                task_output_dir.mkdir(parents=True, exist_ok=True)
                task_results_path = task_output_dir / "results.yaml"
                with task_results_path.open("w") as f:
                    f.write(to_yaml(task_results))

            except Exception as e:
                logger.exception("Failed to evaluate task %s: %s", task_type, e)
                results[task_type] = {"error": str(e)}

        results_path = self.output_dir / "ltswap_results.yaml"
        with results_path.open("w") as f:
            f.write(to_yaml(results))

        logger.info("LT-Swap evaluation complete. Results saved to %s", results_path)

        return results
