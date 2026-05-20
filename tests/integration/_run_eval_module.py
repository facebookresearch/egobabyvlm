# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run a single EvalModule directly (bypasses Stopes/submitit) for golden comparisons.

Used by ``tests/integration/test_real_data_local.py`` to invoke each eval the
same way a developer would when debugging locally — via Hydra compose, with the
launcher module imported just for ConfigStore registration.
"""

from __future__ import annotations

import argparse
import importlib
import json

from hydra import compose, initialize_config_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--launcher-module", default="evaluation.eval_launcher")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", default="model", help="`+model` for legacy, `model` for ported.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    importlib.import_module(args.launcher_module)

    overrides = [
        f"eval={args.eval}",
        f"{args.model_key}={args.model}",
        f"eval.output_dir={args.output_dir}",
        *args.extra,
    ]

    with initialize_config_dir(version_base=None, config_dir=args.config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    from evaluation.base.eval_module import EvalModule

    eval_module = EvalModule.build(cfg.eval)
    results = eval_module.run(iteration_value=0, iteration_index=0)

    print("==RESULTS==")
    print(json.dumps(results, default=str, indent=2))


if __name__ == "__main__":
    main()
