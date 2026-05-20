# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import hydra
from omegaconf import MISSING, OmegaConf
from stopes.core import StopesModule


@dataclass
class EvalConfig:
    _target_: str = MISSING
    name: str = MISSING

    #: Output directory for results.
    output_dir: str = MISSING

    seed: int = 42


class EvalModule(StopesModule, ABC):
    def __init__(self, config: EvalConfig, config_class: type[EvalConfig]) -> None:
        super().__init__(config, config_class)

    @abstractmethod
    def run(self, iteration_value: Any, iteration_index: int) -> Any:
        pass

    def name(self) -> str:
        return self.config.name

    def array(self) -> list[dict[str, Any]] | None:
        return None

    def get_config_for_cache(self):
        """
        Return a dictionary corresponding to the config. Here transient attributes,
        attributes of stopes.core.stopes_modules.Requirements will be excluded
        """
        config_for_cache = OmegaConf.to_container(self.config, resolve=False)
        assert isinstance(config_for_cache, dict), "StopesModule.config need to be a dict config."
        OVERWRITE_VALUE_FOR_CACHE = -1

        def deep_overwrite(dct: dict, transient_cfg: dict):
            for k, v in dct.items():
                if k in transient_cfg:
                    if transient_cfg[k] is True:
                        dct[k] = OVERWRITE_VALUE_FOR_CACHE
                        continue
                    if isinstance(transient_cfg[k], dict):
                        transient_cfg = transient_cfg[k]
                if isinstance(v, dict):
                    # Some configs are defined dynamically within the root module's compose config
                    if "_target_" in v:
                        sub_transient_cfg = {}
                        target = v["_target_"]

                        try:
                            # First try to get the target as a class
                            sub_cls = hydra.utils.get_class(target)

                            # If sub_cls is a config / dataclass
                            if dataclasses.is_dataclass(sub_cls):
                                sub_transient_cfg = self._get_transient_configs(sub_cls)
                            # sub_cls is a normal Python class, merge from the class property
                            # `transient_attributes` if exists
                            elif hasattr(sub_cls, "transient_attributes"):
                                sub_transient_cfg = dict.fromkeys(sub_cls.transient_attributes, True)
                        except (ImportError, AttributeError, ValueError):
                            # Target might be a method (e.g., transformers.AutoModelForMaskedLM.from_pretrained)
                            # Try to get it as a method instead
                            try:
                                sub_method = hydra.utils.get_method(target)
                                # Methods typically don't have transient_attributes, but check just in case
                                if hasattr(sub_method, "transient_attributes"):
                                    sub_transient_cfg = dict.fromkeys(sub_method.transient_attributes, True)
                            except (ImportError, AttributeError, ValueError):
                                pass  # Neither valid class nor method, skip

                        transient_cfg = {**transient_cfg, **sub_transient_cfg}
                    deep_overwrite(v, transient_cfg)

        deep_overwrite(config_for_cache, self.transient_configs)
        return config_for_cache
