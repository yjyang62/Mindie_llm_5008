# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from typing import Any
from dataclasses import dataclass, field

from mindie_llm.runtime.utils.helpers.safety.path import standardize_path


@dataclass
class LoadConfig:
    """
    Configuration class for loading models, encapsulating all parameters related to model loading.

    Attributes:
        model_name_or_path (str): Name or local path of the model
        max_position_embeddings (int): Maximum position embedding length, can be None to use the model's default
        trust_remote_code (bool): Whether to trust remote code (for custom models), default is False
        load_tokenizer (bool): Whether to load the tokenizer, default is True
        tokenizer_path (str): Custom path for the tokenizer, can be None to use the same path as the model
        llm_config_path (str): Path to the LLM configuration file for custom model configuration, can be None
        models_dict (dict): Dictionary of models for multi-model management or preloaded configuration
    """

    model_name_or_path: str
    max_position_embeddings: int | None = None
    trust_remote_code: bool = False
    load_tokenizer: bool = True
    tokenizer_path: str | None = None
    llm_config_path: str | None = None
    models_dict: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_name_or_path

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LoadConfig":
        """
        Create a LoadConfig instance from a dictionary.

        Args:
            config_dict (dict[str, Any]): Configuration dictionary

        Returns:
            LoadConfig: Configuration instance
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def validate(self) -> None:
        """
        Validate the legitimacy of configuration parameters.

        Raises:
            ValueError: When parameters are invalid
        """
        self.model_name_or_path = standardize_path(self.model_name_or_path, check_link=False)

        if self.tokenizer_path is not None:
            self.tokenizer_path = standardize_path(self.tokenizer_path, check_link=False)

        if self.llm_config_path is not None:
            self.llm_config_path = standardize_path(self.llm_config_path, check_link=False)

        if self.max_position_embeddings is not None and self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be a positive integer, current value: {self.max_position_embeddings}"
            )

        if self.models_dict is not None:
            max_json_length = 4096
            if len(self.models_dict) > max_json_length:
                raise ValueError(
                    f"The length of plugin_params ({len(self.models_dict)}) is too long, "
                    f"it should be within (0, {max_json_length}]"
                )
