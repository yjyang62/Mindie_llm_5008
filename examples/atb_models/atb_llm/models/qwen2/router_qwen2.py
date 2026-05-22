# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from ..base.model_utils import safe_get_tokenizer_from_pretrained
from ..base.router import BaseRouter
from .config_qwen2 import Qwen2Config
from .input_builder_qwen2 import Qwen2InputBuilder


QWEN_TEST_INTERMEDIATE_SIZE = "intermediate_size"
QWEN_TEST_MAX_WINDOW_LAYERS = "max_window_layers"
QWEN_TRANSFORMERS_VERSION = "transformers_version"
MAX_INT = 2147483647



@dataclass
class Qwen2Router(BaseRouter):

    def __post_init__(self):
        super().__post_init__()
        self.transformers_version = self.config_dict["transformers_version"]
        self.prealloc_weight_mem_on_npu = True

    @property
    def embedding_model_name(self):
        """
        次级模型:主要用于区分qwen-gte
        """
        auto_map = "auto_map"
        sequence_classification = "AutoModelForSequenceClassification"
        if auto_map in self.config_dict and \
            sequence_classification in self.config_dict[auto_map]:
            text_embedding_model_name = "gte"
        else:
            text_embedding_model_name = ""
            
        return text_embedding_model_name

    @property
    def is_qwen1_5_or_2(self):
        qwen1_5_or_2 = False
        if self.config_dict['model_type'] == "qwen2" and QWEN_TEST_INTERMEDIATE_SIZE in self.config_dict:
            if self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 2816:
                qwen1_5_or_2 = True  # "qwen1.5_0.5b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 5504:
                qwen1_5_or_2 = True  # "qwen1.5_1.8b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 6912:
                qwen1_5_or_2 = True  # "qwen1.5_4b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 11008:
                qwen1_5_or_2 = True  # "qwen1.5_7b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 13696:
                qwen1_5_or_2 = True  # "qwen1.5_14b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 27392:
                qwen1_5_or_2 = True  # "qwen1.5_32b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 24576:
                qwen1_5_or_2 = True  # "qwen1.5_72b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 49152:
                qwen1_5_or_2 = True  # "qwen1.5_110b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 4864:
                if QWEN_TEST_MAX_WINDOW_LAYERS in self.config_dict and \
                        self.config_dict[QWEN_TEST_MAX_WINDOW_LAYERS] == 24:
                    qwen1_5_or_2 = True  # "qwen2_0.5b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 8960:
                if QWEN_TEST_MAX_WINDOW_LAYERS in self.config_dict and \
                        self.config_dict[QWEN_TEST_MAX_WINDOW_LAYERS] == 28:
                    qwen1_5_or_2 = True  # "qwen2_1.5b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 18944:
                if QWEN_TRANSFORMERS_VERSION in self.config_dict and \
                        self.config_dict[QWEN_TRANSFORMERS_VERSION] == "4.41.2":
                    qwen1_5_or_2 = True  # "qwen2_7b"
            elif self.config_dict[QWEN_TEST_INTERMEDIATE_SIZE] == 29568:
                if QWEN_TEST_MAX_WINDOW_LAYERS in self.config_dict and \
                        self.config_dict[QWEN_TEST_MAX_WINDOW_LAYERS] == 80:
                    qwen1_5_or_2 = True  # "qwen2_72b"
        return qwen1_5_or_2

    def get_config(self):
        self.config_dict.update({"transformers_version": self.transformers_version})
        config = Qwen2Config.from_dict(self.config_dict)
        self.checkout_config_qwen(config)
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.tokenizer_path,
            padding_side='left',
            trust_remote_code=self.trust_remote_code
        )

    def checkout_config_qwen(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attention_dropout': (0, MAX_INT),
            'max_window_layers': (1, MAX_INT),
            'num_key_value_heads': (1, MAX_INT),
            'rope_theta': (1, MAX_INT),
            'sliding_window': (1, MAX_INT)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")

    def get_input_builder(self):
        kwargs = {"is_qwen1_5_or_2": self.is_qwen1_5_or_2}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        if hasattr(self.config, "max_position_embeddings") and self.config.max_position_embeddings:
            kwargs["max_length"] = self.config.max_position_embeddings
        return Qwen2InputBuilder(self.tokenizer, **kwargs)
        
    def get_tool_call_parser(self):
        if self.is_qwen1_5_or_2:
            return "qwen1_5_or_2"
        else:
            return "qwen2_5"