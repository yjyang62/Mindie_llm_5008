# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import importlib
from dataclasses import dataclass
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode
from ..base.router import BaseRouter
from .config_chatglm import ChatglmConfig
from .input_builder_chatglm import ChatglmInputBuilder


@dataclass
class ChatglmRouter(BaseRouter):
    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if '_name_or_path' not in self.config_dict.keys() and self.config_dict['model_type'] == 'glm':
            model_ver_str = "glm-4"
        else:
            model_ver_str = self.config_dict['_name_or_path'].lower()
        if 'chatglm2' in model_ver_str or 'codegeex2' in model_ver_str:
            model_ver = "v2_6b"
        elif 'chatglm3' in model_ver_str:
            model_ver = "v3_6b"
        elif 'glm-4' in model_ver_str:
            model_ver = 'v4_9b'
        else:
            msg = ("Currently only chatglm2_6b, chatglm3_6b, codegeex2_6b, glm-4-9b are supported. "
                    "If it is the above model, "
                    "please check whether the content of the _name_or_path field in config.json is standardized")
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(msg)
        return model_ver

    def get_config(self):
        config = ChatglmConfig.from_dict(self.config_dict)
        self.check_config_chatglm(config)
        return config

    def check_config_chatglm(self, config):
        super().check_config(config)
        attribute_ranges = {
            'kv_channels': (1, 2147483647),
            'multi_query_group_num': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
            'layernorm_epsilon': (0, 1),
            'attention_dropout': (0, 1),
            'ffn_hidden_size': (1, 2147483647),
            'hidden_dropout': (0, 1),
            'rope_ratio': (0, 2147483647),
            'seq_length': (1, 2147483647),
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if isinstance(value, list):
                value = max(value)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)

    def get_input_builder(self):
        kwargs = {"model_version": self.model_version}
        if self.custom_chat_template:
            kwargs["chat_template"] = self.custom_chat_template
        return ChatglmInputBuilder(self.tokenizer, **kwargs)

    def get_model_cls(self):
        """
        get_model_cls
        """
        model_file_dir_name = f"atb_llm.models.{self.model_type}."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        module_path = f"{model_file_dir_name}{model_file_name}_{self.model_type}"
        module = importlib.import_module(module_path)
        model_cls_name = f"{self.model_type_cap}ForCausalLM"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)
        
    def get_tool_call_parser(self):
        if self.model_version == "v2_6b":
            return "chatglm2_6b"
        elif self.model_version == "v3_6b":
            return "chatglm3_6b"
        elif self.model_version == "v4_9b":
            return "glm_4_9b"
        else:
            return "chatglm"