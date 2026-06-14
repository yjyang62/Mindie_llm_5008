# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass
import torch

from atb_llm.models.qwen3_vl.router_qwen3_vl import Qwen3vlRouter
from .config_qwen3_vl_moe import Qwen3vlmoeConfig


@dataclass
class Qwen3vlmoeRouter(Qwen3vlRouter):

    def __post_init__(self):
        super().__post_init__()
        self.model_type = "qwen3_vl_moe"
        self.model_type_cap = "Qwen3vlmoe"
    
    def get_config(self):
        config = Qwen3vlmoeConfig.from_dict(self.config_dict)
        config.model_name_or_path = self.model_name_or_path
        if config.text_config.dtype == torch.bfloat16:
            config.torch_dtype = torch.bfloat16
        elif config.text_config.dtype == torch.float16:
            config.torch_dtype = torch.float16
        else:
            err_msg = "`torch_dtype` is only supported for type `float16` and `bfloat16`"
            raise NotImplementedError(err_msg)
        super().check_config(config)
        return config
