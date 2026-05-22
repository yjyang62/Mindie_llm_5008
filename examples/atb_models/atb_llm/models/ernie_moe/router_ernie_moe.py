# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from ..base.router import BaseRouter
from .config_ernie_moe import ErniemoeConfig
from ...utils.log import logger
from ...utils.log.error_code import ErrorCode


@dataclass
class ErniemoeRouter(BaseRouter):
    def get_config(self):
        config = ErniemoeConfig.from_dict(self.config_dict)
        self.check_config_ernie(config)
        return config

    def check_config_ernie(self, config):
        super().check_config(config)
        attribute_ranges = {
            "moe_num_experts": (1, 64),
            "moe_layer_start_index": (0, 53),
            "moe_intermediate_size": (1, 2147483647),
            "moe_k": (1, 64)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr) or getattr(config, attr) is None:
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                msg = f"self._config.{attr} must be between {min_val} and {max_val}"
                logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
                raise ValueError(msg)
            
        if getattr(config, "moe_k", 0) > getattr(config, "moe_num_experts", 0):
            msg = "config.moe_k must be smaller than or equal to config.moe_num_experts!"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)

        if getattr(config, "moe_layer_start_index") > getattr(config, "num_hidden_layers") - 1:
            msg = f"config.moe_layer_start_index should be less than config.num_hidden_layers - 1, " \
                  f"but {config.moe_layer_start_index=}, {config.num_hidden_layers=}"
            logger.error(msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(msg)