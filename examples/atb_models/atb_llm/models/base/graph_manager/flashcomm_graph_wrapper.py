# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.models.base.graph_manager.graph_wrapper import ATBGraphWrapper
from atb_llm.models.base.graph_manager.compatible_matrix import FeatureType
from atb_llm.models.base.flash_causal_lm import FlashForCausalLM


class FlashCommGraphWrapper(ATBGraphWrapper):
    def __init__(self):
        super().__init__()

        self.feature_name = FeatureType.FLASHCOMM
        self.feature_params = {"enableFlashComm": True, "backend": "hccl"}
    
    def activate(self, context: FlashForCausalLM, runtime_params, **kwargs) -> bool:
        if context.flash_comm_modifier.active:
            return True
        else:
            return False