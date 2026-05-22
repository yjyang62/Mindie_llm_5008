# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from atb_llm.models.qwen3_vl.flash_causal_qwen3_vl import FlashQwen3vlForCausalLM
from .modeling_qwen3_vl_moe_text import FlashQwen3VLMOETextModelForCausalLM


class FlashQwen3vlmoeForCausalLM(FlashQwen3vlForCausalLM):

    def init_llm(self):
        self.language_model = FlashQwen3VLMOETextModelForCausalLM(self.config.text_config,
                                                                  self.weights,
                                                                  llm_config=self.llm_config,
                                                                  inference_mode=self.inference_mode)
