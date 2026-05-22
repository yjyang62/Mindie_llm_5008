# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from ... import nn
from ...nn.functional import split, cat
from ...utils.initial import NPUSocInfo
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...layers import InferenceMode
from ...models.base.config import BaseConfig
from ...models.base.mindie_llm_config import ModelStatus
from ...nn.distributed import distributed as dist


class Mlp(nn.Module):
    def __init__(
            self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str,
            config_metadata: ModelStatus, **kwargs):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.mapping = file_loader.mapping
        self.need_nz = NPUSocInfo().need_nz

        self.gate_up = None
        self.down = None

    def forward(self, inputs, is_prefill: bool = True, **kwargs):
        """
        Args:
            kwargs:
                enable_lora: exists when LoRA is enabled
                group_list: exists when multi-LoRA is activated
        """
        inference_mode = InferenceMode.PREFILL if is_prefill else InferenceMode.DECODE
        if len(self.gate_up) == 1:
            gate_up_out = self.gate_up(inputs, inference_mode=inference_mode, **kwargs)[0]
            if self.need_nz:
                gate_out, up_out = split(gate_up_out, dim=-1, split_size_or_sections=2)
        else:
            gate_out, up_out = self.gate_up(inputs, inference_mode=inference_mode, **kwargs)
            if not self.need_nz:
                gate_up_out = cat([gate_out, up_out], dim=-1)

        if self.need_nz:
            gate_out_ = nn.functional.activation(gate_out, nn.functional.ActType.SWISH)
            act_out = gate_out_ * up_out
        else:
            act_out = nn.functional.activation(gate_up_out, nn.functional.ActType.SWIGLU)

        down_out = self.down(act_out, inference_mode=inference_mode, **kwargs)
        down_out_ = dist.all_reduce(down_out, process_group=self.mapping.mlp_tp.process_group)
        return down_out_