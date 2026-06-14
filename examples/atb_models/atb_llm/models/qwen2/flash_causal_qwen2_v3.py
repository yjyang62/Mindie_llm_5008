# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.nn.functional import gather
from atb_llm.nn.distributed import distributed as dist
from atb_llm.models.qwen2.modeling_qwen2_python import Qwen2Model
from atb_llm.models.base.flash_causal_lm_v3 import FlashCausalLMV3, torch_to_mindie_graph
from atb_llm.models.base.feature_decorator.v3.singlelora_decorator import SingleLoraDecorator
from atb_llm.models.base.feature_decorator.v3.multilora_decorator import MultiLoraDecorator
from atb_llm.layers.linear.linear import ColumnParallelLinear
from atb_llm.models.base.mindie_llm_config import ModelStatus, MindIELLMConfig
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader


class Qwen2ModelStatus(ModelStatus):
    enable_intra_layer_add_norm: bool = False
    enable_inter_layer_add_norm: bool = False

    @classmethod
    def from_config(cls, mindie_llm_config):
        model_status = super().from_config(mindie_llm_config)
        model_status.enable_intra_layer_add_norm = False
        model_status.enable_inter_layer_add_norm = False
        return model_status


@torch_to_mindie_graph(SingleLoraDecorator, MultiLoraDecorator)
class FlashQwen2ForCausalLMV3(FlashCausalLMV3):
    model_status_cls = Qwen2ModelStatus

    def __init__(self, mindie_llm_config: MindIELLMConfig, weight_loader: SafetensorFileLoader, **kwargs) -> None:
        super().__init__(mindie_llm_config, weight_loader, **kwargs)
        self.model = Qwen2Model(
            config=mindie_llm_config.hf_config,
            file_loader=weight_loader,
            mapping=mindie_llm_config.mapping,
            prefix="model",
            config_metadata=self.model_status,
            **kwargs
        )
        self.lm_head = ColumnParallelLinear(mindie_llm_config.hf_config, weight_loader, ["lm_head"],
                                            llm_config=mindie_llm_config.llm_config)

    def forward(self, **kwargs):
        is_prefill = kwargs.get("is_prefill", True)
            
        if is_prefill:
            lm_head_indices = kwargs.get("lm_head_indices")
            hidden_states = self.model(**kwargs)
            hidden_states_ = gather(hidden_states, 0, lm_head_indices)
            lm_head_out = self.lm_head(hidden_states_)
        else:
            hidden_states = self.model(**kwargs)
            lm_head_out = self.lm_head(hidden_states)
        if self.mindie_llm_config.mapping.world_size > 1:
            logits = dist.all_gather(lm_head_out)
            logits = logits.permute([1, 0, 2])
            return {"model_out": logits}
        return {"model_out": lm_head_out}

