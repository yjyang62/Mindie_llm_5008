# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from atb_llm import nn
from atb_llm.models.base.mindie_llm_config import ModelStatus
from atb_llm.models.base.config import BaseConfig
from atb_llm.nn.functional import (softmax, sum_, moe_init_routing, moe_token_unpermute,
                                   gating, argsort, sort, gather)
from atb_llm.utils.loader.safetensor_file_loader import SafetensorFileLoader
from atb_llm.utils.quantize.pack_type import DataType


class Moe(nn.Module):
    def __init__(
            self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str,
            config_metadata: ModelStatus, **kwargs):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.mapping = file_loader.mapping
        self.topk_num = getattr(config, "num_experts_per_tok", 8)
        self.expert_num = getattr(config, "num_experts", 64)
        self.has_shared_expert = False
        self.enable_fused_routing = True

        self.gate = None
        self.fused_experts = None
        self.shared_expert = None

    def moe_gate(self, inputs, **kwargs):
        router_logits = self.gate(inputs)
        return router_logits

    def moe_route(self, router_logits, **kwargs):
        router_weights = softmax(router_logits, dims=[1])
        expert_weights, select_experts = sort(router_weights, self.topk_num)
        return expert_weights, select_experts

    def moe_block(self, inputs, expert_weights, select_experts, **kwargs):
        if self.enable_fused_routing:
            sorted_hidden_states, idx, group_list = moe_init_routing(inputs, select_experts,
                                                                     self.topk_num, self.expert_num)
            group_list_ = group_list.to(DataType.ACL_INT64)
            mlp_out = self.fused_experts(sorted_hidden_states, group_list_)
            moe_out = moe_token_unpermute(mlp_out, idx, expert_weights)
        else:
            expert_array = kwargs.get("expert_array", None)
            idx, group_list, weight_idx = gating(select_experts, expert_array, self.topk_num, self.expert_num)
            sorted_hidden_states = gather(inputs, 0, idx)
            mlp_out = self.fused_experts(sorted_hidden_states, group_list)
            sorted_weights = gather(expert_weights, 0, weight_idx)
            mlp_out_weighted = mlp_out * sorted_weights
            dummy_zero, dummy_one, rev_idx = argsort(weight_idx)
            rev_sorted_hidden_states = gather(mlp_out_weighted, 0, rev_idx)
            moe_out = sum_(rev_sorted_hidden_states, dims=[-1])
        return moe_out

    def moe_shared_expert(self, inputs, moe_out, **kwargs):
        share_expert_out = self.shared_expert(inputs)
        moe_out = moe_out + share_expert_out
        return moe_out

    def forward(self, inputs, **kwargs):
        router_logits = self.moe_gate(inputs, **kwargs)
        expert_weights, select_experts = self.moe_route(router_logits, **kwargs)
        moe_out = self.moe_block(inputs, expert_weights, select_experts, **kwargs)
        if self.has_shared_expert:
            moe_out = self.moe_shared_expert(inputs, moe_out, **kwargs)
        return moe_out
