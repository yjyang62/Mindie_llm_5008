#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import json
import torch
import numpy as np
from atb_llm.utils.env import ENV
from atb_llm.utils.singleton import Singleton
from atb_llm.utils.prof.profiler import prof_expert_routing, is_profiler_enable


class EplbExpertDataCollect(Singleton):
    _initialized = False

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.total_prefill_cumsum_per_expert = None
            self.total_decode_cumsum_per_expert = None
            self.total_warmup_cumsum_per_expert = None
            self.cumsum_list = None
            self.topk_list = None
            self.mtp_cumsum_list = None
            self.mtp_topk_list = None
            self._model = None
            self.is_data_integrity_valid = True
            self.prefill_forward_count = 0
            self.decode_forward_count = 0
            self.last_expert_routing = None

    def set_model_ref(self, model):
        if self._model is None:
            self._model = model

    def split_eplb_expert_data(self, acl_model_out, is_mtp_layer=False, with_topk=False):
        if is_mtp_layer:
            if with_topk:
                self.mtp_cumsum_list = acl_model_out[-2]
                self.mtp_topk_list = acl_model_out[-1]
                return acl_model_out[:-2]
            else:
                self.mtp_cumsum_list = acl_model_out[-1]
                return acl_model_out[:-1]

        first_k_dense_replace = getattr(self._model.config, 'first_k_dense_replace', 0)
        moe_layer_num = self._model.config.num_hidden_layers - first_k_dense_replace

        if with_topk:
            self.cumsum_list = acl_model_out[-2*moe_layer_num::2] # moelayer层cumsum和moelayer层topk交错排列
            self.topk_list = acl_model_out[-2*moe_layer_num+1::2]
            return acl_model_out[:-2*moe_layer_num]
        else:
            self.cumsum_list = acl_model_out[-moe_layer_num:]
            return acl_model_out[:-moe_layer_num]

    def accumulation_expert_cumsum(self, is_prefill=False):
        if is_prefill:
            self.prefill_forward_count += 1
        else:
            self.decode_forward_count += 1
        first_k_dense_replace = getattr(self._model.config, 'first_k_dense_replace', 0)
        moe_layer_num = self._model.config.num_hidden_layers - first_k_dense_replace
        cumsum_list = self.cumsum_list
        if hasattr(self._model, 'num_speculative_tokens') and self._model.num_speculative_tokens:
            if self.mtp_cumsum_list is None:
                self.mtp_cumsum_list = torch.zeros(self._model.num_of_device_expert, dtype=torch.int64, device="npu")
            moe_layer_num += 1
            cumsum_list.append(self.mtp_cumsum_list)
            self.mtp_cumsum_list = None
        whole_model_cumsum_list = torch.cat(cumsum_list).reshape(moe_layer_num, -1)
        self.cumsum_list = None

        if self._model.warmup_is_end and is_prefill:
            if self.total_prefill_cumsum_per_expert is None:
                self.total_prefill_cumsum_per_expert = whole_model_cumsum_list
            else:
                self.total_prefill_cumsum_per_expert += whole_model_cumsum_list
        elif self._model.warmup_is_end:
            if self.total_decode_cumsum_per_expert is None:
                self.total_decode_cumsum_per_expert = whole_model_cumsum_list
            else:
                self.total_decode_cumsum_per_expert += whole_model_cumsum_list
        else:
            if self.total_warmup_cumsum_per_expert is None:
                self.total_warmup_cumsum_per_expert = whole_model_cumsum_list
            else:
                self.total_warmup_cumsum_per_expert += whole_model_cumsum_list

    def get_prefill_token_num_per_expert(self):
        return self._diff_cumsum(self.total_prefill_cumsum_per_expert)

    def get_decode_token_num_per_expert(self):
        return self._diff_cumsum(self.total_decode_cumsum_per_expert)

    def get_topk(self):
        topk_list = self.topk_list
        if hasattr(self._model, 'num_speculative_tokens') and self._model.num_speculative_tokens:
            if self.mtp_topk_list is not None:
                topk_list.append(self.mtp_topk_list)
            return topk_list
        return topk_list
    
    def get_warmup_token_num_per_expert(self):
        return self._diff_cumsum(self.total_warmup_cumsum_per_expert)

    def reset_expert_data(self):
        self.total_warmup_cumsum_per_expert = None
        self.total_prefill_cumsum_per_expert = None
        self.total_decode_cumsum_per_expert = None

    def all_gather_token_num_per_expert(self, is_prefill=False):
        if self._model.acl_all_gather_operation is not None:
            if self._model.warmup_is_end:
                if is_prefill:
                    all_gather_input_tensor = self.get_prefill_token_num_per_expert()
                else:
                    all_gather_input_tensor = self.get_decode_token_num_per_expert()
            else:
                all_gather_input_tensor = self.get_warmup_token_num_per_expert()
            if all_gather_input_tensor is None:
                return None
            output_tensor = self._model.acl_all_gather_operation.execute(
                [all_gather_input_tensor], json.dumps({}))[0]
            self.reset_expert_data()
            return output_tensor.transpose(0, 1)
        else:
            raise RuntimeError('acl_all_gather_operation is None')
        
    def collect_routing_map(self, routing_map, rank):
        if (not is_profiler_enable()) or (not ENV.enable_expert_hotpot_gather):
            self.last_expert_routing = None
            return 
        expert_routing = routing_map.cpu().numpy()
        if self.last_expert_routing is not None:
            if np.array_equal(self.last_expert_routing, expert_routing):
                return
        self.last_expert_routing = expert_routing
        prof_expert_routing(self.last_expert_routing.tolist(), rank)

    def _diff_cumsum(self, cumsum_list):
        if cumsum_list is None:
            return None
        return cumsum_list.diff(prepend=torch.zeros(cumsum_list.shape[0], 1, dtype=torch.int64, device='npu'))
