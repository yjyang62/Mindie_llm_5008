# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List
import numpy as np

import torch


class QLenModifier:
    @staticmethod
    def modify_inputs(
            inputs: List[torch.Tensor],
            runtime_param,
            device,
            **kwargs
        ) -> None:
        q_lens = kwargs.get("q_lens", None)
        mask = kwargs.get("attn_mask", None)
        if q_lens is None or mask is None:
            return
        is_prefill = kwargs.get("is_prefill", False)
        enable_prefill_pa = kwargs.get("enable_prefill_pa", False)
        enable_splitfuse_pa = kwargs.get("enable_splitfuse_pa", False)
        # splitfuse and prefixcache (A2/A3) will use splitfusepa
        if is_prefill and enable_prefill_pa and enable_splitfuse_pa:
            q_len_cumsum = np.cumsum(np.array(q_lens))
            q_len_tensor = torch.from_numpy(q_len_cumsum).to(device).to(torch.int32)
            inputs.append(q_len_tensor)
            runtime_param.update({"qLen": q_len_cumsum.tolist()})
        # lookahead, memory_decoding and prefixcache (300I) will use pa_decoder
        else:
            q_len_tensor = torch.tensor(q_lens).to(device).to(torch.int32)

            inputs.append(q_len_tensor)
            runtime_param.update({"qLen": q_lens})