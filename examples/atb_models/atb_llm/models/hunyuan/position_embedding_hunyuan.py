# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
from atb_llm.utils.layers import PositionRotaryEmbedding


class HunyuanRotaryEmbedding(PositionRotaryEmbedding):
    @classmethod
    def static(cls, dim, base, device, scaling_factor=1.0, **kwargs):
        scaling_alpha = kwargs.get("scaling_alpha", 1.0)
        try:
            new_base = base * scaling_alpha ** (dim / (dim - 2))
            inv_freq = 1.0 / (
                    new_base 
                    ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
            ).to(torch.float)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        return cls(inv_freq=inv_freq, base=new_base)
    
    def update_cos_sin_cache_total(self, dtype, device, seqlen):
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached_total.device != device
                or self._cos_cached_total.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached_total = (torch.cos(emb)).to(dtype)
            self._sin_cached_total = (torch.sin(emb)).to(dtype)