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
from torch import nn
import numpy as np

from ..layers.linear.linear_utils import LinearUtils
from .pack_type import TransposeType
from .quant_type import LinearTypeV2
from ..initial import NPUSocInfo


class W8A8LinearStatic(nn.Module, LinearUtils):
    def __init__(self, weight, deq_scale, input_scale, quant_bias=None,
                 input_offset=None, bias=None, inter_type=None):
        super().__init__()
        super(nn.Module, self).__init__()
        self.trans_flag = self.check_transpose(weight)
        self.linear_desc = LinearTypeV2.W8A8

        self.register_buffer('weight', weight.to(torch.int8)
                             if self.trans_flag == TransposeType.TRANSPOSE
                             else weight.transpose(-1, -2).contiguous().to(torch.int8))
        self.act_quant_name = 'per_tensor'
        self.register_buffer('input_scale', input_scale)

        if input_offset is not None:
            self.register_buffer('input_offset', input_offset.to(torch.int8))
        else:
            self.register_buffer('input_offset', torch.tensor([], dtype=torch.int8))

        self.weight_quant_name = 'per_channel'

        if NPUSocInfo().soc_version in (100, 101, 102, 103, 104):
            deq_scale = self._transform_deqscale_dtype_to_float(deq_scale)
        elif inter_type == torch.float16 and weight.dtype == torch.int8 and deq_scale.dtype != torch.int64:
            deq_scale = torch.from_numpy(
                np.frombuffer(deq_scale.to(torch.float32).numpy().tobytes(), dtype=np.int32).astype(np.int64))

        self.register_buffer('deq_scale', deq_scale)

        if quant_bias is not None:
            self.register_buffer('quant_bias', quant_bias)
            self.has_bias = True
        else:
            self.quant_bias = None

        self.output_quant_name = 'per_channel'

        if bias is not None:
            self.register_buffer('bias', bias)

    def _transform_deqscale_dtype_to_float(self, deq_scale_int64):
        deq_scale_cpu_int64 = deq_scale_int64.cpu().numpy()
        deq_scale_int32 = np.uint32(deq_scale_cpu_int64)
        original_deq_scale = np.frombuffer(deq_scale_int32.tobytes(), dtype=np.float32).copy()
        tmp_deq_scale = torch.from_numpy(original_deq_scale)
        return tmp_deq_scale.to(deq_scale_int64.device)