# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from collections import OrderedDict

import torch
from torch import nn
import torch_npu

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger
from atb_llm.utils.initial import NPUSocInfo
from atb_llm.utils.quantize.pack_type import TransposeType
from atb_llm.utils.quantize.quant_type import LinearTypeV2


class LinearUtils:
    soc_info = None
    quant_version: str = "0.0.0"

    def __init__(self):
        self.prefix = None
        self.linear_desc = LinearTypeV2.INVALID
        self.trans_flag = TransposeType.TRANSPOSE
        self.has_bias = False
        if not LinearUtils.soc_info:
            LinearUtils.set_soc_info()
        self.prefixes = []
        self.num_linear_before_pack = 1
        self.tensor_parallel_dim = 0
        self.align_size = 1
        self.nd_weight = False

    @classmethod
    def set_soc_info(cls):
        cls.soc_info = NPUSocInfo()
    
    @classmethod
    def weight_format_cast(cls, tensor):
        if not cls.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        return tensor

    def set_transpose(self, trans_flag):
        if trans_flag != TransposeType.INVALID and self.trans_flag != trans_flag:
            if trans_flag != TransposeType.TRANSPOSE:
                self.trans_flag = trans_flag
            if not trans_flag and self.weight.dim() != 3:
                # 3: dimension of the weight tensor
                self.weight = nn.Parameter(torch.transpose(self.weight, -2, -1).contiguous(), requires_grad=False)

    def check_transpose(self, weight):
        if self.soc_info.need_nz:
            return TransposeType.TRANSPOSE

        if not ENV.auto_transpose_enable:
            if self.soc_info.matmul_nd_nz:
                logger.warning("NZ weight format is enabled. To ensure hardware compatibility, "
                               "weights must be transposed to [k, n]. The environment variable "
                               "`ATB_LLM_ENABLE_AUTO_TRANSPOSE=0` is being ignored.")
                ENV.auto_transpose_enable = True
                return TransposeType.NOT_TRANSPOSE
            return TransposeType.TRANSPOSE
        
        if self.soc_info.matmul_nd_nz:
            # transpose weights to [k, n] when using nz format
            return TransposeType.NOT_TRANSPOSE
        
        is_k_divisible = weight.shape[-1] % 256 == 0
        is_n_divisible = weight.shape[-2] % 256 == 0
        if not is_k_divisible and is_n_divisible and weight.dim() != 3:
            return TransposeType.NOT_TRANSPOSE
        return TransposeType.TRANSPOSE

    def transpose_weight_as_need(self):
        self.set_transpose(self.check_transpose(self.weight))

    def get_weights(self, prefix):
        self.prefix = prefix
        weights_dict = OrderedDict()
        for name, buf in self.named_buffers():
            weights_dict[f"{prefix}.{name}"] = self.weight_format_cast(buf.data)
        return weights_dict
