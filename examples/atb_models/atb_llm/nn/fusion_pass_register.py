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

from .fusion_pass.add_norm_quant_per_tensor import AddNormQuantPerTensorPass
from .fusion_pass.add_norm_quant_per_token import AddNormQuantPerTokenPass
from .fusion_pass.add_norm_with_bias_quant_per_tensor import AddNormWithBiasQuantPerTensorPass
from .fusion_pass.add_norm import AddRmsNormPass
from .fusion_pass.norm_with_bias_quant_per_tensor import NormWithBiasQuantPerTensorPass
from .fusion_pass.w8a8_linear_with_bias_dequant_per_tensor import W8A8LinearWitBiasDequantPerTensorPass
from .fusion_pass.w8a8_linear_dequant_per_token import W8A8LinearDequantPerTokenPass
from .fusion_pass.w8a8_linear_dequant_per_token_with_bias import W8A8LinearDequantPerTokenWithBiasPass
from .fusion_pass.matmul_all_reduce import MatmulAllReducePass
from .fusion_pass.swiglu_weight_pack import SwigluWeightPackPass
from .fusion_pass.swiglu_weight_no_pack import SwigluWeightNoPackPass
from .fusion_pass.swiglu_weight_pack_quant_per_token import SwigluWeightPackQuantPerTokenPass
from .fusion_pass.swiglu_weight_pack_quant_per_channel import SwigluWeightPackQuantPerChannelPass
from .fusion_pass.swiglu_weight_no_pack_quant_per_token import SwigluWeightNoPackQuantPerTokenPass
from .fusion_pass.swiglu_weight_no_pack_quant_per_channel import SwigluWeightNoPackQuantPerChannelPass


def register_all_fusion_pass(fusion_pass_manager):
    fusion_pass_manager.register_fusion_pass("AddNormQuantPerTensorPass", AddNormQuantPerTensorPass())
    fusion_pass_manager.register_fusion_pass("AddNormQuantPerTokenPass", AddNormQuantPerTokenPass())
    fusion_pass_manager.register_fusion_pass("AddNormWithBiasQuantPerTensorPass", AddNormWithBiasQuantPerTensorPass())
    fusion_pass_manager.register_fusion_pass("AddRmsNormPass", AddRmsNormPass())
    fusion_pass_manager.register_fusion_pass("NormWithBiasQuantPerTensorPass", NormWithBiasQuantPerTensorPass())
    fusion_pass_manager.register_fusion_pass("W8A8LinearWitBiasDequantPerTensorPass", W8A8LinearWitBiasDequantPerTensorPass(), True)
    fusion_pass_manager.register_fusion_pass("W8A8LinearDequantPerTokenWithBiasPass", W8A8LinearDequantPerTokenWithBiasPass(), True)
    fusion_pass_manager.register_fusion_pass("W8A8LinearDequantPerTokenPass", W8A8LinearDequantPerTokenPass(), True)
    fusion_pass_manager.register_fusion_pass("MatmulAllReducePass", MatmulAllReducePass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightPackPass", SwigluWeightPackPass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightNoPackPass", SwigluWeightNoPackPass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightPackQuantPerTokenPass", SwigluWeightPackQuantPerTokenPass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightPackQuantPerChannelPass", SwigluWeightPackQuantPerChannelPass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightNoPackQuantPerTokenPass", SwigluWeightNoPackQuantPerTokenPass())
    fusion_pass_manager.register_fusion_pass("SwigluWeightNoPackQuantPerChannelPass", SwigluWeightNoPackQuantPerChannelPass())
