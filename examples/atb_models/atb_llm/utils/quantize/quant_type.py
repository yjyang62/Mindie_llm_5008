# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum


class QuantType(str, Enum):
    FLOAT = "float"
    W4A16 = "w4a16"
    W8A8 = "w8a8"
    W8A16 = "w8a16"
    W8A8S = "w8a8s"
    W8A8SC = "w8a8sc"
    W8A8_DYNAMIC = "w8a8_dynamic"
    W8A8_PDMIX = "w8a8_pdmix"
    W4A8_DYNAMIC = "w4a8_dynamic"
    W8A8_MIX = "w8a8_mix"
    W16A16S = "w16a16s"
    W16A16SC = "w16a16sc"



QUANT_W8A8_DESC_LIST = [QuantType.W8A8.upper(), QuantType.W8A8S.upper(), QuantType.W8A8_MIX.upper()]


QUANTIZE_DESC_REQUIRED_LIST = [
    QuantType.W8A8,
    QuantType.W8A8S,
    QuantType.W8A8SC,
    QuantType.W8A16,
    QuantType.W4A16,
    QuantType.W8A8_DYNAMIC,
    QuantType.W8A8_PDMIX,
    QuantType.W4A8_DYNAMIC,
    QuantType.W8A8_MIX,
    QuantType.W16A16S,
    QuantType.W16A16SC
]


class LinearTypeV2(int, Enum):
    INVALID = -1
    FLOAT16 = 0
    BFLOAT16 = 1
    W4A16 = 2
    W8A16 = 3
    W8A8 = 4
    W8A8S = 5
    W8A8SC = 6
    W8A8_DYNAMIC = 7
    W8A8_PDMIX = 8
    W4A8_DYNAMIC = 9
    W16A16S = 10
    W16A16SC = 11


class QuantTypeV2(int, Enum):
    NO_QUANT = 0
    LINEAR_W8A8_DEQUANT = 1
    LINEAR_W8A8_QUANT = 2
    W4A16 = 3
    W8A16 = 4
    LINEAR_W8A8_SC_DEQUANT = 5
    LINEAR_W8A8_SC_QUANT = 6
    LINEAR_W8A8_DYNAMIC_DEQUANT = 7
    LINEAR_W8A8_DYNAMIC_QUANT = 8
    W4A8_DYNAMIC = 9
    W16A16SC = 10


class GmmQuantType(int, Enum):
    NONE = 0,
    W8A8_CHANNEL = 1,
    W8A16_CHANNEL = 2,
    W8A8_TOKEN = 3,


def is_same_type(linear_desc_list):
    desc_type = None
    for linear_desc in linear_desc_list:
        if desc_type is None:
            desc_type = linear_desc
        if linear_desc != desc_type:
            return False
    return True


def is_all_float(linear_desc_list):
    for linear_desc in linear_desc_list:
        if linear_desc != LinearTypeV2.FLOAT:
            return False
    return True