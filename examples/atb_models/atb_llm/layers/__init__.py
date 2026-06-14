# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from enum import Enum


class InferenceMode(str, Enum):
    ANY = "ANY"
    PREFILL = "PREFILL"
    DECODE = "DECODE"


class QuantTypeV3(str, Enum):
    INVALID = "INVALID"
    FLOAT16 = "FLOAT16"
    BFLOAT16 = "BFLOAT16"
    W8A8 = "W8A8"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W8A8_MIX = "W8A8_MIX"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
