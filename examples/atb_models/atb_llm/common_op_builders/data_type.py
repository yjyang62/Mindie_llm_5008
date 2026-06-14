# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum


class CommonOpBuilderType(str, Enum):
    BASE = "BASE"
    LINEAR = "LINEAR"
    LINEAR_PARALLEL = "LINEAR_PARALLEL"
    FUSION_LINEAR = "FUSION_LINEAR"
    WORD_EMBEDDING = "WORD_EMBEDDING"
    POSITIONAL_EMBEDDING = "POSITIONAL_EMBEDDING"
    NORM = "NORM"
    QKV = "QKV"
    ATTENTION = "ATTENTION"
    LM_HEAD = "LM_HEAD"
    GATE_UP = "GATE_UP"
    ACTIVATION = "ACTIVATION"
    ROPE = "ROPE"
    INTEGRATED_GMM = "INTEGRATED_GMM"


class NormType(str, Enum):
    RMSNORM = "RmsNorm"
    LAYERNORM = "LayerNorm"


class CommonOpBuilderOwner(str, Enum):
    DEFAULT = "DEFAULT"


class ActivationType(str, Enum):
    SWIGLU = "swiglu"
    SWISH = "swish"


class OperationBackend(str, Enum):
    ATB = "ATB"
    ACLNN = "ACLNN"