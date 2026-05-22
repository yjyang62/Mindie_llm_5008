# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from .math import (cos, sin, neg, logical_not, logical_or, logical_and, equal, amax, amin, sum_,
                   std, norm, linear, rms_norm, grouped_matmul)
from .position_embedding import rope
from .attention.paged_attention import paged_attention, MaskType, reshape_and_cache
from .activation import ActType, GeluMode, activation, softmax
from .attention.multi_latent_attention import multi_latent_attention
from .tensor_manipulation import cat, split
from .block_copy import copy_blocks
from .index import sort, argsort, gather
from .moe import moe_topk_softmax, moe_token_unpermute, gating, moe_init_routing, group_topk