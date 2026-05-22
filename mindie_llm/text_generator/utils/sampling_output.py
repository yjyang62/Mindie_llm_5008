# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SamplingOutput:
    sequence_ids: np.ndarray
    parent_sequence_ids: np.ndarray
    group_indices: List[Tuple[int, int]]
    repeating_indices: np.ndarray
    token_ids: np.ndarray
    logprobs: np.ndarray
    top_token_ids: np.ndarray
    top_logprobs: np.ndarray
    cumulative_logprobs: np.ndarray
    num_new_tokens: np.ndarray
    num_top_tokens: np.ndarray = None
    seeds: np.ndarray = None

    def to_deprecated(self):
        return self.token_ids, self.logprobs