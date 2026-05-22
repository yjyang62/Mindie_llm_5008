# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import MagicMock

import torch

from atb_llm.utils.layers import FA3


class TestFA3(unittest.TestCase):
    def test_load(self):
        def get_tensor(prefix, ignore_tensor_correction=False):
            if "q." in prefix:
                return torch.ones(64, 1)
            elif ("k." in prefix) or ("v." in prefix):
                return torch.ones(8, 1)
            return None
        weights = MagicMock()
        weights.sharded = False
        weights.get_tensor = get_tensor
        weights.process_group.size.return_value = 8
        weights.process_group.rank.return_value = 3
        FA3.load("q", "k", "v", weights, 4)
        weights.process_group.size.return_value = 128
        FA3.load("q", "k", "v", weights, 4)
    
    def test_mla_load(self):
        def get_tensor(prefix, ignore_tensor_correction=False):
            if "q." in prefix:
                return torch.ones(64, 1)
            elif ("k." in prefix) or ("v." in prefix):
                return torch.ones(8, 1)
            return None
        weights = MagicMock()
        weights.sharded = False
        weights.get_tensor = get_tensor
        weights.process_group.size.return_value = 8
        weights.process_group.rank.return_value = 3
        weights.mapping.attn_inner_sp.group_size = 2
        FA3.load_mla("q", "k", "v", weights, 4)
        weights.mapping.attn_inner_sp.group_size = 1
        FA3.load_mla("q", "k", "v", weights, 4)

    def test_mla_load_sharded(self):
        weights = MagicMock()
        weights.sharded = True
        weights.get_tensor.return_value = torch.ones([1])
        module, per_layer_quant = FA3.load_mla("q", "k", "v", weights, 4)
        self.assertFalse(per_layer_quant)


if __name__ == '__main__':
    unittest.main()