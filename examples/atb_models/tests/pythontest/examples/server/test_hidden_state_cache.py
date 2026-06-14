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

import torch
import numpy as np

from examples.server.cache import HiddenStateCache


class TestHiddenStateCache(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 129280
        self.mtp = 1
        self.max_batch_size = 1024
        self.dtype = torch.float16
        self.device = "cpu"
        self.hidden_state_cache = HiddenStateCache(
            self.hidden_size,
            self.mtp,
            self.max_batch_size,
            self.dtype,
            self.device
        )

    def test_hidden_state_cache(self):
        req_ids = [0, 1, 2, 3]
        batch = len(req_ids)
        cache_ids = self.hidden_state_cache.save_input_cache(req_ids, is_prefill=True)
        self.assertTrue(np.array_equal(cache_ids, np.array([0, 1, 2, 3], dtype=np.int32)))

        hidden_states = torch.rand(batch * (self.mtp + 1), self.hidden_size, dtype=self.dtype)
        self.hidden_state_cache.save_hidden_states(cache_ids, hidden_states)
        result = self.hidden_state_cache.get_hidden_states(cache_ids)

        self.assertTrue(torch.allclose(hidden_states, result))
        self.hidden_state_cache.clear_cache(req_ids)


if __name__ == "__main__":
    unittest.main()