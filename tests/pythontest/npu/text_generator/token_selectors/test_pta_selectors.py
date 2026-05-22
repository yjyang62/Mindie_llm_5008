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

import numpy as np
import torch
import torch_npu

from mindie_llm.text_generator.samplers.sampler_params import HandlerParams, SelectorParams
from mindie_llm.text_generator.samplers.token_selectors.pta_selectors import PTA_SELECTOR_REGISTRY
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestPtaHandlers(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.test_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        ).to(self.device)
        self.expected_logits = [3, 2, 3]
        self.expected_beam_search_logits = [3, 2, 1, 0, 2, 3, 1, 0, 3, 2, 1, 0]
        self.params = HandlerParams(backend_type='atb', rank=0)
        self.selector_params = SelectorParams()
        self.params.batch_size = 3
        self.params.vocab_size = 4
        self.params.output_token_ids = torch.tensor(
            [[0, 0, 1, 1, 2, 2, 3, 3],
             [2, 3, 3, 3, 3, 3, 4, 4],
             [4, 4, 4, 4, 4, 4, 4, 4]], dtype=torch.int64
        ).to(self.device)
        self.metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            is_prefill=False,
            to_tensor=torch.tensor
        )
        self.metadata.max_beam_width = 2
        self.metadata.beam_width_array = np.array([2, 2, 2])
        self.metadata.use_beam_search_array = np.array([1, 1, 1])

    def test_greedy_search(self):
        greedy_search_lh = PTA_SELECTOR_REGISTRY['greedy_search'](self.selector_params)
        output = greedy_search_lh(self.test_logits, self.metadata)
        torch_npu.npu.synchronize()
        self.assertTrue(np.array_equal(output.token_ids, self.expected_logits))


if __name__ == '__main__':
    unittest.main()