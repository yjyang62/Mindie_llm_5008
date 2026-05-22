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

from mindie_llm.text_generator.samplers.token_selectors.cpu_selectors import CPU_SELECTOR_REGISTRY
from mindie_llm.text_generator.samplers.sampler_params import HandlerParams, SelectorParams
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestCpuHandlers(unittest.TestCase):
    def setUp(self):
        self.test_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4],
             [-0.1, 0.01, 0.2, 0.02],
             [-0.3, -0.2, -0.1, 0.01]]
        ).npu()
        self.params = HandlerParams(backend_type='atb', rank=0)
        self.selector_params = SelectorParams()
        self.metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=[np.array([0]), np.array([1]), np.array([2])],
            reserved_sequence_ids=[np.array([4]), np.array([5]), np.array([6])],
            is_prefill=True,
            to_tensor=torch.tensor
        )
        self.speed_mode = 2
        self.use_approx = 1

    def assert_almost_equal_1d(self, res_list, exp_list):
        for r, e in zip(res_list, exp_list):
            self.assertAlmostEqual(r, e, places=6, msg=f'\nres:\n{res_list}\nexp:\n{exp_list}\n')

    def test_top_k_top_p_sampling(self):
        self.params.num_threads = 16
        self.params.npu_id = self.params.rank
        self.params.request_ids = np.array([0, 1, 2])
        self.params.sampling_method = 'multinomial'

        top_k_array = np.array([1, 2, 0])
        top_k_tensor = torch.tensor(top_k_array - 1).npu()
        top_p_array = np.array([0.8, 0.1, 0.5])
        top_p_tensor = torch.tensor(top_p_array).npu().unsqueeze(1)
        do_sample_array = np.array([True, False, True])

        self.metadata.top_k_array = top_k_array
        self.metadata.top_k_tensor = top_k_tensor.unsqueeze(1)
        self.metadata.top_k_disabled_mask_tensor = torch.tensor(top_k_array < 0).npu().unsqueeze(1)
        self.metadata.top_p_array = top_p_array
        self.metadata.top_p_tensor = top_p_tensor
        self.metadata.do_sample_array = do_sample_array
        self.metadata.do_sample_tensor = torch.tensor(do_sample_array).npu()
        self.metadata.seed_array = np.array([10, 1, 1024])

        topk_topp_sampling_lh = CPU_SELECTOR_REGISTRY['top_k_top_p_sampling'](self.selector_params)
        topk_topp_sampling_lh.configure(self.metadata)
        sampling_output = topk_topp_sampling_lh(self.test_logits, self.metadata)
        expected_tokens = [3, 2, 2]
        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)

    def test_top_approx(self):
        self.params.num_threads = 16
        self.params.npu_id = self.params.rank
        self.params.request_ids = np.array([0, 1, 2])
        self.params.sampling_method = 'multinomial'

        top_k_array = np.array([1, 2, 0])
        top_k_tensor = torch.tensor(top_k_array - 1).npu()
        top_p_array = np.array([0.8, 0.1, 0.5])
        top_p_tensor = torch.tensor(top_p_array).npu().unsqueeze(1)
        do_sample_array = np.array([True, False, True])

        self.metadata.top_k_array = top_k_array
        self.metadata.top_k_tensor = top_k_tensor.unsqueeze(1)
        self.metadata.top_k_disabled_mask_tensor = torch.tensor(top_k_array < 0).npu().unsqueeze(1)
        self.metadata.top_p_array = top_p_array
        self.metadata.top_p_tensor = top_p_tensor
        self.metadata.do_sample_array = do_sample_array
        self.metadata.do_sample_tensor = torch.tensor(do_sample_array).npu()
        self.metadata.seed_array = np.array([10, 1, 1024])

        topk_topp_sampling_lh = CPU_SELECTOR_REGISTRY['top_k_top_p_sampling'](self.selector_params)
        topk_topp_sampling_lh.speed_mode = self.speed_mode
        topk_topp_sampling_lh.use_approx = self.use_approx
        topk_topp_sampling_lh.configure(self.metadata)
        sampling_output = topk_topp_sampling_lh(self.test_logits, self.metadata)
        expected_tokens = [3, 2, 2]
        self.assert_almost_equal_1d(sampling_output.token_ids.tolist(), expected_tokens)


if __name__ == '__main__':
    unittest.main()