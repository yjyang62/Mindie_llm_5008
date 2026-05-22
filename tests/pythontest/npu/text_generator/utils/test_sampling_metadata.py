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

from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata
from mindie_llm.utils.validation import UnsupportedTypeError, OutOfBoundsError


class TestSamplingMetadata(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_validation(self):
        with self.assertRaises(UnsupportedTypeError):
            SamplingMetadata.from_numpy(np.array([[100]]), is_prefill=None)

        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), repetition_penalty=np.array([-1]))

        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), do_sample=np.array([True]), temperature=np.array([-1]))

        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), do_sample=np.array([True]), top_k=np.array([-1]))

        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), do_sample=np.array([True]), top_p=np.array([2]))

        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), do_sample=np.array([True]), seeds=np.array([-1]))

        SamplingMetadata.from_numpy(np.array([[100]]), top_logprobs=np.array([20], dtype=np.int32))
        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), top_logprobs=np.array([21]))

        use_beam_search = np.array([1], dtype=np.int32)
        SamplingMetadata.from_numpy(np.array([[100]]), use_beam_search=use_beam_search, n=np.array([1], dtype=np.int32))
        SamplingMetadata.from_numpy(np.array([[100]]), use_beam_search=use_beam_search)
        with self.assertRaises(OutOfBoundsError):
            SamplingMetadata.from_numpy(np.array([[100]]), use_beam_search=use_beam_search, n=np.array([0]))

    def test_update_beam_search(self):
        sampling_metadata = SamplingMetadata.from_numpy(
            np.array([[100]]), cumulative_logprobs=np.array([0]), output_lengths=np.array([0]))
        sampling_metadata.update_beam_search(np.array([-0.1]), np.array([1]))


if __name__ == '__main__':
    unittest.main()