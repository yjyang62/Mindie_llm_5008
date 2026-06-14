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

from mindie_llm.text_generator.utils.stopping_criteria import single_eos, strings_eos, continuous_eos, make_mixed_eos


class TestStopCriteria(unittest.TestCase):
    def test_single_eos(self):
        output_ids_without_padding = np.array([])
        eos_token_id = 1
        result = single_eos(output_ids_without_padding, eos_token_id)
        self.assertFalse(result)

        output_ids_without_padding = np.array([12800, 2, 3, 4, 1])
        result = single_eos(output_ids_without_padding, eos_token_id)
        self.assertTrue(result)

    def test_continues_eos(self):
        output_ids_without_padding = np.array([12800, 30])
        eos_token_id = [0, 1, 2]
        result = continuous_eos(output_ids_without_padding, eos_token_id)
        self.assertFalse(result)

        output_ids_without_padding = np.array([12800, 30, 0, 1, 2])
        result = continuous_eos(output_ids_without_padding, eos_token_id)
        self.assertTrue(result)

    def test_make_mixed_eos(self):
        eos_token_id = [0, [1, 2]]
        func = make_mixed_eos(eos_token_id)
        output_ids_without_padding = np.array([12800, 30, 0, 1, 2])
        self.assertTrue(func(output_ids_without_padding))

    def test_strings_eos(self):
        output_text = "My name is Oliver and I am a student from ShangHai."
        new_token_string = "a student from ShangHai."
        stop_strings = ["student"]
        idx = strings_eos(output_text, new_token_string, stop_strings)
        self.assertEqual(idx, -22)


if __name__ == '__main__':
    unittest.main()