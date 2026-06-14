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
from examples.convert.model_slim.get_razor_attention_wins import get_global_wins


class TestGetGlobalWins(unittest.TestCase):

    def test_llama_model_80_layers(self):
        """Test case for model_type='llama' and num_layers=80"""
        model_type = 'llama'
        num_layers = 80
        head_dict, first_sink = get_global_wins(model_type, num_layers)

        # 检查 first_sink 是否正确
        self.assertEqual(first_sink, 40)

        # 检查 head_dict 是否包含指定的 keys
        self.assertIn("prefix_matching", head_dict)
        self.assertIn("copying", head_dict)

        # 检查部分内容是否正确
        prefix_matching = head_dict["prefix_matching"]
        copying = head_dict["copying"]
        self.assertIn(0, prefix_matching)
        self.assertEqual(prefix_matching[0], [0, 1, 2, 3, 4, 5, 6, 7])

        self.assertIn(14, copying)
        self.assertEqual(copying[14], [2])

    def test_chatglm_model_40_layers(self):
        """Test case for model_type='chatglm' and num_layers=40"""
        model_type = 'chatglm'
        num_layers = 40
        head_dict, first_sink = get_global_wins(model_type, num_layers)

        # 检查 first_sink 是否正确
        self.assertEqual(first_sink, 4)

        # 检查 head_dict 是否包含指定的 keys
        self.assertIn("prefix_matching", head_dict)
        self.assertIn("copying", head_dict)

        # 检查部分内容是否正确
        prefix_matching = head_dict["prefix_matching"]
        copying = head_dict["copying"]
        self.assertIn(19, prefix_matching)
        self.assertEqual(prefix_matching[19], [0, 1])

        self.assertIn(5, copying)
        self.assertEqual(copying[5], [2])

    def test_invalid_model_type(self):
        """Test case for an invalid model_type"""
        model_type = 'unknown_model'
        num_layers = 80
        head_dict, first_sink = get_global_wins(model_type, num_layers)

        # 检查返回值是否为空
        self.assertEqual(head_dict, None)
        self.assertEqual(first_sink, None)

    def test_invalid_num_layers(self):
        """Test case for an invalid num_layers"""
        model_type = 'llama'
        num_layers = 0  # 不支持的层数
        head_dict, first_sink = get_global_wins(model_type, num_layers)

        # 检查返回值是否为空
        self.assertEqual(head_dict, None)
        self.assertEqual(first_sink, None)


if __name__ == "__main__":
    unittest.main()