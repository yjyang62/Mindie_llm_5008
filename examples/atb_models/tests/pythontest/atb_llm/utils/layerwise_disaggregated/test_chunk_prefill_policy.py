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
from unittest.mock import Mock, patch
from atb_llm.utils.layerwise_disaggregated.chunk_prefill_policy import (
    ChunkPrefilPolicy,
    RatioInfo,
)


class TestChunkPrefilPolicy(unittest.TestCase):
    @patch("atb_llm.utils.layerwise_disaggregated.chunk_prefill_policy.acl")
    def setUp(self, mock_acl):
        mock_acl.get_soc_name = Mock()
        mock_acl.get_soc_name.return_value = "Ascend910B4"
        self.qwen_policy = ChunkPrefilPolicy()
        self.deepseek_policy = ChunkPrefilPolicy("deepseek", 1, None)

    @classmethod
    def setUpClass(cls):
        print("TestChunkPrefilPolicy start")

    @classmethod
    def tearDownClass(cls):
        print("TestChunkPrefilPolicy end")

    def test_map_prefill_chunk_num(self):
        edge_policy, cloud_policy, eq_list = self.deepseek_policy.get_chunk_len_policy(16384)
        self.assertEqual(len(edge_policy), 5)
        self.assertEqual(len(cloud_policy), 5)
        self.assertEqual(len(eq_list), 5)

    def test_initialize(self):
        self.deepseek_policy.initialize(False)
        self.deepseek_policy.initialize_standard_card()
        edge_policy, cloud_policy, eq_list = self.deepseek_policy.get_chunk_len_policy(8192)
        self.assertEqual(len(edge_policy), 2)
        self.assertEqual(len(cloud_policy), 2)
        self.assertEqual(len(eq_list), 2)

    def test_normal_case(self):
        """测试正常情况：按比例切分，剩余补到最后一份"""
        total_len = 32468
        edge_info = RatioInfo([22, 16, 12, 8, 4, 2], 2048)
        cloud_info = RatioInfo([22, 16, 12, 8, 4, 2], 2048)

        edge_result, cloud_result, _ = ChunkPrefilPolicy.split_long_seq_by_ratio(total_len, edge_info, cloud_info)

        self.assertEqual(len(edge_result), len(edge_info.ratio_list))
        self.assertEqual(sum(edge_result), total_len)
        self.assertEqual(len(cloud_result), len(cloud_info.ratio_list))
        self.assertEqual(sum(cloud_result), total_len)

    def test_exact_division(self):
        """测试整除情况：没有剩余长度"""
        total_len = 12288
        edge_info = RatioInfo([2, 1], 2048)
        cloud_info = RatioInfo([2, 1], 2048)

        edge_result, _, _ = ChunkPrefilPolicy.split_long_seq_by_ratio(total_len, edge_info, cloud_info)

        self.assertEqual(len(edge_result), len(edge_info.ratio_list))
        self.assertEqual(sum(edge_result), total_len)

    def test_with_remainder(self):
        """测试有剩余长度的情况"""
        total_len = 10000
        edge_info = RatioInfo([3, 2, 1], 1000)
        cloud_info = RatioInfo([3, 2, 1], 1000)

        edge_result, _, _ = ChunkPrefilPolicy.split_long_seq_by_ratio(total_len, edge_info, cloud_info)

        self.assertEqual(len(edge_result), len(edge_info.ratio_list))
        self.assertEqual(sum(edge_result), total_len)
        for length in edge_result:
            self.assertEqual(length % edge_info.min_unit, 0)
            self.assertGreaterEqual(length, edge_info.min_unit)
            self.assertEqual(length % cloud_info.min_unit, 0)
            self.assertGreaterEqual(length, cloud_info.min_unit)

    def test_equal_ratios(self):
        """测试等比例切分"""
        total_len = 12288
        edge_info = RatioInfo([1, 1, 1], 2048)
        cloud_info = RatioInfo([1, 1, 1], 2048)

        edge_result, cloud_result, eq_list = ChunkPrefilPolicy.split_long_seq_by_ratio(total_len, edge_info, cloud_info)

        self.assertEqual(len(edge_result), len(edge_info.ratio_list))
        self.assertEqual(sum(edge_result), total_len)
        self.assertEqual(len(edge_result), len(cloud_result))
        self.assertEqual(len(cloud_result), len(eq_list))
        for length in edge_result:
            self.assertEqual(length % edge_info.min_unit, 0)
            self.assertGreaterEqual(length, edge_info.min_unit)
            self.assertEqual(length % cloud_info.min_unit, 0)
            self.assertGreaterEqual(length, cloud_info.min_unit)

    def test_empty_ratio_list(self):
        """测试空比例列表的情况，应该抛出异常"""
        total_len = 10000
        edge_info = RatioInfo([], 1000)
        cloud_info = RatioInfo([], 1000)

        with self.assertRaises(ValueError) as context:
            ChunkPrefilPolicy.split_long_seq_by_ratio(total_len, edge_info, cloud_info)

        self.assertIn("边侧和云侧的总比例和不一致, 边侧0, 云侧0", str(context.exception))
