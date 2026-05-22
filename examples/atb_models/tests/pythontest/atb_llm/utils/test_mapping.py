# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import patch
import pytest
from atb_llm.utils.mapping import Mapping, ParallelInfo, PipelineParallelInfo
from ddt import ddt, data, unpack


@ddt
class TestMapping(unittest.TestCase):
    def test_parse_no_args(self):
        
        mapping = Mapping(4, 0)
        self.assertEqual(mapping.attn_dp.group_size, 1)
        self.assertEqual(mapping.attn_tp.group_size, 4)
        self.assertEqual(mapping.mlp_tp.group_size, 4)

    def test_parse_tp(self):
        mapping = Mapping(8, 0, tp=8)
        self.assertEqual(mapping.attn_dp.group_size, 1)
        self.assertEqual(mapping.attn_tp.group_size, 8)
        self.assertEqual(mapping.mlp_tp.group_size, 8)

    def test_parse_dp_tp(self):
        mapping = Mapping(8, 0, dp=8, tp=1, moe_tp=8)
        self.assertEqual(mapping.attn_dp.group_size, 8)
        self.assertEqual(mapping.attn_tp.group_size, 1)
        self.assertEqual(mapping.mlp_tp.group_size, 8)

    def test_validate_attn_world_size(self):
        with self.assertRaises(ValueError):
            _ = Mapping(8, 0, dp=4, tp=1, moe_tp=8)

    def test_validate_mlp_world_size(self):
        with self.assertRaises(ValueError):
            _ = Mapping(8, 0, dp=8, tp=1, moe_tp=4)

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_get_tp_group(self):
        mapping = Mapping(0, 0)
        mapping.world_size = 4
        mapping.rank = 1
        mapping.attn_tp = ParallelInfo(buffer_size=128, group_size=4)
        mapping.get_tp_group(mapping.attn_tp)
        self.assertEqual(mapping.attn_tp.num_group, 1)
        self.assertEqual(mapping.attn_tp.rank_per_group, [[0, 1, 2, 3]])
        self.assertEqual(mapping.attn_tp.rank, 1)

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_get_dp_group(self):
        mapping = Mapping(0, 0)
        mapping.world_size = 4
        mapping.rank = 1
        mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=1)
        mapping.get_dp_group(mapping.attn_dp)
        self.assertEqual(mapping.attn_dp.num_group, 4)
        self.assertEqual(mapping.attn_dp.rank_per_group, [[0], [1], [2], [3]])
        self.assertEqual(mapping.attn_dp.rank, 0)

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_get_prefix_cache_cp_group(self):
        mapping = Mapping(0, 0)
        mapping.world_size = 4
        mapping.rank = 1
        mapping.attn_prefix_cache_cp = ParallelInfo(buffer_size=128, group_size=1)
        mapping.get_dp_group(mapping.attn_prefix_cache_cp)
        self.assertEqual(mapping.attn_prefix_cache_cp.num_group, 4)
        self.assertEqual(mapping.attn_prefix_cache_cp.rank_per_group, [[0], [1], [2], [3]])
        self.assertEqual(mapping.attn_prefix_cache_cp.rank, 0)

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_get_dp_tp_group(self):
        mapping = Mapping(0, 0)
        mapping.world_size = 8
        mapping.rank = 6
        mapping.attn_tp = ParallelInfo(buffer_size=128, group_size=4)
        mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=2)
        mapping.get_tp_group(mapping.attn_tp)
        self.assertEqual(mapping.attn_tp.num_group, 2)
        self.assertEqual(mapping.attn_tp.rank_per_group, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(mapping.attn_tp.rank, 2)
        mapping.get_dp_group(mapping.attn_dp)
        self.assertEqual(mapping.attn_dp.num_group, 4)
        self.assertEqual(mapping.attn_dp.rank_per_group, [[0, 4], [1, 5], [2, 6], [3, 7]])
        self.assertEqual(mapping.attn_dp.rank, 1)

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_has_attn_tp(self):
        mapping = Mapping(0, 0)
        mapping.attn_tp = ParallelInfo(buffer_size=128, group_size=4)
        self.assertTrue(mapping.has_attn_tp())
        mapping.attn_tp = ParallelInfo(buffer_size=128, group_size=-1)
        self.assertFalse(mapping.has_attn_tp())

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_has_attn_dp(self):
        mapping = Mapping(0, 0)
        mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=4)
        self.assertTrue(mapping.has_dp())
        mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=-1)
        self.assertFalse(mapping.has_dp())

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_has_mlp_tp(self):
        mapping = Mapping(0, 0)
        mapping.mlp_tp = ParallelInfo(buffer_size=128, group_size=4)
        self.assertTrue(mapping.has_mlp_tp())
        mapping.mlp_tp = ParallelInfo(buffer_size=128, group_size=-1)
        self.assertFalse(mapping.has_mlp_tp())

    @patch.object(Mapping, "__init__", lambda x, y, z: None)
    def test_has_pp(self):
        mapping = Mapping(0, 0)
        mapping.world_size = 8
        mapping.num_nodes = 2
        mapping.pp = PipelineParallelInfo(hccl_buffer=128)
        mapping.pp.group_size = 4
        self.assertTrue(mapping.has_pp())
        with pytest.raises(ValueError):
            mapping.update_pp()

    def test_validate1(self):
        mapping = Mapping(8, 0)
        with pytest.raises(ValueError):
            mapping.num_nodes = 3
            mapping.validate()

    def test_validate2(self):
        mapping = Mapping(8, 0)
        with pytest.raises(ValueError):
            mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=2)
            mapping.attn_cp = ParallelInfo(buffer_size=128, group_size=4)
            mapping.validate()

    def test_validate3(self):
        mapping = Mapping(8, 0)
        with pytest.raises(ValueError):
            mapping.pp = PipelineParallelInfo(hccl_buffer=128)
            mapping.attn_dp = ParallelInfo(buffer_size=128, group_size=2)
            mapping.pp.group_size = 2
            mapping.validate()

    def test_validate_pp_1(self):
        mapping = Mapping(4, 0)
        with pytest.raises(ValueError):
            mapping.num_nodes = 2
            mapping.pp.group_size = 2
            mapping.pp.tp.group_size = 1
            mapping.validate_pp()

    def test_validate_pp_2(self):
        mapping = Mapping(4, 0)
        mapping.num_nodes = 2
        mapping.pp.group_size = 2
        mapping.pp.tp.group_size = 2
        self.assertEqual(mapping.prev_pp_rank(), 2)
        self.assertEqual(mapping.next_pp_rank(), 2)
        self.assertEqual(mapping.pp_layers(60), list(range(0,30)))
        self.assertTrue(mapping.is_first_pp_rank())
        self.assertTrue(not mapping.is_last_pp_rank())

    def test_validate_pp_3(self):
        mapping = Mapping(8, 0)
        with pytest.raises(ValueError):
            mapping.local_world_size = 2
            mapping.attn_tp = ParallelInfo(buffer_size=128, group_size=4)
            mapping.validate_pp()

    @data((1, 8, 4, 4), (1, 8, 4, 4))
    @unpack
    def test_validate_all(self, attn_dp, attn_tp, moe_ep, moe_tp):
        with pytest.raises(ValueError):
            Mapping(8, 0, attn_tp=attn_tp, attn_dp=attn_dp, moe_ep=moe_ep, moe_tp=moe_tp)

    def test_validate_innersp(self):
        with pytest.raises(ValueError):
            mapping = Mapping(8, 0, attn_tp=2)
            mapping.attn_inner_sp = ParallelInfo(buffer_size=128, group_size=4)
            mapping.validate_pp()

    def test_to_dict(self):
        mapping = Mapping(8, 0, dp=8, tp=1, moe_ep=2, moe_tp=4)
        _ = mapping.__repr__()
        parallel_dict = mapping.to_dict()
        self.assertEqual(parallel_dict['worldSize'], 8)

    @patch('atb_llm.utils.mapping.atb_llm_dist.init_process_group', return_value=None)
    @patch('atb_llm.utils.mapping.atb_llm_dist.new_group', return_value='-1')
    def test_init_process_group(self, _1, _2):
        mapping = Mapping(8, 0)
        mapping.init_python_comm_process_group()
        self.assertEqual(mapping.attn_tp.process_group, '-1')

    def test_to_dict_v2(self):
        mapping = Mapping(8, 0, dp=8, tp=1, moe_ep=2, moe_tp=4)
        parallel_dict = mapping.to_dict_v2()
        golden_parallel_dict = {
            'worldSize': 8, 'rank': 0, 'rankTableFile': '', 'localWorldSize': 8,
            'lcclCommDomainLowerBound': 0, 'lcclCommDomainUpperBound': 65536,
            'wordEmbedTp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'wordEmbedDp': {'groupId': 0, 'rankIds': [0, 1, 2, 3, 4, 5, 6, 7], 'rank': 0, 'bufferSize': 128},
            'attnTp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'attnDp': {'groupId': 0, 'rankIds': [0, 1, 2, 3, 4, 5, 6, 7], 'rank': 0, 'bufferSize': 128},
            'attnCp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'attnPrefixcacheCp': {'groupId': 0, 'rankIds': [0, 1, 2, 3, 4, 5, 6, 7], 'rank': 0, 'bufferSize': 128},
            'attnInnerSp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'attnOProjTp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'attnOProjDp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'mlpTp': {'groupId': 0, 'rankIds': [0, 1, 2, 3, 4, 5, 6, 7], 'rank': 0, 'bufferSize': 128},
            'mlpDp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'moeTp': {'groupId': 0, 'rankIds': [0, 1, 2, 3], 'rank': 0, 'bufferSize': 64},
            'moeEp': {'groupId': 0, 'rankIds': [0, 4], 'rank': 0, 'bufferSize': 512},
            'moeEpIntraNode': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'moeEpInterNode': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'lmHeadTp': {'groupId': 0, 'rankIds': [0, 1, 2, 3, 4, 5, 6, 7], 'rank': 0, 'bufferSize': 128},
            'lmHeadDp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'denseTp': {'groupId': 0, 'rankIds': [0], 'rank': 0, 'bufferSize': 128},
            'dynamicEplb': {'groupId': 0, 'rankIds': [0, 4], 'rank': 0, 'bufferSize': 128},
        }
        self.assertDictEqual(golden_parallel_dict, parallel_dict)


if __name__ == "__main__":
    unittest.main()