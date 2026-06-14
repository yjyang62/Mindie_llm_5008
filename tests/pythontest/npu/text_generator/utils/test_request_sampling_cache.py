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

from mindie_llm.text_generator.utils.request_sampling_cache import RequestsSamplingCache
from mindie_llm.text_generator.utils.sampling_metadata import SamplingMetadata


class TestRequestsSamplingCache(unittest.TestCase):

    def setUp(self):
        def to_tensor_torch(data_):
            return torch.tensor(data_)
        self.to_tensor = to_tensor_torch
        # 创建一个 RequestsSamplingCache 对象
        self.cache = RequestsSamplingCache()
        
        # 创建一些测试数据
        self.batch_sequence_ids = [np.array([1, 2, 3])]
        self.different_sequence_ids = [np.array([4, 5, 6])]
        
        self.sampling_metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=self.batch_sequence_ids,
            is_prefill=True,
            to_tensor=self.to_tensor
        )
        # 添加数据到缓存
        self.cache.add_to_cache(self.batch_sequence_ids, self.sampling_metadata)

    def test_initialization(self):
        # 测试初始化
        self.assertIsNotNone(self.cache.cached_sequence_ids)
        self.assertIsNotNone(self.cache.sampling_metadata)

    def test_get_from_cache_hit(self):
        # 测试缓存命中
        result = self.cache.get_from_cache(self.batch_sequence_ids)
        self.assertEqual(result, self.sampling_metadata)

    def test_get_from_cache_miss(self):
        # 测试缓存不命中
        result = self.cache.get_from_cache(self.different_sequence_ids)
        self.assertIsNone(result)

    def test_add_to_cache(self):
        # 测试添加数据到缓存
        new_sampling_metadata = SamplingMetadata.from_numpy(
            batch_sequence_ids=self.different_sequence_ids,
            is_prefill=True,
            to_tensor=self.to_tensor
        )
        self.cache.add_to_cache(self.different_sequence_ids, new_sampling_metadata)
        
        result = self.cache.get_from_cache(self.different_sequence_ids)
        self.assertEqual(result, new_sampling_metadata)

    def test_clear(self):
        # 测试清空缓存
        self.cache.clear()
        self.assertIsNone(self.cache.cached_sequence_ids)
        self.assertIsNone(self.cache.sampling_metadata)

    def test_repr(self):
        # 测试 __repr__ 方法
        expected_repr = (f"SamplingCache:\n"
                         f"cached_sequence_ids: {np.array(self.batch_sequence_ids, dtype=np.int_)}, "
                         f"sampling_metadata: {self.sampling_metadata}.")
        self.assertEqual(repr(self.cache), expected_repr)

if __name__ == '__main__':
    unittest.main()