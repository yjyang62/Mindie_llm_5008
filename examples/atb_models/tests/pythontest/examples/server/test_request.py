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
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from examples.server.request import (
    MultiModalRequest,
    MultiModalRequestParams,
    MultiModalReqParams,
    request_from_multimodalinputs,
    request_from_token,
    Request,
)


@dataclass
class MappingParams:
    sp: bool = False
    cp: bool = False
    sp_rank: int = 0
    sp_size: int = 1
    cp_rank: int = 0
    cp_size: int = 1


def create_mock_mapping(params: MappingParams):
    """
    快速生成不会踩 MagicMock 比较坑的 mapping
    """
    mock = MagicMock()
    
    # 配置SP相关属性
    mock.has_attn_inner_sp.return_value = params.sp
    if params.sp:
        mock.attn_inner_sp.rank = params.sp_rank
        mock.attn_inner_sp.group_size = params.sp_size
    
    # 配置CP相关属性
    mock.has_attn_cp.return_value = params.cp
    if params.cp:
        mock.attn_cp.rank = params.cp_rank
        mock.attn_cp.group_size = params.cp_size
    
    return mock



class TestRequest(unittest.TestCase):
    def setUp(self):
        self.golden_mm_req = MultiModalRequest(
            max_out_length=500,
            block_size=128,
            req_id=0,
            input_ids=torch.tensor(list(range(4)), dtype=torch.int64),
            adapter_id='0'
        )
        self.mm_request_params = MultiModalRequestParams(
            text='Describe the image in 500 words.',
            image='image.jpg',
            video='',
            max_out_length=500,
            block_size=128,
            req_idx=0,
            adapter_id='0',
            batch_size=1
        )
        self.mm_req_params = MultiModalReqParams(
            text=['Describe the image in 500 words.'],
            image=['image.jpg'],
            video=[],
            audio=[],
            max_out_length=500,
            block_size=128,
            req_idx=0,
            adapter_id='0',
            batch_size=1
        )

        self.sp_params = MappingParams(sp=True, sp_rank=1, sp_size=2)
        self.cp_params = MappingParams(cp=True, cp_rank=0, cp_size=2)

    def test_request_from_multimodalinputs(self):
        mock_model, mock_processor = self.init_model_and_processor()
        mock_request = request_from_multimodalinputs(mock_processor, mock_model, self.mm_req_params)
        self.compare_mm_request(mock_request, self.golden_mm_req)

    def init_model_and_processor(self):
        mock_model = MagicMock()
        mock_model.model.prepare_prefill_token.return_value = torch.tensor(list(range(4)), dtype=torch.int64)
        mock_processor = MagicMock()
        return mock_model, mock_processor

    def compare_mm_request(self, request_a, request_b):
        self.assertEqual(request_a.req_id, request_b.req_id)
        self.assertTrue(torch.equal(request_a.input_ids, request_b.input_ids))
        self.assertEqual(request_a.adapter_id, request_b.adapter_id)
        self.assertEqual(request_a.input_length, request_b.input_length)
        self.assertEqual(request_a.position_ids, request_b.position_ids)

    def test_need_blocks_and_slots_calculation(self):
        """测试块和槽位计算"""
        # 测试正常情况
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        request = Request(max_out_length=10, block_size=4, req_id=0, input_ids=input_ids, adapter_id=None)
        self.assertEqual(request.need_blocks, 4)  # (4 + 10) / 4 = 3.5 -> ceil(3.5) = 4
        self.assertEqual(request.need_slots, 16)  # 4 * 4 = 16
        
        # 测试边界情况
        request = Request(max_out_length=0, block_size=4, req_id=0, input_ids=input_ids, adapter_id=None)
        self.assertEqual(request.need_blocks, 1)  # (4 + 0) / 4 = 1
        self.assertEqual(request.need_slots, 4)   # 1 * 4 = 4

    def test_zero_division_error(self):
        """测试block_size为0时的异常处理"""
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        with self.assertRaises(ZeroDivisionError):
            Request(max_out_length=10, block_size=0, req_id=0, input_ids=input_ids, adapter_id=None)

    def test_sp_processing(self):
        """测试序列并行处理"""
        mock_mapping = create_mock_mapping(self.sp_params)

        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        request = Request(max_out_length=10, block_size=4, req_id=0,
                        input_ids=input_ids, adapter_id=None,
                        mapping=mock_mapping)
        
        # 验证SP相关属性
        self.assertEqual(request.sp_rank, 1)
        self.assertEqual(request.input_len_per_sp, [3, 2])  # 5 // 2 = 2, 5 % 2 = 1 -> [2+1, 2]
        self.assertEqual(request.input_length_sp, 2)  # sp_rank=1 -> input_len_per_sp[1] = 2
        self.assertEqual(request.input_indices_sp, [3, 4])  # 第二个SP的索引

    def test_cp_processing(self):
        """测试上下文并行处理"""
        mock_mapping = create_mock_mapping(self.cp_params)

        mock_post = MagicMock()
        mock_post.pad_token_id = 0

        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        request = Request(max_out_length=10, block_size=4, req_id=0,
                        input_ids=input_ids, adapter_id=None,
                        mapping=mock_mapping, postprocessor=mock_post)
        
        # 验证CP相关属性
        self.assertEqual(request.cp_rank, 0)
        # 验证序列被分割和填充
        # 输入长度5，cp_size=2，num_chunks=4，chunk_length=2
        # 填充后长度=8，cp_rank=0处理索引[0,1]和[6,7]
        self.assertEqual(request.input_length, 4)  # 2 + 2 = 4
        self.assertTrue(torch.equal(request.position_ids, torch.tensor([0, 1, 6, 7])))

    def test_request_from_token_with_tensor_input(self):
        """测试使用tensor输入创建请求"""
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        request = request_from_token(input_ids, max_out_length=100, block_size=16)
        
        # 验证请求属性
        self.assertEqual(request.req_id, 0)  # 默认req_idx
        self.assertEqual(request.input_length, 4)
        self.assertTrue(torch.equal(request.input_ids, input_ids))
        self.assertIsNone(request.adapter_id)  # 默认adapter_id

    def test_request_from_token_with_list_input(self):
        """测试使用列表输入创建请求"""
        input_ids = [1, 2, 3, 4]
        request = request_from_token(input_ids, max_out_length=100, block_size=16)
        
        # 验证请求属性
        self.assertEqual(request.req_id, 0)  # 默认req_idx
        self.assertEqual(request.input_length, 4)
        self.assertTrue(torch.equal(request.input_ids, torch.tensor([1, 2, 3, 4], dtype=torch.int64)))
        self.assertIsNone(request.adapter_id)  # 默认adapter_id

    def test_request_from_token_with_custom_params(self):
        """测试使用自定义参数创建请求"""
        input_ids = [1, 2, 3, 4]
        request = request_from_token(input_ids, max_out_length=200, block_size=32, 
                                    req_idx=5, adapter_id="test_adapter")
        
        # 验证请求属性
        self.assertEqual(request.req_id, 5)
        self.assertEqual(request.input_length, 4)
        self.assertTrue(torch.equal(request.input_ids, torch.tensor([1, 2, 3, 4], dtype=torch.int64)))
        self.assertEqual(request.adapter_id, "test_adapter")


if __name__ == "__main__":
    unittest.main()