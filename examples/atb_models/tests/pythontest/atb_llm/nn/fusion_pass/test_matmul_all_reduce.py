#!/usr/bin/env python
# coding=utf-8
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
import logging
import torch
import torch_npu
import torch.multiprocessing as mp
import atb_llm.nn as nn
import atb_llm.nn.distributed as dist
from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.modules import Linear
from atb_llm.nn.tensor import Tensor
from atb_llm.nn.node import Node
from ddt import ddt, data
from tests.pythontest.atb_llm.utils import NO_BF16_SUPPORT_SOC_VERSION

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()

NPU = 'npu'


def get_matmul_all_reduce_fusion_engine(x, rank, world_size):
    linear = Linear('weight')
    linear_out = linear(x)
    out = nn.distributed.all_reduce(send_tensor=linear_out)
    get_default_net().mark_output(out, "matmul_all_reduce_out")
    matmul_all_reduce_fusion_engine = get_default_net().build_engine()
    return matmul_all_reduce_fusion_engine


def get_matmul_all_reduce_engine(x, rank, world_size):
    linear = Linear('weight')
    linear_out = linear(x)
    out = nn.distributed.all_reduce(send_tensor=linear_out)
    get_default_net().mark_output(out, "matmul_all_reduce_out")
    matmul_all_reduce_engine = get_default_net().build_engine(del_fpass_keys=["MatmulAllReducePass"])
    return matmul_all_reduce_engine


def get_matmul_all_reduce_bias_engine(x, rank, world_size):
    linear = Linear('weight', bias=True)
    linear_out = linear(x)
    out = nn.distributed.all_reduce(send_tensor=linear_out)
    get_default_net().mark_output(out, "matmul_all_reduce_out")
    matmul_all_reduce_fusion_engine = get_default_net().build_engine()
    return matmul_all_reduce_fusion_engine


def matmul_all_reduce_worker(rank, world_size, data_type):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='lccl', init_method="", world_size=world_size, rank=rank)
    input_tensor = Tensor("x")
    matmul_all_reduce_fusion_engine = get_matmul_all_reduce_fusion_engine(
        input_tensor, rank, world_size
    )
    assert "LinearParallel" in str(matmul_all_reduce_fusion_engine)

    matmul_all_reduce_engine = get_matmul_all_reduce_engine(
        input_tensor, rank, world_size
    )
    assert "LinearParallel" not in str(matmul_all_reduce_engine)

    # input
    x = torch.rand(100, 1024).to(data_type).to(NPU)
    weight = torch.rand(100, 1024).to(data_type).to(NPU)
    # output
    matmul_all_reduce_fusion_out = torch.rand(100, 100).to(data_type).to(NPU)
    matmul_all_reduce_out = torch.rand(100, 100).to(data_type).to(NPU)

    inputs = {}
    inputs["x"] = x
    fusion_output = {"matmul_all_reduce_out": matmul_all_reduce_fusion_out}
    output = {"matmul_all_reduce_out": matmul_all_reduce_out}

    matmul_all_reduce_fusion_engine.set_weights({"weight.weight": weight})
    matmul_all_reduce_fusion_engine.forward(inputs, fusion_output)

    matmul_all_reduce_engine.set_weights({"weight.weight": weight})
    matmul_all_reduce_engine.forward(inputs, output)

    torch.npu.synchronize()
    assert torch.allclose(fusion_output["matmul_all_reduce_out"], output["matmul_all_reduce_out"], rtol=1e-02, atol=1e-02)


def matmul_all_reduce_hccl_worker(rank, world_size):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', init_method="", world_size=world_size, rank=rank)
    input_tensor = Tensor("x")
    matmul_all_reduce_fusion_engine = get_matmul_all_reduce_fusion_engine(
        input_tensor, rank, world_size
    )
    assert "LinearParallel" not in str(matmul_all_reduce_fusion_engine)


def matmul_all_reduce_bias_worker(rank, world_size):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='lccl', init_method="", world_size=world_size, rank=rank)
    input_tensor = Tensor("x")
    matmul_all_reduce_fusion_engine = get_matmul_all_reduce_bias_engine(
        input_tensor, rank, world_size
    )
    assert "LinearParallel" not in str(matmul_all_reduce_fusion_engine)


@ddt
class TestMatmulAllReducePass(unittest.TestCase):
    def setUp(self):
        self.world_size = 2

    @data(torch.float16, torch.bfloat16)
    def test_matmul_all_reduce(self, data_type):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("Atlas 300I DUO doesn't support lccl.")
        mp.spawn(matmul_all_reduce_worker, nprocs=self.world_size, args=(self.world_size, data_type))

    def test_matmul_all_reduce_hccl(self):
        mp.spawn(matmul_all_reduce_hccl_worker, nprocs=self.world_size, args=(self.world_size,))

    def test_matmul_all_reduce_bias(self):
        soc_version = torch_npu._C._npu_get_soc_version()
        if soc_version in NO_BF16_SUPPORT_SOC_VERSION:
            self.skipTest("Atlas 300I DUO doesn't support lccl.")
        mp.spawn(matmul_all_reduce_bias_worker, nprocs=self.world_size, args=(self.world_size,))


if __name__ == "__main__":
    unittest.main()
