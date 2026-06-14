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
import torch
import torch_npu
import torch.multiprocessing as mp

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.tensor import Tensor
import atb_llm.nn.distributed as dist
import pytest
from ddt import ddt, data, unpack

INPUT_KEY = "input"
OUTPUT_KEY = "out"


def all_reduce_worker(rank, world_size, world_input, backend):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend=backend, init_method="", world_size=world_size, rank=rank)
    _, m, n = world_input.shape

    out = dist.all_reduce(send_tensor=Tensor(INPUT_KEY))
    get_default_net().mark_output(out, OUTPUT_KEY)
    engine = get_default_net().build_engine()

    # input
    in_tensor = world_input[rank].clone().npu()
    out_tensor = torch.empty(m, n, dtype=torch.float16).npu()
    inputs = {INPUT_KEY: in_tensor}
    outputs = {'out': out_tensor}

    engine.forward(inputs, outputs)

    # gloden
    golden = torch.sum(world_input, dim=0)
    out_ = outputs.get(OUTPUT_KEY)

    torch.npu.synchronize()
    assert torch.allclose(out_.cpu(), golden.cpu(), rtol=1e-02, atol=1e-02)


def all_gather_worker(rank, world_size, world_input, backend):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend=backend, init_method="", world_size=world_size, rank=rank)
    _, m, n = world_input.shape

    out = dist.all_gather(send_tensor=Tensor(INPUT_KEY))
    get_default_net().mark_output(out, OUTPUT_KEY)
    engine = get_default_net().build_engine()

    # input
    in_tensor = world_input[rank].clone().npu()
    out_tensor = torch.empty(world_size, m, n, dtype=torch.float16).npu()
    inputs = {INPUT_KEY: in_tensor}
    outputs = {'out': out_tensor}

    engine.forward(inputs, outputs)

    # gloden
    golden = world_input
    out_ = outputs.get(OUTPUT_KEY)

    torch.npu.synchronize()
    assert torch.allclose(out_.cpu(), golden.cpu(), rtol=1e-02, atol=1e-02)


def all_reduce_subcomm_worker(rank, world_size, world_input, backend):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend=backend, init_method="", world_size=world_size, rank=rank, buffer_size=128)

    _, m, n = world_input.shape
    if rank in [0, 1]:
        process_group = dist.new_group(ranks=[0, 1], backend=backend, buffer_size=128)
    else:
        process_group = dist.new_group(ranks=[2, 3], backend=backend, buffer_size=128)

    out = dist.all_reduce(send_tensor=Tensor(INPUT_KEY), process_group=process_group)
    get_default_net().mark_output(out, OUTPUT_KEY)
    engine = get_default_net().build_engine()

    # input
    in_tensor = world_input[rank].clone().npu()
    out_tensor = torch.empty(m, n, dtype=torch.float16).npu()
    inputs = {INPUT_KEY: in_tensor}
    outputs = {'out': out_tensor}

    engine.forward(inputs, outputs)

    # gloden
    if rank in [0, 1]:
        golden = torch.sum(world_input[:2,], dim=0)
    else:
        golden = torch.sum(world_input[2:,], dim=0)
    out_ = outputs.get(OUTPUT_KEY)

    torch.npu.synchronize()
    assert torch.allclose(out_.cpu(), golden.cpu(), rtol=1e-02, atol=1e-02)


def all_reduce_reuse_subcomm_worker(rank, world_size, world_input, backend, op_name_pair):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend=backend, init_method="", world_size=world_size, rank=rank, buffer_size=128)

    _, m, n = world_input.shape
    if rank in [0, 1]:
        process_group = dist.new_group(ranks=[0, 1], backend=backend)
    else:
        process_group = dist.new_group(ranks=[2, 3], backend=backend)

    dist.distributed._update_pg_ops_map(process_group, op_name_pair[0])
    with pytest.raises(ValueError):
        dist.distributed._update_pg_ops_map(process_group, op_name_pair[1])


@ddt
class TestDistributedFunction(unittest.TestCase):
    def setUp(self):
        self.world_size = 4
        self.m = 4
        self.n = 10

    @data(('hccl', 4), ('hccl', 2))
    @unpack
    def test_all_reduce(self, backend, world_size):
        world_input = torch.rand(world_size, self.m, self.n, dtype=torch.float16)
        mp.spawn(all_reduce_worker, nprocs=world_size, args=(world_size, world_input, backend))

    @data(('hccl', 2))
    @unpack
    def test_all_gather(self, backend, world_size):
        world_input = torch.rand(world_size, self.m, self.n, dtype=torch.float16)
        mp.spawn(all_gather_worker, nprocs=world_size, args=(world_size, world_input, backend))

    @data(('hccl'))
    def test_all_reduce_subcomm(self, backend):
        world_input = torch.rand(self.world_size, self.m, self.n, dtype=torch.float16)
        mp.spawn(all_reduce_subcomm_worker, nprocs=self.world_size, args=(self.world_size, world_input, backend))

    @data(('hccl', ["MoEDistribute", "AllReduce"]), ('hccl', ["AllReduce", "MoEDistribute"]))
    @unpack
    def test_reuse_subcomm(self, backend, op_name_pair):
        world_input = torch.rand(self.world_size, self.m, self.n, dtype=torch.float16)
        mp.spawn(all_reduce_reuse_subcomm_worker, nprocs=self.world_size, args=(self.world_size, world_input,
            backend, op_name_pair))


if __name__ == '__main__':
    unittest.main()