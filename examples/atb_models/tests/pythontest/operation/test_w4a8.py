# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
import unittest
import torch
import torch_npu
import numpy as np
from loguru import logger
CPU_DEVICE = "cpu"
M = 8
K = 8192
N = 128*8


def a8w4_quant_golden(x, weight, scale, perTokenScale, groupList, bias, output_dtype):
    m, k = x.shape
    k, n = weight.shape
    x = x.numpy()
    weight = weight.cpu().numpy()
    scale = scale.numpy()
    perTokenScale = perTokenScale.numpy()
    bias = bias.numpy()
    bias = bias.reshape(1, -1)
    weight_int8 = weight.astype(np.int8)
    x = np.concatenate([x.reshape(m, 1, k) // 16, (x.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)

    groupNum = 1
    quantGroupNum = scale.shape[0]
    index = np.cumsum(groupList)
    xSplit = np.split(x, index * 2, axis=0)
    xSplit[0] = x
    weightGroup = weight_int8.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    mmi = []
    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
            mm = mm.astype(np.float32) * scale[j, :].reshape(1, -1)
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

        mmi = mmi.reshape(-1, 2, n).astype(np.float32)
        mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
        mmi = mmi * perTokenScale
        mmOuts.append(mmi)
    golden_tensor = torch.from_numpy(mmi)

    return golden_tensor.to(output_dtype)


class TestW4A8Pertoken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_npu.npu.set_device(0)
        atb_speed_home_path = os.environ.get("ATB_SPEED_HOME_PATH")
        if atb_speed_home_path is None:
            raise RuntimeError("env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
        lib_path = os.path.join(atb_speed_home_path, "lib/libatb_speed_torch.so")
        torch.classes.load_library(lib_path)

    def setUp(self):
        self.w4a8_op = torch.classes.OperationTorch.OperationTorch("W4A8Operation")
    
    def test_w4a8_pertoken(self):
        self.w4a8_op.set_param(json.dumps({}))

        x1 = torch.randint(-128, 128, [M, K], dtype=torch.int8)
        x2 = torch.randint(1, 8, (K, N), dtype=torch.int32)

        y_offset = torch.zeros([N,], dtype=torch.float32)
        x1_scale = torch.randn([M, 1], dtype=torch.float32) * 0.01
        x2_scale = torch.randn([K // 256, N], dtype=torch.float32).uniform_(0, 1)
        expanded = x2_scale.unsqueeze(1).repeat(1, 256, 1).view(-1, N)
        group_size_list = [0, 0, 256]
        x1_npu = x1.npu()
        x2_npu = torch_npu.npu_convert_weight_to_int4pack(x2.npu())
        y_offset_npu = y_offset.npu()
        x1_scale_npu = x1_scale.npu()
        x2_scale_tmp = torch_npu.npu_trans_quant_param(x2_scale.npu().reshape([K // 256 * N,]))
        x2_scale_npu = x2_scale_tmp.reshape([K // 256, N])
        
        cpu_out = a8w4_quant_golden(x=x1, weight=x2, scale=x2_scale, perTokenScale=x1_scale,
                                         groupList=group_size_list, bias=y_offset, output_dtype=torch.float16)
        x1_float = x1 * x1_scale
        x2_float = x2 * expanded
        logger.info(f'{(x1_float @ x2_float)=}')
        w4a8_output = self.w4a8_op.execute([
            x1_npu,                 # int8
            x2_npu,                 # int32
            x1_scale_npu,           # fp32
            x2_scale_npu,           # uint64
            y_offset_npu
        ])[0]
        
        logger.info("w4a8 output: %s, %s, %s" % (w4a8_output.shape, w4a8_output.dtype, w4a8_output))
        logger.info("golden_out output: %s, %s, %s" % (cpu_out.shape, cpu_out.dtype, cpu_out))
        logger.info(f"{w4a8_output.min().item()=}")
        logger.info(f"{w4a8_output.max().item()=}")

        self.assertTrue(golden_compare(w4a8_output.cpu(), cpu_out, K).item())


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    result = EB <= 2 ** (-10)
    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result


def golden_compare(out_tensor, golden_out_tensor, ksize):
    eb = get_eb(golden_out_tensor, out_tensor)
    cmp = ref_compare(golden_out_tensor, out_tensor, 2 ** -7 if ksize < 2048 else 2**-6)
    return eb and cmp


if __name__ == "__main__":
    unittest.main()