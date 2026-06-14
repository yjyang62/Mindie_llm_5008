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
from types import SimpleNamespace

import torch

from atb_llm.utils.data.weight_wrapper import WeightWrapper, AttnWrapper, MlpWrapper
from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import LinearType, TransposeType, PackType


class FakeLinear:
    def __init__(self, dtype=torch.float16, has_bias=True, trans_flag=TransposeType.NOT_TRANSPOSE):
        self.trans_flag = trans_flag
        self.nd_weight = False
        self.linear_desc = 123

        self.weight = torch.zeros((2, 2), dtype=dtype)
        self.quant_bias = torch.ones((2,), dtype=torch.int32)
        self.deq_scale = torch.ones((2,), dtype=torch.float32)
        self.input_offset = torch.zeros((2,), dtype=torch.int32)
        self.input_scale = torch.ones((2,), dtype=torch.float16)
        self.index = torch.arange(2, dtype=torch.int32)

        self.bias = torch.zeros((2,), dtype=dtype) if has_bias else None

        self.weight_offset = torch.zeros((2,), dtype=torch.int32)
        self.weight_scale = torch.ones((2,), dtype=torch.float16)


class FakeModule:
    def __init__(self, linear: FakeLinear):
        self.linear = linear


class FakeReduceQuant:
    def __init__(self):
        self.reduce_quant_scale = torch.ones((2,), dtype=torch.float16)
        self.gather_quant_scale = torch.ones((2,), dtype=torch.float16)


class FakeKVCacheQuant:
    def __init__(self):
        self.k_quant_scale = torch.ones((2,), dtype=torch.float16)
        self.k_dequant_scale = torch.ones((2,), dtype=torch.float16)
        self.v_quant_scale = torch.ones((2,), dtype=torch.float16)
        self.v_dequant_scale = torch.ones((2,), dtype=torch.float16)
        self.k_quant_offset = torch.zeros((2,), dtype=torch.float16)
        self.k_dequant_offset = torch.zeros((2,), dtype=torch.float16)
        self.v_quant_offset = torch.zeros((2,), dtype=torch.float16)
        self.v_dequant_offset = torch.zeros((2,), dtype=torch.float16)


class FakeFA3:
    def __init__(self):
        self.q_scale = torch.ones((2,), dtype=torch.float16)
        self.k_scale = torch.ones((2,), dtype=torch.float16)
        self.v_scale = torch.ones((2,), dtype=torch.float16)
        self.qk_scale = torch.ones((2,), dtype=torch.float16)
        self.q_offset = torch.zeros((2,), dtype=torch.float16)
        self.kv_offset = torch.zeros((2,), dtype=torch.float16)
        self.fa3_v_scale = torch.ones((2,), dtype=torch.float16)
        self.fa3_offset = torch.zeros((2,), dtype=torch.float16)


class TestRegisterLinearWrapper(unittest.TestCase):
    def setUp(self):
        self.soc_info = SimpleNamespace(need_nz=False, matmul_nd_nz=False)
        self.attn_wrapper = AttnWrapper(norm_name="norm", wrapper_name="attn")
        self.mlp_wrapper = MlpWrapper(norm_name="norm", wrapper_name="mlp")

    def _new_wrapper(self, enable_swiglu_quant=True, enable_rope_quant_kvcache=False):
        return WeightWrapper(self.soc_info, tp_rank=0,
                             attn_wrapper=self.attn_wrapper,
                             mlp_wrapper=self.mlp_wrapper,
                             enable_swiglu_quant=enable_swiglu_quant,
                             enable_rope_quant_kvcache=enable_rope_quant_kvcache)

    def test_fp_branch_float16(self):
        ww = self._new_wrapper()
        lin = FakeLinear(dtype=torch.float16)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.FLOAT)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.FP)
        self.assertEqual(ww.layer_linear_descs[-1], lin.linear_desc)
        self.assertEqual(ww.layer_linear_transpose_types[-1], lin.trans_flag)

    def test_w4a16_branch(self):
        ww = self._new_wrapper()
        lin = FakeLinear(dtype=torch.int8)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.W4A16)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)
        self.assertTrue(torch.allclose(ww.weights[before + 3], lin.weight_offset))
        self.assertTrue(torch.allclose(ww.weights[before + 4], lin.weight_scale))
    
    def test_w8a16_branch(self):
        ww = self._new_wrapper()
        lin = FakeLinear(dtype=torch.int8)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.W8A16)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)
        self.assertTrue(torch.allclose(ww.weights[before + 3], lin.weight_offset))
        self.assertTrue(torch.allclose(ww.weights[before + 4], lin.weight_scale))

    def test_w8a8_dynamic_branch_with_down_and_swiglu(self):
        ww = self._new_wrapper(enable_swiglu_quant=True)
        lin = FakeLinear(dtype=torch.int8)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.W8A8_DYNAMIC, is_down=True)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)
        offset_tensor = ww.weights[before + 3]
        scale_tensor = ww.weights[before + 4]
        self.assertEqual(offset_tensor.dtype, torch.float32)
        self.assertEqual(scale_tensor.dtype, torch.float32)

    def test_w4a8_dynamic_branch(self):
        ww = self._new_wrapper()
        lin = FakeLinear(dtype=torch.int8)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.W4A8_DYNAMIC)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)

        self.assertTrue(torch.allclose(ww.weights[before + 4], lin.weight_scale))

    def test_default_int_branch_w8a8sc_with_index_and_down_swiglu(self):
        ww = self._new_wrapper(enable_swiglu_quant=True)
        lin = FakeLinear(dtype=torch.int8)
        before = len(ww.weights)
        ww.register_linear_wrapper(lin, QuantType.W8A8SC, is_down=True)
        after = len(ww.weights)
        self.assertEqual(after - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)
        self.assertTrue(torch.allclose(ww.weights[before + 0], lin.weight))
        self.assertTrue(torch.allclose(ww.weights[before + 1], lin.quant_bias))
        self.assertTrue(torch.allclose(ww.weights[before + 2], lin.deq_scale))
        self.assertEqual(ww.weights[before + 3].dtype, torch.float32)
        self.assertEqual(ww.weights[before + 4].dtype, torch.float32)
        self.assertTrue(torch.allclose(ww.weights[before + 5], lin.index))

    def test_register_linear_bias_no_bias_placeholder(self):
        ww = self._new_wrapper()
        lin = FakeLinear(dtype=torch.float16, has_bias=False)
        before = len(ww.weights)
        ww.register_linear_bias(lin, enable_nz=False)
        after = len(ww.weights)
        self.assertEqual(after - before, 2)
        self.assertTrue(torch.allclose(ww.weights[before], lin.weight))
        self.assertEqual(ww.weights[before + 1].shape, ww.placeholder.shape)

    def test_register_norm_with_and_without_bias(self):
        ww = self._new_wrapper()
        
        class FakeNorm:
            def __init__(self, with_bias=True):
                self.weight = torch.ones((2,), dtype=torch.float16)
                self.bias = torch.zeros((2,), dtype=torch.float16) if with_bias else None
        before = len(ww.weights)
        ww.register_norm(FakeNorm(with_bias=True))
        ww.register_norm(FakeNorm(with_bias=False))
        after = len(ww.weights)
        self.assertEqual(after - before, 4)
        self.assertTrue(torch.allclose(ww.weights[before], torch.ones((2,), dtype=torch.float16)))
        self.assertEqual(ww.weights[before + 3].shape, ww.placeholder.shape)

    def _build_attn_layer_pack(self):
        qkv = FakeModule(FakeLinear(dtype=torch.int8))
        output_proj = FakeModule(FakeLinear(dtype=torch.int8))
        attn_mod = SimpleNamespace(pack_type=PackType.ALL_W8A8, qkv=qkv, o=output_proj)
        layer = SimpleNamespace(attn=attn_mod)
        return layer

    def _build_attn_layer_sep(self):
        q = FakeModule(FakeLinear(dtype=torch.int8))
        k = FakeModule(FakeLinear(dtype=torch.int8))
        v = FakeModule(FakeLinear(dtype=torch.int8))
        output_proj = FakeModule(FakeLinear(dtype=torch.int8))
        attn_mod = SimpleNamespace(pack_type=PackType.INVALID, q=q, k=k, v=v, o=output_proj)
        layer = SimpleNamespace(self_attn=attn_mod, attn=attn_mod)
        return layer

    def _build_mlp_layer_pack(self):
        gate_up = FakeModule(FakeLinear(dtype=torch.int8))
        down = FakeModule(FakeLinear(dtype=torch.int8))
        mlp_mod = SimpleNamespace(pack_type=PackType.ALL_W8A8, gate_up=gate_up, down=down)
        layer = SimpleNamespace(mlp=mlp_mod)
        return layer

    def _build_mlp_layer_sep(self):
        gate = FakeModule(FakeLinear(dtype=torch.int8))
        up = FakeModule(FakeLinear(dtype=torch.int8))
        down = FakeModule(FakeLinear(dtype=torch.int8))
        mlp_mod = SimpleNamespace(pack_type=PackType.INVALID, gate=gate, up=up, down=down)
        layer = SimpleNamespace(mlp=mlp_mod)
        return layer

    def test_register_layer_kvquant_true_false(self):
        ww_true = self._new_wrapper(enable_rope_quant_kvcache=True)
        attn = SimpleNamespace(kv_cache_quant=FakeKVCacheQuant())
        layer = SimpleNamespace(self_attn=attn, attn=attn)
        before = len(ww_true.weights)
        ww_true.register_layer_kvquant(layer)
        self.assertEqual(len(ww_true.weights) - before, 8)
        ww_false = self._new_wrapper(enable_rope_quant_kvcache=False)
        layer2 = SimpleNamespace(self_attn=attn, attn=attn)
        before2 = len(ww_false.weights)
        ww_false.register_layer_kvquant(layer2)
        self.assertEqual(len(ww_false.weights) - before2, 8)

    def test_register_layer_qkvquant(self):
        ww = self._new_wrapper()
        attn = SimpleNamespace(fa3=FakeFA3())
        layer = SimpleNamespace(self_attn=attn, attn=attn)
        before = len(ww.weights)
        ww.register_layer_qkvquant(layer)
        self.assertGreater(len(ww.weights) - before, 0)

    def test_register_layer_reducequant(self):
        ww = self._new_wrapper()
        reduce_q = FakeReduceQuant()
        layer = SimpleNamespace(self_attn=SimpleNamespace(reduce_quant=reduce_q),
                                mlp=SimpleNamespace(reduce_quant=reduce_q))
        before = len(ww.weights)
        ww.register_layer_reducequant(layer)
        self.assertEqual(len(ww.weights) - before, 8)

    def test_register_model_norm_and_lmhead(self):
        ww = self._new_wrapper()
        norm = SimpleNamespace(weight=torch.ones((2,), dtype=torch.float16), bias=None)
        before = len(ww.weights)
        ww.register_model_norm(norm)
        self.assertEqual(len(ww.weights) - before, 1)
        lmhead = SimpleNamespace(linear=SimpleNamespace(weight=torch.ones((2, 2), dtype=torch.float16)))
        before2 = len(ww.weights)
        ww.register_model_lmhead(lmhead)
        self.assertEqual(len(ww.weights) - before2, 1)

    def test_register_model_lmhead_quant(self):
        ww = self._new_wrapper()
        lin = SimpleNamespace(weight=torch.zeros((2, 2), dtype=torch.int8),
                              quant_bias=torch.ones((2,), dtype=torch.int32),
                              deq_scale=torch.ones((2,), dtype=torch.float32),
                              input_offset=torch.zeros((2,), dtype=torch.int32),
                              input_scale=torch.ones((2,), dtype=torch.float16))
        lmhead = SimpleNamespace(linear=lin)
        before = len(ww.weights)
        ww.register_model_lmhead_quant(lmhead)
        self.assertEqual(len(ww.weights) - before, 6)
        self.assertEqual(ww.layer_linear_type[-1], LinearType.INT)

    def test_register_layer_addrmsnormquant_paths(self):
        ww = self._new_wrapper()
        lin_int = FakeLinear(dtype=torch.int8)
        lin_fp = FakeLinear(dtype=torch.float16)
        pack_mod_int = SimpleNamespace(linear=lin_int)
        pack_mod_fp = SimpleNamespace(linear=lin_fp)
        layer = SimpleNamespace()
        setattr(layer, 'attn', SimpleNamespace(pack_type=PackType.ALL_W8A8, pack_mod=pack_mod_int))
        setattr(layer, 'mlp', SimpleNamespace(pack_type=PackType.ALL_W8A8, pack_mod=pack_mod_fp))
        wrapper_attn = AttnWrapper(norm_name="norm", wrapper_name="attn", pack_name="pack_mod")
        wrapper_mlp = MlpWrapper(norm_name="norm", wrapper_name="mlp", pack_name="pack_mod")
        before = len(ww.weights)
        ww.register_layer_addrmsnormquant_by_wrapper(layer, wrapper_attn, QuantType.W8A8)
        ww.register_layer_addrmsnormquant_by_wrapper(layer, wrapper_mlp, QuantType.W4A16)
        after = len(ww.weights)
        self.assertEqual(after - before, 4)
        self.assertTrue(ww.weights[-1].equal(ww.placeholder))
        self.assertTrue(ww.weights[-2].equal(ww.placeholder))


if __name__ == "__main__":
    unittest.main()