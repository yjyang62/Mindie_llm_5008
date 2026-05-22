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

from atb_llm.models.base.inputs_modifier.qlen_modifier import QLenModifier


class TestQLenModifier(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("npu:0")

    def test_modify_inputs_with_no_mask(self):
        qlen_modifier = QLenModifier()
        engine_inputs = []
        engine_param = {}
        qlen_modifier.modify_inputs(engine_inputs, engine_param, self.device)
        self.assertEqual(len(engine_inputs), 0)

    def test_modify_inputs_with_splitfuse(self):
        qlen_modifier = QLenModifier()
        engine_inputs = []
        engine_param = {}
        q_lens = [2, 3, 4]
        attn_mask = torch.ones(7)
        qlen_modifier.modify_inputs(
            engine_inputs,
            engine_param,
            self.device,
            is_prefill=True,
            enable_prefill_pa=True,
            enable_splitfuse_pa=True,
            q_lens=q_lens,
            attn_mask=attn_mask
        )
        self.assertEqual(len(engine_inputs), 1)
        self.assertEqual(engine_param.get("qLen", None), [2, 5, 9])

    def test_modify_inputs_with_la(self):
        qlen_modifier = QLenModifier()
        engine_inputs = []
        engine_param = {}
        q_lens = [10, 11, 12]
        attn_mask = torch.ones(7)
        qlen_modifier.modify_inputs(
            engine_inputs,
            engine_param,
            self.device,
            q_lens=q_lens,
            attn_mask=attn_mask
        )
        self.assertEqual(len(engine_inputs), 1)
        self.assertEqual(engine_param.get("qLen", None), [10, 11, 12])