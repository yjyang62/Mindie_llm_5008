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
from unittest.mock import MagicMock

from atb_llm.models.base.inputs_modifier.long_seq_modifier import LongSeqModifier


class TestLongSeqModifier(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()

    def test_init_with_no_adapter(self):
        long_seq_modifier = LongSeqModifier(self.config)
        self.assertFalse(long_seq_modifier.active)

    def test_init_with_adapter(self):
        self.config.rope_scaling.rope_type = "dynamic"
        long_seq_modifier = LongSeqModifier(self.config)
        self.assertTrue(long_seq_modifier.active)

    def test_modify_inputs(self):
        self.config.rope_scaling.rope_type = "dynamic"
        long_seq_modifier = LongSeqModifier(self.config)

        engine_inputs = [None] * 16
        placeholder = "DUMMY_PLACEHOLDER"

        mock_pos_embed = MagicMock()
        mock_pos_embed.position_ids_expanded = "POS_EXPANDED"
        mock_pos_embed.ntk_inv_freqs = "NTK_FREQS"
        mock_pos_embed.pos_lens = "POS_LEN"

        long_seq_modifier.modify_inputs(
            engine_inputs,
            mock_pos_embed,
            position_ids=None,
            placeholder=placeholder
        )

        self.assertEqual(engine_inputs[3], placeholder)
        self.assertEqual(engine_inputs[4], placeholder)
        self.assertEqual(engine_inputs[-3], "POS_EXPANDED")
        self.assertEqual(engine_inputs[-2], "NTK_FREQS")
        self.assertEqual(engine_inputs[-1], "POS_LEN")

    def test_modify_inputs_yarn(self):
        self.config.rope_scaling.rope_type = "yarn"
        long_seq_modifier = LongSeqModifier(self.config)
        engine_inputs = [1, 2]
        long_seq_modifier.modify_inputs(engine_inputs, MagicMock(), MagicMock())
        self.assertEqual(len(engine_inputs), 5)