# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from ddt import ddt, data, unpack

from mindie_llm.runtime.config.lora_config import LoraConfig


@ddt
class TestLoraConfig(unittest.TestCase):
    def setUp(self):
        self.lora_config = LoraConfig.from_dict({})

    @data(
        ('r', 0),
        ('lora_alpha', 0),
        ('rank_pattern', '42'),
        ('rank_pattern', {'1': 0}),
        ('alpha_pattern', {'1': 0}),
        ('target_modules', ['lm_head'])
        )
    @unpack
    def test_validate_fail(self, key, value):
        setattr(self.lora_config, key, value)
        with self.assertRaises(ValueError) as _:
            self.lora_config._validate()


if __name__ == '__main__':
    unittest.main()