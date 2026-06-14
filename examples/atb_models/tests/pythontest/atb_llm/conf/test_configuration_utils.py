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

import os
import json
import tempfile
import unittest
from atb_llm.utils.configuration_utils import LLMConfig
from atb_llm.utils.parameter_validators import (
    DictionaryParameterValidator, BooleanParameterValidator, RangeParamaterValidator
)


class TestConfiguratioinUtils(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.empty_config_path = os.path.join(self.temp_dir.name, 'empty_config.json')
        with open(self.empty_config_path, 'w') as f:
            json.dump({}, f)

        self.config_path1 = os.path.join(self.temp_dir.name, 'test_config1.json')
        self.test_json_config1 = {
            "llm": {
                "ccl": {
                    'backend': 'lccl',
                    "enable_mc2": "false"
                },
                "engine": {
                    "graph": "cpp"
                },
                "plugins": {
                    "plugin_type": "mtp",
                    "num_speculative_tokens": "1"
                },
                "enable_reasoning": "false"
            },
        }
        with open(self.config_path1, 'w') as f:
            json.dump(self.test_json_config1, f)

        self.config_path2 = os.path.join(self.temp_dir.name, 'test_config2.json')
        self.test_json_config2 = {
            "llm": {
                "ccl": {
                    'backend': 'lccl',
                    "enable_mc2": "false"
                },
                "engine": {
                    "graph": "cpp"
                },
                "enable_reasoning": "false"
            },
            "models": {
                "deepseekv2": {
                    "ccl": {
                        'backend': 'hccl',
                        "enable_mc2": "true"
                    },
                    "enable_reasoning": "true"
                }
            },
        }
        with open(self.config_path2, 'w') as f:
            json.dump(self.test_json_config2, f)

        self.validators = {"llm": DictionaryParameterValidator({
            "ccl": DictionaryParameterValidator({
                "backend": RangeParamaterValidator(range_list=["lccl", "hccl"]),
                "enable_mc2": BooleanParameterValidator()
            }),
            "engine": DictionaryParameterValidator({
                "graph": RangeParamaterValidator(range_list=["cpp", "python"])
            })
        }), "models": {}}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config(self):
        llm_config = LLMConfig(self.empty_config_path)
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, None)

    def test_load_config(self):
        llm_config = LLMConfig(self.config_path1)
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')

    def test_update_dict_replace(self):
        llm_config = LLMConfig(self.config_path1)
        replace_dict = {
            "llm": {
                "engine": {
                    "graph": "python"
                },
                "enable_reasoning": "true"
            }
        }
        llm_config.update(replace_dict, allow_new_keys=False)
        self.assertEqual(llm_config.llm.engine.graph, 'python')
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')
        self.assertTrue(llm_config.llm.enable_reasoning)

    def test_update_dict_add_llm(self):
        llm_config = LLMConfig(self.config_path1)
        add_dict = {
            "llm": {
                "stream_options": {
                    "cv_dual": "false",
                    "micro_batch": "false"
                },
            }
        }
        llm_config.update(add_dict, allow_new_keys=True)
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')
        self.assertFalse(llm_config.llm.stream_options.cv_dual)

    def test_update_dict_add_models(self):
        llm_config = LLMConfig(self.config_path1)
        add_models_dict = {
            "models": {
                "deepseekv2": {
                    "eplb": {
                        "level": 0,
                        "expert_map_file": "",
                        "num_of_redundant_experts": 0
                    },
                    "ep_level": 1,
                }
            }
        }
        llm_config.update(add_models_dict, allow_new_keys=True)
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')

        self.assertEqual(llm_config.models.deepseekv2.eplb.level, 0)
        self.assertEqual(llm_config.models.deepseekv2.eplb.expert_map_file, None)
        self.assertEqual(llm_config.models.deepseekv2.eplb.num_of_redundant_experts, 0)
        self.assertEqual(llm_config.models.deepseekv2.ep_level, 1)

    def test_update_dict_add_none(self):
        llm_config = LLMConfig(self.config_path1)
        add_models_dict = None
        llm_config.update(add_models_dict, allow_new_keys=True, current_path='models')
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')
        self.assertEqual(llm_config.llm.plugins.plugin_type, 'mtp')

    def test_merge_correct_model_config(self):
        llm_config = LLMConfig(self.config_path2)
        model_name = 'deepseekv2'
        llm_config.merge_models_config(model_name)
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.ccl.backend, 'hccl')
        self.assertTrue(llm_config.llm.enable_reasoning)

    def test_merge_incorrect_model_config(self):
        llm_config = LLMConfig(self.config_path2)
        model_name = 'error_name'
        llm_config.merge_models_config(model_name)
        self.assertEqual(llm_config.llm.engine.graph, 'cpp')
        self.assertEqual(llm_config.llm.ccl.backend, 'lccl')

    def test_check_config(self):
        llm_config = LLMConfig(self.config_path2)
        
        llm_config.check_config(self.validators)


if __name__ == "__main__":
    unittest.main()
