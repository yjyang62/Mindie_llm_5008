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
from copy import deepcopy
from atb_llm.models.base.feature_decorator.v3.base_decorator import FeatureDecorator
from atb_llm.models.base.flash_causal_lm_v3 import EngineWrapper


class AliceFeatureDecorator(FeatureDecorator):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.feature_stackable_whitelist = ["prefill", "decode", "bob"]
        self.need_additional_engine = True
    
    @property
    def feature_name(self):
        return "alice"
    
    def is_enabled(self):
        return True

    def generate_engine_wrapper(self, engine_wrapper):
        engine_wrapper_copy = deepcopy(engine_wrapper)
        engine_wrapper_copy.feature_list.append(self.feature_name)
        engine_wrapper_copy.input_keys.add("alice_input")
        return engine_wrapper_copy

    def is_activated(self, input_metadata):
        return False


class TestFeatureDecorator(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_decorator = AliceFeatureDecorator(None)
    
    def test_is_stackable(self):
        self.assertTrue(self.feature_decorator.is_stackable(["prefill"]))
        self.assertFalse(self.feature_decorator.is_stackable(["prefill", "other_mock"]))

    def test_generate_engine_wrappers(self):
        engine_wrappers = [
            EngineWrapper(["prefill"], {"input_ids"}, {}),
            EngineWrapper(["decode"], {"input_ids"}, {}),
            EngineWrapper(["prefill", "other_mock"], {"input_ids"}, {})
        ]
        self.feature_decorator.expand_engine_wrapper_collections(engine_wrappers)
        self.assertAlmostEqual(len(engine_wrappers), 5)
    
    def test_is_activated(self):
        input_metadata = {}
        self.assertFalse(self.feature_decorator.is_activated(input_metadata))

    def test_modify_inputs(self):
        self.feature_decorator.modify_inputs({}, {}, {}, {})


if __name__ == "__main__":
    unittest.main()