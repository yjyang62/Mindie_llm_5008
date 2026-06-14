# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from abc import ABC, abstractmethod


class FeatureDecorator(ABC):
    def __init__(self, model) -> None:
        self.feature_stackable_whitelist: list = []
        self.need_additional_engine: bool = False
        self.model = model

    @property
    @abstractmethod
    def feature_name(self):
        pass

    @property
    @abstractmethod
    def is_enabled(self):
        return False
    
    def is_stackable(self, feature_list):
        return all(feature in self.feature_stackable_whitelist for feature in feature_list)
    
    def expand_engine_wrapper_collections(self, engine_wrappers: list):
        for engine_wrapper in engine_wrappers[:]:
            if self.is_stackable(engine_wrapper.feature_list):
                engine_wrappers.append(self.generate_engine_wrapper(engine_wrapper))

    def generate_engine_wrapper(self, engine_wrapper):
        raise NotImplementedError

    def modify_inputs(self, engine_inputs, engine_outputs, engine_runtime_params, input_metadata):
        pass
    
    @abstractmethod
    def is_activated(self, input_metadata):
        return False