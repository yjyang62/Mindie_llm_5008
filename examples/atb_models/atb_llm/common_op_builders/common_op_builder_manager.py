# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from atb_llm.common_op_builders.base_common_op_builder import BaseCommonOpBuilder
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.env import ENV


class CommonOpBuilderManager:
    _common_op_builders = []

    @classmethod
    def register(cls, common_op_builder_class):
        cls._common_op_builders.append(common_op_builder_class())

    @classmethod
    def get_builder(cls, param: dict) -> BaseCommonOpBuilder | None:
        for common_op_builder in cls._common_op_builders:
            if common_op_builder.is_match(param):
                return common_op_builder
        print_log(ENV.rank, logger.debug, f"CommonOpBuilder not found for param: {param}")
        raise RuntimeError(f"CommonOpBuilder not found for param: {param}")