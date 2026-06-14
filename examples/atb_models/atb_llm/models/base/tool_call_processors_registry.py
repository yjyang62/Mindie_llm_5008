# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
ToolCallProcessor Registry

Import all ToolCallProcessors to ensure that all ToolCallProcessors are registered to ToolParserManager.
"""

from ...utils.log import logger


def register_all_tool_call_processors():
    """
    Import to register all ToolCallProcessors
    """
    # ChatGLM
    from ..chatglm.tool_call_process_chatglm import (
        ToolsCallProcessorChatglmV2, 
        ToolsCallProcessorChatglmV3,
        ToolsCallProcessorChatglmV4, 
        ToolsCallProcessorChatglm
    )
    logger.debug("Successfully imported ChatGLM ToolCallProcessors")

    # Qwen2
    from ..qwen2.tool_call_process_qwen2 import (
        ToolsCallProcessorQwen1_5_or_2, 
        ToolsCallProcessorQwen2_5, 
        ToolsCallProcessorQwenAuto
    )
    logger.debug("Successfully imported Qwen2 ToolCallProcessors")

    # Qwen3
    from ..qwen3.tool_call_process_qwen3 import ToolsCallProcessorQwen3
    from ..qwen3_moe.tool_call_process_qwen3_coder import ToolsCallProcessorQwen3Coder
    logger.debug("Successfully imported Qwen3 ToolCallProcessors")

    # DeepSeekV2
    from ..deepseekv2.tool_call_process_deepseekv2 import (
        ToolsCallProcessorDeepseekv3, 
        ToolsCallProcessorDeepseekv31
    )
    logger.debug("Successfully imported DeepSeekV2 ToolCallProcessors")

    # Llama
    from ..llama.tool_call_process_llama import ToolsCallProcessorLlama
    logger.debug("Successfully imported Llama ToolCallProcessors")


register_all_tool_call_processors()