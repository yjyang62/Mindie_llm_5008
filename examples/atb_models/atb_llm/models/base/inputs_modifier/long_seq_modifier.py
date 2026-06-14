# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from typing import List

import torch

from atb_llm.utils.layers.embedding.position_rotary_embedding import PositionRotaryEmbedding
from atb_llm.utils.log import logger


ROPE_TYPE = "rope_type"


class LongSeqModifier:
    """
    This class contains methods and attributes required to enable long sequences functionality in a model.
    """
    def __init__(self, config, **kwargs):
        self.config = config
        rope_scaling = getattr(config, 'rope_scaling', None)
        self.is_yarn = bool(
            rope_scaling and (
                getattr(rope_scaling, ROPE_TYPE, None) == "yarn" or
                getattr(rope_scaling, "type", None) == "yarn"
            )
        )
        self.is_dynamicntk = bool(
            rope_scaling and (
                getattr(rope_scaling, ROPE_TYPE, None) == "dynamic" or
                getattr(rope_scaling, ROPE_TYPE, None) == "llama3" or
                getattr(rope_scaling, ROPE_TYPE, None) == "dynamic"
            )
        )
        if self.is_yarn or self.is_dynamicntk:
            self.active = True
        else:
            self.active = False

    def modify_inputs(
            self,
            inputs: List[torch.Tensor],
            pos_embed: PositionRotaryEmbedding,
            position_ids=None,
            **kwargs
        ) -> None:
        if not self.active:
            return
        if self.is_yarn:
            inputs[1] = pos_embed.position_ids_expanded
            inputs.append(pos_embed.ntk_inv_freqs)
            inputs.append(pos_embed.pos_lens)
            inputs.append(position_ids)
        elif self.is_dynamicntk:
            placeholder = kwargs.get('placeholder')
            if placeholder is None:
                err_msg = "If long_seq_enable and ROPE_TYPE is llama3 or dynamic, kwargs must has key: placeholder."
                logger.info(err_msg)
                raise ValueError(err_msg)
            inputs[3] = placeholder
            inputs[4] = placeholder
            inputs.append(pos_embed.position_ids_expanded)
            inputs.append(pos_embed.ntk_inv_freqs)
            inputs.append(pos_embed.pos_lens)
        else:
            err_msg = "As for long sequence, only Yarn, DynamicNTK and llama3 are supported now."
            logger.error(err_msg)
            raise ValueError(err_msg)