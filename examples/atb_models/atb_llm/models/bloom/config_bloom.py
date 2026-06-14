# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from dataclasses import dataclass

from ..base.config import BaseConfig


@dataclass
class BloomConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.seq_length = 4096
        self.attribute_map = {
            'max_position_embeddings': 'seq_length',
            'num_hidden_layers': 'n_layer',
            'n_head': 'num_attention_heads',
            'hidden_size': 'n_embed',
        }
        super().__init__(**kwargs)
