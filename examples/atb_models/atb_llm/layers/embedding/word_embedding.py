# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch


from ...models.base.config import BaseConfig
from ...layers.base_layer import BaseLayer
from ... import nn
from ...nn.tensor import Tensor
from ...nn.distributed import all_gather
from ...utils.loader.safetensor_file_loader import SafetensorFileLoader
from ...utils.loader.weight_loader import sharded_loader, replicated_loader
from ...utils.mapping import Mapping


class ParallelEmbedding(BaseLayer):
    def __init__(self, config: BaseConfig, file_loader: SafetensorFileLoader, prefix: str, mapping: Mapping, **kwargs):
        super().__init__(config, file_loader, prefix=prefix, **kwargs)
        self.mapping = mapping
        self.parallel_embedding = kwargs.get("parallel_embedding", False)

    def create_module(self, prefix: str, **kwargs) -> nn.Module:
        self.weight = nn.Parameter(prefix=prefix, suffix="weight")
        self.weight.register_processor(lambda tensor: tensor.to(self.config.torch_dtype))

    def weight_loader(self,
        parameter: nn.Parameter, prefix: str, **kwargs
    ) -> torch.Tensor:
        parallel_embedding = kwargs.get("parallel_embedding", False)
        if parallel_embedding:
            return sharded_loader(parameter, self.file_loader, [prefix], dim=1, **kwargs)
        return replicated_loader(parameter, self.file_loader, [prefix], **kwargs)

    def forward(self, input_ids: Tensor) -> Tensor:
        out = nn.functional.gather(self.weight.get_tensor(), dim=0, index=input_ids, batch_dims=0)
        if self.parallel_embedding and self.mapping.lm_head_tp.group_size > 1:
            out = all_gather(out)
            out = out.permute([1, 0, 2]).reshape(
                lambda org_shape: [org_shape[0], org_shape[1] * org_shape[2]]
            )
        return out
