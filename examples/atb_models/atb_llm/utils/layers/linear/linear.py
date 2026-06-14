#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnLinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 gather_output=True,
                 process_group=None):
        super().__init__()
        self.process_group = process_group
        self.tp_size = self.process_group.size()
        self.in_features = in_features
        self.out_features = out_features // self.tp_size
        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.gather_output = gather_output
        self.bias = nn.Parameter(torch.ones(self.out_features)) if bias else None

    def multiply_gather(self, x):
        if self.bias is not None:
            x = F.linear(x, self.weight, self.bias)
        else:
            x = F.linear(x, self.weight)

        if self.gather_output and self.tp_size > 1:
            world_output = [
                torch.empty_like(x)
                for _ in range(self.process_group.size())
            ]
            torch.distributed.all_gather(world_output, x, group=self.process_group)
            x = torch.cat(world_output, dim=-1)

        return x

    def forward(self, x):
        return self.multiply_gather(x)


class RowLinear(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 process_group=None):
        super().__init__()
        self.process_group = process_group
        self.tp_size = self.process_group.size()
        self.in_features = in_features // self.tp_size
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.ones(self.out_features)) if bias else None

    def multiply_reduce(self, x):
        if self.bias is not None and self.process_group.rank() == 0:
            x = F.linear(x, self.weight, self.bias)
        else:
            x = F.linear(x, self.weight)

        if self.tp_size > 1:
            torch.distributed.all_reduce(x, group=self.process_group)

        return x

    def forward(self, x):
        return self.multiply_reduce(x)