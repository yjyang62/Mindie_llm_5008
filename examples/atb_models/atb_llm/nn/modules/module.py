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

from torch import nn


class Module(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._aliases = {}
    
    def add_aliases(self, alias_map: dict[str, str]) -> None:
        '''
        Update a mapping from alias to attribute name. For example, to get the attribute
        `foo` from alias `bar`, use `add_aliases({"bar": "foo"})` to register the alias.
        Then access the attribute `foo` by `getattr_by_alias("bar")`.
        '''
        self._aliases.update(alias_map)

    def getattr_by_alias(self, alias: str) -> None:
        if alias in self._aliases:
            attribute_name = self._aliases[alias]
            try:
                return getattr(self, attribute_name)
            except AttributeError as e:
                raise AttributeError(f"Alias mapping error: `{alias}` -> `{attribute_name}`."
                    f"Missing attribute: '{attribute_name}' not found in module. Possible causes: "
                    "1. Typo in attribute name when `add_aliases`. 2. Attribute not initialized in __init__."
                ) from e
        else:
            try:
                return getattr(self, alias)
            except AttributeError as e:
                raise AttributeError(
                    f"Attribute '{alias}' not found in module. Possible causes: 1. Typo in attribute name;"
                    "2. Attribute not initialized in __init__; 3. Alias not registered (use add_alias() to register)."
                ) from e


class ModuleList(nn.ModuleList):
    def __init__(self, modules):
        super().__init__(modules=modules)


class ModuleDict(nn.ModuleDict):
    def __init__(self, modules):
        super().__init__(modules=modules)
