# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import copy
from .fusion_pass_register import register_all_fusion_pass
from .fusion_pass_base import FusionPassBase
from ..utils.log.logging import logger


class FusionPassManager:
    __instance = None

    def __init__(self, del_fpass_keys: list[str] = None):
        self.fusion_pass_list = {}
        self.unremovable_fusion_pass_list = {}
        self.del_fpass_keys = del_fpass_keys if del_fpass_keys is not None else {}

    @staticmethod
    def get_instance(del_fpass_keys: list[str] = None):
        if FusionPassManager.__instance is None:
            FusionPassManager.__instance = FusionPassManager(del_fpass_keys)
        FusionPassManager.__instance.del_fpass_keys = del_fpass_keys if del_fpass_keys is not None else {}
        return FusionPassManager.__instance

    @classmethod
    def _get_fused_nodes_by_pass(cls, node_num: int, target, fusion_pass: FusionPassBase):
        num_nodes = len(fusion_pass.nodes_before_fusing)
        for i in range(len(target) - num_nodes + 1):
            sub_of_target = target[i:i + num_nodes]
            fused_nodes = fusion_pass.get_fused_nodes_by_pass(sub_of_target)
            if fused_nodes is not None:
                return sub_of_target, fused_nodes
        return None, None

    def fuse_network(self, network):
        nodes = network.nodes
        sub_nodes = copy.deepcopy(nodes)
        cut_points = network.cut_points
        self._fuse_sub_network(sub_nodes, cut_points)
        new_sub_nodes = copy.deepcopy(sub_nodes)
        network.nodes = new_sub_nodes

    def register_fusion_pass(self, key: str, fusion_pass: FusionPassBase, unremovable: bool = False):
        self.fusion_pass_list[key] = fusion_pass
        if unremovable:
            self.unremovable_fusion_pass_list[key] = fusion_pass
        return 

    def _fuse_sub_network(self, nodes, cut_points):
        if self.del_fpass_keys == ["All"]:
            del_fpass_keys = self.fusion_pass_list
            fusion_pass_list = self.unremovable_fusion_pass_list
        else:
            del_fpass_keys = self.del_fpass_keys
            fusion_pass_list = self.fusion_pass_list
        for fusion_pass_key in fusion_pass_list.keys():
            for node_num in range(2, len(nodes) + 1):
                fusion_pass = None
                if fusion_pass_key not in del_fpass_keys:
                    fusion_pass = fusion_pass_list[fusion_pass_key]
                elif fusion_pass_key in del_fpass_keys and fusion_pass_key in self.unremovable_fusion_pass_list:
                    fusion_pass = fusion_pass_list[fusion_pass_key]
                    logger.warning("The %s is unremovable.", fusion_pass)
                while True:
                    if fusion_pass is None:
                        break
                    src_nodes, fused_nodes = self._get_fused_nodes_by_pass(node_num, nodes, fusion_pass)
                    if fused_nodes is not None:
                        _fuse_nodes(nodes, src_nodes, fused_nodes, cut_points)
                    else:
                        break


def _fuse_nodes(nodes, src_nodes, fused_nodes, cut_points):
    if len(cut_points) == 0:
        index = nodes.index(src_nodes[-1])
        nodes[index + 1: index + 1] = fused_nodes
        for node in src_nodes:
            nodes.remove(node)
    else:
        start_pos = nodes.index(src_nodes[0])
        end_pos = nodes.index(src_nodes[-1])

        nodes[end_pos + 1: end_pos + 1] = fused_nodes
        for node in src_nodes:
            nodes.remove(node)
        
        nodes_removed = len(src_nodes) - len(fused_nodes)
        for i, cut_point in enumerate(cut_points):
            if start_pos <= cut_point < end_pos:
                cut_points[i] = start_pos + len(fused_nodes) - 1
            elif cut_point >= end_pos:
                cut_points[i] -= nodes_removed


register_all_fusion_pass(FusionPassManager.get_instance())