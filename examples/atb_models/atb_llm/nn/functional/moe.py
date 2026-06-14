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

from atb_llm.nn.network_manager import get_default_net
from atb_llm.nn.node import Node
from atb_llm.nn.tensor import Tensor


def moe_topk_softmax(input_: Tensor, k: int) -> list[Tensor]:
    """
    Performs the MoeTopkSoftmax operation on the input tensor and returns a list containing the output tensor, expert indices, and row indices.

    Args:
        input_ (Tensor): The input tensor.
        k (int): The number used for Top-k selection.

    Returns:
        list[Tensor]: A list containing the output tensor, expert indices, and row indices.
    
    Restrictions:
        This operation doesn't support Atlas 300I DUO.
    """
    y_out = Tensor()
    expert_idx_out = Tensor()
    row_idx_out = Tensor()
    outputs = [y_out, expert_idx_out, row_idx_out]
    param = {
        "topkNum": k
    }
    node = Node("MoeTopkSoftmax", param, [input_], outputs)
    get_default_net().push_node(node)
    return outputs


def moe_init_routing(
        input_: Tensor,
        expert_idx: Tensor,
        topk_num: int,
        expert_num: int) -> list[Tensor]:
    """
    Initialization function for routing in a Mixture of Experts (MoE) model.

    Args:
        input_ (Tensor): Input tensor containing data to be routed.
        expert_idx (Tensor): Tensor indicating which expert each input should be routed to.
        topk_num (int): Number of top-k experts to select.
        expert_num (int): Total number of experts.

    Returns:
        list[Tensor]: A list containing three tensors:
            - expanded_x_out (Tensor): Expanded output tensor.
            - expanded_row_idx_out (Tensor): Expanded row index output tensor.
            - expert_tokens_count_or_cumsum_out (Tensor): Expert token count or cumulative sum output tensor.
    
    Restrictions:
        This operation doesn't support Atlas 300I DUO.
    """
    expanded_x_out = Tensor()
    expanded_row_idx_out = Tensor()
    expert_tokens_count_or_cumsum_out = Tensor()
    outputs = [expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum_out]
    param = {
        "topkNum": topk_num,
        "expertNum": expert_num
    }
    node = Node("MoeInitRouting", param, [input_, expert_idx], outputs)
    get_default_net().push_node(node)
    return outputs


def moe_token_unpermute(
        permuted_tokens: Tensor,
        sorted_indices: Tensor,
        experts_weights: Tensor) -> Tensor:
    """
    Unpermute tokens in a Mixture of Experts (MoE) model.

    Args:
        permuted_tokens (Tensor): Permuted tokens tensor.
        sorted_indices (Tensor): Tensor containing sorted indices for unpermuting.
        experts_weights (Tensor): Tensor of expert weights.

    Returns:
        Tensor: The unpermuted tokens tensor.

    Restrictions:
        This operation doesn't support Atlas 300I DUO.
    """
    param = {}
    out = Tensor()
    node = Node("MoeTokenUnpermute", param, [permuted_tokens, sorted_indices, experts_weights], [out])
    get_default_net().push_node(node)
    return out


def gating(
        topk: Tensor,
        idx_arr: Tensor,
        topk_expert_num: int,
        cum_sum_num: int,
        cum_sum_int64: bool = False,
        device_expert: list[int] = None) -> list[Tensor]:
    """
    Perform gating operation in a Mixture of Experts (MoE) model.

    Args:
        topk (Tensor): Tensor containing top-k values.
        idx_arr (Tensor): Tensor containing expert indices.
        topk_expert_num (int): Number of top-k experts.
        cum_sum_num (int): Number for cumulative sum calculation.
        cum_sum_int64 (bool, optional): Whether to use int64 for cumulative sum. Defaults to False.
        device_expert (list[int], optional): List of expert device IDs. Defaults to None.

    Returns:
        list[Tensor]: A list containing tensors:
            - token_index (Tensor): Tensor of token indices.
            - cum_sum (Tensor): Tensor of cumulative sums.
            - original_index (Tensor): Tensor of original indices.
            - valid_index (Tensor, optional): Tensor of valid indices if `device_expert` is provided.
    """
    param = {
        "topkExpertNum": topk_expert_num,
        "cumSumNum": cum_sum_num,
        "cumSumInt64": cum_sum_int64
    }
    token_index = Tensor()
    cum_sum = Tensor()
    original_index = Tensor()
    outputs = [token_index, cum_sum, original_index]
    if device_expert:
        param["deviceExpert"] = device_expert
        valid_index = Tensor()
        outputs.append(valid_index)
    node = Node("Gating", param, [topk, idx_arr], outputs)
    get_default_net().push_node(node)
    return outputs


def group_topk(
        input_: Tensor,
        idx_arr: Tensor,
        group_num: int = 1,
        k: int = 0,
        group_multi_flag: int = 0,
        n: int = 1):
    """
    Performs a grouped Top-k operation on the input tensor.

    Args:
        input_ (Tensor): Input tensor data.
        idx_arr (Tensor): Index tensor used for grouping.
        group_num (int): Number of groups, default is 1.
        k (int): Number of top elements to select per group, default is 0.
        group_multi_flag (int): Group multi-select flag, default is 0.
        n (int): Interval for selecting elements in each group, default is 1.

    Returns:
        None: This function does not return a value but adds the created node to the default network.

    Restrictions:
        This operation doesn't support Atlas 300I DUO.
    """
    # Create a dictionary to store the parameters for the grouped Top-k operation
    param = {
        "groupNum": group_num,
        "k": k,
        "groupMultiFlag": group_multi_flag,
        "n": n
    }
    node = Node('GroupTopk', param, [input_, idx_arr], [input_])
    get_default_net().push_node(node)
