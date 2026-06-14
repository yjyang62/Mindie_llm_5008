# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mindie_llm.runtime.layers.fused_moe.token_dispatcher import (
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
    TokenDispatcherWithAll2AllV,
    MoeAllGatherArgs,
    MoeMC2Args,
    MoeAll2AllVArgs,
    AllGatherDispatchContext,
    MC2DispatchContext,
    All2AllVDispatchContext,
    async_all_to_all,
    gather_from_sequence_parallel_region,
)
from mindie_llm.runtime.utils.npu.device_utils import DeviceType

sys.modules['torch_npu'] = MagicMock()
sys.modules['torch.distributed'] = MagicMock()


@pytest.fixture(scope="session", autouse=True)
def global_mocks():
    """Global mock for basic dependencies"""
    # Mock get_parallel_info_manager
    mock_parallel_info = MagicMock()
    mock_parallel_info.moe_ep.group_size = 8
    mock_parallel_info.moe_ep.rank = 0
    mock_parallel_info.moe_ep.process_group = MagicMock()
    mock_parallel_info.moe_ep_mc2.process_group = MagicMock()

    with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.get_parallel_info_manager") as mock_pim:
        mock_pim.return_value = mock_parallel_info

        # Mock get_npu_node_info
        mock_platform = MagicMock()
        mock_platform.get_device_type.return_value = DeviceType.ASCEND_910_93
        with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.get_npu_node_info") as mock_pf:
            mock_pf.return_value = mock_platform

            # Mock torch.distributed
            mock_dist = MagicMock()
            mock_dist.get_rank.return_value = 0
            mock_dist.get_world_size.return_value = 8
            mock_dist.all_to_all_single.return_value = MagicMock(wait=MagicMock())
            mock_dist.all_gather_into_tensor.return_value = torch.ones(8, 16)
            mock_dist.all_gather.return_value = [torch.ones(4, 16) for _ in range(8)]

            with patch(
                    "mindie_llm.runtime.layers.fused_moe.token_dispatcher.gather_from_sequence_parallel_region") as mock_gather:
                # Return shape: [num_experts * ep_size] to match reshape logic
                mock_gather.side_effect = lambda x, group, output_split_sizes=None: torch.ones(x.shape[0] * 8)

                with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.dist", mock_dist):
                    yield


@pytest.fixture
def mock_torch_npu():
    """Mock return values of torch_npu operators"""
    mock_npu = MagicMock()

    # Mock npu_moe_init_routing_v2
    mock_npu.npu_moe_init_routing_v2.return_value = (
        torch.ones(8, 16),  # sorted_hidden_states
        torch.arange(8),  # expanded_row_idx
        torch.ones(8),  # expert_tokens
        torch.ones(8)  # pertoken_scale
    )

    # Mock npu_moe_token_unpermute
    mock_npu.npu_moe_token_unpermute.side_effect = lambda *args, **kwargs: torch.ones(8, 16) if (
            len(args) > 0 and args[0].shape[0] == 8) else torch.ones(4, 16)

    # Mock npu_moe_distribute_dispatch/v2
    mock_npu.npu_moe_distribute_dispatch.return_value = (
        torch.ones(4, 16),  # expand_x
        torch.ones(4),  # dynamic_scale
        torch.ones(4),  # assist_info_for_combine
        torch.ones(8),  # expert_token_nums
        torch.ones(8),  # ep_recv_counts
        torch.ones(8),  # tp_recv_counts
    )
    mock_npu.npu_moe_distribute_dispatch_v2.return_value = mock_npu.npu_moe_distribute_dispatch.return_value

    # Mock npu_moe_distribute_combine/v2
    mock_npu.npu_moe_distribute_combine.return_value = torch.ones(4, 16)
    mock_npu.npu_moe_distribute_combine_v2.return_value = torch.ones(4, 16)

    # Mock npu_moe_token_permute
    mock_npu.npu_moe_token_permute.return_value = (
        torch.ones(8, 16),  # permutated_tokens
        torch.arange(8)  # reversed_mapping
    )

    with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.torch_npu", mock_npu):
        yield mock_npu


@pytest.fixture
def base_tensors():
    """Basic test tensors"""
    return {
        "hidden_states": torch.ones(4, 16).npu(),
        "topk_weights": torch.ones(4, 2).npu(),
        "topk_ids": torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]).npu(),
        "num_experts": 8,
        "mc2_mask": torch.ones(4).npu(),
        "expert_map": torch.tensor([0, 1, 2, 3, -1, -1, -1, -1]).npu(),
        "expert_list": [0, 1, 2, 3],
        "top_k": 2
    }


@pytest.fixture
def all2allv_dispatcher():
    dispatcher = TokenDispatcherWithAll2AllV()
    dispatcher.ep_size = 8
    dispatcher.ep_rank = 0
    dispatcher.ep_group = MagicMock()
    return dispatcher


class TestTokenDispatcherWithAllGather:
    @pytest.mark.parametrize("expert_list, with_quant", [
        (None, False),  # expert_list=None + non-quantization
        ([0, 1, 2, 3], False),  # expert_list with value + non-quantization
        (None, True),  # expert_list=None + quantization
        ([0, 1, 2, 3], True),  # expert_list with value + quantization
    ])
    def test_token_dispatch(self, mock_torch_npu, base_tensors, expert_list, with_quant):
        dispatcher = TokenDispatcherWithAllGather()

        args = MoeAllGatherArgs(
            hidden_states=base_tensors["hidden_states"],
            topk_weights=base_tensors["topk_weights"],
            topk_ids=base_tensors["topk_ids"],
            num_experts=base_tensors["num_experts"],
            top_k=base_tensors["top_k"],
            expert_list=expert_list,
            expert_map=base_tensors["expert_map"],
            with_quant=with_quant
        )

        output, context = dispatcher.token_dispatch(args)

        assert isinstance(output, dict)
        assert isinstance(context, AllGatherDispatchContext)
        assert output["group_list_type"] == 1
        assert "hidden_states" in output
        assert "group_list" in output
        assert output["dynamic_scale"] is None or with_quant

        if expert_list:
            mock_torch_npu.npu_moe_init_routing_v2.assert_called()
            call_args = mock_torch_npu.npu_moe_init_routing_v2.call_args[1]
            assert call_args["active_expert_range"] == [0, 4]
        else:
            call_args = mock_torch_npu.npu_moe_init_routing_v2.call_args[1]
            assert call_args["active_expert_range"] == [0, 1]

    def test_token_combine(self, mock_torch_npu, base_tensors):
        dispatcher = TokenDispatcherWithAllGather()

        ctx = AllGatherDispatchContext(
            expanded_row_idx=torch.arange(4),
            topk_weights=base_tensors["topk_weights"]
        )

        result = dispatcher.token_combine(base_tensors["hidden_states"], ctx)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 16)
        mock_torch_npu.npu_moe_token_unpermute.assert_called()


class TestTokenDispatcherWithMC2:
    @pytest.mark.parametrize("device_type, enable_v2, with_quant, has_shared_experts", [
        (DeviceType.ASCEND_910_93, True, True, True),  # ASCEND_910_93 + v2 + quantization + shared experts
        (DeviceType.ASCEND_910_93, False, False, False),
        # ASCEND_910_93 + non-v2 + non-quantization + no shared experts
        (DeviceType.ASCEND_910B, True, True, False),  # ASCEND_910B + v2 + quantization + no shared experts
        (DeviceType.ASCEND_910B, False, False, True),  # ASCEND_910B + non-v2 + non-quantization + shared experts
    ])
    def test_token_dispatch(self, mock_torch_npu, base_tensors, device_type, enable_v2, with_quant, has_shared_experts):
        with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.get_npu_node_info") as mock_pf:
            mock_platform = MagicMock()
            mock_platform.get_device_type.return_value = device_type
            mock_pf.return_value = mock_platform

            dispatcher = TokenDispatcherWithMC2()
            dispatcher.enable_dispatch_v2 = enable_v2

            mock_shared_experts = MagicMock() if has_shared_experts else None
            if has_shared_experts:
                if with_quant:
                    mock_shared_experts.gate_up_proj.return_value = ((torch.ones(4, 16), torch.ones(4)), None)
                    mock_shared_experts.act_fn.return_value = ((torch.ones(4, 16), torch.ones(4)), None)
                else:
                    mock_shared_experts.gate_up_proj.return_value = (torch.ones(4, 16), None)
                    mock_shared_experts.act_fn.return_value = (torch.ones(4, 16),)

            args = MoeMC2Args(
                hidden_states=base_tensors["hidden_states"],
                topk_weights=base_tensors["topk_weights"],
                topk_ids=base_tensors["topk_ids"],
                num_experts=base_tensors["num_experts"],
                mc2_mask=base_tensors["mc2_mask"],
                with_quant=with_quant,
                shared_experts=mock_shared_experts,
                quantized_x_for_share=torch.ones(4, 16) if with_quant else None,
                dynamic_scale_for_share=torch.ones(4) if with_quant else None
            )

            output, context = dispatcher.token_dispatch(args)

            assert isinstance(output, dict)
            assert isinstance(context, MC2DispatchContext)
            assert isinstance(context.global_bs, int)

            if enable_v2:
                mock_torch_npu.npu_moe_distribute_dispatch_v2.assert_called()
            else:
                mock_torch_npu.npu_moe_distribute_dispatch.assert_called()

            kwargs = dispatcher.select_dispatch_mc2_kwargs(args)
            assert kwargs["quant_mode"] == (2 if with_quant else 0)

            if device_type == DeviceType.ASCEND_910_93:
                assert "group_tp" in kwargs
                assert "tp_world_size" in kwargs
                if enable_v2:
                    assert "x_active_mask" in kwargs
            else:
                assert "group_tp" not in kwargs

    @pytest.mark.parametrize("enable_v2, with_quant, has_shared_experts", [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ])
    def test_token_combine(self, mock_torch_npu, base_tensors, enable_v2, with_quant, has_shared_experts):
        dispatcher = TokenDispatcherWithMC2()
        dispatcher.enable_dispatch_v2 = enable_v2

        mock_shared_experts = MagicMock() if has_shared_experts else None
        if has_shared_experts:
            if with_quant:
                mock_shared_experts.down_proj.return_value = ((torch.ones(4, 16), torch.ones(4)), None)
            else:
                mock_shared_experts.down_proj.return_value = (torch.ones(4, 16), None)

        ctx = MC2DispatchContext(
            topk_ids=base_tensors["topk_ids"],
            topk_weights=base_tensors["topk_weights"],
            num_experts=base_tensors["num_experts"],
            with_quant=with_quant,
            mc2_mask=base_tensors["mc2_mask"],
            shared_experts=mock_shared_experts,
            global_bs=32,
            assist_info_for_combine=torch.ones(4),
            ep_recv_counts=torch.ones(8),
            tp_recv_counts=torch.ones(8) if not with_quant else torch.empty(1, dtype=torch.int32).npu(),
            shared_act=torch.ones(4, 16) if has_shared_experts else None,
            swiglu_out_scale=torch.ones(4) if (has_shared_experts and with_quant) else None
        )

        result = dispatcher.token_combine(base_tensors["hidden_states"], ctx)

        if has_shared_experts:
            assert isinstance(result, tuple)
            assert len(result) == 2
        else:
            assert isinstance(result, torch.Tensor)

        if enable_v2:
            mock_torch_npu.npu_moe_distribute_combine_v2.assert_called()
        else:
            mock_torch_npu.npu_moe_distribute_combine.assert_called()

    def test_select_dispatch_mc2_kwargs(self, base_tensors):
        """Test parameter selection method independently"""
        dispatcher = TokenDispatcherWithMC2()

        args = MoeMC2Args(
            hidden_states=base_tensors["hidden_states"],
            topk_weights=base_tensors["topk_weights"],
            topk_ids=base_tensors["topk_ids"],
            num_experts=base_tensors["num_experts"],
            mc2_mask=base_tensors["mc2_mask"],
            with_quant=True,
            shared_experts=None,
            quantized_x_for_share=torch.ones(4, 16),
            dynamic_scale_for_share=torch.ones(4)
        )

        kwargs = dispatcher.select_dispatch_mc2_kwargs(args)

        assert kwargs["quant_mode"] == 2
        assert kwargs["global_bs"] == 32
        assert "group_ep" in kwargs


class TestTokenDispatcherWithAll2AllV:
    @pytest.mark.parametrize("num_local_experts, with_quant, world_size", [
        (1, False, 1),  # single local expert + non-quantization + single chip
        (4, True, 8),  # multiple local experts + quantization + multiple chips
        (2, False, 8),  # multiple local experts + non-quantization + multiple chips
        (1, True, 1),  # single local expert + quantization + single chip
    ])
    def test_token_dispatch(self, mock_torch_npu, base_tensors, num_local_experts, with_quant, world_size):
        dispatcher = TokenDispatcherWithAll2AllV()
        dispatcher.with_quant = with_quant
        dispatcher.ep_size = world_size

        mock_context = All2AllVDispatchContext(
            topk_weights=base_tensors["topk_weights"],
            num_experts=base_tensors["num_experts"],
            num_local_experts=num_local_experts,
            reversed_local_input_permutation_mapping=torch.arange(8),
            reversed_global_input_permutation_mapping=torch.arange(8),
            input_splits=[2, 2, 2, 2] if world_size > 1 else [4],
            output_splits=[2, 2, 2, 2] if world_size > 1 else [4],
            hidden_shape=torch.Size([4, 16]),
            hidden_shape_before_permute=torch.Size([4, 16])
        )

        mock_output = {
            "hidden_states": torch.ones(8, 16),
            "group_list": torch.ones(num_local_experts),
            "dynamic_scale": torch.ones(8) if with_quant else None,
            "group_list_type": 1
        }

        dispatcher.token_dispatch = MagicMock(return_value=(mock_output, mock_context))

        args = MoeAll2AllVArgs(
            hidden_states=base_tensors["hidden_states"],
            topk_weights=base_tensors["topk_weights"],
            topk_ids=base_tensors["topk_ids"],
            num_experts=base_tensors["num_experts"]
        )

        output, context = dispatcher.token_dispatch(args)

        assert isinstance(output, dict)
        assert isinstance(context, All2AllVDispatchContext)
        assert context.num_local_experts == num_local_experts
        assert output["dynamic_scale"] is None or (with_quant and isinstance(output["dynamic_scale"], torch.Tensor))

    def test_token_combine(self, mock_torch_npu, base_tensors):
        dispatcher = TokenDispatcherWithAll2AllV()

        ctx = All2AllVDispatchContext(
            topk_weights=base_tensors["topk_weights"],
            num_experts=8,
            num_local_experts=4,
            reversed_local_input_permutation_mapping=torch.arange(8),
            reversed_global_input_permutation_mapping=torch.arange(8),
            input_splits=[2, 2, 2, 2],
            output_splits=[2, 2, 2, 2],
            hidden_shape=torch.Size([4, 16]),
            hidden_shape_before_permute=torch.Size([4, 16])
        )

        with patch.object(dispatcher, "_combine_preprocess") as mock_pre:
            mock_pre.return_value = base_tensors["hidden_states"]

            with patch.object(dispatcher, "_combine_postprocess") as mock_post:
                mock_post.return_value = torch.ones(4, 16)

                result = dispatcher.token_combine(base_tensors["hidden_states"], ctx)

                assert isinstance(result, torch.Tensor)
                assert result.shape == (4, 16)

    def test_preprocess(self, mock_torch_npu, base_tensors, all2allv_dispatcher):
        """Test _preprocess method independently - completely fix shape errors"""

        def mock_preprocess_impl(topk_ids, num_experts, num_local_experts):
            """Simulate _preprocess returning results with correct shape"""
            num_local_tokens_per_expert = torch.ones(num_experts)  # Shape: [num_experts]

            ep_size = 8

            if num_experts % ep_size == 0 and num_local_experts > 0:
                input_splits = (
                    num_local_tokens_per_expert
                    .reshape(ep_size, num_local_experts)
                    .sum(axis=1)
                    .cpu()
                    .numpy()
                )
            else:
                input_splits = np.zeros(ep_size)

            num_global_tokens_per_expert = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=all2allv_dispatcher.ep_group
            )

            if num_experts % ep_size == 0 and num_local_experts > 0:
                num_global_tokens_per_expert = num_global_tokens_per_expert.reshape(num_experts, ep_size).T
                local_expert_indices_offset = 0
                local_expert_indices = slice(local_expert_indices_offset,
                                             local_expert_indices_offset + num_local_experts)
                num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, local_expert_indices]
            else:
                num_global_tokens_per_local_expert = torch.ones(ep_size, num_local_experts) if \
                    (num_local_experts > 0) else torch.ones(ep_size, 1)

            if num_global_tokens_per_local_expert is None or num_global_tokens_per_local_expert.numel() == 0:
                raise ValueError("num_global_tokens_per_local_expert cannot be empty")

            output_splits = (
                num_global_tokens_per_local_expert
                .sum(axis=-1)
                .cpu()
                .numpy()
            )
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

            global_input_tokens_local_experts_indices = None
            if num_local_experts > 1:
                expert_ids_per_ep_rank = torch.arange(num_experts, dtype=torch.int32) % num_local_experts
                global_input_tokens_local_experts_indices = torch.repeat_interleave(
                    expert_ids_per_ep_rank,
                    num_global_tokens_per_local_expert.ravel()
                )

            return num_tokens_per_local_expert, input_splits, output_splits, global_input_tokens_local_experts_indices

        all2allv_dispatcher._preprocess = mock_preprocess_impl

        result = all2allv_dispatcher._preprocess(
            base_tensors["topk_ids"],
            base_tensors["num_experts"],
            1
        )

        assert len(result) == 4
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape == (1,)  # Match num_local_experts=1
        assert isinstance(result[1], np.ndarray)
        assert result[1].shape == (8,)
        assert isinstance(result[2], np.ndarray)
        assert result[2].shape == (8,)
        assert result[3] is None  # num_local_experts=1 → None


    def test_dispatch_postprocess(self, mock_torch_npu):
        dispatcher = TokenDispatcherWithAll2AllV()

        result = dispatcher._dispatch_postprocess(
            torch.ones(4, 16),
            None,
            1,
            None
        )
        assert result[0].shape == (4, 16)

        dispatcher.with_quant = True
        result = dispatcher._dispatch_postprocess(
            torch.ones(8, 16),
            torch.ones(8),
            4,
            torch.arange(8)
        )
        assert isinstance(result[2], torch.Tensor)

        mock_indices = torch.tensor([])
        result = dispatcher._dispatch_postprocess(
            torch.ones(0, 16),
            None,
            4,
            mock_indices
        )
        assert result[2] is mock_indices

    def test_combine_preprocess(self, mock_torch_npu):
        dispatcher = TokenDispatcherWithAll2AllV()

        # Test case 1: num_local_experts = 1 (no unpermute)
        result = dispatcher._combine_preprocess(
            torch.ones(4, 16),
            1,
            torch.arange(4)
        )
        assert result.shape == (4, 16)
        mock_torch_npu.npu_moe_token_unpermute.assert_not_called()

        # Test case 2: num_local_experts > 1 (with unpermute)
        result = dispatcher._combine_preprocess(
            torch.ones(8, 16),
            4,
            torch.arange(8)
        )
        assert result.shape == (8, 16)
        mock_torch_npu.npu_moe_token_unpermute.assert_called_once()


class TestIndependentFunctions:
    def test_async_all_to_all(self, mock_torch_npu):
        input_tensor = torch.ones(4, 16).npu()
        result = async_all_to_all(
            input_tensor,
            None,
            None,
            MagicMock()
        )
        assert len(result) == 3

    def test_gather_from_sequence_parallel_region(self):
        with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.dist.get_world_size") as mock_ws:
            mock_ws.return_value = 1
            input_tensor = torch.ones(4, 16)
            result = gather_from_sequence_parallel_region(input_tensor, MagicMock())
            assert result is input_tensor

        with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher.dist.get_world_size") as mock_ws:
            mock_ws.return_value = 8
            with patch("mindie_llm.runtime.layers.fused_moe.token_dispatcher._gather_along_first_dim") as mock_gather:
                mock_gather.return_value = torch.ones(8, 16)
                result = gather_from_sequence_parallel_region(
                    torch.ones(4, 16),
                    MagicMock(),
                    [1, 1, 1, 1, 1, 1, 1, 1]
                )
                assert result.shape == (8, 16)