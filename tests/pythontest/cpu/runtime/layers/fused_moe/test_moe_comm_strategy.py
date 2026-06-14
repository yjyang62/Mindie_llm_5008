"""
Unit tests for MoE communication strategy selection logic.

Mocks all NPU/distributed dependencies to ensure fast, isolated testing.
Uses unittest assertions (assertTrue/assertFalse/assertEqual) per coding standards.
"""

import unittest
from unittest.mock import patch, MagicMock

from mindie_llm.runtime.layers.fused_moe.moe_comm_strategy import (
    MoECommType,
    MoECommStrategyBase,
    FusedMC2Strategy,
    MC2Strategy,
    All2AllStrategy,
    AllGatherStrategy,
    MOE_COMM_STRATEGIES,
)


class TestMoECommStrategies(unittest.TestCase):
    """Test suite for MoE communication strategy selection."""

    def setUp(self) -> None:
        """Set up common mocks for NPU/distributed dependencies."""
        # Mock objects
        self.mock_device_info = MagicMock()
        self.mock_parallel_mgr = MagicMock()
        self.mock_forward_ctx = MagicMock()
        self.mock_batch_desc = MagicMock()
        self.mock_forward_ctx.batch_descriptor = self.mock_batch_desc
        self.mock_dp_metadata = MagicMock()
        self.mock_forward_ctx.dp_metadata = self.mock_dp_metadata

        # Patchers list for cleanup
        self.patchers = []

        # Helper to register and start a patcher
        def _patch(target, **kwargs):
            p = patch(target, **kwargs)
            self.patchers.append(p)
            return p.start()

        # Start all patchers and collect mocks
        _patch(
            "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_npu_node_info",
            return_value=self.mock_device_info,
        )
        _patch(
            "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_parallel_info_manager",
            return_value=self.mock_parallel_mgr,
        )
        _patch(
            "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_forward_context",
            return_value=self.mock_forward_ctx,
        )
        _patch(
            "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.get_mc2_token_capacity",
            return_value=8192,
        )
        _patch("mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.logger")

        # Special: cal_num_tokens_per_device - store mock for easy override
        self.mock_cal_num_tokens = _patch(
            "mindie_llm.runtime.layers.fused_moe.moe_comm_strategy.cal_num_tokens_per_device",
            return_value=1024,  # Default value
        )

    def tearDown(self) -> None:
        """Stop all patchers after each test."""
        for p in self.patchers:
            p.stop()

    def _setup_parallel_info(self, **kwargs) -> None:
        """Helper: configure mock parallel manager."""
        ep_enabled = kwargs.get("ep_enabled", True)
        ep_mc2_group_size = kwargs.get("ep_mc2_group_size", 16)
        moe_tp_enabled = kwargs.get("moe_tp_enabled", False)
        attn_tp_group_size = kwargs.get("attn_tp_group_size", 1)
        attn_dp_enabled = kwargs.get("attn_dp_enabled", False)
        world_size = kwargs.get("world_size", 8)
        attn_cp_enabled = kwargs.get("attn_cp_enabled", False)

        def get_side_effect(pt):
            mock_pt = MagicMock()
            if pt.name == "MOE_EP":
                mock_pt.is_enabled.return_value = ep_enabled
            elif pt.name == "MOE_EP_MC2":
                mock_pt.group_size = ep_mc2_group_size
            elif pt.name == "MOE_TP":
                mock_pt.is_enabled.return_value = moe_tp_enabled
            elif pt.name == "ATTN_TP":
                mock_pt.group_size = attn_tp_group_size
                mock_pt.is_enabled.return_value = attn_tp_group_size > 1
            elif pt.name == "ATTN_DP":
                mock_pt.is_enabled.return_value = attn_dp_enabled
            elif pt.name == "ATTN_CP":
                mock_pt.is_enabled.return_value = attn_cp_enabled
            return mock_pt

        self.mock_parallel_mgr.get.side_effect = get_side_effect
        self.mock_parallel_mgr.world_size = world_size

    def _setup_device(self, device_type: str) -> None:
        """Helper: set mock device type."""
        from mindie_llm.runtime.utils.npu.device_utils import DeviceType

        self.mock_device_info.get_device_type.return_value = getattr(
            DeviceType, device_type
        )

    def _setup_forward_context(self, is_prefill: bool, num_tokens: int) -> None:
        """Helper: configure mock forward context."""
        self.mock_forward_ctx.is_prefill = is_prefill
        self.mock_batch_desc.num_tokens = num_tokens
        self.mock_batch_desc.is_flash_comm_enabled = False
        self.mock_dp_metadata.max_tokens_across_dp_cpu = num_tokens

    def _set_num_tokens_per_device(self, value: int) -> None:
        """Helper: override cal_num_tokens_per_device return value."""
        self.mock_cal_num_tokens.return_value = value

    # ==================== Base Class Tests ====================

    def test_base_class_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            MoECommStrategyBase.is_applicable(2)
        with self.assertRaises(NotImplementedError):
            MoECommStrategyBase.get_comm_type()

    # ==================== FusedMC2Strategy Tests ====================

    def test_fused_mc2_happy_path(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(
            ep_enabled=True, ep_mc2_group_size=16, moe_tp_enabled=False
        )
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = FusedMC2Strategy.is_applicable(2)
        self.assertTrue(result)
        self.assertEqual(FusedMC2Strategy.get_comm_type(), MoECommType.FUSED_MC2)

    def test_fused_mc2_reject_wrong_device(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = FusedMC2Strategy.is_applicable(2)
        self.assertFalse(result)

    def test_fused_mc2_reject_large_ep_group(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_mc2_group_size=64)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = FusedMC2Strategy.is_applicable(2)
        self.assertFalse(result)

    def test_fused_mc2_reject_token_overflow(self) -> None:
        """FusedMC2: raise RuntimeError when tokens >= 4096."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(attn_tp_group_size=1)
        self._setup_forward_context(is_prefill=False, num_tokens=4096)
        # FIXED: Must be >= 4096 to trigger RuntimeError
        self._set_num_tokens_per_device(4096)

        with self.assertRaises(RuntimeError) as ctx:
            FusedMC2Strategy.is_applicable(2)
        self.assertIn("num_tokens_per_device", str(ctx.exception))
        self.assertIn("MAX_FUSED_MC2_OPERATOR_CAPACITY", str(ctx.exception))

    def test_fused_mc2_reject_moe_tp(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(moe_tp_enabled=True)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = FusedMC2Strategy.is_applicable(2)
        self.assertFalse(result)

    # ==================== MC2Strategy Tests ====================

    def test_mc2_910b_decode_large_cluster(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=16)
        self._setup_forward_context(is_prefill=False, num_tokens=512)
        self._set_num_tokens_per_device(512)

        result = MC2Strategy.is_applicable(2)
        self.assertTrue(result)
        self.assertEqual(MC2Strategy.get_comm_type(), MoECommType.MC2)

    def test_mc2_910b_prefill_allowed(self) -> None:
        """MC2 on 910B: prefill phase is ALLOWED (no is_prefill check in code)."""
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=16)
        self._setup_forward_context(is_prefill=True, num_tokens=512)  # prefill=True
        self._set_num_tokens_per_device(512)

        # FIXED: Source code does NOT check is_prefill, so this should return True
        result = MC2Strategy.is_applicable(2)
        self.assertTrue(result, "MC2 on 910B does not reject prefill phase")

    def test_mc2_910b_reject_small_cluster(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(world_size=8)  # < 16
        self._setup_forward_context(is_prefill=False, num_tokens=512)
        self._set_num_tokens_per_device(512)

        result = MC2Strategy.is_applicable(2)
        self.assertFalse(result)

    def test_mc2_910c_decode_valid(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=4096)
        self._set_num_tokens_per_device(4096)  # <= 8192

        result = MC2Strategy.is_applicable(2)
        self.assertTrue(result)

    def test_mc2_910c_reject_token_exceed(self) -> None:
        """MC2 on 910C: return False when tokens > get_mc2_token_capacity()."""
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=10000)
        # FIXED: Ensure mock returns value > 8192
        self._set_num_tokens_per_device(10000)

        result = MC2Strategy.is_applicable(2)
        self.assertFalse(result, "MC2 should reject tokens > capacity on 910C")

    def test_mc2_reject_moe_tp(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(moe_tp_enabled=True)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = MC2Strategy.is_applicable(2)
        self.assertFalse(result)

    def test_mc2_unsupported_device(self) -> None:
        """MC2: return False on unsupported device type."""
        # Use a device type that doesn't match any branch
        self.mock_device_info.get_device_type.return_value = MagicMock()
        self._setup_parallel_info()
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        result = MC2Strategy.is_applicable(2)
        self.assertFalse(result)

    # ==================== All2AllStrategy Tests ====================

    def test_all2all_910c_prefill_valid(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_enabled=True, moe_tp_enabled=False)
        self._setup_forward_context(is_prefill=True, num_tokens=1024)

        result = All2AllStrategy.is_applicable(2)
        self.assertTrue(result)
        self.assertEqual(All2AllStrategy.get_comm_type(), MoECommType.ALLTOALL)

    def test_all2all_910c_reject_decode(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_enabled=True, moe_tp_enabled=False)
        self._setup_forward_context(is_prefill=False, num_tokens=1024)

        result = All2AllStrategy.is_applicable(2)
        self.assertFalse(result)

    def test_all2all_reject_wrong_device(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(ep_enabled=True, moe_tp_enabled=False)
        self._setup_forward_context(is_prefill=True, num_tokens=1024)

        result = All2AllStrategy.is_applicable(2)
        self.assertFalse(result)

    def test_all2all_reject_moe_tp(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_enabled=True, moe_tp_enabled=True)
        self._setup_forward_context(is_prefill=True, num_tokens=1024)

        result = All2AllStrategy.is_applicable(2)
        self.assertFalse(result)

    def test_all2all_reject_ep_disabled(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(ep_enabled=False, moe_tp_enabled=False)
        self._setup_forward_context(is_prefill=True, num_tokens=1024)

        result = All2AllStrategy.is_applicable(2)
        self.assertFalse(result)

    # ==================== AllGatherStrategy Tests ====================

    def test_allgather_910b_fallback(self) -> None:
        self._setup_device("ASCEND_910B")  # FIXED: Must be 910B
        self._setup_parallel_info(moe_tp_enabled=False, attn_dp_enabled=False)

        result = AllGatherStrategy.is_applicable(2)
        self.assertTrue(result)
        self.assertEqual(AllGatherStrategy.get_comm_type(), MoECommType.ALLGATHER)

    def test_allgather_reject_910c(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(moe_tp_enabled=False, attn_dp_enabled=False)

        result = AllGatherStrategy.is_applicable(2)
        self.assertFalse(result)

    def test_allgather_reject_moe_tp(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(moe_tp_enabled=True, attn_dp_enabled=False)

        result = AllGatherStrategy.is_applicable(2)
        self.assertFalse(result)

    def test_allgather_reject_attn_dp(self) -> None:
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(moe_tp_enabled=False, attn_dp_enabled=True)

        result = AllGatherStrategy.is_applicable(2)
        self.assertFalse(result)

    # ==================== Strategy Selection Order Tests ====================

    def test_strategy_priority_order(self) -> None:
        expected_order = [
            FusedMC2Strategy,
            MC2Strategy,
            All2AllStrategy,
            AllGatherStrategy,
        ]
        self.assertEqual(MOE_COMM_STRATEGIES, expected_order)

    def test_first_applicable_strategy_selection(self) -> None:
        self._setup_device("ASCEND_910_93")
        self._setup_parallel_info(
            ep_enabled=True, ep_mc2_group_size=16, moe_tp_enabled=False
        )
        self._setup_forward_context(is_prefill=False, num_tokens=1024)
        self._set_num_tokens_per_device(1024)

        selected = None
        for strategy in MOE_COMM_STRATEGIES:
            if strategy.is_applicable(2):
                selected = strategy.get_comm_type()
                break
        self.assertEqual(selected, MoECommType.FUSED_MC2)

    def test_fallback_to_allgather(self) -> None:
        """When no strategy matches, AllGather should be selected as fallback."""
        # FIXED: Must set device to 910B for AllGather to be applicable
        self._setup_device("ASCEND_910B")
        self._setup_parallel_info(
            ep_enabled=False, moe_tp_enabled=False, attn_dp_enabled=False
        )

        selected = None
        for strategy in MOE_COMM_STRATEGIES:
            if strategy.is_applicable(2):
                selected = strategy.get_comm_type()
                break
        self.assertEqual(selected, MoECommType.ALLGATHER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
