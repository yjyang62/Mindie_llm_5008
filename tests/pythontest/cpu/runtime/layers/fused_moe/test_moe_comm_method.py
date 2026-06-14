# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Unit tests for moe_comm_method module.

This module contains test cases for MoE communication method selection
and dispatcher caching functionality.
"""

import unittest
from unittest.mock import Mock, patch
from enum import Enum


class MockMoECommType(Enum):
    """Mock enumeration for MoE communication types."""

    ALLGATHER = "allgather"
    MC2 = "mc2"
    ALLTOALL = "alltoall"
    FUSED_MC2 = "fused_mc2"


class TestMoECommMethod(unittest.TestCase):
    """Test cases for MoE communication method selection and dispatcher."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock strategy classes
        self.mock_strategy_allgather = Mock()
        self.mock_strategy_allgather.is_applicable = Mock(return_value=True)
        self.mock_strategy_allgather.get_comm_type = Mock(
            return_value=MockMoECommType.ALLGATHER
        )

        self.mock_strategy_mc2 = Mock()
        self.mock_strategy_mc2.is_applicable = Mock(return_value=False)
        self.mock_strategy_mc2.get_comm_type = Mock(return_value=MockMoECommType.MC2)

        self.mock_strategy_alltoall = Mock()
        self.mock_strategy_alltoall.is_applicable = Mock(return_value=False)
        self.mock_strategy_alltoall.get_comm_type = Mock(
            return_value=MockMoECommType.ALLTOALL
        )

        # Mock dispatcher classes
        self.mock_dispatcher_allgather = Mock()
        self.mock_dispatcher_mc2 = Mock()
        self.mock_dispatcher_alltoall = Mock()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        pass

    @patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES")
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_select_moe_comm_method_returns_first_applicable(self, mock_strategies):
        """Test selection returns first applicable strategy."""
        # FIXED: Use side_effect to return fresh iterator each time
        mock_strategies.__iter__ = Mock(
            side_effect=lambda: iter(
                [
                    self.mock_strategy_allgather,
                    self.mock_strategy_mc2,
                    self.mock_strategy_alltoall,
                ]
            )
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method,
        )

        result = select_moe_comm_method(num_experts_per_ep_rank=2)

        self.assertEqual(result, MockMoECommType.ALLGATHER)
        self.mock_strategy_allgather.is_applicable.assert_called_with(
            num_experts_per_ep_rank=2
        )

    @patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES")
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_select_moe_comm_method_no_applicable_raises_error(self, mock_strategies):
        """Test selection raises RuntimeError when no strategy matches."""
        self.mock_strategy_allgather.is_applicable = Mock(return_value=False)
        self.mock_strategy_mc2.is_applicable = Mock(return_value=False)
        self.mock_strategy_alltoall.is_applicable = Mock(return_value=False)

        mock_strategies.__iter__ = Mock(
            side_effect=lambda: iter(
                [
                    self.mock_strategy_allgather,
                    self.mock_strategy_mc2,
                    self.mock_strategy_alltoall,
                ]
            )
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method,
        )

        with self.assertRaises(RuntimeError) as ctx:
            select_moe_comm_method(num_experts_per_ep_rank=2)

        self.assertIn("MoE strategy selection failed", str(ctx.exception))

    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER"
    )
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_get_cached_dispatcher_returns_instance(self, mock_dispatcher_map):
        """Test dispatcher retrieval returns correct instance."""
        mock_dispatcher_map.__getitem__ = Mock(
            return_value=self.mock_dispatcher_allgather
        )
        mock_dispatcher_map.__contains__ = Mock(return_value=True)

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher,
        )

        result = get_cached_dispatcher(MockMoECommType.ALLGATHER)

        self.assertIsNotNone(result)
        self.assertTrue(mock_dispatcher_map.__getitem__.called)

    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER"
    )
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_get_cached_dispatcher_none_type_returns_none(self, mock_dispatcher_map):
        """Test dispatcher retrieval with None type returns None."""
        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher,
        )

        result = get_cached_dispatcher(None)

        self.assertIsNone(result)
        self.assertFalse(
            mock_dispatcher_map.__getitem__.called,
            "Should not lookup dispatcher for None input",
        )

    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method._COMM_TYPE_TO_DISPATCHER"
    )
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_get_cached_dispatcher_unsupported_type_returns_none(
        self, mock_dispatcher_map
    ):
        """Test dispatcher retrieval with unsupported type returns None."""
        mock_dispatcher_map.__contains__ = Mock(return_value=False)

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            get_cached_dispatcher,
        )

        result = get_cached_dispatcher(MockMoECommType.MC2)

        self.assertIsNone(result)

    @patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES")
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_select_moe_comm_method_with_different_expert_count(self, mock_strategies):
        """Test selection with different num_experts_per_ep_rank values."""
        # FIXED: side_effect ensures fresh iterator for each call in subTest loop
        mock_strategies.__iter__ = Mock(
            side_effect=lambda: iter([self.mock_strategy_allgather])
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method,
        )

        # Test with different expert counts
        for expert_count in [1, 2, 8, 16]:
            with self.subTest(expert_count=expert_count):
                result = select_moe_comm_method(num_experts_per_ep_rank=expert_count)
                self.assertEqual(result, MockMoECommType.ALLGATHER)
                self.mock_strategy_allgather.is_applicable.assert_called_with(
                    num_experts_per_ep_rank=expert_count
                )
                # Reset for next subTest iteration
                self.mock_strategy_allgather.is_applicable.reset_mock()

    @patch("mindie_llm.runtime.layers.fused_moe.moe_comm_method.MOE_COMM_STRATEGIES")
    @patch(
        "mindie_llm.runtime.layers.fused_moe.moe_comm_method.MoECommType",
        MockMoECommType,
    )
    def test_select_moe_comm_method_strategy_order(self, mock_strategies):
        """Test that strategies are checked in order and first applicable wins."""
        self.mock_strategy_allgather.is_applicable = Mock(return_value=False)
        self.mock_strategy_mc2.is_applicable = Mock(return_value=True)

        mock_strategies.__iter__ = Mock(
            side_effect=lambda: iter(
                [
                    self.mock_strategy_allgather,
                    self.mock_strategy_mc2,
                    self.mock_strategy_alltoall,
                ]
            )
        )

        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            select_moe_comm_method,
        )

        result = select_moe_comm_method(num_experts_per_ep_rank=2)

        self.assertEqual(result, MockMoECommType.MC2)
        self.mock_strategy_allgather.is_applicable.assert_called_once_with(
            num_experts_per_ep_rank=2
        )
        self.mock_strategy_mc2.is_applicable.assert_called_once_with(
            num_experts_per_ep_rank=2
        )
        self.mock_strategy_alltoall.is_applicable.assert_not_called()

    def test_comm_type_to_dispatcher_mapping_exists(self):
        """Test communication type to dispatcher mapping is defined."""
        from mindie_llm.runtime.layers.fused_moe.moe_comm_method import (
            _COMM_TYPE_TO_DISPATCHER,
        )

        self.assertIsInstance(_COMM_TYPE_TO_DISPATCHER, dict)
        self.assertGreater(len(_COMM_TYPE_TO_DISPATCHER), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
