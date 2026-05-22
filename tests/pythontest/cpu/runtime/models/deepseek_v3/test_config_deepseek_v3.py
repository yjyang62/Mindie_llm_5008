import unittest
from mindie_llm.runtime.models.deepseek_v3.config_deepseek_v3 import DeepseekV3Config


class TestDeepseekV3Config(unittest.TestCase):

    def setUp(self):
        self.config = DeepseekV3Config(
            rope_scaling={"type": "dynamic"},
            model_type="deepseek_v3",
            n_routed_experts=256,
            num_experts_per_tok=4,
            first_k_dense_replace=5,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            topk_method="greedy",
            topk_group=1,
            n_group=1
        )

    def test_default_values(self):
        self.assertEqual(self.config.model_type, "deepseek_v3")
        self.assertEqual(self.config.vocab_size, 102400)
        self.assertEqual(self.config.hidden_size, 5120)
        self.assertEqual(self.config.num_hidden_layers, 60)
        self.assertEqual(self.config.bos_token_id, None)
        self.assertEqual(self.config.eos_token_id, None)

    def test_custom_values(self):
        self.assertEqual(self.config.n_routed_experts, 256)
        self.assertEqual(self.config.num_experts_per_tok, 4)
        self.assertEqual(self.config.first_k_dense_replace, 5)
        self.assertEqual(self.config.qk_nope_head_dim, 64)
        self.assertEqual(self.config.qk_rope_head_dim, 32)

    def test_validate_passes(self):
        # 没有抛出异常即为通过
        try:
            self.config.validate()
        except Exception as e:
            self.fail(f"validate() raised an exception: {e}")

    def test_validate_num_experts_per_tok_too_high(self):
        config = DeepseekV3Config(
            rope_scaling={"type": "dynamic"},
            model_type="deepseek_v3",
            n_routed_experts=256,
            num_experts_per_tok=257,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64
        )
        with self.assertRaises(AttributeError):
            config.validate()

    def test_validate_first_k_dense_replace_too_high(self):
        config = DeepseekV3Config(
            rope_scaling={"type": "dynamic"},
            model_type="deepseek_v3",
            num_hidden_layers=5,
            first_k_dense_replace=6,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_experts_per_tok=8,
            n_routed_experts=256
        )
        with self.assertRaises(AttributeError):
            config.validate()

    def test_validate_topk_method_invalid(self):
        config = DeepseekV3Config(
            rope_scaling={"type": "dynamic"},
            model_type="deepseek_v3",
            topk_method="invalid_method",
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_experts_per_tok=8,
            n_routed_experts=256
        )
        with self.assertRaises(AttributeError):
            config.validate()

    def test_validate_topk_method_greedy_with_wrong_group_settings(self):
        config = DeepseekV3Config(
            rope_scaling={"type": "dynamic"},
            model_type="deepseek_v3",
            topk_method="greedy",
            topk_group=3,  # should be 1
            n_group=2,  # should be 1
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_experts_per_tok=8,
            n_routed_experts=256
        )
        with self.assertRaises(AttributeError):
            config.validate()

    def test_validate_q_head_dim_before_and_index_n_heads(self):
        self.assertEqual(self.config.index_n_heads, 64)
        self.assertEqual(self.config.index_head_dim, 128)
        self.assertEqual(self.config.index_topk, 2048)


if __name__ == '__main__':
    unittest.main()