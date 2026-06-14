import unittest
from unittest.mock import patch, MagicMock
import torch
from mindie_llm.runtime.models.deepseek_v3.deepseek_v3_mtp import DeepseekV3MtpLayer, DeepseekV3MtpModel, DeepseekV3MTP, SharedHead
from mindie_llm.runtime.models.deepseek_v3.config_deepseek_v3 import DeepseekV3Config
from mindie_llm.runtime.utils.distributed import set_parallel_info_manager
from mindie_llm.runtime.layers.quantization.ms_model_slim.quantization_config import QuantizationConfig
from mindie_llm.runtime.layers.quantization.ms_model_slim.quant_type import QuantType, InferenceMode
from mindie_llm.runtime.layers.normalization import RMSNorm
from mindie_llm.runtime.layers.linear.linear import ReplicatedLinear
from mindie_llm.runtime.layers.embedding.embedding import VocabParallelEmbedding
from mindie_llm.runtime.model_runner.forward_context import set_forward_context


class TestDeepseekV3Mtp(unittest.TestCase):
    def setUp(self):
        config_dict = {
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "kv_lora_rank": 512,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_hidden_layers": 28,
            "topk_method": "greedy",
            "topk_group": 1,
            "n_group": 1,
            "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
            "parallel_embedding": True
            }
        }
        self.config = DeepseekV3Config.from_dict(config_dict)
        self.config.routed_scaling_factor = 2
        self.config.hidden_size = 7168
        self.config.vocab_size = 50000
        self.config.rms_norm_eps = 1e-8
        self.config.max_position_embeddings = 2048
        self.config.torch_dtype = torch.bfloat16
        self.config.num_attention_heads = 128
        self.config.qk_nope_head_dim = 128
        self.config.qk_rope_head_dim = 64
        self.config.index_n_heads = 64
        self.config.index_head_dim = 128
        self.config.q_lora_rank = 1536
        self.config.intermediate_size = 18432
        self.config.moe_intermediate_size = 2048

        # Mock the parallel info and parallel info manager
        self.mock_parallel_info = MagicMock()
        self.mock_parallel_info.rank = 0
        self.mock_parallel_info.group_size = 1 

        self.mock_parallel_info_manager = MagicMock()
        self.mock_parallel_info_manager.rank = 0
        self.mock_parallel_info_manager.world_size = 1
        self.mock_parallel_info_manager.attn_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.mlp_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.lm_head_tp = self.mock_parallel_info
        self.mock_parallel_info_manager.moe_ep = self.mock_parallel_info
        self.mock_parallel_info_manager.moe_tp = self.mock_parallel_info

        # Set the global parallel info manager
        set_parallel_info_manager(self.mock_parallel_info_manager)

        self.quant_config = None
        self.prefix = "model"
        self.layer_idx = 61

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3MtpLayer_init(self, mock_dist_get_rank, mock_check):
        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0
        layer = DeepseekV3MtpLayer(
            config=self.config,
            prefix=self.prefix,
            layer_idx=self.layer_idx,
            quant_config=self.quant_config
        )
        self.assertIsInstance(layer.embed_tokens, type(VocabParallelEmbedding(self.config.hidden_size, self.config.vocab_size)))
        self.assertIsInstance(layer.enorm, type(RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)))
        self.assertIsInstance(layer.hnorm, type(RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)))
        self.assertIsInstance(layer.shared_head, type(SharedHead(self.config, self.prefix, self.quant_config)))
        self.assertIsInstance(layer.eh_proj, type(ReplicatedLinear(2 * self.config.hidden_size, self.config.hidden_size)))

    @patch('torch_npu.npu_rms_norm')
    def test_SharedHead_forward(self, mock_npu_rms_norm):
        shared_head = SharedHead(self.config, self.prefix, self.quant_config, )

        # Create input
        hidden_states = torch.randn(2, 3, 512)

        # Mock npu_rms_norm to return expected output
        mock_npu_rms_norm.return_value = (torch.randn(2, 3, 512), None)

        output = shared_head(hidden_states)

        # Verify npu_rms_norm was called
        mock_npu_rms_norm.assert_called_once()
        self.assertEqual(output.shape, (2, 3, 512))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    def test_DeepseekV3MtpModel_forward(self, mock_get_forward_context, mock_check):
        with patch('torch.distributed.get_rank') as mock_dist_get_rank, \
             patch('torch_npu.npu_rms_norm') as mock_npu_rms_norm, \
             patch('torch.nn.functional.embedding') as mock_embedding, \
             patch('torch_npu.npu_add_rms_norm') as mock_npu_add_rms_norm:
            
            # Mock dist get rank to return local rank 0
            mock_dist_get_rank = MagicMock()
            mock_dist_get_rank.return_value = 0

            model = DeepseekV3MtpModel(
                config=self.config,
                prefix="model",
                quant_config=self.quant_config
            )

            # Create inputs
            num_tokens = 10
            input_ids = torch.randint(0, self.config.vocab_size, (num_tokens,))
            positions = torch.randint(0, 2048, (num_tokens,))

            # Mock forward_context
            forward_context = MagicMock()
            forward_context.mtp_metadata = MagicMock()
            forward_context.mtp_metadata.last_hidden_states = torch.randn(num_tokens, self.config.hidden_size)
            forward_context.enable_flash_comm = False
            forward_context.seq_lens = torch.tensor([num_tokens])
            forward_context.lm_head_indices = torch.tensor([0])
            mock_get_forward_context.return_value = forward_context
            set_forward_context(forward_context)

            # Create layers contents
            model.layers[str(61)].embed_tokens = VocabParallelEmbedding(self.config.hidden_size, self.config.vocab_size)
            mock_embedding.return_value = torch.randn(num_tokens, self.config.hidden_size)
            model.layers[str(61)].enorm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
            model.layers[str(61)].hnorm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
            model.layers[str(61)].shared_head = SharedHead(self.config, self.prefix, self.quant_config)
            layer = DeepseekV3MtpLayer(
                config=self.config,
                prefix=self.prefix,
                layer_idx=self.layer_idx,
                quant_config=self.quant_config
            )
            layer.forward = MagicMock()

            # Mock the layer forward
            layer.forward.return_value = (torch.randn(num_tokens, self.config.hidden_size), torch.randn(num_tokens, self.config.hidden_size))
            model.layers[str(61)] = layer

            # Mock add rms norm and rms norm
            mock_npu_rms_norm.return_value = (torch.randn(num_tokens, self.config.hidden_size), None)
            mock_npu_add_rms_norm.return_value = (
                torch.randn(num_tokens, self.config.hidden_size),
                None,
                torch.randn(num_tokens, self.config.hidden_size)
            )

            output = model(input_ids, positions)
            self.assertEqual(output.shape, (num_tokens, self.config.hidden_size))

    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('torch.distributed.get_rank')
    def test_DeepseekV3MTP_forward(self, mock_dist_get_rank, mock_check):
        # Create inputs
        num_tokens = 10
        input_ids = torch.randint(0, self.config.vocab_size, (num_tokens,))
        positions = torch.randint(0, 2048, (num_tokens,))

        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        mindie_llm_config = MagicMock()
        mindie_llm_config.hf_config = self.config
        mindie_llm_config.quant_config = self.quant_config

        mtp = DeepseekV3MTP(mindie_llm_config)
        model = DeepseekV3MtpModel(
            config=self.config,
            prefix="model",
            quant_config=self.quant_config
        )
        model.forward = MagicMock()

        # Mock model forward
        model.forward.return_value = torch.randn(num_tokens, self.config.hidden_size)

        mtp.model = model

        output = mtp(input_ids, positions)
        self.assertEqual(output.shape, (num_tokens, self.config.hidden_size))
    
    @patch('mindie_llm.runtime.layers.attention.backend.sparse_attention.SfaBackendImpl.check_parallel_info')
    @patch('torch.nn.functional.linear')
    @patch('mindie_llm.runtime.model_runner.forward_context.get_forward_context')
    @patch('torch.distributed.get_rank')
    def test_LmHead_forward(self, mock_dist_get_rank, 
        mock_get_forward_context, mock_linear, mock_check):
        # Create inputs
        num_tokens = 10
        hidden_states = torch.randn(num_tokens, self.config.hidden_size)

        # Mock dist get rank to return local rank 0
        mock_dist_get_rank = MagicMock()
        mock_dist_get_rank.return_value = 0

        mindie_llm_config = MagicMock()
        mindie_llm_config.hf_config = self.config
        mindie_llm_config.quant_config = self.quant_config

        mtp = DeepseekV3MTP(mindie_llm_config)

        # Mock forward_context
        forward_context = MagicMock()
        forward_context.lm_head_indices = torch.tensor([0,1])
        mock_get_forward_context.return_value = forward_context
        set_forward_context(forward_context)

        # Mock linear
        mock_linear.return_value = torch.randn(2, self.config.vocab_size)

        output = mtp.compute_logits(hidden_states)
        self.assertEqual(output.shape, (2, self.config.vocab_size))


if __name__ == '__main__':
    unittest.main()