# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import typing
import unittest
from unittest.mock import MagicMock, patch
import torch

from examples.run_pa import PARunner, parse_arguments, cmd_bool, parse_ids, ENV


class TestPARunner(unittest.TestCase):
    @patch("examples.run_pa.ModelRunner", return_value=MagicMock())
    def setUp(self, modelrunner):
        self.args = {
            'input_texts': ["What's deep learning?"],
            'max_batch_size': 1,
            'max_input_length': 1024,
            'max_output_length': 20,
            'max_position_embeddings': 64,
            'max_prefill_tokens': -1,
            "block_size": 128
        }
        input_dict = {
            'rank': 0,
            'world_size': 8,
            'local_rank': 0,
            **(self.args)
        }
        self.pa_runner = PARunner(**input_dict)
        self.pa_runner.model.model.total_prefill_token_num_per_expert = torch.Tensor([[1, 1, 1], [1, 1, 1]])
        self.pa_runner.model.model.total_decode_token_num_per_expert = torch.Tensor([[1, 1, 1], [1, 1, 1]])
        
        self.pa_runner.model.mapping = MagicMock()
        self.pa_runner.model.mapping.has_attn_inner_sp.return_value = False

    @patch("examples.run_pa.CacheManager", return_value=MagicMock())
    @patch("examples.run_pa.generate_req")
    def test_warm_up(self, cachemanager, req: typing.List):
        self.pa_runner.warm_up()

    @patch("examples.run_pa.file_utils.safe_open")
    @patch("typing.IO.write")
    @patch("examples.run_pa.CacheManager", return_value=MagicMock())
    @patch("examples.run_pa.generate_req")
    def test_infer(self, mock_cache, mock_req, mock_write, mock_open):
        ENV.enable_expert_hotpot_gather = True
        ENV.expert_hotpot_dump_path = "./"
        infer_params = {
            "inputs": ["what is deeplearn"],
            "batch_size": self.args["max_batch_size"],
            "max_output_length": self.args["max_output_length"],
            "ignore_eos": False,
            "is_chat_model": False
        }
        self.pa_runner.infer(**infer_params)

    @patch("examples.run_pa.ArgumentParser", return_value=MagicMock())
    def test_parse_arguments(self, _):
        parse_arguments()

    def test_cmd_bool(self):
        cmd_bool("True")
        cmd_bool("False")
        try:
            cmd_bool("hahahaha")
        except ValueError as e:
            self.assertEqual(type(e), ValueError)
    
    def test_parse_ids(self):
        parse_ids("1,2,3,4,5")

    def test_build_model_inputs(self):
        try:
            self.pa_runner._build_model_inputs("test", False)
        except ValueError as e:
            self.assertEqual(type(e), ValueError)

    @patch("examples.run_pa.os.chmod")
    @patch("examples.run_pa.torch.save")
    @patch("examples.run_pa.file_utils.safe_open")
    @patch("typing.IO.write")
    def test_save_input_output_ids(self, mock_write, mock_open, mock_save, mock_chmod):
        self.pa_runner.save_input_output_ids([MagicMock()])

    @patch("examples.run_pa.ModelRunner", return_value=MagicMock())
    def test_check_limit(self, modelrunner):
        args = {
            'input_texts': ["What's deep learning?"],
            'max_batch_size': -1,
            'max_input_length': -1,
            'max_output_length': -1,
            'max_position_embeddings': 64,
            'max_prefill_tokens': -2,
            "block_size": -1,
            "max_prefill_batch_size": -1
        }
        input_dict = {
            'rank': 0,
            'world_size': 8,
            'local_rank': 0,
            **(args)
        }
        pa_runner = PARunner(**input_dict)
        pa_runner.check_limits()



if __name__ == '__main__':
    unittest.main()