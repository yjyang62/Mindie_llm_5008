# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper, REASON_CONTENT_KEY, CONTENT_KEY


FAKE_MODEL_NAME_OR_PATH = 'fake/path'
FAKE_INPUT_TOKENS = [1, 2, 3]
DEFAULT_OUTPUT = '123'
PREV_DECODE_INDEX = -2
CURR_DECODE_INDEX = -1
METADATA = "metadata"
REASONING_TOKEN_OBJ = {"reasoning_tokens": 3}


def mock_decode(token_list, **kwargs):
    return ''.join(str(x) for x in token_list)


class MockRouter():
    def __init__(self):
        self.config = MagicMock()
        self.tokenizer = MagicMock()
        self.tokenizer.decode.side_effect = mock_decode
        self.input_builder = MagicMock()
        self.postprocessor = MagicMock()
        self.tokenize = MagicMock()
        self.toolscallprocessor = MagicMock()
        self.reasoning_parser = MagicMock()
        self.llm_config = MagicMock()
        self.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir = None
        self.toolscallprocessor.decode.side_effect = lambda x: {"tool_call": x}
        self.toolscallprocessor.decode_stream.return_value = {"tool_call": "something"}
        self.reasoning_parser.count_reasoning_tokens.side_effect = lambda x: len(x)
        self.reasoning_parser.single_process_reasoning.side_effect = lambda x: (x[1:], x)
        self.reasoning_parser.stream_process_reasoning.return_value = ('789', '654')


class TestTokenizerWrapper(unittest.TestCase):
    def setUp(self):
        with patch('atb_llm.runner.tokenizer_wrapper.get_model') as mock_get_model:
            mock_get_model.return_value = MockRouter()
            self.tokenizer_wrapper = TokenizerWrapper(FAKE_MODEL_NAME_OR_PATH)
    
    @patch("atb_llm.runner.tokenizer_wrapper.check_file_safety", MagicMock())
    def test_init_with_obfuscation(self):
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_instance = MagicMock()

        mock_instance.set_seed_safer.return_value = (0, 'fake_str')
        mock_class.return_value = mock_instance
        mock_module.data_asset_obfuscation.DataAssetObfuscation = mock_class

        sys.modules['ai_asset_obfuscate'] = mock_module
        sys.modules['ai_asset_obfuscate.data_asset_obfuscation'] = mock_module.data_asset_obfuscation
        mock_router = MockRouter()
        mock_router.llm_config.llm.pmcc_obfuscation_options.data_obfuscation_ca_dir = "/tmp"
        mock_router.llm_config.llm.pmcc_obfuscation_options.kms_agent_port = "12345"

        with patch('atb_llm.runner.tokenizer_wrapper.get_model') as mock_get_model_with_obfuscation:
            mock_get_model_with_obfuscation.return_value = mock_router
            _ = TokenizerWrapper(FAKE_MODEL_NAME_OR_PATH)
    
    def test_encode(self):
        inputs = "what is deep learning?"
        self.tokenizer_wrapper.tokenizer = MagicMock(return_value={"input_ids": [torch.tensor((0, 1, 2, 3))]})
        self.tokenizer_wrapper.input_builder.make_context = MagicMock(return_value=[5, 6, 7, 8])
        result = self.tokenizer_wrapper.encode(inputs)
        self.assertEqual(result, [0, 1, 2, 3])
        result = self.tokenizer_wrapper.encode(inputs, is_chatting=True)
        self.assertEqual(result, [5, 6, 7, 8])

    def test_decode_no_stream(self):
        token_ids = FAKE_INPUT_TOKENS
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=False,
                                                   is_chat_req=False, stream=False)
        golden = {CONTENT_KEY: DEFAULT_OUTPUT, METADATA: REASONING_TOKEN_OBJ}
        self.assertDictEqual(result, golden)

    def test_decode_no_stream_use_tool_calls(self):
        token_ids = FAKE_INPUT_TOKENS
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=True,
            is_chat_req=True, stream=False, metadata={"req_enable_thinking": False})
        golden = {"tool_call": DEFAULT_OUTPUT, METADATA: REASONING_TOKEN_OBJ}
        self.assertDictEqual(result, golden)

    def test_decode_no_stream_use_reasoning_parser(self):
        token_ids = FAKE_INPUT_TOKENS
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=False,
            is_chat_req=True, stream=False, metadata={"req_enable_thinking": True})
        golden = {REASON_CONTENT_KEY: '23', CONTENT_KEY: DEFAULT_OUTPUT, METADATA: REASONING_TOKEN_OBJ}
        self.assertDictEqual(result, golden)

    def test_decode_no_stream_use_both(self):
        token_ids = FAKE_INPUT_TOKENS
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=True,
            is_chat_req=True, stream=False, metadata={"req_enable_thinking": True})
        golden = {'tool_call': DEFAULT_OUTPUT, REASON_CONTENT_KEY: '23', METADATA: REASONING_TOKEN_OBJ}
        self.assertDictEqual(result, golden)

    def test_decode_stream(self):
        token_ids = FAKE_INPUT_TOKENS
        prev_decode_index = PREV_DECODE_INDEX
        curr_decode_index = CURR_DECODE_INDEX
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=False,
            is_chat_req=False, stream=True, curr_decode_index=curr_decode_index, prev_decode_index=prev_decode_index)
        golden = {CONTENT_KEY: '3'}
        self.assertDictEqual(result, golden)

    def test_decode_stream_use_tool_calls(self):
        token_ids = FAKE_INPUT_TOKENS
        prev_decode_index = PREV_DECODE_INDEX
        curr_decode_index = CURR_DECODE_INDEX
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=True,
            is_chat_req=True, stream=True, curr_decode_index=curr_decode_index, prev_decode_index=prev_decode_index,
            metadata={"req_enable_thinking": False})
        golden = {'tool_call': 'something', 'metadata': {'current_tool_name_sent': None, 
                                                         'current_tool_arguments_sent': None, 'current_tool_id': None}}
        self.assertDictEqual(result, golden)

    def test_decode_stream_use_reasoning_parser(self):
        token_ids = FAKE_INPUT_TOKENS
        prev_decode_index = PREV_DECODE_INDEX
        curr_decode_index = CURR_DECODE_INDEX
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=False,
            is_chat_req=True, stream=True, curr_decode_index=curr_decode_index, prev_decode_index=prev_decode_index,
            metadata={"req_enable_thinking": True})
        golden = {'reasoning_content': '789', 'content': '654'}
        self.assertDictEqual(result, golden)

    def test_decode_stream_use_both(self):
        token_ids = FAKE_INPUT_TOKENS
        prev_decode_index = PREV_DECODE_INDEX
        curr_decode_index = CURR_DECODE_INDEX
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=True,
            is_chat_req=True, stream=True, curr_decode_index=curr_decode_index, prev_decode_index=prev_decode_index,
            metadata={"req_enable_thinking": True})
        golden = {'tool_call': 'something', 'metadata': {'current_tool_name_sent': None,
            'current_tool_arguments_sent': None, 'current_tool_id': None}, 'reasoning_content': '789'}
        self.assertDictEqual(result, golden)
    
    @patch("atb_llm.runner.tokenizer_wrapper.logger.warning")
    def test_decode_stream_fc_not_implemented(self, mocked_warning):
        del self.tokenizer_wrapper.tool_calls_parser.decode_stream
        token_ids = FAKE_INPUT_TOKENS
        prev_decode_index = PREV_DECODE_INDEX
        curr_decode_index = CURR_DECODE_INDEX
        result = self.tokenizer_wrapper.decode(token_ids, skip_special_tokens=True, use_tool_calls=True,
            is_chat_req=True, stream=True, curr_decode_index=curr_decode_index, prev_decode_index=prev_decode_index,
            metadata={"req_enable_thinking": False})
        golden = {CONTENT_KEY: '3'}
        self.assertDictEqual(result, golden)
        mocked_warning.assert_called_once()



if __name__ == '__main__':
    unittest.main()