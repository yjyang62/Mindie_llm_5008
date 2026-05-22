# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import unittest
from unittest.mock import Mock

from atb_llm.models.base.reasoning_parser import CommonReasoningParser


class TestCommonReasoningParser(unittest.TestCase):
    def setUp(self) -> None:
        config = Mock()
        config.reasoning_config.start_reasoning_token_id = 123
        config.reasoning_config.end_reasoning_token_id = 321
        self.parser = CommonReasoningParser(config)

    def test_reasoning_config_init(self):
        self.assertEqual(self.parser.start_reasoning_token_id, 123)
        self.assertEqual(self.parser.end_reasoning_token_id, 321)

    def test_is_reasoning_end(self):
        mock_false_token_ids = [1, 2, 3]
        mock_true_token_ids = [1, 2, 3, 321]
        self.assertFalse(self.parser.is_reasoning_end(mock_false_token_ids))
        self.assertTrue(self.parser.is_reasoning_end(mock_true_token_ids))

    def test_single_process_reasoning_case1(self):
        # common case
        mock_reasoning_token_ids = [1, 2, 3]
        mock_content_token_ids = [4, 5, 6]
        all_token_ids = [self.parser.start_reasoning_token_id] + mock_reasoning_token_ids + [
            self.parser.end_reasoning_token_id] + mock_content_token_ids
        reasoning_content_token_ids, content_token_ids = self.parser.single_process_reasoning(all_token_ids)
        self.assertEqual(reasoning_content_token_ids, mock_reasoning_token_ids)
        self.assertEqual(content_token_ids, mock_content_token_ids)

    def test_single_process_reasoning_case2(self):
        # no start <think>
        mock_reasoning_token_ids = [1, 2, 3]
        mock_content_token_ids = [4, 5, 6]
        all_token_ids = mock_reasoning_token_ids + [self.parser.end_reasoning_token_id] + mock_content_token_ids
        reasoning_content_token_ids, content_token_ids = self.parser.single_process_reasoning(all_token_ids)
        self.assertEqual(reasoning_content_token_ids, mock_reasoning_token_ids)
        self.assertEqual(content_token_ids, mock_content_token_ids)

    def test_single_process_reasoning_case3(self):
        # limited by length with unfinished
        mock_reasoning_token_ids = [1, 2, 3]
        all_token_ids = [self.parser.start_reasoning_token_id] + mock_reasoning_token_ids
        reasoning_content_token_ids, content_token_ids = self.parser.single_process_reasoning(all_token_ids)
        self.assertEqual(reasoning_content_token_ids, mock_reasoning_token_ids)
        self.assertEqual(content_token_ids, [])

    def test_stream_process_reasoning_case1(self):
        # before </think>
        all_token_ids = [1, 2, 3]
        delta_size = 1
        current_index = len(all_token_ids) - delta_size
        reasoning_content_token_ids, content_token_ids = self.parser.stream_process_reasoning(all_token_ids,
                                                                                              current_index)
        self.assertEqual(reasoning_content_token_ids, [3])
        self.assertEqual(content_token_ids, [])

    def test_stream_process_reasoning_case2(self):
        # end </think>
        all_token_ids = [123, 1, 2, 3, 321, 4, 5]
        delta_size = 2
        current_index = len(all_token_ids) - delta_size
        reasoning_content_token_ids, content_token_ids = self.parser.stream_process_reasoning(all_token_ids,
                                                                                              current_index)
        self.assertEqual(reasoning_content_token_ids, [])
        self.assertEqual(content_token_ids, [4, 5])

    def test_stream_process_reasoning_case3(self):
        # current </think>
        all_token_ids = [123, 1, 2, 3, 321, 4]
        current_index = len(all_token_ids) - 3
        reasoning_content_token_ids, content_token_ids = self.parser.stream_process_reasoning(all_token_ids,
                                                                                              current_index)
        self.assertEqual(reasoning_content_token_ids, [3])
        self.assertEqual(content_token_ids, [4])
    
    def test_count_reasoning_tokens(self):
        all_token_ids = [123, 1, 2, 3, 321, 4]
        self.assertEqual(self.parser.count_reasoning_tokens(all_token_ids), 4)
        
        all_token_ids = [123]
        self.assertEqual(self.parser.count_reasoning_tokens(all_token_ids), 0)


if __name__ == '__main__':
    unittest.main()