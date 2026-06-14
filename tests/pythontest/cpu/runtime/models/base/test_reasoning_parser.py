# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from unittest import mock

from mindie_llm.runtime.models.base.reasoning_parser import (
    ReasoningParser,
    CommonReasoningParser
)


START_TOKEN = 1001
END_TOKEN = 1002


def test_reasoning_parser_is_reasoning_end():
    parser = ReasoningParser(START_TOKEN, END_TOKEN)
    
    # End token present
    assert parser.is_reasoning_end([1, 2, END_TOKEN, 3]) is True
    
    # End token absent
    assert parser.is_reasoning_end([1, 2, 3]) is False
    
    # Empty list
    assert parser.is_reasoning_end([]) is False


def test_stream_process_with_start_and_end():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [START_TOKEN, 10, 20, END_TOKEN, 30, 40]
    
    # current_index=0: includes start token
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=0)
    assert reasoning == [START_TOKEN, 10, 20]
    assert content == [30, 40]

    # current_index=1: skips start token (typical usage)
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=1)
    assert reasoning == [10, 20]
    assert content == [30, 40]

    # current_index=3: at end token
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=3)
    assert reasoning == []
    assert content == [30, 40]

    # current_index=5: after end
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=5)
    assert reasoning == []
    assert content == [40]


def test_stream_process_without_start_token():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [5, 10, END_TOKEN, 20]  # no START_TOKEN at beginning
    
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=0)
    assert reasoning == [5, 10]       # valid_text_start_index = 0
    assert content == [20]


def test_stream_process_no_end_token():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [START_TOKEN, 10, 20, 30]
    
    reasoning, content = parser.stream_process_reasoning(all_tokens, current_index=1)
    assert reasoning == [10, 20, 30]  # all treated as reasoning
    assert content == []


def test_stream_process_empty_input():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    reasoning, content = parser.stream_process_reasoning([START_TOKEN], current_index=0)
    assert reasoning == []
    assert content == []


def test_single_process_with_start_and_end():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [START_TOKEN, 10, 20, END_TOKEN, 30, 40]
    
    reasoning, content = parser.single_process_reasoning(all_tokens)
    assert reasoning == [10, 20]
    assert content == [30, 40]


def test_single_process_without_start_token():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [5, 10, END_TOKEN, 20]
    
    reasoning, content = parser.single_process_reasoning(all_tokens)
    assert reasoning == [5, 10]  # no start token → include from beginning
    assert content == [20]


def test_single_process_no_end_token():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [START_TOKEN, 10, 20, 30]
    
    reasoning, content = parser.single_process_reasoning(all_tokens)
    assert reasoning == [10, 20, 30]  # unfinished → all as reasoning
    assert content == []


def test_single_process_end_at_last():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    all_tokens = [START_TOKEN, 10, END_TOKEN]
    
    reasoning, content = parser.single_process_reasoning(all_tokens)
    assert reasoning == [10]
    assert content == []  # nothing after end_token


def test_single_process_empty_input():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    reasoning, content = parser.single_process_reasoning([])
    assert reasoning == []
    assert content == []


def test_single_process_no_end_token_id():
    # Test error handling when end_reasoning_token_id is None
    parser = CommonReasoningParser(START_TOKEN, None)
    with mock.patch("mindie_llm.runtime.models.base.reasoning_parser.logger.error") as mock_log:
        reasoning, content = parser.single_process_reasoning([1, 2, 3])
        assert reasoning == []
        assert content == [1, 2, 3]
        mock_log.assert_called_once_with("ERROR: now in reasoning parser without given end_reasoning_token id.")


def test_count_reasoning_tokens_found():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    assert parser.count_reasoning_tokens([1, 2, END_TOKEN, 3]) == 2  # index of END_TOKEN


def test_count_reasoning_tokens_not_found():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    assert parser.count_reasoning_tokens([1, 2, 3]) == 0


def test_count_reasoning_tokens_empty():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    assert parser.count_reasoning_tokens([]) == 0


def test_count_reasoning_tokens_end_at_start():
    parser = CommonReasoningParser(START_TOKEN, END_TOKEN)
    assert parser.count_reasoning_tokens([END_TOKEN, 1, 2]) == 0