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
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Union

from atb_llm.models.base.postprocessor import Postprocessor
from atb_llm.utils.env import ENV
from atb_llm.utils.argument_utils import MAX_KEY_LENGTH


@dataclass
class MockGenerationConfig:
    """Mock generation config for testing"""
    max_new_tokens: int
    pad_token_id: int
    eos_token_id: Union[int, List[Union[int, List[int]]]]


class TestPostprocessor(unittest.TestCase):
    """Test cases for Postprocessor class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.eos_token_id = 2
        self.mock_generation_config = MockGenerationConfig(
            max_new_tokens=100,
            pad_token_id=0,
            eos_token_id=2
        )

    def tearDown(self):
        """Clean up after each test method."""
        ENV.modeltest_dataset_specified = None
        ENV.rank = 0

    def test_init_with_valid_config(self):
        """Test initialization with valid config."""
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        self.assertEqual(processor.max_new_tokens, 100)
        self.assertEqual(processor.pad_token_id, 0)
        self.assertEqual(processor.eos_token_id, 2)

    def test_init_with_none_eos_token_id(self):
        """Test initialization uses tokenizer eos_token_id when config eos_token_id is None."""
        self.mock_generation_config.eos_token_id = None
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        self.assertEqual(processor.eos_token_id, self.mock_tokenizer.eos_token_id)

    def test_init_with_empty_eos_token_id(self):
        """Test initialization uses tokenizer eos_token_id when config eos_token_id is empty."""
        self.mock_generation_config.eos_token_id = []
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        self.assertEqual(processor.eos_token_id, self.mock_tokenizer.eos_token_id)

    def test_stopping_criteria_with_int_eos_match(self):
        """Test stopping_criteria with integer eos_token_id that matches."""
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 2]
        self.assertTrue(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_int_eos_no_match(self):
        """Test stopping_criteria with integer eos_token_id that doesn't match."""
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 4]
        self.assertFalse(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_list_eos_int_match(self):
        """Test stopping_criteria with list eos_token_id containing integer that matches."""
        self.mock_generation_config.eos_token_id = [2, 3, 4]
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 2]
        self.assertTrue(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_list_eos_list_match(self):
        """Test stopping_criteria with list eos_token_id containing list that matches."""
        self.mock_generation_config.eos_token_id = [[1, 2], [3, 4]]
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2]
        self.assertTrue(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_list_eos_no_match(self):
        """Test stopping_criteria with list eos_token_id that doesn't match."""
        self.mock_generation_config.eos_token_id = [5, 6, 7]
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 4]
        self.assertFalse(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_invalid_eos_type(self):
        """Test stopping_criteria with invalid eos_token_id type."""
        self.mock_generation_config.eos_token_id = "invalid"
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        
        with patch('atb_llm.models.base.postprocessor.print_log') as mock_log:
            output_ids = [1, 2, 3]
            result = processor.stopping_criteria(output_ids)
            self.assertFalse(result)
            mock_log.assert_called_once()

    @patch.dict('os.environ', {'MODELTEST_DATASET_SPECIFIED': 'OtherDataset_123_python'})
    def test_stopping_criteria_with_non_humaneval_env(self):
        """Test stopping_criteria with non-HumanEval environment variable."""
        ENV.update()
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 2]
        result = processor.stopping_criteria(output_ids)
        self.assertTrue(result)

    @patch.dict('os.environ', {'MODELTEST_DATASET_SPECIFIED': 'H' * (MAX_KEY_LENGTH + 1)})
    def test_stopping_criteria_with_too_long_env_var(self):
        """Test stopping_criteria with too long environment variable."""
        ENV.update()
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        
        with self.assertRaises(ValueError):
            processor.stopping_criteria([1, 2, 3])

    def test_stopping_criteria_with_single_token_match(self):
        """Test stopping_criteria with single token that matches eos."""
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [2]
        self.assertTrue(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_short_sequence_list_match(self):
        """Test stopping_criteria with sequence shorter than list eos_token."""
        self.mock_generation_config.eos_token_id = [[1, 2, 3]]
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2]
        self.assertFalse(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_rank_not_zero(self):
        """Test stopping_criteria with non-zero rank."""
        ENV.rank = 1
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        output_ids = [1, 2, 3, 2]
        self.assertTrue(processor.stopping_criteria(output_ids))

    def test_stopping_criteria_with_multiple_list_matches(self):
        """Test stopping_criteria with multiple list eos_token_id matches."""
        self.mock_generation_config.eos_token_id = [[1, 2], [2, 3], [3, 4]]
        processor = Postprocessor(self.mock_tokenizer, self.mock_generation_config)
        
        # Test each possible match
        self.assertTrue(processor.stopping_criteria([1, 2]))
        self.assertTrue(processor.stopping_criteria([2, 3]))
        self.assertTrue(processor.stopping_criteria([3, 4]))
        self.assertFalse(processor.stopping_criteria([1, 3]))


if __name__ == '__main__':
    unittest.main()