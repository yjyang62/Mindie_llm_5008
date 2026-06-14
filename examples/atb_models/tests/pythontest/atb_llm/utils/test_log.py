# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch
from mindie_llm.utils.log.logging_base import get_logger, Component


class TestLoggerInitialization(unittest.TestCase):
    def test_get_llmmodels_logger(self):
        """Test getting the logger for LLMMODELS component"""
        logger = get_logger(Component.LLMMODELS)
        
        # Verify logger instance exists
        self.assertIsNotNone(logger)
        
        # Verify logger configuration is correct
        self.assertEqual(logger.extra['component'], Component.LLMMODELS)
        
        # Verify logger can be used normally
        with patch.object(logger, 'info') as mock_info:
            logger.info("test message")
            mock_info.assert_called_once_with("test message")

if __name__ == '__main__':
    unittest.main()
