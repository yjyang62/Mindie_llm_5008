#!/usr/bin/env python
# coding=utf-8
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
from unittest.mock import patch, MagicMock
import types

from atb_llm.utils import multimodal_utils
from atb_llm.utils.multimodal_utils import (
    ImageType,
    VideoType,
    AudioType,
    is_image,
    is_video,
    is_audio,
    is_multimodal_source_path,
    is_image_path,
    is_video_path,
    is_audio_path,
    safe_open_audio,
    safe_open_image,
    check_video_path,
    validate_image_loader,
    safe_load_multimodal_source
)


class TestMultimodalUtils(unittest.TestCase):
    
    def test_extensions(self):
        self.assertTrue(ImageType.has_extension(".jpg"))
        self.assertTrue(ImageType.has_extension(".png"))
        self.assertFalse(ImageType.has_extension(".txt"))
        
        self.assertTrue(VideoType.has_extension(".mp4"))
        self.assertFalse(VideoType.has_extension(".jpg"))
        
        self.assertTrue(AudioType.has_extension(".mp3"))
        self.assertFalse(AudioType.has_extension(".mp4"))

    def test_is_type_checks(self):
        self.assertTrue(is_image("test.jpg"))
        self.assertFalse(is_image("test.mp4"))
        
        self.assertTrue(is_video("test.mp4"))
        self.assertFalse(is_video("test.mp3"))
        
        self.assertTrue(is_audio("test.mp3"))
        self.assertFalse(is_audio("test.jpg"))

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_is_multimodal_source_path(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["test.jpg"]
        
        # Success case
        self.assertTrue(is_multimodal_source_path("/dummy", is_image))
        
        # Failure case (wrong extension)
        mock_listdir.return_value = ["test.txt"]
        self.assertFalse(is_multimodal_source_path("/dummy", is_image))
        
        # Path not exists
        mock_exists.return_value = False
        with self.assertRaises(RuntimeError):
            is_multimodal_source_path("/dummy", is_image)

    @patch("atb_llm.utils.multimodal_utils.is_multimodal_source_path")
    def test_is_specific_path_checks(self, mock_is_multi):
        mock_is_multi.return_value = True
        
        self.assertTrue(is_image_path("/dummy"))
        mock_is_multi.assert_called_with("/dummy", is_image)
        
        self.assertTrue(is_video_path("/dummy"))
        mock_is_multi.assert_called_with("/dummy", is_video)
        
        self.assertTrue(is_audio_path("/dummy"))
        mock_is_multi.assert_called_with("/dummy", is_audio)

    @patch("atb_llm.utils.file_utils.check_file_safety")
    @patch("atb_llm.utils.file_utils.standardize_path")
    def test_safe_open_audio(self, mock_std, mock_check):
        mock_std.return_value = "/abs/test.mp3"
        mock_audio_cls = MagicMock()
        
        safe_open_audio(mock_audio_cls, "test.mp3")
        
        mock_std.assert_called()
        mock_check.assert_called()
        mock_audio_cls.load.assert_called_with("/abs/test.mp3")

    def test_safe_open_image(self):
        with patch("atb_llm.utils.multimodal_utils.warnings"), \
             patch("atb_llm.utils.multimodal_utils.Image") as mock_pil_image, \
             patch("atb_llm.utils.file_utils.check_file_safety") as mock_check, \
             patch("atb_llm.utils.file_utils.standardize_path") as mock_std:
            mock_std.return_value = "/abs/test.jpg"
            
            # Construct a dummy module for image_cls to satisfy type checks
            dummy_module = types.ModuleType("PIL.Image")
            dummy_module.open = MagicMock()
            
            # Setup the exception on the mocked Image class
            class MockBombWarning(Warning): 
                pass
            mock_pil_image.DecompressionBombWarning = MockBombWarning
            
            # Case 1: Success
            safe_open_image(dummy_module, "test.jpg")
            mock_check.assert_called()
            dummy_module.open.assert_called_with("/abs/test.jpg")
            
            # Case 2: Invalid image_cls
            with self.assertRaises(ValueError):
                safe_open_image(MagicMock(), "test.jpg") # MagicMock is not a ModuleType

            # Case 3: DecompressionBombWarning
            dummy_module.open.side_effect = MockBombWarning("Too big")
            with self.assertRaises(RuntimeError):
                safe_open_image(dummy_module, "test.jpg")
                
            # Case 4: Other Exception
            dummy_module.open.side_effect = Exception("Unknown error")
            with self.assertRaises(RuntimeError):
                safe_open_image(dummy_module, "test.jpg")

    @patch("atb_llm.utils.file_utils.check_file_safety")
    @patch("atb_llm.utils.file_utils.standardize_path")
    def test_check_video_path(self, mock_std, mock_check):
        mock_std.return_value = "/abs/test.mp4"
        res = check_video_path("test.mp4")
        self.assertEqual(res, "/abs/test.mp4")
        mock_check.assert_called()

    @patch("atb_llm.utils.multimodal_utils.warnings")
    @patch("atb_llm.utils.multimodal_utils.Image")
    def test_validate_image_loader(self, mock_pil_image, _):
        # Valid case
        def valid_loader(): 
            pass
        valid_loader.__module__ = "PIL.Image"
        validate_image_loader(valid_loader, {})
        
        # Verify assignments on Image mock
        self.assertEqual(mock_pil_image.MAX_IMAGE_PIXELS, multimodal_utils.MAX_IMAGE_PIXELS)

        # Invalid case
        def invalid_loader(): 
            pass
        with self.assertRaises(ValueError):
            validate_image_loader(invalid_loader, {})

    def test_safe_load_multimodal_source(self):
        with patch("atb_llm.utils.multimodal_utils.is_image") as mock_is_img, \
             patch("atb_llm.utils.multimodal_utils.validate_image_loader") as mock_val, \
             patch("atb_llm.utils.file_utils.check_file_safety"), \
             patch("atb_llm.utils.file_utils.standardize_path") as mock_std:
            mock_std.return_value = "/abs/file"
            target_func = MagicMock()
            
            # Image
            mock_is_img.return_value = True
            safe_load_multimodal_source(target_func, "test.jpg")
            mock_val.assert_called()
            target_func.assert_called_with("/abs/file")
            
            # Video (Need to patch is_video)
            with patch("atb_llm.utils.multimodal_utils.is_video") as mock_is_vid:
                mock_is_img.return_value = False
                mock_is_vid.return_value = True
                safe_load_multimodal_source(target_func, "test.mp4")
                target_func.assert_called_with("/abs/file")
                
            # Audio (Need to patch is_audio)
            with patch("atb_llm.utils.multimodal_utils.is_video") as mock_is_vid, \
                 patch("atb_llm.utils.multimodal_utils.is_audio") as mock_is_aud:
                mock_is_img.return_value = False
                mock_is_vid.return_value = False
                mock_is_aud.return_value = True
                safe_load_multimodal_source(target_func, "test.mp3")
                target_func.assert_called_with("/abs/file")
                
            # Invalid
            with patch("atb_llm.utils.multimodal_utils.is_video") as mock_is_vid, \
                 patch("atb_llm.utils.multimodal_utils.is_audio") as mock_is_aud:
                mock_is_img.return_value = False
                mock_is_vid.return_value = False
                mock_is_aud.return_value = False
                with self.assertRaises(ValueError):
                    safe_load_multimodal_source(target_func, "test.txt")
