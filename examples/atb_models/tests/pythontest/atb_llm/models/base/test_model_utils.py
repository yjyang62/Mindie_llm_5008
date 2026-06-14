#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import patch, Mock, MagicMock, call

import torch
from torch import nn
from transformers import AutoTokenizer

# 待测函数
from atb_llm.models.base.model_utils import (
    EXTRA_EXP_INFO,
    BaseModel,
    filter_urls_from_error,
    safe_get_tokenizer_from_pretrained,
    safe_get_model_from_pretrained,
    safe_get_auto_model_from_pretrained,
    safe_get_auto_model_for_sequence_classification_from_pretrained,
    safe_get_config_from_pretrained,
    safe_get_config_dict,
    safe_from_pretrained,
    safe_open_clip_from_pretrained,
    AttributeMapUtils,
    get_leaf_modules_recursive,
    get_module_quant_type
)
from atb_llm.utils.quantize.quant_type import LinearTypeV2
from atb_llm.models.base.config import QuantizationConfig


class FakeLayer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.layer_id = i


class FakeModel(AttributeMapUtils):
    def __init__(self):
        super().__init__()
        self.attribute_map = {"h": "layers"}
        self.h = nn.ModuleList([FakeLayer(i) for i in range(2)])


class FakeFlashCausalLmModel(AttributeMapUtils):
    def __init__(self):
        super().__init__()
        self.attribute_map = {"transformer": "model"}
        self.transformer = FakeModel()


class FakeConfig:
    def __init__(self):
        self.quantization_config = QuantizationConfig(
            group_size=0, kv_quant_type=None, fa_quant_type=None, reduce_quant_type=None
        )


class FakeMapping:
    def __init__(self):
        self.moe_tp = Mock(rank=1, group_size=2)
        self.moe_ep = Mock(rank=2, group_size=3)
        self.attn_tp = Mock(rank=3, group_size=4)
        self.attn_dp = Mock(group_size=5)
        self.rank = 0
        self.local_world_size = 1
        self.world_size = 8


class FakeBaseModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.quantize = "W8A8"
        self.quant_version = "1.0.0"
        self.config = FakeConfig()
        self.mapping = FakeMapping()
        self._state_dict = {
            "norm.weight": torch.tensor([1.0]),
            "model.weight": torch.tensor([2.0]),
        }

    def state_dict(self):
        return self._state_dict


class TestModelUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mocked_standardize_path = patch('atb_llm.utils.file_utils.standardize_path')
        cls.mocked_check_path_permission = patch('atb_llm.utils.file_utils.check_path_permission')
        cls.mocked_standardize_path.start()
        cls.mocked_check_path_permission.start()

    @classmethod
    def tearDownClass(cls):
        cls.mocked_standardize_path.stop()
        cls.mocked_check_path_permission.stop()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_leaf_modules_recursive(self):
        leaf_modules = get_leaf_modules_recursive(FakeFlashCausalLmModel())
        self.assertIn("transformer.h.0", leaf_modules)
        self.assertIn("transformer.h.1", leaf_modules)

    def test_get_module_quant_type(self):
        default_quantize_type = "FLOAT"
        quant_type = get_module_quant_type(None, "", "FLOAT")
        self.assertEqual(quant_type, default_quantize_type)

        quant_type = get_module_quant_type(None, "linear.weight", "FLOAT")
        self.assertEqual(quant_type, default_quantize_type)

        linear = MagicMock()
        linear.linear_desc = LinearTypeV2.FLOAT16
        quant_type = get_module_quant_type(linear, "linear.weight", "FLOAT")
        self.assertEqual(quant_type, "FLOAT")

        linear.linear_desc = LinearTypeV2.W4A8_DYNAMIC
        quant_type = get_module_quant_type(linear, "linear.weight", "FLOAT")
        self.assertEqual(quant_type, "W4A8_DYNAMIC")

    @patch("os.makedirs")
    @patch("os.path.realpath")
    @patch("atb_llm.models.base.model_utils.file_utils.standardize_path")
    @patch("atb_llm.models.base.model_utils.file_utils.check_path_permission")
    @patch("atb_llm.models.base.model_utils.safe_save_file")
    @patch("atb_llm.models.base.model_utils.file_utils.safe_open")
    def test_save_sharded(self, mock_save_open, mock_save_file, mock_check_path_permission,
        mock_standardize_path, mock_realpath, mock_mkdirs):
        save_dir = "/save_dir"
        mock_standardize_path.return_value = save_dir
        mock_check_path_permission.return_value = "None"
        mock_realpath.return_value = save_dir
        model = FakeBaseModel()
        model.save_sharded(save_dir)

        mock_mkdirs.assert_has_calls([
            call('/save_dir', exist_ok=True),
            call('/save_dir/model-moe-tp-001-ep-002', exist_ok=True),
            call('/save_dir/model-dense-tp-003', exist_ok=True),
            call('/save_dir/model-attn-tp-003', exist_ok=True),
            call('/save_dir/model-norm', exist_ok=True),
            call('/save_dir/model-000', exist_ok=True)
        ], any_order=True)
        mock_save_open.assert_has_calls([
            call("/save_dir/model_sharded_metadata.json",
            "w", encoding="utf-8", is_exist_ok=True),
        ])

    def test_filter_urls_from_error_case_exp_with_empty_args_result_success(self):
        error_message = Exception()

        filtered_error = filter_urls_from_error(error_message)

        self.assertEqual(filtered_error.args, ())
    
    def test_filter_urls_from_error_case_exp_with_no_url_result_success(self):
        arg = "Load tokenizer faild. Please check tokenizer files in model path."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        self.assertEqual(filtered_error.args, (arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_domain_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://huggingface.co for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_domain_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://huggingface.co:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_ipv4_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://255.200.100.0 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_ipv4_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://255.200.100.0:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_ipv6_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://[0000:1234:5678:90ab:cdef:90AB:CDEF:0000] " \
            + "for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_ipv6_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://[1234::5678]:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_multi_args_and_urls_result_success(self):
        arg0 = "Load tokenizer faild. Please check tokenizer files in model path."
        arg1 = "Load tokenizer faild. Please visit http://huggingface.co/abcd for more information."
        arg2 = "Load tokenizer faild. Please visit https://huggingface.co:1234/efgh for more information."
        arg3 = "Load tokenizer faild. Please visit http://255.200.100.0/ijkl for more information."
        arg4 = "Load tokenizer faild. Please visit https://255.200.100.0:1234/mnop for more information."
        arg5 = "Load tokenizer faild. Please visit http://[0000:1234:5678:90ab:cdef:90AB:CDEF:0000]/qrst " \
               + "for more information."
        arg6 = "Load tokenizer faild. Please visit https://[1234::5678]:1234/uvwx for more information."
        error_message = Exception(arg0, arg1, arg2, arg3, arg4, arg5, arg6)

        filtered_error = filter_urls_from_error(error_message)

        filtered_args = (
            "Load tokenizer faild. Please check tokenizer files in model path.",
            "Load tokenizer faild. Please visit http***/abcd for more information.",
            "Load tokenizer faild. Please visit https***/efgh for more information.",
            "Load tokenizer faild. Please visit http***/ijkl for more information.",
            "Load tokenizer faild. Please visit https***/mnop for more information.",
            "Load tokenizer faild. Please visit http***/qrst for more information.",
            "Load tokenizer faild. Please visit https***/uvwx for more information."
        )
        self.assertEqual(filtered_error.args, filtered_args)
    
    def validate_param_local_files_only(self, *args, **kwargs):
        if 'local_files_only' not in kwargs or kwargs['local_files_only'] is not True:
            return False
        return True
    
    def validate_param_local_files_only_return_two_value(self, *args, **kwargs):
        if 'local_files_only' not in kwargs or kwargs['local_files_only'] is not True:
            return False, False
        return True, True
    
    def test_safe_get_tokenizer_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            tokenizer = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertTrue(tokenizer)
    
    def test_safe_get_tokenizer_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_tokenizer_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_tokenizer_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_tokenizer_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_model_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_model_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_model_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_model_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_auto_model_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_for_sequence_classification_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_for_sequence_classification_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            config = safe_get_config_from_pretrained(model_path)
        
        self.assertTrue(config)
    
    def test_safe_get_config_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_config_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_config_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_config_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_config_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_dict_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only_return_two_value
            config = safe_get_config_dict(model_path)
        
        self.assertTrue(config)
    
    def test_safe_get_config_dict_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_config_dict(model_path)
        
        self.assertIn(
            f"{safe_get_config_dict.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_dict_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_config_dict(model_path)
        
        self.assertIn(
            f"{safe_get_config_dict.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_from_pretrained_case_success(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            tokenizer = safe_from_pretrained(target_cls, model_path)
        
        self.assertTrue(tokenizer)
    
    def test_safe_from_pretrained_case_raise_environment_error(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_from_pretrained(target_cls, model_path)
        
        self.assertIn(
            f"Get instance from {target_cls.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_from_pretrained_case_raise_value_error(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_from_pretrained(target_cls, model_path)
        
        self.assertIn(
            f"Get instance from {target_cls.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_open_clip_from_pretrained_case_model_name_illegal_raise_value_error(self):
        open_clip_method = Mock(__name__='open_clip_method')
        model_name = 'hf-hub:llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(ValueError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            "Model name should not start with hf-hub: to avoid internet connection.", str(context.exception.__cause__))
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))

    def test_safe_open_clip_from_pretrained_case_raise_environment_error(self):
        open_clip_method = Mock(side_effect=EnvironmentError, __name__='open_clip_method')
        model_name = 'llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(EnvironmentError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_open_clip_from_pretrained_case_raise_value_error(self):
        open_clip_method = Mock(side_effect=ValueError, __name__='open_clip_method')
        model_name = 'llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(ValueError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))

    def test_attribute_map_utils(self):
        model = FakeFlashCausalLmModel()
        self.assertEqual(model.model, model.transformer)
        self.assertEqual(model.model.layers, model.transformer.h)
        AttributeMapUtils.attribute_map.clear()
        AttributeMapUtils.reversed_attribute_map.clear()
    
    def test_gengerate_description_case_sparse_type_w16a16s(self):
        model = FakeBaseModel()
        model.quantize = 'W16A16S'
        model.generate_module = MagicMock(return_value={"model.weight": "Linear", "norm.weight": "Norm"})
        model_description = model.generate_description()
        self.assertEqual(model_description["model.weight"], 'W16A16S')
        self.assertEqual(model_description["norm.weight"], 'FLOAT')
    
    def test_generate_outsdim(self):
        model = FakeBaseModel()
        model.generate_module = MagicMock(return_value={"model.weight": "Linear", "norm.weight": "Norm"})
        expect_output = {'model.weight.outdim': 1}
        self.assertEqual(model.generate_outsdim(), expect_output)

    def test_generate_module(self):
        model = FakeBaseModel()
        linear_mock = MagicMock()
        linear_mock.__class__.__name__ = "Linear"
        norm_mock = MagicMock()
        norm_mock.__class__.__name__ = "Norm"
        model.named_modules = MagicMock(return_value={"model": linear_mock, "norm": norm_mock})
        module_type_map = model.generate_module()
        self.assertEqual(module_type_map["model.weight"], 'Linear')
        self.assertEqual(module_type_map["norm.weight"], 'Norm')

if __name__ == '__main__':
    unittest.main()
