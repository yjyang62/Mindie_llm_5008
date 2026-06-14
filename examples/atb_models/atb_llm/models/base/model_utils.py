# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Implement part of this file based on tramsformers
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import collections
import json
import os
import re
from typing import Union, Callable, Optional, Type, Any, Dict
from dataclasses import dataclass, asdict
from functools import wraps
import importlib.metadata
from tqdm.auto import tqdm


import torch
import transformers
from torch import nn
from safetensors.torch import save_file as safe_save_file
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig
import packaging

from atb_llm.utils.weights import QUANTIZE_DTYPE_LIST
from atb_llm.utils import file_utils
from atb_llm.utils.log.logging import logger


transformers_version = importlib.metadata.version("transformers")
if packaging.version.parse(transformers_version) >= packaging.version.parse("4.47.0"):
    def shard_checkpoint(state_dict, max_shard_size, weights_name):
        from huggingface_hub import split_torch_state_dict_into_shards
        state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size,
                                                              filename_pattern=weights_name)
        shards = {}
        for shard_file, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            shards[shard_file] = shard
        return shards, None
else:
    from transformers.modeling_utils import shard_checkpoint


WEIGHTS_NAME = "pytorch_model.bin"
SAFE_WEIGHTS_NAME = "model.safetensors"
EXTRA_EXP_INFO = "Please check the input parameters model_path, kwargs and the version of transformers. " \
                 + "If the input parameters are valid, the required files exist in model_path and " \
                 + "the version of transformers is correct, make sure the folder's owner has execute permission. " \
                 + "Otherwise, please check the function stack for detailed exception information " \
                 + "and the logs of the llmmodels."


def disable_logger_decorator(func: Callable):
    """Set logger.disable to True in the wrapped function."""
    def wrapper(name):
        disabled_logger = func(name)
        disabled_logger.disabled = True
        return disabled_logger
    return wrapper

transformers.utils.logging.get_logger = disable_logger_decorator(transformers.utils.logging.get_logger)


def unwrap_model_state_dict(state_dict: dict) -> dict:	
    """Remove the prefix 'model.' from the keys of the state dict."""	
    new_state_dict = {}	
    for name, tensor in state_dict.items():	
        new_name = name.replace('.linear.', '.')	
        new_state_dict[new_name] = tensor	
    return new_state_dict


def get_leaf_modules_recursive(model, prefix=""):
    leaf_modules = {}
    if len(list(model.children())) == 0:
        leaf_modules[prefix] = model
    else:
        for name, child in model.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            leaf_modules.update(get_leaf_modules_recursive(child, child_prefix))
    return leaf_modules


def get_module_quant_type(module, name, default_quantize_type):
    if '.weight' in name or '.bias' in name:
        if hasattr(module, "linear_desc"):
            quantize_type = module.linear_desc.name
            if quantize_type == "FLOAT16" or quantize_type == "BFLOAT16":
                return 'FLOAT'
            return quantize_type
    if module is None:
        logger.warning(f"Cannot find parent module for {name}. "
                        f"Use default quantization type: {default_quantize_type}.")
    return default_quantize_type


RESERVED_ATTRIBUTE_MAP_KEY = "reversed_attribute_map"


def get_tqdm_iterator(load_list, disable_cond):
    return tqdm(
        load_list, 
        desc="Loading selected layers", 
        unit="layer", 
        leave=True, 
        dynamic_ncols=True, 
        disable=disable_cond
    )
    

class AttributeMapUtils(nn.Module):
    attribute_map: Dict[str, str] = {}
    reversed_attribute_map: Dict[str, str] = {}

    def __setattr__(self, key: str, value: Any):
        if key == "attribute_map":
            for k, v in value.items():
                if v in super().__getattribute__(RESERVED_ATTRIBUTE_MAP_KEY):
                    raise ValueError(
                        f"{v} is already registered in the attribute map "
                        f"with value {super().__getattribute__('reversed_attribute_map')[v]}"
                    )
                super().__getattribute__(RESERVED_ATTRIBUTE_MAP_KEY)[v] = k
        super().__setattr__(key, value)

    def __getattribute__(self, key: str) -> Any:
        if key != RESERVED_ATTRIBUTE_MAP_KEY and key in super().__getattribute__('reversed_attribute_map'):
            key = super().__getattribute__('reversed_attribute_map')[key]
        return super().__getattribute__(key)

    def __getattr__(self, key: str) -> Any:
        if key != RESERVED_ATTRIBUTE_MAP_KEY and key in super().__getattribute__('reversed_attribute_map'):
            key = super().__getattribute__('reversed_attribute_map')[key]
        return super().__getattr__(key)


class BaseModel(nn.Module):
    """Base class for all models."""
    def save_pretrained(self,
                        save_directory: str,
                        max_shard_size: Union[int, str] = "10GB",
                        save_function: Callable = torch.save,
                        safe_serialization: bool = False):
        """
        Save a model and its configuration file to `save_directory`.

        Args:
            save_directory (str): Directory to which to save, will be created if it doesn't exist.
            max_shard_size (Union[int, str], optional): Maximum size for an individual saved shard file.
                If the model size is larger than this ,the model will be split into multiple shards.
                Can be an integer (in bytes) or string with units ('42GB' or '42MB'), defaults to `10GB`.
            save_function (Callable, optional): The function to use to save the model files.
                Defaults to `torch.save`.
            safe_serialization (bool): Whether to use the safe serialization method, defaults to `Fasle`.
        """
        os.makedirs(save_directory, exist_ok=True)
        save_directory = file_utils.standardize_path(save_directory, check_link=False)
        file_utils.check_path_permission(save_directory)
        state_dict = self.unwrap_model_state_dict(self.state_dict())
        self.save_state_dict(state_dict, save_directory, max_shard_size)
        if self.quantize:
            self.generate_description(save_directory)

    def save_sharded(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        save_directory = file_utils.standardize_path(save_directory, check_link=False)
        file_utils.check_path_permission(save_directory)

        save_directory_dict = {
            # shared experts：
            # ep level 1 mlp_tp
            # ep level 2 moe_tp 1 moe_ep 卡数，全局做dp，每张卡一样
            "moe": os.path.join(
                save_directory,
                f'model-moe-tp-{str(self.mapping.moe_tp.rank).zfill(3)}-ep-{str(self.mapping.moe_ep.rank).zfill(3)}'
            ),
            # mlp full tp不开启时，dense层按照attn_tp切分
            "dense": os.path.join(save_directory, f'model-dense-tp-{str(self.mapping.attn_tp.rank).zfill(3)}'),
            "attn": os.path.join(save_directory, f'model-attn-tp-{str(self.mapping.attn_tp.rank).zfill(3)}'),
            "norm": os.path.join(save_directory, 'model-norm'),
            "model": os.path.join(save_directory, f'model-{str(self.mapping.rank).zfill(3)}')
        }
        for weight_type in save_directory_dict:
            path = save_directory_dict[weight_type]
            os.makedirs(path, exist_ok=True)
            path = file_utils.standardize_path(path, check_link=False)
            file_utils.check_path_permission(path)
            full_state_dict = self.unwrap_model_state_dict(self.state_dict())
            state_dict = {}
            for name, tensor in full_state_dict.items():
                if self.get_module_save_dir(name).lower() == weight_type:
                    state_dict[name] = tensor.contiguous()

            if state_dict:
                self.save_state_dict(state_dict, path)
    
        if self.mapping.rank % self.mapping.local_world_size == 0:
            self.save_sharded_metadata(save_directory)
        if self.quantize and self.mapping.rank % self.mapping.local_world_size == 0:
            self.generate_description(save_directory)

    def unwrap_model_state_dict(self, state_dict: dict) -> dict:
        return unwrap_model_state_dict(state_dict)

    def get_module_save_dir(self, tensor_name):
        if "norm" in tensor_name:
            return "norm"
        return "model"

    def save_state_dict(
            self,
            state_dict,
            save_directory: str,
            max_shard_size: Union[int, str] = "10GB"
    ):
        ptrs = collections.defaultdict(list)
        for name, tensor in state_dict.items():
            ident = (tensor.data_ptr(), tensor.device, tensor.shape, tensor.stride())
            ptrs[ident].append(name)

        shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
        warn_names = set()
        for names in shared_ptrs.values():
            found = 0
            for name in names:
                if name in state_dict:
                    found += 1
                    if found > 1:
                        del state_dict[name]
                        warn_names.add(name)

        shards, _ = shard_checkpoint(
            state_dict, max_shard_size=max_shard_size, weights_name="model{suffix}.safetensors")

        for shard_file, shard in shards.items():
            save_shard_file = os.path.join(save_directory, shard_file)
            save_shard_file = file_utils.standardize_path(save_shard_file)
            file_utils.check_file_safety(save_shard_file, 'w', is_check_file_size=False)
            safe_save_file(shard, save_shard_file, metadata={"format": "pt"})

    def generate_description(self, save_directory: Optional[str] = None):
        """Generate description file of saved quant model."""
        model_description = {}
        state_dict = self.unwrap_model_state_dict(self.state_dict())
        quantize_type = self.quantize.upper()
        model_description['model_quant_type'] = quantize_type
        model_description['version'] = getattr(self, "quant_version", "0.0.0")
        if quantize_type == "W16A16S":
            module_type_map = self.generate_module()
            for name in state_dict.keys():
                if 'Linear' in module_type_map[name]:
                    model_description[name] = quantize_type
                else:
                    model_description[name] = 'FLOAT'
        else:
            for name, tensor in state_dict.items():
                if '.weight' in name or '.bias' in name:
                    if tensor.dtype in QUANTIZE_DTYPE_LIST:
                        model_description[name] = quantize_type
                    else:
                        model_description[name] = 'FLOAT'
                else:
                    model_description[name] = quantize_type

        model_description.update(asdict(self.config.quantization_config))
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            save_directory = os.path.realpath(save_directory)
            if self.quant_version == "0.0.0":
                save_path = os.path.join(save_directory, f'quant_model_description_{quantize_type.lower()}.json')
            else:
                save_path = os.path.join(save_directory, 'quant_model_description.json')
            with file_utils.safe_open(save_path, 'w', encoding='utf-8', is_exist_ok=True) as fw:
                json.dump(model_description, fw, indent=4)
        return model_description
    
    def generate_outsdim(self, save_directory: Optional[str] = None):
        """Generate outsdim description file of saved quant model."""
        model_outs_description = {}
        state_dict = unwrap_model_state_dict(self.state_dict())
        module_type_map = self.generate_module()
        for name, tensor in state_dict.items():
            if 'Linear' in module_type_map[name]:
                if name.endswith('.weight'):
                    name = name + '.outdim'
                model_outs_description[name] = tensor.shape[0]
        return model_outs_description

    def generate_module(self):
        """Generate module description of quant weight."""
        module_type_map = {}
        module_dict = dict(self.named_modules())
        state_dict = unwrap_model_state_dict(self.state_dict())
        for name in state_dict.keys():
            module_path = name.rsplit('.', 1)[0]
            module = module_dict.get(module_path)
            if module:
                module_type_map[name] = type(module).__name__
        return module_type_map      

    def save_sharded_metadata(self, save_directory: str):
        model_sharded_metadata = {}
        model_sharded_metadata["world_size"] = self.mapping.world_size
        parallel_option = dict()
        parallel_option["dp"] = self.mapping.attn_dp.group_size
        parallel_option["tp"] = self.mapping.attn_tp.group_size
        parallel_option["moe_tp"] = self.mapping.moe_tp.group_size
        parallel_option["moe_ep"] = self.mapping.moe_ep.group_size
        model_sharded_metadata["parallel_option"] = parallel_option

        model_sharded = {}
        for root, _, files in os.walk(save_directory):
            model_sharded[os.path.relpath(root, save_directory)] = files
        model_sharded_metadata["model_sharded"] = model_sharded
        file_name = os.path.join(save_directory, "model_sharded_metadata.json")

        with file_utils.safe_open(file_name, 'w', encoding='utf-8', is_exist_ok=True) as fw:
            json.dump(model_sharded_metadata, fw, indent=4)


@dataclass
class LinearInfo:
    """Dataclass maintaining linear layer information."""
    is_pack: bool = True
    is_all_float: bool = False
    pack_linear: nn.Module = None
    location: str = None
    split_num: int = 1


@dataclass
class AttnLinearInfo(LinearInfo):
    """Dataclass maintaining attention linear layer information."""
    q_linear: nn.Module = None
    k_linear: nn.Module = None
    v_linear: nn.Module = None
    dense_linear: nn.Module = None
    location: str = "BEFORE_ATTN"


@dataclass
class MlpLinearInfo(LinearInfo):
    """Dataclass maintaining MlP linear layer information."""
    up_weight_only: bool = False
    gate_linear: nn.Module = None
    up_linear: nn.Module = None
    down_linear: nn.Module = None
    location: str = "BEFORE_MLP"


@dataclass
class LmHeadLinearInfo(LinearInfo):
    """Dataclass maintaining languange model head linear layer information."""
    lm_head_name: str = None
    lm_head_linear: nn.Module = None
    location: str = "BEFORE_LMHEAD"


def filter_urls_from_error(error_message):
    domain_pattern = r'://(?:[a-zA-Z0-9.-]{1,253}(?:\.[a-zA-Z]{2,63})(?::[0-9]{1,5})?)'
    ipv4_pattern = \
        r'://(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::[0-9]{1,5})?'
    ipv6_pattern = r'://(?:\[[0-9a-fA-F:]{3,39}\])(?::[0-9]{1,5})?'
    url_pattern = rf'{domain_pattern}|{ipv4_pattern}|{ipv6_pattern}'
    args = list(error_message.args)
    for i, arg in enumerate(args):
        if isinstance(arg, str):
            args[i] = re.sub(url_pattern, '***', arg)
    error_message.args = tuple(args)
    return error_message


def check_path_and_catch_exception_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_path = file_utils.standardize_path(args[0], check_link=False)
        file_utils.check_path_permission(model_path)
        try:
            return func(model_path, **kwargs)
        except EnvironmentError as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise EnvironmentError(f"{func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error
        except Exception as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise ValueError(f"{func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error
    return wrapper


@check_path_and_catch_exception_decorator
def safe_get_tokenizer_from_pretrained(model_path: str, **kwargs) -> AutoTokenizer:
    """A wrapper function of `AutoTokenizer.from_pretrained` which validates the path."""
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_path_and_catch_exception_decorator
def safe_get_model_from_pretrained(model_path: str, **kwargs) -> AutoModelForCausalLM:
    """A wrapper of `AutoModelForCausalLM.from_pretrained` which validates the path."""
    return AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_path_and_catch_exception_decorator
def safe_get_auto_model_from_pretrained(model_path: str, **kwargs) -> AutoModel:
    """A wrapper of `AutoModel.from_pretrained` which validates the path."""
    return AutoModel.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_path_and_catch_exception_decorator
def safe_get_auto_model_for_sequence_classification_from_pretrained(
        model_path: str, **kwargs
    ) -> AutoModelForSequenceClassification:
    """A wrapper of `AutoModelForSequenceClassification.from_pretrained` which validates the path."""
    return AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_path_and_catch_exception_decorator
def safe_get_config_from_pretrained(model_path: str, **kwargs) -> AutoConfig:
    """A wrapper of `AutoConfig.from_pretrained` which will validate the path."""
    return AutoConfig.from_pretrained(model_path, local_files_only=True, **kwargs)


@check_path_and_catch_exception_decorator
def safe_get_config_dict(model_path: str, **kwargs) -> PretrainedConfig:
    """A wrapper of `PretrainedConfig.get_config_dict` which will validate the path."""
    config, _ = PretrainedConfig.get_config_dict(model_path, local_files_only=True, **kwargs)
    return config


def check_optional_path_and_catch_exception_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        target_cls_or_func = args[0]
        model_path = kwargs.get("model_path", None)
        if model_path:
            model_path = file_utils.standardize_path(model_path, check_link=False)
            file_utils.check_path_permission(model_path)
            kwargs["model_path"] = model_path

        try:
            return func(*args, **kwargs)
        except EnvironmentError as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise EnvironmentError(
                f"Get instance from {target_cls_or_func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error
        except Exception as error:
            filtered_error = filter_urls_from_error(error)
            logger.error(filtered_error)
            raise ValueError(
                f"Get instance from {target_cls_or_func.__name__} failed. " + EXTRA_EXP_INFO) from filtered_error
    return wrapper 


@check_optional_path_and_catch_exception_decorator
def safe_from_pretrained(target_cls: Type, model_path: Optional[str] = None, **kwargs) -> Any:
    """
    A wrapper of `target_cls.from_pretrained` which will validate the path.
    
    Args:
        target_cls (Type): The class to be instantiated.
        model_path (str, optional): The path to the pretrained model, defaults to None.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Any: The instantiated class.
    """
    if model_path:
        return target_cls.from_pretrained(model_path, local_files_only=True, **kwargs)
    else:
        return target_cls.from_pretrained(**kwargs, local_files_only=True)


@check_optional_path_and_catch_exception_decorator
def safe_open_clip_from_pretrained(open_clip_method: Callable, model_name: str, 
                                   model_path: Optional[str] = None, **kwargs) -> Any:
    """A wrapper of `open_clip_method` which will validate the model_name and model_path."""
    # 对model_name进行检查，防止open_clip创建该模型时存在联网操作
    hf_hub_prefix = 'hf-hub:'
    if model_name.startswith(hf_hub_prefix):
        raise ValueError(f"Model name should not start with {hf_hub_prefix} to avoid internet connection.")

    if model_path:
        # 如果model_path存在，则判断输入的方法为模型创建方法
        return open_clip_method(model_name, pretrained=model_path, **kwargs)
    else:
        # 判断为其他方法
        return open_clip_method(model_name, **kwargs)
