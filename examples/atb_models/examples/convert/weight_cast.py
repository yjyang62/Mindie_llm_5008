# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import json
from pathlib import Path
import re
from collections import defaultdict
import torch
import torch_npu
import acl
from tqdm import tqdm
import safetensors
from safetensors.torch import save_file

from atb_llm.utils.argument_utils import ArgumentParser, StringArgumentValidator
from atb_llm.utils.file_utils import MAX_PATH_LENGTH, standardize_path, safe_open
from atb_llm.utils.log import logger


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--save_directory', type=str, help="save the converted weights to the new path",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--index_file', type=str, help="the index file name of model weights",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    return parser.parse_args()


def check_path_valid(dir_path):
    if not Path(dir_path).is_absolute():
        raise ValueError(f"The path of {dir_path} must be absolute.")
    if not Path(dir_path).exists() or not Path(dir_path).is_dir():
        raise ValueError(f"The path of {dir_path} is not a valid directory.")


def reshape_fusion_gmm_weight(weight, dim):
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)
    weight = weight.view(*original_shape[:dim], 2, 16, 128, *original_shape[dim + 1:])
    weight = weight.transpose(dim, dim + 1).contiguous()
    weight = weight.view(*original_shape[:dim], -1, *original_shape[dim + 1:])
    return weight.contiguous()


class WeightCast:
    def __init__(self, args):
        self.model_path = args.model_path
        self.save_directory = args.save_directory
        self.index_file = args.index_file
        self.pattern = r"model\.layers\.(\d+)\.mlp\.(experts\.(\d+)|shared_experts)\.(gate_proj|up_proj|down_proj)"

    def cast(self):
        check_path_valid(self.model_path)
        check_path_valid(self.save_directory)
        index_path = os.path.join(self.model_path, self.index_file)
        index_path = standardize_path(index_path)
        if not os.path.exists(index_path):
            raise FileExistsError(f"{index_path} does not exists.")
        with safe_open(index_path, 'r', encoding='utf-8') as file:
            index_json = json.load(file)

        weight_map = index_json["weight_map"]
        safetensor_names = list(set(weight_map.values()))

        # === Step 1: Load all tensors into global dict ===
        global_tensor_map = {}
        tensor2file_map = defaultdict(list)  # key -> [file1, file2] 便于写回
        logger.info("Step 1: Load all tensors into global dict")
        for sf_name in tqdm(safetensor_names):
            sf_path = os.path.join(self.model_path, sf_name)
            if not os.path.exists(sf_path):
                continue
            with safetensors.safe_open(sf_path, framework="torch") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    global_tensor_map[key] = tensor
                    tensor2file_map[sf_name].append(key)

        # === Step 2: 组织为专家结构 ===
        expert_gateup_groups = defaultdict(lambda: defaultdict(dict))
        expert_down_groups = defaultdict(lambda: defaultdict(dict))
        expert_gateup_scale_groups = defaultdict(lambda: defaultdict(dict))
        expert_down_scale_groups = defaultdict(lambda: defaultdict(dict))
        logger.info("Step 2: Extraxt Expert Weights")
        for key, tensor in tqdm(global_tensor_map.items()):
            match = re.match(self.pattern, key)
            if not match:
                continue
            layer_id = int(match.group(1))
            expert_id = match.group(3) if match.group(3) else "shared"
            proj_type = match.group(4)

            if "down_" in key:
                if "weight_scale" in key:
                    expert_down_scale_groups[layer_id][expert_id][proj_type] = (key, tensor)
                elif "weight_offset" in key:
                    pass
                else:
                    expert_down_groups[layer_id][expert_id][proj_type] = (key, tensor)
            else:
                if "weight_scale" in key:
                    expert_gateup_scale_groups[layer_id][expert_id][proj_type] = (key, tensor)
                elif "weight_offset" in key:
                    pass
                else:
                    expert_gateup_groups[layer_id][expert_id][proj_type] = (key, tensor)

        # === Step 3: Cast & reshape ===
        logger.info("Step 3: Cast Format & Reshape")
        torch.npu.config.allow_internal_format = True
        for layer in tqdm(expert_gateup_groups):
            for expert in expert_gateup_groups[layer]:
                # gate + up
                gate_key, gate_tensor = expert_gateup_groups[layer][expert]["gate_proj"]
                up_key, up_tensor = expert_gateup_groups[layer][expert]["up_proj"]
                gateup_tensor = torch.concat([gate_tensor, up_tensor], dim=0).permute(1, 0).contiguous()
                gateup_tensor = reshape_fusion_gmm_weight(gateup_tensor.unsqueeze(0), -1).npu()[0]
                gateup_tensor = torch_npu.npu_format_cast_(gateup_tensor, 29)
                if gateup_tensor.untyped_storage().size() != gateup_tensor.numel() * gateup_tensor.element_size():
                    raise RuntimeError("Cast failed due to insufficient NZ format memory.")
                gateup_tensor_cpu = torch.empty_like(gateup_tensor, device='cpu')
                torch.npu.synchronize()
                acl.rt.memcpy(gateup_tensor_cpu.untyped_storage().data_ptr(),
                              gateup_tensor_cpu.untyped_storage().nbytes(),
                              gateup_tensor.untyped_storage().data_ptr(),
                              gateup_tensor.untyped_storage().nbytes(), 2)
                global_tensor_map[gate_key] = gateup_tensor_cpu[:, :gate_tensor.shape[0]].contiguous()
                global_tensor_map[up_key] = gateup_tensor_cpu[:, gate_tensor.shape[0]:].contiguous()

                # down
                down_key, down_tensor = expert_down_groups[layer][expert]["down_proj"]
                down_tensor = down_tensor.npu()
                down_tensor = torch_npu.npu_format_cast_(down_tensor, 29)
                if down_tensor.untyped_storage().size() != down_tensor.numel() * down_tensor.element_size():
                    raise RuntimeError("Cast failed due to NZ format memory.")
                down_tensor_cpu = torch.empty_like(down_tensor, device='cpu')
                torch.npu.synchronize()
                acl.rt.memcpy(down_tensor_cpu.untyped_storage().data_ptr(),
                              down_tensor_cpu.untyped_storage().nbytes(),
                              down_tensor.untyped_storage().data_ptr(),
                              down_tensor.untyped_storage().nbytes(), 2)
                global_tensor_map[down_key] = down_tensor_cpu.contiguous()

                # gateup scale（可选）
                if expert_gateup_scale_groups[layer][expert]:
                    scale_g_key, scale_g_tensor = expert_gateup_scale_groups[layer][expert]["gate_proj"]
                    scale_u_key, scale_u_tensor = expert_gateup_scale_groups[layer][expert]["up_proj"]
                    gateup_scale_tensor = torch.concat([scale_g_tensor, scale_u_tensor], dim=0).contiguous()
                    gateup_scale_tensor = reshape_fusion_gmm_weight(gateup_scale_tensor.unsqueeze(0), -2).npu()[0]
                    gateup_scale_tensor = torch_npu.npu_format_cast_(gateup_scale_tensor, 2)
                    global_tensor_map[scale_g_key] = gateup_scale_tensor[:scale_g_tensor.shape[0]].contiguous()
                    global_tensor_map[scale_u_key] = gateup_scale_tensor[scale_g_tensor.shape[0]:].contiguous()

        # === Step 4: 写回保存 ===
        logger.info("Step 4: Saving To File")
        for sf_name in tqdm(safetensor_names):
            save_keys = tensor2file_map[sf_name]
            save_tensor_map = {k: global_tensor_map.get(k, None) for k in save_keys}
            new_sf_path = os.path.join(self.save_directory, sf_name)
            new_sf_path = standardize_path(new_sf_path)
            save_file(save_tensor_map, new_sf_path)
            logger.info(f"Saved to: {new_sf_path}")

        return 0


if __name__ == '__main__':
    args = parse_arguments()
    WeightCast(args).cast()
    describe_json = {"AtlasGMMPermute": True, "is_nzcasted": True}
    describe_file = [i for i in Path(args.save_directory).glob("quant_model_description*")][0]
    with safe_open(str(describe_file), "r") as f:
        d = json.load(f)
        describe_json.update(d)

    with safe_open(str(describe_file), "w", encoding="utf-8") as f:
        json.dump(describe_json, f, indent=4)