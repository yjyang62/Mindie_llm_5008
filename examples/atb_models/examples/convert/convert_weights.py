# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import argparse

from atb_llm.utils.convert import convert_files
from atb_llm.utils.hub import weight_files
from atb_llm.utils.log import logger
from atb_llm.utils import file_utils
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path")
    return parser.parse_args()


def convert_bin2st(model_path):
    local_pt_files = weight_files(model_path, extension=".bin")
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
        for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])
    _ = weight_files(model_path)


def convert_bin2st_from_pretrained(model_path):
    model = safe_get_model_from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype="auto"
    )
    model.save_pretrained(model_path, safe_serialization=True)


if __name__ == '__main__':
    args = parse_arguments()

    input_model_path = file_utils.standardize_path(args.model_path, check_link=False)
    file_utils.check_path_permission(input_model_path)
    try:
        convert_bin2st(input_model_path)
    except RuntimeError:
        logger.warning("Convert weights failed with 'torch.load' method, need model loaded to convert again.")
        convert_bin2st_from_pretrained(input_model_path)