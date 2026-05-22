# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import argparse
from pathlib import Path
import shutil
import traceback

from atb_llm.utils.hub import weight_files
from encrypt import EncryptTools
from custom_crypt import CustomEncrypt
from atb_llm.utils import file_utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_path', type=str, help="input model and tokenizer path")
    parser.add_argument('--encrypted_model_weights_path', type=str, help="output model and tokenizer path")
    parser.add_argument('--key_path', type=str, help="")
    return parser.parse_args()


def encrypt_weights(model_weights_path, encrypted_model_weights_path, key_path=None):
    file_suffix = ['.bin', '.safetensors']
    try:
        local_weight_files = weight_files(model_weights_path, extension=file_suffix[0])
    except FileNotFoundError:
        local_weight_files = weight_files(model_weights_path, extension=file_suffix[1])
    encrypted_weight_files = [Path(encrypted_model_weights_path) / f"{p.stem.lstrip('pytorch_')}.safetensors"
                              if p.suffix == file_suffix[0] else Path(encrypted_model_weights_path) / f"{p.name}"
                              for p in local_weight_files]

    local_all_files = list(Path(model_weights_path).glob('*'))
    for local_all_file in local_all_files:
        if file_suffix[0] == local_all_file.suffix or file_suffix[1] == local_all_file.suffix: 
            continue
        # skip nested directory
        if local_all_file.is_dir():
            print(f"! Warning: Nested directory skipped - all internal content (subdirs/files) will not be processed: {local_all_file.resolve()}")
            continue
        output_file = os.path.join(encrypted_model_weights_path, local_all_file.name)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        output_file = file_utils.standardize_path(str(output_file), check_link=False)
        file_utils.check_file_safety(output_file, 'w', is_check_file_size=False)
        shutil.copyfile(local_all_file, output_file)
    try:
        my_encrypt = CustomEncrypt()
    except NotImplementedError:
        print('! Warning: you have not implemented the custom method for encrypting tensors.')
        answer = input('Type "yes/y" to continue to use the demo encryption method with a required parameter of key_path.' \
        ' Type others to terminate the process. If you are in a production environment, please be cautious about using demo methods!')
        if answer.lower() == 'yes' or answer.lower() == 'y':
            my_encrypt = EncryptTools(key_path)
        else:
            print('!Warning: You did not choose to continue, so no encryption was performed. Welcome to use it again.')
            return
    except Exception:
        error_msg = traceback.format_exc()
        print('The custom encryption method was executed incorrectly. The error message is as follows:', error_msg)
    my_encrypt.encrypt_files(local_weight_files, encrypted_weight_files, discard_names=[])
    _ = weight_files(encrypted_model_weights_path)


def main():
    args = parse_arguments()
    model_weights_path = file_utils.standardize_path(args.model_weights_path, check_link=False)
    file_utils.check_path_permission(model_weights_path)

    os.makedirs(os.path.dirname(args.encrypted_model_weights_path), exist_ok=True)
    encrypted_model_weights_path = file_utils.standardize_path(args.encrypted_model_weights_path, check_link=False)
    file_utils.check_path_permission(encrypted_model_weights_path)

    if args.key_path:
        os.makedirs(os.path.dirname(args.key_path), exist_ok=True)
        key_path = file_utils.standardize_path(args.key_path, check_link=False)
        file_utils.check_path_permission(key_path)
    else:
        key_path = None

    encrypt_weights(model_weights_path, encrypted_model_weights_path, key_path)

if __name__ == '__main__':
    main()