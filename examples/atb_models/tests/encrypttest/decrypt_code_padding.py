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

import sys
import os

from atb_llm.utils.loader import safetensor_file_loader
patch_operations = [
    {
        'type': 'check_modified',
        'target_block': """    @property
    def encrypt_enable(self):
        return self.model_weight_path.endswith('crypt') or self.model_weight_path.endswith('crypt/')"""
    },
    {
        'type': 'insert',
        'target_block': """# Part of this file was copied from project text-generation-inference 0.9.1""",
        'code': """import importlib"""
    },
    {
        'type': 'insert',
        'target_block': """        super().__init__(model_weight_path)
        self._filenames = get_weight_filenames(self.model_weight_path, self.extension)""",
        'code': """        if self.encrypt_enable:
            decrypt_script = importlib.import_module('tests.encrypttest.custom_crypt')
            decrypt_cls = getattr(decrypt_script, 'CustomDecrypt')
            self.decrypt_ins = decrypt_cls()
        self.sf_metadata = {}"""
    },
    {
        'type': 'insert',
        'target_block': """        self._routing = self._load_weight_file_routing()
        self.mapping = mapping
        self.device = device""",
        'code': """    @property
    def encrypt_enable(self):
        return self.model_weight_path.endswith('crypt') or self.model_weight_path.endswith('crypt/')"""
    },
    {
        'type': 'remove',
        'target_block': """def get_tensor(self, tensor_name: str) -> Any:
        filename, tensor_name = self.get_filename(tensor_name)
        f = self.get_handler(filename)""",
        'code': """        tensor = f.get_tensor(tensor_name)"""
    },
    {
        'type': 'insert',
        'target_block': """    def get_tensor(self, tensor_name: str) -> Any:
        filename, tensor_name = self.get_filename(tensor_name)
        f = self.get_handler(filename)""",
        'code': """        if self.encrypt_enable:
            tensor = f.get_tensor(tensor_name)
            tensor = self.decrypt_ins.decrypt(tensor)
            if tensor_name in self.sf_metadata:
                module_name, attribute_name = self.sf_metadata[tensor_name].split(".")
                module = importlib.import_module(module_name)
                dtype_ = getattr(module, attribute_name)
            else:
                raise AssertionError(f"{tensor_name} does not exist in metadata")
            tensor = tensor.to(dtype_)
        else:
            tensor = f.get_tensor(tensor_name)
        del self._handlers[filename]"""
    },
    {
        'type': 'insert',
        'target_block': """    def get_sharded(self, tensor_name: str, dim: int, chunk_id: int, num_chunk: int) -> Any:
        if dim not in [0, 1]:
            raise AssertionError(f"Dimension {dim} is invalid in `get_sharded`.")
        slice_ = self._get_slice(tensor_name)
        group_size = slice_.get_shape()[dim]""",
        'code': """        if self.encrypt_enable:
            slice_ = self.get_tensor(tensor_name)"""
    },
    {
        'type': 'insert',
        'target_block': """        for filename in self._filenames:
            filename = file_utils.standardize_path(str(filename), check_link=False)
            file_utils.check_path_permission(filename)
            with safe_open(filename, framework="pytorch") as f:""",
        'code': """                self.sf_metadata.update(f.metadata())"""
    },
]


def process_file(filename, operations):
    with open(filename, 'r') as f:
        content = f.read()
      
    current_content = backup_content = content
    modified = False
    for i, op in enumerate(operations):
        print(f"\nExecute operation {i+1}/{len(operations)}")

        if op['type'] == 'check_modified':
            if op['target_block'] not in current_content:
                print('  -> The modification check has been passed.')
            else:
                print("  ! Warning: The file has been modified. It can not be modified twice.")
                break
        elif op['type'] == 'remove':
            tmp_gather = op['target_block'] + '\n' + op['code']
            if tmp_gather in current_content:
                current_content = current_content.replace(tmp_gather, op['target_block'])
                print("  -> The deletion check has been passed")
            else:
                print("  ! Warning: The code block to be deleted was not found.")
                break
        
        elif op['type'] == 'insert':
            tmp_gather = op['target_block']+'\n'+op['code']+'\n'
            if op['target_block'] in current_content:
                current_content = current_content.replace(op['target_block'], tmp_gather)
                print("  -> The insertion check has been passed.")
            else:
                print("  ! Warning: The target position was not found")
                break
    else:
        modified = True
    if modified:
        backup_file = filename + ".bak"
        with open(backup_file, 'w') as f:
            f.write(backup_content)
        print(f"A backup file has been created: {backup_file}")
        with open(filename, 'w') as f:
            f.write(current_content)
        print("\nThe file modification has been completed.!")
        return True
    else:
        print("\nNo modifications were executed")
        return False


if __name__ == "__main__":
    target_file = safetensor_file_loader.__file__
    if not os.path.exists(target_file):
        print(f"Error: file {target_file} does not exists")
        sys.exit(1)
    print(f"Start processing the file: {target_file}")
    success = process_file(target_file, patch_operations)
    if success:
        print("All operations have been completed.!")
    else:
        print("No modifications were executed")