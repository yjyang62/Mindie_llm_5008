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

import torch

from tests.encrypttest.encrypt import Encrypt
from tests.encrypttest.decrypt import Decrypt


class CustomEncrypt(Encrypt):

    def __init__(self):
        super().__init__()
        self.generate_keys()

    def generate_keys(self):
        raise NotImplementedError("Please implement your method for gnerating secret keys.")

    def encrypt(self, tensor: torch.Tensor):
        """
        Implemet your encrypting method here

        Returns an encrypted tensor

        Args:
            tensor(`torch.Tensor`):
                The tensor you want to encrypt

        Returns:
            (`torch.Tensor`):
                The encrypted tensor, obtained by encrypting the input tensor

        """
        raise NotImplementedError("Please implement your method to encrypt tesnors.")
    


class CustomDecrypt(Decrypt):

    def __init__(self):
        super().__init__()
        self.get_key_paths()

    def get_key_paths(self):
        raise NotImplementedError("Please implemet your method to get your keys.")

    def decrypt(self, encrypted_tensor: torch.Tensor):
        """
        Implemet your decrypting method here

        Returns an decrypted tensor

        Args:
            encrypted_tensor(`torch.Tensor`):
                The encrypted tensor you want to decrypt

        Returns:
            (`torch.Tensor`):
                The decrypted tensor, obtained by decrypting the input encrypted tesnor

        """
        raise NotImplementedError("Please implement your method to decrypt tesnors.")