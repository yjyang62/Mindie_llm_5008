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

from enum import Enum
from .network_manager import get_default_net
from .node import Node
from ..utils.quantize.pack_type import DataType


class Format(str, Enum):
    """Tensor data format enumeration."""
    ND = "ND"
    NZ = "NZ"


class Tensor:
    def __init__(self, name="?"):
        """
        Initialize a Tensor object.
        
        Args:
            name (str): Name identifier for the tensor (default: "?")
        """
        self.name = name
        self.reshape_func = None
        self.shape = None
        self.view_tensor = None
        self.view_father = None

    def __eq__(self, other):
        """
        Check equality between two tensors based on name.
        
        Args:
            other (Tensor): Another tensor to compare with
            
        Returns:
            bool: True if names are equal, False otherwise
        """
        return self.name == other.name

    def __gt__(self, other):
        """
        Perform element-wise greater than operation.
        
        Args:
            other (Tensor): Right-hand side tensor for comparison
            
        Returns:
            Tensor: Output tensor with comparison results
        """
        out = Tensor()
        param = {"elewiseType": "ELEWISE_GREATER"}
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out
    
    def __lt__(self, other):
        """
        Perform element-wise less than operation.
        
        Args:
            other (Tensor): Right-hand side tensor for comparison
            
        Returns:
            Tensor: Output tensor with comparison results
        """
        out = Tensor()
        param = {"elewiseType": "ELEWISE_LESS"}
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __add__(self, other):
        """
        Perform element-wise addition.
        
        Args:
            other (Tensor): Tensor to add to current tensor
            
        Returns:
            Tensor: Output tensor with addition results
        """
        out = Tensor()
        node = Node("Elewise", {'elewiseType': 'ELEWISE_ADD'}, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __sub__(self, other):
        """
        Perform element-wise subtraction.
        
        Args:
            other (Tensor): Tensor to subtract from current tensor
            
        Returns:
            Tensor: Output tensor with subtraction results
        """
        out = Tensor()
        param = {"elewiseType": "ELEWISE_SUB"}
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __mul__(self, value):
        """
        Perform element-wise multiplication.
        
        Args:
            value (Tensor|int|float): Multiplicand value or tensor
            
        Returns:
            Tensor: Output tensor with multiplication results
            
        Raises:
            ValueError: If value type is not supported
        """
        out = Tensor()
        if isinstance(value, Tensor):
            param = {'elewiseType': 'ELEWISE_MUL'}
            node = Node('Elewise', param, [self, value], [out])
            get_default_net().push_node(node)
        elif isinstance(value, int) or isinstance(value, float):
            param = {
                'elewiseType': 'ELEWISE_MULS',
                'mulsParam': {'varAttr': float(value)}
            }
            node = Node('Elewise', param, [self], [out])
            get_default_net().push_node(node)
        else:
            raise ValueError("* operator only support Tensor * Tensor, Tensor * float")
        return out

    def __truediv__(self, value):
        """
        Perform element-wise division.
        
        Args:
            value (Tensor|int|float): Divisor value or tensor
            
        Returns:
            Tensor: Output tensor with division results
            
        Raises:
            ValueError: If value type is not supported
        """
        out = Tensor()
        if isinstance(value, Tensor):
            param = {'elewiseType': 'ELEWISE_REALDIV'}
            node = Node('Elewise', param, [self, value], [out])
            get_default_net().push_node(node)
        elif isinstance(value, int) or isinstance(value, float):
            param = {
                'elewiseType': 'ELEWISE_MULS',
                'mulsParam': {'varAttr': float(1 / value)}
            }
            node = Node('Elewise', param, [self], [out])
            get_default_net().push_node(node)
        else:
            raise ValueError("/ operator only support Tensor / Tensor, Tensor / float")
        return out

    def __invert__(self):
        """
        Perform element-wise logical NOT operation.
        
        Returns:
            Tensor: Output tensor with logical NOT results
        """
        out = Tensor()
        param = {"elewiseType": "ELEWISE_LOGICAL_NOT"}
        node = Node('Elewise', param, [self], [out])
        get_default_net().push_node(node)
        return out

    def __getitem__(self, slices):
        """
        Slice the tensor using given slice indices.
        
        Args:
            slices: Slice indices for each dimension
            
        Returns:
            Tensor: Sliced output tensor
        """
        out = Tensor()
        param = {}
        offsets = []
        sizes = []
        for slice_item in slices:
            start = slice_item.start
            stop = slice_item.stop
            if start is None:
                start = 0
            if stop is None:
                stop = 0
            if stop - start > 0:
                offsets.append(start)
                sizes.append(stop - start)
            else:
                offsets.append(0)
                sizes.append(-1)
        param["offsets"] = offsets
        param["size"] = sizes
        node = Node("Slice", param, [self], [out])
        get_default_net().push_node(node)
        return out

    def write_inplace(self, tensor):
        """
        Replace current tensor's name with another tensor's name (in-place operation).
        
        Args:
            tensor (Tensor): Source tensor whose name will be copied
        """
        self.name = tensor.name

    def to(self, dtype: DataType):
        """
        Cast tensor to specified data type.
        
        Args:
            dtype (DataType): Target data type for conversion
            
        Returns:
            Tensor: Output tensor with converted data type
        """
        out = Tensor()
        param = {"dtype": dtype}
        node = Node("Cast", param, [self], [out])
        get_default_net().push_node(node)
        return out

    def permute(self, dims: list[int]):
        """
        Permute tensor dimensions according to given order.
        
        Args:
            dims (list[int]): New dimension order
            
        Returns:
            Tensor: Output tensor with permuted dimensions
        """
        out = Tensor()
        node = Node("Transpose", {'perm': dims}, [self], [out])
        get_default_net().push_node(node)
        return out

    def reshape(self, reshape_func):
        """
        Create a view tensor with reshape function (lazy reshape).
        
        Args:
            reshape_func: Function defining the reshape operation
            
        Returns:
            Tensor: View tensor with reshape function applied
        """
        self.view_tensor = Tensor(f"{self.name}_view")
        self.view_tensor.view_father = self
        self.view_tensor.reshape_func = reshape_func
        return self.view_tensor

    def repeat(self, repeats: list[int] = None):
        """
        Repeat tensor along specified dimensions.
        
        Args:
            repeats (list[int]): Number of repetitions for each dimension
            
        Returns:
            Tensor: Output tensor with repeated values
        """
        out = Tensor()
        param = {"multiples": repeats}
        node = Node('Repeat', param, [self], [out])
        get_default_net().push_node(node)
        return out


class Engine:
    def __init__(self, atb_engine, weights_keys) -> None:
        """
        Initialize an Engine object.
        
        Args:
            atb_engine: Underlying ATB engine instance
            weights_keys: Keys for weight parameters
        """
        self.engine = atb_engine
        self.weights_keys = weights_keys

    def __str__(self):
        """
        Get string representation of the engine's graph.
        
        Returns:
            str: String representation of the ATB graph
        """
        return self.engine.atb_graph_string

    def set_weights(self, weights):
        """
        Set weights for the engine.
        
        Args:
            weights: Weight parameters to set
        """
        self.engine.set_weights(weights)

    def load_weights(self, weights):
        """
        Load weights into the engine (placeholder implementation).
        
        Args:
            weights: Weight parameters to load
        """
        pass

    def forward(self, inputs, outputs, bind_map=None):
        """
        Execute forward pass through the engine.
        
        Args:
            inputs: Input tensors for the forward pass
            outputs: Output tensors to store results
            bind_map (dict, optional): Binding map for tensor relationships
            
        Returns:
            Result of the forward operation
        """
        if bind_map is None:
            bind_map = {}
        return self.engine.forward(inputs, outputs, bind_map)