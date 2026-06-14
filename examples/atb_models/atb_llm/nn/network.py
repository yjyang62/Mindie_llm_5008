# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import os
import sys

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger

path = ENV.atb_speed_home_path
sys.path.append(os.path.join(path, 'lib'))
import _libatb_torch as atb
from .fusion_pass_manager import FusionPassManager


class Engine:
    def __init__(self, atb_engine, weights_keys) -> None:
        self.engine = atb_engine
        self.weights_keys = weights_keys

    def __str__(self):
        return self.engine.atb_graph_string

    def set_weights(self, weights: dict):
        real_weights = {}
        for k, v in weights.items():
            if k in self.weights_keys:
                real_weights[k] = v
        self.engine.set_weights(real_weights)

    def set_kv_caches(self, kv_caches: dict):
        self.engine.set_weights(kv_caches)

    def load_weights(self, weights):
        pass

    def forward(self, inputs, outputs, bind_map=None):
        if bind_map is None:
            bind_map = {}
        return self.engine.forward(inputs, outputs, bind_map)


class Network:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.cut_points = []
        self.in_tensors = []
        self.out_tensors = []
        self.tmp_name = "?"
        self.cut_step = 8
        self.build_status = False
        self.weights_keys = []

    def __str__(self):
        self._auto_generate_internal_keys()
        return self.to_string()

    @classmethod
    def _push_tensor(cls, tensor, tensor_list):
        for tsr in tensor_list:
            if tsr.name == tensor.name:
                return
        tensor_list.append(tensor)

    def mark_output(self, *args, **kwargs):
        if len(args) == 2:
            args[0].name = args[1]
        self._push_tensor(args[0], self.out_tensors)

    def push_node(self, node):
        self.nodes.append(node)
        for tensor in node.in_tensors:
            if tensor.name != self.tmp_name and tensor.name != f"{self.tmp_name}_view" and tensor.view_father is None:
                self._push_tensor(tensor, self.in_tensors)
            elif tensor.view_father is not None and tensor.name != f"{self.tmp_name}_view":
                self._push_tensor(tensor.view_father, self.in_tensors)

    def push_weight_key(self, key: str):
        if key not in self.weights_keys:
            self.weights_keys.append(key)

    def cut(self):
        self.cut_points.append(len(self.nodes) - 1)

    def to_string(self):
        rt_string = ""
        for i, node in enumerate(self.nodes):
            str_rt = f"{i}:{node.op_type}:{node.op_param}:in -> "
            for tensor in node.in_tensors:
                str_rt += f"{tensor.name},"
                if tensor.view_father is not None:
                    str_rt += f"reshape: {tensor.view_father.name} -> {tensor.name}\n"
            str_rt += "out -> "
            for tensor in node.out_tensors:
                str_rt += f"{tensor.name},"
            rt_string += str_rt
            rt_string += "\n"

        str_rt = "net wort inputs: "
        for tensor in self.in_tensors:
            str_rt += f"{tensor.name},"
        rt_string += str_rt
        rt_string += "\n"

        str_rt = "net wort outputs: "
        for tensor in self.out_tensors:
            str_rt += f"{tensor.name},"
        rt_string += str_rt
        rt_string += "\n"

        str_rt = "net work cut points: "
        for point in self.cut_points:
            str_rt += f"{point},"
        return rt_string + "\n"

    def build_engine(self, del_fpass_keys: list[str] = None):
        engine = None
        self._auto_generate_internal_keys()
        fusion_pass_manager = FusionPassManager.get_instance(del_fpass_keys)
        fusion_pass_manager.fuse_network(self)
        cut_num = len(self.cut_points)
        if cut_num == 0:
            self._auto_cut()
        else:
            if self.cut_points[cut_num - 1] != len(self.nodes) - 1:
                self.cut_points.append(len(self.nodes) - 1)
        converter = Converter()
        engine = converter.network_to_atbgraph(self)
        self.build_status = True
        return Engine(engine, self.weights_keys)

    def _auto_cut(self):
        if len(self.nodes) <= self.cut_step:
            self.cut_points.append(len(self.nodes) - 1)
            return
        cut_point = self.cut_step - 1
        for _ in range(len(self.nodes)):
            if cut_point < len(self.nodes):
                self.cut_points.append(cut_point)
                cut_point += self.cut_step
            else:
                break
        cut_num = len(self.cut_points)
        if cut_num > 0 and self.cut_points[cut_num - 1] != len(self.nodes) - 1:
            self.cut_points.append(len(self.nodes) - 1)

    def _is_output(self, tensor):
        for tsr in self.out_tensors:
            if tsr == tensor:
                return True
        return False

    def _auto_generate_internal_keys(self):
        for i, node in enumerate(self.nodes):
            for j, tensor in enumerate(node.in_tensors):
                if tensor.name == self.tmp_name:
                    tensor.name = f"in{j}@" + f"node{i}_{node.op_type}"
                    if tensor.view_tensor is not None:
                        tensor.view_tensor.name = f"{tensor.name}_view"
            for k, tensor in enumerate(node.out_tensors):
                if tensor.name == self.tmp_name:
                    tensor.name = f"out{k}@" + f"node{i}_{node.op_type}"
                    if tensor.view_tensor is not None:
                        tensor.view_tensor.name = f"{tensor.name}_view"


class AtbGraph(atb.GraphOperation):
    def __init__(self, network: Network, cut_point_idx, sub_ops, post_fix):
        super().__init__(f"{network.name}_{post_fix}")
        self.sub_ops = []
        self.atb_graph_string = ""

        self.in_tensors = []
        self.out_tensors = []
        if cut_point_idx is not None:
            self._init_sub_graph(network, cut_point_idx)
        elif sub_ops is not None:
            self._init_final_graph(network, sub_ops)

        self.atb_graph_string = self.atb_graph_string + "build start" + "\n"
        logger.debug(f"atb graph string is {self.atb_graph_string}")
        self.build()
        self.atb_graph_string = self.atb_graph_string + "build success" + "\n"

    def _init_sub_graph(self, network, cut_point_idx):
        start_node = 0
        if cut_point_idx > 0:
            start_node = network.cut_points[cut_point_idx - 1] + 1
        end_node = network.cut_points[cut_point_idx]
        ops = network.nodes[start_node:end_node + 1]
        self._init_sub_graph_inputs_outputs(network, cut_point_idx, ops)
        graph_inputs = "inputs: "
        for tensor in self.in_tensors:
            graph_inputs += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + graph_inputs + "\n"
        graph_outputs = "outputs: "
        for tensor in self.out_tensors:
            graph_outputs += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + graph_outputs + "\n"
        self.add_input_output(input=[tensor.name for tensor in self.in_tensors],
                                output=[tensor.name for tensor in self.out_tensors])

        for idx, op in enumerate(self.sub_ops):
            graph_info = f"{idx}:{ops[idx].op_type}:{ops[idx].op_param}: in -> "
            for tensor in ops[idx].in_tensors:
                if tensor.view_father is not None:
                    self.atb_graph_string = self.atb_graph_string + \
                        f"reshape:{tensor.view_father.name}->{tensor.name}" + "\n"
                    self.add_reshape(tensor.view_father.name, tensor.name, tensor.reshape_func)
            ins = []
            for tensor in ops[idx].in_tensors:
                graph_info += f"{tensor.name},"
                ins.append(tensor.name)
            outs = []
            graph_info += " : out -> "
            for tensor in ops[idx].out_tensors:
                graph_info += f"{tensor.name},"
                outs.append(tensor.name)
            self.add_operation(op, ins, outs)
            self.atb_graph_string = self.atb_graph_string + graph_info + "\n"

    def _init_sub_graph_inputs_outputs(self, network, cut_point_idx, ops):
        for _, node in enumerate(ops):
            self.sub_ops.append(atb.BaseOperation(
                op_type=node.op_type,
                op_param=json.dumps(node.op_param),
                op_name=node.op_type
            ))
            for in_tensor in node.in_tensors:
                if (in_tensor.view_father is not None and in_tensor.view_father not in self.in_tensors and
                    self._is_subgraph_intensor(in_tensor.view_father, network, cut_point_idx)):
                    self.in_tensors.append(in_tensor.view_father)
                if (in_tensor not in self.in_tensors and
                    self._is_subgraph_intensor(in_tensor, network, cut_point_idx)):
                    self.in_tensors.append(in_tensor)

            for out_tensor in node.out_tensors:
                if (out_tensor not in self.out_tensors and
                    self._is_subgraph_outtensor(out_tensor, network, cut_point_idx)):
                    self.out_tensors.append(out_tensor)
                if (out_tensor.view_tensor is not None and
                    self._is_subgraph_outtensor(out_tensor.view_tensor, network, cut_point_idx) and
                    out_tensor not in self.out_tensors):
                    self.out_tensors.append(out_tensor)

    def _init_final_graph(self, network, sub_ops):
        self.sub_ops = sub_ops
        self.in_tensors = network.in_tensors
        self.out_tensors = network.out_tensors
        str_rt = "inputs: "
        for tensor in self.in_tensors:
            str_rt += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + str_rt + "\n"
        str_rt = "outputs: "
        for tensor in self.out_tensors:
            str_rt += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + str_rt + "\n"
        self.add_input_output(input=[tensor.name for tensor in self.in_tensors],
                                output=[tensor.name for tensor in self.out_tensors])
        for _, op in enumerate(self.sub_ops):
            for tensor in op.in_tensors:
                if tensor.view_father is not None:
                    self.atb_graph_string = self.atb_graph_string + \
                        f"reshape:{tensor.view_father.name}->{tensor.name}" + "\n"
                    self.add_reshape(tensor.view_father.name, tensor.name, tensor.reshape_func)
            self.add_operation(op,
                [tensor.name for tensor in op.in_tensors],
                [tensor.name for tensor in op.out_tensors])

            self.atb_graph_string = "\n" + self.atb_graph_string + op.atb_graph_string + "\n"

        self.execute_as_single = False

    def _is_subgraph_intensor(self, tensor, network, cut_point_idx):
        for tsr in network.in_tensors:
            if tensor == tsr:
                return True
        if cut_point_idx > 0:
            end_node_idx = network.cut_points[cut_point_idx - 1]
            for node in network.nodes[:end_node_idx + 1]:
                for tsr in node.out_tensors:
                    if tsr == tensor:
                        return True
        return False

    def _is_subgraph_outtensor(self, tensor, network, cut_point_idx):
        if tensor in self.in_tensors:
            return False
        for tsr in network.out_tensors:
            if tensor == tsr:
                return True
        if cut_point_idx < len(network.cut_points) - 1:
            start_node_idx = network.cut_points[cut_point_idx]
            for node in network.nodes[start_node_idx + 1:]:
                for tsr in node.in_tensors:
                    if tsr == tensor:
                        return True
                for tsr in node.out_tensors:
                    if tsr == tensor:
                        return True
        return False


class Converter:
    @classmethod
    def network_to_atbgraph(cls, network: Network) -> AtbGraph:
        sub_ops = []
        count = 0
        for cut_point_idx in range(len(network.cut_points)):
            sub_ops.append(AtbGraph(network, cut_point_idx, None, str(count)))
            count += 1
        if count == 1:
            return sub_ops[0]
        atb_graph = AtbGraph(network, None, sub_ops, network.name)
        return atb_graph