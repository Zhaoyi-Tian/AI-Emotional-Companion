# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import os
import sys
import logging
import json
from collections import OrderedDict
from enum import Enum
from typing import Callable
import torch
import torch_npu
import torch.nn as nn
from atb_llm.utils.env import ENV

path = ENV.atb_speed_home_path
sys.path.append(os.path.join(path, 'lib'))
import _libatb_torch as atb


class Dtype(Enum):
    FLOAT = 1
    FLOAT16 = 2
    INT8 = 3
    INT32 = 4
    UINT8 = 5
    INT16 = 6
    UINT16 = 7
    UINT32 = 8
    INT64 = 9
    UINT64 = 10
    DOUBLE = 11
    BOOL = 12
    BF16 = 13
    INT4 = 14
    UINT1 = 15

dtype_map = {
    Dtype.FLOAT: "ACL_FLOAT",
    Dtype.FLOAT16: "ACL_FLOAT16",
    Dtype.INT8: "ACL_INT8",
    Dtype.INT32: "ACL_INT32",
    Dtype.UINT8: "ACL_UINT8",
    Dtype.INT16: "ACL_INT16",
    Dtype.UINT16: "ACL_UINT16",
    Dtype.UINT32: "ACL_UINT32",
    Dtype.INT64: "ACL_INT64",
    Dtype.UINT64: "ACL_UINT64",
    Dtype.DOUBLE: "ACL_DOUBLE",
    Dtype.BOOL: "ACL_BOOL",
    Dtype.BF16: "ACL_BF16",
    Dtype.INT4: "ACL_INT4",
    Dtype.UINT1: "ACL_UINT1",
}


class Node:
    def __init__(self, op_type, op_param, in_tensors, out_tensors):
        self.op_type = op_type
        self.op_param = op_param
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors


class Tensor:
    def __init__(self, name="?"):
        self.name = name
        self.reshape_func = None
        self.shape = None
        self.view_tensor = None
        self.view_father = None

    def __eq__(self, other):
        return self.name == other.name

    def __gt__(self, other):
        out = Tensor()
        param = {
            "elewiseType":"ELEWISE_GREATER",
        }
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out
    
    def __lt__(self, other):
        out = Tensor()
        param = {
            "elewiseType":"ELEWISE_LESS",
        }
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __add__(self, other):
        out = Tensor()
        node = Node("Elewise", {'elewiseType':'ELEWISE_ADD'}, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __sub__(self, other):
        out = Tensor()
        param = {
            "elewiseType":"ELEWISE_SUB",
        }
        node = Node('Elewise', param, [self, other], [out])
        get_default_net().push_node(node)
        return out

    def __mul__(self, mul_target):
        out = Tensor()
        if isinstance(mul_target, Tensor):
            param = {
                'elewiseType': 'ELEWISE_MUL'
            }
            node = Node('Elewise', param, [self, mul_target], [out])
            get_default_net().push_node(node)
        elif isinstance(mul_target, int) or isinstance(mul_target, float):
            param = {
                'elewiseType': 'ELEWISE_MULS',
                'mulsParam':{
                    'varAttr': float(mul_target)
                }
            }
            node = Node('Elewise', param, [self], [out])
            get_default_net().push_node(node)
        else:
            raise ValueError("* operator only support Tensor * Tensor, Tensor * float")
        return out

    def __truediv__(self, div_target):
        out = Tensor()
        if isinstance(div_target, Tensor):
            param = {
                'elewiseType': 'ELEWISE_REALDIV'
            }
            node = Node('Elewise', param, [self, div_target], [out])
            get_default_net().push_node(node)
        elif isinstance(div_target, int) or isinstance(div_target, float):
            param = {
                'elewiseType': 'ELEWISE_MULS',
                'mulsParam':{
                    'varAttr': float(1 / div_target)
                }
            }
            node = Node('Elewise', param, [self], [out])
            get_default_net().push_node(node)
        else:
            raise ValueError("/ operator only support Tensor / Tensor, Tensor / float")
        return out

    def __invert__(self):
        out = Tensor()
        param = {
            "elewiseType":"ELEWISE_LOGICAL_NOT",
        }
        node = Node('Elewise', param, [self], [out])
        get_default_net().push_node(node)
        return out

    def __getitem__(self, slices):
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
        self.name = tensor.name

    def to(self, dtype: Dtype):
        out = Tensor()
        param = {
            "elewiseType": "ELEWISE_CAST",
            "outTensorType": dtype_map[dtype]
        }
        node = Node("Elewise", param, [self], [out])
        get_default_net().push_node(node)
        return out

    def permute(self, perm: list[int]):
        out = Tensor()
        node = Node("Transpose", {'perm':perm}, [self], [out])
        get_default_net().push_node(node)
        return out

    def reshape(self, reshape_func):
        self.view_tensor = Tensor(f"{self.name}_view")
        self.view_tensor.view_father = self
        self.view_tensor.reshape_func = reshape_func
        return self.view_tensor


class Engine:
    def __init__(self, atb_engine, weights_keys) -> None:
        self.engine = atb_engine
        self.weights_keys = weights_keys

    def __str__(self):
        return self.engine.atb_graph_string

    def set_weights(self, weights):
        self.engine.set_weights(weights)

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

    def build_engine(self):
        engine = None
        self._auto_generate_internal_keys()
        self._auto_cut()
        converter = Converter()
        engine = converter.network_to_atbgraph(self)
        self.build_status = True
        return Engine(engine, self.weights_keys)

    def _auto_cut(self):
        if len(self.nodes) <= self.cut_step:
            self.cut_points.append(len(self.nodes) - 1)
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


class AtbGraph(atb._GraphOperation):
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
        self.build()
        self.atb_graph_string = self.atb_graph_string + "build success" + "\n"

    def _init_sub_graph(self, network, cut_point_idx):
        start_node = 0
        if cut_point_idx > 0:
            start_node = network.cut_points[cut_point_idx - 1] + 1
        end_node = network.cut_points[cut_point_idx]
        ops = network.nodes[start_node:end_node + 1]
        self._init_sub_graph_inputs_outputs(network, cut_point_idx, ops)
        graph_inputs = f"inputs: "
        for tensor in self.in_tensors:
            graph_inputs += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + graph_inputs + "\n"
        graph_outputs = f"outputs: "
        for tensor in self.out_tensors:
            graph_outputs += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + graph_outputs + "\n"
        self.add_input_output(input=[tensor.name for tensor in self.in_tensors],
                                output=[tensor.name for tensor in self.out_tensors])

        for idx, op in enumerate(self.sub_ops):
            graph_info = f"{idx}:{ops[idx].op_type}: in -> "
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
            self.sub_ops.append(atb._BaseOperation(
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
        str_rt = f"inputs: "
        for tensor in self.in_tensors:
            str_rt += f"{tensor.name},"
        self.atb_graph_string = self.atb_graph_string + str_rt + "\n"
        str_rt = f"outputs: "
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


default_network = Network("default_network")


def get_default_net():
    global default_network
    if default_network.build_status is True:
        del default_network
        default_network = Network("default_network")
    return default_network