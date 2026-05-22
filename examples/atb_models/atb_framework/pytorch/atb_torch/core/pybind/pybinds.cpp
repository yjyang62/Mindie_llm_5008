/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#pragma GCC diagnostic push
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma GCC diagnostic pop
#include "base_operation.h"
#include "graph_operation.h"
#include "atb_comm_manager.h"

namespace atb_torch {
PYBIND11_MODULE(_libatb_torch, m)
{
    pybind11::class_<atb_torch::Operation>(m, "Operation")
        .def_property("op_name", &atb_torch::Operation::GetOpName, &atb_torch::Operation::SetOpName)
        .def_property_readonly("input_names", &atb_torch::Operation::GetInputNames)
        .def_property_readonly("output_names", &atb_torch::Operation::GetOutputNames)
        .def("pre_input", &atb_torch::Operation::PreInputTensor)
        .def("pre_output", &atb_torch::Operation::PreOutputTensor)
        .def("pre_bind", &atb_torch::Operation::PreBindTensor)
        .def("set_weights", &atb_torch::Operation::SetWeights, pybind11::arg("weights") = atb_torch::TorchTensorMap())
        .def("forward", &atb_torch::Operation::Forward, pybind11::arg("input"),
             pybind11::arg("output") = atb_torch::TorchTensorMap(),
             pybind11::arg("bind") = atb_torch::TorchTensorMap());

    pybind11::class_<atb_torch::BaseOperation, atb_torch::Operation>(m, "BaseOperation")
        .def(pybind11::init<std::string, std::string, std::string>(), pybind11::arg("op_type"),
             pybind11::arg("op_param"), pybind11::arg("op_name"))
        .def_property_readonly("op_type", &atb_torch::BaseOperation::GetOpType)
        .def_property_readonly("op_param", &atb_torch::BaseOperation::GetOpParam);

    pybind11::class_<atb_torch::GraphOperation, atb_torch::Operation>(m, "GraphOperation")
        .def(pybind11::init<std::string>(), pybind11::arg("op_name") = "")
        .def("add_input_output", &atb_torch::GraphOperation::AddInputOutput, pybind11::arg("input"),
             pybind11::arg("output"))
        .def("add_operation", &atb_torch::GraphOperation::AddOperation, pybind11::arg("operation"),
             pybind11::arg("input"), pybind11::arg("output"))
        .def("add_reshape", &atb_torch::GraphOperation::AddReshape, pybind11::arg("input"), pybind11::arg("ouput"),
             pybind11::arg("func"))
        .def("build", &atb_torch::GraphOperation::Build)
        .def_property("execute_as_single", &atb_torch::GraphOperation::GetExecuteAsSingle,
                      &atb_torch::GraphOperation::SetExecuteAsSingle);
     
    m.def("init_process_group", &atb_torch::initProcessGroup);
    m.def("new_group", &atb_torch::newGroup);
    m.def("get_backend", &atb_torch::getBackend);
    m.def("is_process_group_initialized", &atb_torch::isPGInitialized);
}
}