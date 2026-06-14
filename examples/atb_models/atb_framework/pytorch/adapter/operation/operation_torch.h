/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_OPERATION_TORCH_H
#define ATB_SPEED_OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/operation.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
class OperationTorch : public torch::CustomClassHolder {
public:
    using Task = std::function<int()>;
    using RunTaskFunc = std::function<void(const std::string &taskName, Task task)>;
    explicit OperationTorch(std::string opName);
    ~OperationTorch() override;
    void SetName(std::string name);
    void SetParam(std::string param);
    std::vector<torch::Tensor> ExecuteWithParam(std::vector<torch::Tensor> atInTensors, std::string varaintPackParam);
    void ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                             std::string varaintPackParam);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors);
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    c10::intrusive_ptr<OperationTorch> Clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                        const std::string &varaintPackParam = "");
    std::vector<torch::Tensor> ExecuteImpl(std::vector<torch::Tensor> &atInTensors);
    void BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                          atb::VariantPack &variantPack);
    void RunTask(std::string taskName, std::function<int()> task) const;
    atb::Status ExecutePlan();
    void Clear();
    void ExecutePlanASync();

private:
    std::string opName_;
    uint64_t opId_ = 0;
    std::string nodeId_ = "0";
    std::string name_;
    std::string param_;
    std::unique_ptr<atb::Operation> operation_;
    uint64_t executeCount_ = 0;
    bool isTaskQueueEnable_ = false;
    std::unique_ptr<atb_speed::HostTensorBinder> hostTensorBinder_;
    std::shared_ptr<atb::Context> context_;
    atb::VariantPack variantPack_;
    uint64_t workspaceSize_ = 0;
    void *workspace_ = nullptr;
    RunTaskFunc runTaskFunc_ = nullptr;
};
}
#endif