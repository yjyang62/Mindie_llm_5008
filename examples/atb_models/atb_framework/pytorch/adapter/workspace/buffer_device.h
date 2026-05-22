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

#ifndef ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#define ATB_SPEED_CONTEXT_BUFFER_DEVICE_H
#include <torch/torch.h>
#include "buffer_base.h"

namespace atb_speed {
class BufferDevice : public BufferBase {
public:
    explicit BufferDevice(uint64_t bufferSize);
    ~BufferDevice() override;
    void *GetBuffer(uint64_t bufferSize) override;
    void ClearBuffer() override;
    int32_t GetCachedNum() override;
private:
    torch::Tensor CreateAtTensor(const uint64_t bufferSize) const;

private:
    void *buffer_ = nullptr;
    uint64_t bufferSize_ = 0;
    torch::Tensor atTensor_;
    std::vector<torch::Tensor> cachedTensor_;
};
} // namespace atb_speed
#endif