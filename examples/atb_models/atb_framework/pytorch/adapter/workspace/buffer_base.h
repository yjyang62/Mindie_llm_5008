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

#ifndef ATB_SPEED_CONTEXT_BUFFER_BASE_H
#define ATB_SPEED_CONTEXT_BUFFER_BASE_H
#include <cstdint>

namespace atb_speed {
class BufferBase {
public:
    BufferBase();
    virtual ~BufferBase();
    virtual void *GetBuffer(uint64_t bufferSize) = 0;
    virtual void ClearBuffer() = 0;
    virtual int32_t GetCachedNum() = 0;
};
} // namespace atb_speed
#endif