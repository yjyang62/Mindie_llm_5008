/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATB_SPEED_UTILS_TENSOR_UTIL_H
#define ATB_SPEED_UTILS_TENSOR_UTIL_H
#include <string>
#include <atb/types.h>

namespace atb_speed {
class TensorUtil {
public:
    static std::string TensorToString(const atb::Tensor &tensor);
    static std::string TensorDescToString(const atb::TensorDesc &tensorDesc);
    static uint64_t GetTensorNumel(const atb::Tensor &tensor);
    static uint64_t GetTensorNumel(const atb::TensorDesc &tensorDesc);
    static bool TensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB);
};
} // namespace atb_speed
#endif