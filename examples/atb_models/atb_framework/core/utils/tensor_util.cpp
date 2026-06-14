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
#include "atb_speed/utils/tensor_util.h"
#include <sstream>
#include <fstream>
#include <string>
#include <acl/acl.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"

namespace atb_speed {
std::string TensorUtil::TensorToString(const atb::Tensor &tensor)
{
    std::stringstream ss;
    ss << TensorDescToString(tensor.desc) << ", deviceData:" << tensor.deviceData << ", hostData:" << tensor.hostData
       << ", dataSize:" << tensor.dataSize;
    return ss.str();
}

std::string TensorUtil::TensorDescToString(const atb::TensorDesc &tensorDesc)
{
    std::stringstream ss;
    ss << "dtype: " << tensorDesc.dtype << ", format: " << tensorDesc.format << ", shape:[";
    for (size_t i = 0; i < tensorDesc.shape.dimNum; ++i) {
        if (i == 0) {
            ss << tensorDesc.shape.dims[i];
        } else {
            ss << ", " << tensorDesc.shape.dims[i];
        }
    }
    ss << "]";

    return ss.str();
}

uint64_t TensorUtil::GetTensorNumel(const atb::Tensor &tensor) { return GetTensorNumel(tensor.desc); }

uint64_t TensorUtil::GetTensorNumel(const atb::TensorDesc &tensorDesc)
{
    if (tensorDesc.shape.dimNum == 0) {
        return 0;
    }

    int64_t elementCount = 1;
    for (size_t i = 0; i < tensorDesc.shape.dimNum; i++) {
        elementCount = CheckIntMulOverFlow(elementCount, tensorDesc.shape.dims[i]);
    }

    return elementCount;
}

bool TensorUtil::TensorDescEqual(const atb::TensorDesc &tensorDescA, const atb::TensorDesc &tensorDescB)
{
    if (tensorDescA.dtype == tensorDescB.dtype && tensorDescA.format == tensorDescB.format &&
        tensorDescA.shape.dimNum == tensorDescB.shape.dimNum) {
        for (size_t i = 0; i < tensorDescA.shape.dimNum; i++) {
            if (tensorDescA.shape.dims[i] != tensorDescB.shape.dims[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}
} // namespace atb_speed