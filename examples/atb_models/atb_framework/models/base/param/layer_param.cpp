/*
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
#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace base {

void LayerParam::PrintParam()
{
    Param::PrintParam();
    std::stringstream ss;
    ss << "Base Layer Param:"
       << ", tensorParallelInfo.rank:" << this->tensorParallelInfo.rank
       << ", tensorParallelInfo.worldSize:" << this->tensorParallelInfo.worldSize
       << ", tensorParallelInfo.backend:" << this->tensorParallelInfo.backend
       << ", tensorParallelInfo.rankTableFile:" << this->tensorParallelInfo.rankTableFile
       << ", tensorParallelInfo.quantType:" << this->tensorParallelInfo.quantType
       << ", tensorParallelInfo.outDataType:" << this->tensorParallelInfo.outDataType;
    for (size_t i = 0; i < packQuantType.size(); ++i) {
        ss << "packQuantType[" << i << "]:" << packQuantType.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearQuantType.size(); ++i) {
        ss << "linearQuantType[" << i << "]:" << linearQuantType.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearHasBias.size(); ++i) {
        ss << "linearHasBias[" << i << "]:" << linearHasBias.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearTransposeType.size(); ++i) {
        ss << "linearTransposeType[" << i << "]:" << linearTransposeType.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearDescs.size(); ++i) {
        ss << "linearDescs[" << i << "]:" << linearDescs.at(i) << std::endl;
    }
    for (size_t i = 0; i < isAntiOutlier.size(); ++i) {
        ss << "isAntiOutlier[" << i << "]:" << isAntiOutlier.at(i) << std::endl;
    }
    ATB_SPEED_LOG_DEBUG(ss.str());
}

void LayerParam::CheckParam()
{
    if (this->hiddenSizePerAttentionHead == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param hiddenSizePerAttentionHead is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
}
} // namespace base
} // namespace atb_speed