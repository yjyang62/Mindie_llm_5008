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
#include "atb_speed/utils/check_util.h"

#include <iostream>
#include <map>

namespace atb_speed {
// Param Type Size
const size_t PACK_QUANT_TYPE_LENGTH = 2;
const size_t LINEAR_TYPE_LENGTH = 7;
const size_t LINEAR_BIAS_TYPE_LENGTH = 4;
const int MAX_NUM_HIDDEN_LAYER = 1000;

static std::map<std::string, std::pair<std::string, std::string>> g_integerTypeMap = {
    {"int32_t", {"2147483647", "-2147483648"}},
    {"uint32_t", {"4294967295", " "}},
    {"size_t", {"18446744073709551615", " "}},
};


int CheckParamRange(const int &intParam, int min, int max)
{
    if (intParam < min) {
        std::stringstream ss;
        ss << "This param must be a number greater or equal to " << min << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    if (intParam > max) {
        std::stringstream ss;
        ss << "This param must be a number less or equal to " << max << ", please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

int CheckNumHiddenLayersValid(const int &numHiddenLayers)
{
    return CheckParamRange(numHiddenLayers, 1, MAX_NUM_HIDDEN_LAYER);
}

int CheckPositive(const int &intParam)
{
    if (intParam <= 0) {
        std::stringstream ss;
        ss << "This param must be a number greater than 0, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    return intParam;
}

template <typename T>
void CheckLinearParamsSufficient(const std::vector<std::vector<T>> &linearParam, \
    size_t numHiddenLayers, size_t threshold)
{
    if (linearParam.size() != numHiddenLayers) {
        std::stringstream ss;
        ss << "The size of param must be equal to numHiddenLayers, please check." << std::endl;
        throw std::runtime_error(ss.str());
    }
    for (auto item : linearParam) {
        if (item.size() != threshold) {
            std::stringstream ss;
            ss << "The size of vector within param must be equal to " << threshold <<" please check." << std::endl;
            throw std::runtime_error(ss.str());
        }
    }
}

void CheckPackQuantParamsSufficient(const std::vector<std::vector<int>> &packQuantType, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(packQuantType, numHiddenLayers, PACK_QUANT_TYPE_LENGTH);
}

void CheckLinearPackParamsSufficient(const std::vector<std::vector<int>> &linearPackType, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(linearPackType, numHiddenLayers, LINEAR_TYPE_LENGTH);
}

void CheckLinearHasBiasSufficient(const std::vector<std::vector<bool>> &linearHasBias, size_t numHiddenLayers)
{
    CheckLinearParamsSufficient(linearHasBias, numHiddenLayers, LINEAR_BIAS_TYPE_LENGTH);
}

template void CheckLinearParamsSufficient(const std::vector<std::vector<int>> &linearParam, \
    size_t numHiddenLayers, size_t threshold);
template void CheckLinearParamsSufficient(const std::vector<std::vector<bool>> &linearParam, \
    size_t numHiddenLayers, size_t threshold);
} // namespace atb_speed