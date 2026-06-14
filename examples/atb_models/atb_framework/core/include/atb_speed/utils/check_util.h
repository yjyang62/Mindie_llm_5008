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

#ifndef ATB_SPEED_UTILS_CHECK_H
#define ATB_SPEED_UTILS_CHECK_H
#include <vector>
#include <cstddef>
#include <sstream>
#include <limits>

#include "nlohmann/json.hpp"

namespace atb_speed {

using Json = nlohmann::json;

template<typename T, typename U>
typename std::common_type<T, U>::type CheckIntMulOverFlow(const T a, const U b)
{
    if (std::is_signed<T>::value != std::is_signed<U>::value) {
        throw std::runtime_error("Multiplication between signed and unsigned integer not supported, it's not safe");
    }
    using PromotedType = typename std::common_type<T, U>::type;
    if (a == 0 || b == 0) {
        return 0;
    }

    PromotedType pa = static_cast<PromotedType>(a);
    PromotedType pb = static_cast<PromotedType>(b);

    if constexpr (std::is_signed<PromotedType>::value) {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        const PromotedType minVal = std::numeric_limits<PromotedType>::min();
        if (pa > 0 && pb > 0) {
            if (pa > maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < 0 && pb < 0) {
            if (pa < maxVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa > 0 && pb < 0) {
            if (pa > minVal / pb) {
                throw std::overflow_error("Integer overflow detected.");
            }
        } else if (pa < minVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    } else {
        const PromotedType maxVal = std::numeric_limits<PromotedType>::max();
        if (pa > maxVal / pb) {
            throw std::overflow_error("Integer overflow detected.");
        }
    }
    return pa * pb;
}
int CheckParamRange(const int &intParam, int min, int max);
int CheckNumHiddenLayersValid(const int &numHiddenLayers);
int CheckPositive(const int &intParam);
template <typename T>
void CheckLinearParamsSufficient(const std::vector<std::vector<T>> &linearParam, \
    size_t numHiddenLayers, size_t threshold);
void CheckPackQuantParamsSufficient(const std::vector<std::vector<int>> &packQuantType, size_t numHiddenLayers);
void CheckLinearPackParamsSufficient(const std::vector<std::vector<int>> &linearPackType, size_t numHiddenLayers);
void CheckLinearHasBiasSufficient(const std::vector<std::vector<bool>> &linearHasBias, size_t numHiddenLayers);
} // namespace atb_speed
#endif