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
#include "atb_speed/utils/str_split.h"
#include <sstream>

namespace atb_speed {
constexpr int OPGRAPH_NAME_MAX_LENG = 128;

std::string GetFuncNameAndNameSpace(const std::string &inputStr)
{
    int spaceInd = 0;
    int leftBracketInd = 0;
    std::string extractStr;
    int inputStrLen = static_cast<int>(inputStr.size());
    for (int i = 0; i < inputStrLen; i++) {
        if (inputStr.at(i) == ' ') {
            spaceInd = i;
        } else if (inputStr.at(i) == '(') {
            leftBracketInd = i;
            break;
        }
    }
    if (spaceInd >= 0 && (leftBracketInd - spaceInd) > 0) {
        int len;
        if (leftBracketInd - (spaceInd + 1) > OPGRAPH_NAME_MAX_LENG) {
            len = OPGRAPH_NAME_MAX_LENG;
        } else {
            len = leftBracketInd - (spaceInd + 1);
        }
        extractStr = inputStr.substr(spaceInd + 1, len);
    } else {
        extractStr = inputStr;
    }

    for (char &i : extractStr) {
        if (!isalnum(i) && i != '_') {
            i = '_';
        }
    }
    return extractStr;
}

} // namespace atb_speed