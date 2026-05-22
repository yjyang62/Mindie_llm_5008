/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "utils/status.h"
#include "infer_tokenizer.h"

#ifndef OCK_LLM_WORKSPACE_MOCK_UTIL_H
#define OCK_LLM_WORKSPACE_MOCK_UTIL_H

#define MOCKER_CPP_OVERLOAD_EQ(TYPE) \
inline bool operator==(TYPE, const TYPE) {       \
    return true;                                 \
}

using namespace mindie_llm;

namespace mindie_llm {

extern "C++" {
inline bool operator==(const mindie_llm::ModelDeployConfig& lhs, const mindie_llm::ModelDeployConfig& rhs)
{
    return true;
}

inline bool operator==(const mindie_llm::LoraConfig& lhs, const mindie_llm::LoraConfig& rhs)
{
    return true;
}

inline Status MockTokenizerEncodeSuccess(TokenizerProcessPool *pool, const std::string &prompt,
    std::vector<int64_t> &tokenIds, HeadFlag flag, uint64_t &timestamp)
{
    tokenIds = {1, 2, 3};
    return Status(mindie_llm::Error::Code::OK, "Success");
}

inline Status MockTokenizerEncodeFail(TokenizerProcessPool *pool, const std::string &prompt,
    std::vector<int64_t> &tokenIds, HeadFlag flag, uint64_t &timestamp)
{
    return Status(mindie_llm::Error::Code::ERROR, "Failed");
}

inline Status MockTokenizerDecodeSuccess(TokenizerProcessPool *pool, std::vector<int64_t> &tokenIds,
    std::string &output, const uint64_t &timestamp)
{
    tokenIds = {1, 2, 3};
    return Status(mindie_llm::Error::Code::OK, "Success");
}

inline Status MockTokenizerDecodeFail(TokenizerProcessPool *pool, std::vector<int64_t> &tokenIds,
    std::string &output, const uint64_t &timestamp)
{
    return Status(mindie_llm::Error::Code::ERROR, "Failed");
}

inline Status MockTokenizerDecodeOneSuccess(TokenizerProcessPool *pool, std::vector<int64_t> &tokenIds,
    std::string &output, uint32_t prevDecodeIndex, uint32_t currentDecodeIndex, const uint64_t &timestamp,
    const bool &useToolsCall = false, const bool &skipSpecialTokens = true, const bool requestEndFlag = false,
    const DetokenizeExtraInfo &detokenizeStatus = {})
{
    tokenIds = {1, 2, 3};
    return Status(mindie_llm::Error::Code::OK, "Success");
}

inline Status MockTokenizerDecodeOneFail(TokenizerProcessPool *pool, std::vector<int64_t> &tokenIds,
    std::string &output, uint32_t prevDecodeIndex, uint32_t currentDecodeIndex, const uint64_t &timestamp,
    const bool &useToolsCall = false, const bool &skipSpecialTokens = true, const bool requestEndFlag = false,
    const DetokenizeExtraInfo &detokenizeStatus = {})
{
    return Status(mindie_llm::Error::Code::ERROR, "Failed");
}

inline Status MockTokenizerResultsEmpty(TokenizerProcessPool *pool, const std::string &prompt,
    std::vector<int64_t> &tokenIds, HeadFlag flag, uint64_t &timestamp)
{
    tokenIds = {};
    return Status(mindie_llm::Error::Code::OK, "Success");
}
}

} // namespace mindie_llm
#endif //OCK_LLM_WORKSPACE_MOCK_UTIL_H
