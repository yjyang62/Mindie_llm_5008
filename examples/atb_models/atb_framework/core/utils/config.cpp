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
#include "atb_speed/utils/config.h"
#include <string>
#include <iostream>
#include <thread>
#include <atb_speed/utils/str_split.h>
#include "atb_speed/log.h"

namespace atb_speed {
Config::Config()
{
    isConvertNCHWToND_ = true;
    isTorchTensorFormatCast_ = true;
    isUseTilingCopyStream_ = IsEnable("ATB_USE_TILING_COPY_STREAM");
    isLayerInternalTensorReuse_ = true;
    ATB_SPEED_LOG_DEBUG(" \nIsConvertNCHWToND:" << isConvertNCHWToND_
                   << "\nIsTorchTensorFormatCast:" << isTorchTensorFormatCast_
                   << "\nIsLayerInternalTensorReuse:" << isLayerInternalTensorReuse_);
}

Config::~Config() {}

bool Config::IsEnable(const char *env, bool enable)
{
    const char *saveTensor = std::getenv(env);
    if (!saveTensor) {
        return enable;
    }
    return std::string(saveTensor) == "1";
}

bool Config::IsTorchTensorFormatCast() const { return isTorchTensorFormatCast_; };

bool Config::IsConvertNCHWToND() const { return isConvertNCHWToND_; }

bool Config::IsUseTilingCopyStream() const { return isUseTilingCopyStream_; }

bool Config::IsLayerInternalTensorReuse() const
{
    return isLayerInternalTensorReuse_;
}
} // namespace atb_speed