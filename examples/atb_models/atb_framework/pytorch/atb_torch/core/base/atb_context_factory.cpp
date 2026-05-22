/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "atb_context_factory.h"
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
AtbContextFactory &AtbContextFactory::Instance()
{
    static AtbContextFactory instance;
    return instance;
}

std::shared_ptr<atb::Context> AtbContextFactory::GetAtbContext(void *stream)
{
    if (atbContext_) {
        ATB_SPEED_LOG_DEBUG("AtbContextFactory return localContext");
        return atbContext_;
    }

    ATB_SPEED_LOG_DEBUG("AtbContextFactory create atb::Context start");
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("AtbContextFactory create atb::Context fail");
    }
    if (context) {
        context->SetExecuteStream(stream);
        if (Config::Instance().IsUseTilingCopyStream()) {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory use tiling copy stream");
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory not use tiling copy stream");
        }
    }
    this->atbContext_ = std::shared_ptr<atb::Context>(
        context, [](atb::Context* context) {atb::DestroyContext(context);});

    return atbContext_;
}

void AtbContextFactory::FreeAtbContext()
{
    ATB_SPEED_LOG_DEBUG("AtbContextFactory FreeAtbContext start");
    if (!atbContext_) {
        return;
    }

    ATB_SPEED_LOG_DEBUG("AtbContextFactory localContext use_count: " << atbContext_.use_count());
    if (atbContext_.use_count() != 1) {
        return;
    }
    ATB_SPEED_LOG_DEBUG("AtbContextFactory localContext reset");
    atbContext_.reset();
}
} // namespace atb_torch
