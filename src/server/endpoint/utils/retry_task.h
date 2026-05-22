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

#ifndef RETRY_TASK_H
#define RETRY_TASK_H

#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/bind/bind.hpp>
#include <functional>
#include <cmath>
#include <atomic>
#include <map>
#include <thread>
#include <chrono>
#include "log.h"
#include "infer_instances.h"

namespace mindie_llm {
class RetryTask {
public:
    using RetryFunc = std::function<void(GlobalIpInfo &)>;

    RetryTask(RetryFunc retryFunc, std::shared_ptr<boost::asio::io_context> ioContext, GlobalIpInfo &globalIpInfo,
        uint32_t maxRetryInterval = 600U)
        : retryFunc_(std::move(retryFunc)), ioContext_(ioContext),
        steadyTimer_(*ioContext), globalIpInfo_(globalIpInfo), maxRetryTime_(maxRetryInterval) {};

    ~RetryTask() = default;

    void Execute()
    {
        // avoid being destroyed during callback
        auto self = std::shared_ptr<RetryTask>(this, [](RetryTask*) {});
        boost::asio::post(*ioContext_, [self, this]() {
            bool retryTaskExecute = true;
            while (retryTaskExecute) {
                if (Exit()) {
                    retryTaskExecute = false;
                    break;
                }
                if (retries_ == 1) {
                    globalIpInfo_.ResetRetryState();
                }
                globalIpInfo_.retryLinkIpInfo.clear(); // 每次执行前确保retryLinkIpInfo是空的
                retryFunc_(globalIpInfo_);
                auto waitTime = GetRetryTime();
                ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "Wait for " << waitTime << " seconds.");
                retries_ += 1U;
                std::this_thread::sleep_for(std::chrono::seconds(waitTime));
            }
        });
        ioContext_->run();
    }

private:
    RetryFunc retryFunc_;
    std::shared_ptr<boost::asio::io_context> ioContext_;
    boost::asio::steady_timer steadyTimer_;
    GlobalIpInfo globalIpInfo_;

    uint32_t retries_{0U};
    // default max retry time is set to 30(minutes)
    uint32_t maxRetryTime_{10 * 60U};

    inline bool NoPendingTask() const
    {
        return globalIpInfo_.retryLinkIpInfo.empty();
    }

    inline uint32_t GetRetryTime()
    {
        static const uint32_t MAX_RETRIES = 13;
        uint32_t retries = std::min(retries_, MAX_RETRIES);
        return std::min(static_cast<uint32_t>(pow(5U, retries)), maxRetryTime_);
    }

    bool Exit()
    {
        // Only after first role assignment needs to decide whether to retry
        if ((retries_ != 0) && NoPendingTask()) {
            if (GetInferInstance()->GetPDRoleStatus() == PDRoleStatus::SWITCHING) {
                GetInferInstance()->SetPDRoleStatus(PDRoleStatus::READY);
            }
            ULOG_INFO(SUBMODLE_NAME_ENDPOINT, "[RetryTask] finish a retryTask execution.");
            return true;
        } else {
            return false;
        }
    }
};
} // namespace mindie_llm
#endif