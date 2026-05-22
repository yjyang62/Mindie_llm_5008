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

#ifndef DMI_ROLE_H
#define DMI_ROLE_H

#include <map>
#include <vector>
#include <unordered_set>
#include "httplib.h"
#include "common_util.h"
#include "config_manager.h"
#include "http_rest_resource.h"
#include "infer_instances.h"
#include "../utils/retry_task.h"
#include "blocking_queue.h"
#include "global_ip_info.h"
#include "dmi_role.h"
using ordered_json = nlohmann::ordered_json;

namespace mindie_llm {
constexpr uint32_t DEFAULT_PD_ROLE_FLEX_P_PERCENTAGE = 50; // pdRole flex p_percentage 默认值50
class FlexPPercentageProcessor {
public:
    static FlexPPercentageProcessor &GetInstance()
    {
        static FlexPPercentageProcessor instance;
        return instance;
    }
    uint32_t GetPdRoleFlexPPercentage() const
    {
        return this->pdRoleFlexPPercentage;
    }
    void SetPdRoleFlexPPercentage(const uint32_t pPercentage)
    {
        this->pdRoleFlexPPercentage = pPercentage;
    }
private:
    FlexPPercentageProcessor() = default;
    ~FlexPPercentageProcessor() = default;
    uint32_t pdRoleFlexPPercentage{DEFAULT_PD_ROLE_FLEX_P_PERCENTAGE};
};

class DmiRole {
public:
    DmiRole();
    ~DmiRole();
    void HandlePDRoleV1(const ReqCtxPtr &ctx, const std::string &roleName);
    void HandlePDRoleV2(const ReqCtxPtr &ctx, const std::string &roleName);
    const std::map<uint64_t, std::vector<DeviceInfo>> &GetSuccessLinkIp();
    const std::map<uint64_t, std::vector<std::string>> &GetSuccessHostIp();
    const std::map<uint64_t, std::pair<std::string, bool>> &GetRemoteNodeLinkStatus();
    std::map<uint64_t, std::pair<std::string, bool>> GetRemoteNodeLinkStatusV2();
    const std::map<uint32_t, std::string> &GetInstanceIdToServerIp();
    const uint32_t &GetLocalInstanceId();
    void ModifyPullKVFailId(const uint32_t &instanceId);
    void RunThread();
    bool IsHealthy();
    static std::shared_ptr<DmiRole> GetInstance();
private:
    bool UpdatePDInfo(const std::string &roleName, const std::string &preRole, const ordered_json &body,
        GlobalIpInfo &globalIpInfo);
    bool UpdatePDInfoV2(const std::string &roleName, const std::string &preRole, const ordered_json &body,
        GlobalIpInfo &globalIpInfo);
    bool UpdatePDSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
        bool needInit);
    bool UpdatePDSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo,
        bool needInit);
    bool UpdatePDNotSwitchInfo(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo);
    bool UpdatePDNotSwitchInfoV2(const std::string &roleName, const ordered_json &body, GlobalIpInfo &globalIpInfo);
    void UpdateSuccessLinkIp(GlobalIpInfo &globalIpInfo);
    void UpdateSuccessHostIp(GlobalIpInfo &globalIpInfo);
    void RetryThread();
    void ResetContext(boost::system::error_code ec);
    void RetryLinkCallback(GlobalIpInfo &globalIpInfo);
    bool PDParseRequestBodyToJson(const ReqCtxPtr &reqCtx, ordered_json &body) const noexcept;
    void ProcessInitInfo(const ordered_json &body, GlobalIpInfo &globalIpInfo);
    void ProcessInitInfoV2(const ordered_json &body, GlobalIpInfo &globalIpInfo);
    void UpdateIpInfo(std::map<uint64_t, std::vector<DeviceInfo>>& currentLinkIpInfo,
        GlobalIpInfo &globalIpInfo, std::string &superPodId);
    void UpdateHostIpInfo(std::map<uint64_t, std::vector<std::string>>& currentLinkHostIpInfo,
        GlobalIpInfo &globalIpInfo);
    void ProcessPDRoleSwitch(const ReqCtxPtr &ctx, const std::string &roleName,
        GlobalIpInfo &globalIpInfo);
    // already successful linked server ip address
    std::map<uint64_t, std::vector<DeviceInfo>> successLinkIP_;
    std::map<uint64_t, std::vector<std::string>> successHostIP_;

    // [key] is <localDpInstanceId, remoteDpInstanceId> and [value] is {linkStatus, isProcessed}
    // [record remote node status]: key is instanceId, [value] is {linkStatus, isProcessed}
    std::map<uint64_t, std::pair<std::string, bool>> remoteNodeLinkStatus_;
    // example: [10001, 0k] [10002, not ok] will cause [1: not ok]
    std::map<uint32_t, std::string> instanceIdToServerIp_;
    std::atomic<bool> retryTerminate_{false};
    std::shared_ptr<boost::asio::io_context> ioContext_{nullptr};
    std::thread retryThread_;
    BlockingQueue<std::unique_ptr<RetryTask>> taskQueue_;
    std::mutex mtx_;
    // Attention: localInstanceId_ will be updated by [DmiRole::ProcessInitInfo]
    uint32_t localInstanceId_{0};
    // Attention: localDpInstanceIds_ will be updated by [DmiRole::ProcessInitInfo]
    std::vector<uint64_t> localDpInstanceIds_;
    bool abnormalLink_{false};
    std::vector<uint64_t> dpInstanceList;
};
extern std::atomic<bool> keepAlive;
} // namespace mindie_llm

#endif // OCK_ENDPOINT_HTTP_HANDLER_H
