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
#include "atb_speed/utils/speed_probe.h"


namespace atb_speed {


// The actual implementation of the function is in CANN. This is a placeholder and can not be deleted.
bool SpeedProbe::IsReportModelTopoInfo(const std::string &modelName)
{
    (void)modelName;
    return false;
}


// The actual implementation of the function is in CANN. This is a placeholder and can not be deleted.
void SpeedProbe::ReportModelTopoInfo(const std::string &modelName, const std::string &graph)
{
    (void)modelName;
    (void)graph;
    return;
}


} // namespace atb_speed