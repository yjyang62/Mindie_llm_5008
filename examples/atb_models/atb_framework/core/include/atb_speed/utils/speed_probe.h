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

#ifndef ATB_SPEED_PROBE_H
#define ATB_SPEED_PROBE_H

#include <string>
#include <iostream>

namespace atb_speed {

// This class is designed for dumping model topo, actual implementation is in CANN. This can not be deleted.
class SpeedProbe {
public:
    static bool IsReportModelTopoInfo(const std::string &modelName);
    static void ReportModelTopoInfo(const std::string &modelName, const std::string &graph);
};

}

#endif