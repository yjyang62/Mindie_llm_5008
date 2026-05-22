#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

set -e

LOG_DIR="/var/log/mindie"
LOG_FILE="${LOG_DIR}/uninstall.log"
SELF_DIR=$(realpath "$(dirname "$0")")

INSTALL_PATH=$(realpath "${SELF_DIR}/..")

mkdir -p "$LOG_DIR"

print() {
    echo -e "[$(date '+%F %T')] [$1] $2" | tee -a "$LOG_FILE"
}

print "INFO" "Uninstalling MindIE..."

python_bin=$(command -v python3 || command -v python || true)
if [[ -n "$python_bin" ]]; then
    if $python_bin -m pip show mindie-sd &>/dev/null; then
        print "INFO" "Removing mindie-sd wheel..."
        $python_bin -m pip uninstall -y mindiesd 2>&1 | tee -a "$LOG_FILE"
    else
        print "INFO" "mindiesd not installed, skipping pip uninstall."
    fi
else
    print "WARN" "Python not found: wheel uninstall skipped."
fi

if [[ -d "$INSTALL_PATH" ]]; then
    print "INFO" "Removing install directory: $INSTALL_PATH"
    rm -rf "$INSTALL_PATH"
else
    print "INFO" "Install directory not found: skip removal"
fi

print "INFO" "MindIE uninstall SUCCESS!"
