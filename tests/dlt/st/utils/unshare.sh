#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 默认值
DEFAULT_HOST_NAME="node-default"
export HOST_NAME="$DEFAULT_HOST_NAME"
export PROGRAM=""
export ARGS=()

function parse_args() {
    # 参数解析逻辑
    while [[ $# -gt 0 ]]; do
        case "$1" in
        -n | --hostname)
            HOST_NAME="$2"
            shift 2
            ;;
        -h | --help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$PROGRAM" ]]; then
                PROGRAM=$(realpath "$1")
                PROG_DIR=$(dirname "$PROGRAM")
                PROG_NAME=$(basename "$PROGRAM")
            else
                ARGS+=("$1")
            fi
            shift
            ;;
        esac
    done
}

function show_help() {
    echo "Usage: ${0##*/} [OPTIONS] <PROGRAM> [ARGS...]"
    echo "Options:"
    echo "  -n, --hostname NAME   Set node hostname"
    echo "  -h, --help            Show help"
}

function fn_main() {
    parse_args "$@"

    # mount_tmpfs

    # 执行隔离环境
    echo "Starting environment hostname: $HOST_NAME"
    echo "Executing: $PROG_NAME ${ARGS[*]}"

    unshare --uts --ipc --net --mount --pid --user --fork --map-root-user --mount-proc \
        sh -c "
    # 设置主机名
    hostname node-'$HOST_NAME' 2>/dev/null || echo 'Warning: Failed to set hostname'

    # 执行程序
    cd $PROG_DIR
    exec \"./$PROG_NAME\" ${ARGS[@]}
    "
}

fn_main "$@"
