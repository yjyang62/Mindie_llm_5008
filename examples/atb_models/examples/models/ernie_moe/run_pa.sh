#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=20030

export MINDIE_LOG_TO_STDOUT=1

# 解析命令行参数
if [[ "$1" == "--trust_remote_code" ]]; then
    TRUST_REMOTE_CODE="--trust_remote_code"
    echo "[TRUST_REMOTE_CODE]: true"
    shift
else
    TRUST_REMOTE_CODE=""
fi

if [[ ! -z "$1" ]];then
    MODEL_PATH=$1
    echo "[MODEL_PATH]: $MODEL_PATH"
    shift
fi

extra_param="--model_path $MODEL_PATH $TRUST_REMOTE_CODE --is_chat_model"

world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --$extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa $extra_param
fi