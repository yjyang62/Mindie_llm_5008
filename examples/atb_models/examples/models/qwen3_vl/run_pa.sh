#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_PORT=20036
export INF_NAN_MODE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export BIND_CPU=1

MODEL_PATH=""
IMAGE_PATH=""
MAX_BATCH_SIZE=1
MAX_INPUT_LENGTH=3000
MAX_OUTPUT_LENGTH=256
TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
fi

trust_remote_code="False"
if [[ "$1" == "--trust_remote_code" ]]; then
    trust_remote_code="True"
    echo "[TRUST_REMOTE_CODE]: true"
    shift
else
    trust_remote_code="False"
fi

if [[ "$1" == "--model_path" ]]; then
    if [[ -n "$2" && "$2" != --* ]]; then
        MODEL_PATH="$2"
        echo "[MODEL_PATH]: $MODEL_PATH"
        shift 2
    else
        echo "Error: --model_path requires a valid non-empty value"
        exit 1
    fi
fi

if [[ "$1" == "--image_path" ]];then
    if [[ -n "$2" && "$2" != --* ]]; then
        IMAGE_PATH="$2"
        echo "[IMAGE_PATH]: $IMAGE_PATH"
        shift 2
    else
        echo "Error: --image_path requires a valid non-empty value"
        exit 1
    fi
fi

if [[ ! -z "$1" ]];then
    MAX_BATCH_SIZE=$1
    echo "[MAX_BATCH_SIZE]: $MAX_BATCH_SIZE"
    shift
fi

if [[ ! -z "$1" ]];then
    MAX_INPUT_LENGTH=$1
    echo "[MAX_INPUT_LENGTH]: $MAX_INPUT_LENGTH"
    shift
fi

if [[ ! -z "$1" ]];then
    MAX_OUTPUT_LENGTH=$1
    echo "[MAX_OUTPUT_LENGTH]: $MAX_OUTPUT_LENGTH"
    shift
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: --model_path parameter is required"
  exit 1
fi

if [[ -z "$IMAGE_PATH" ]]; then
  echo "Error: --image_path parameter is required"
  exit 1
fi

extra_param=""
if [ "$trust_remote_code" == "True" ]; then
    extra_param="${extra_param} --trust_remote_code"
fi
extra_param="${extra_param} --model_path $MODEL_PATH
                            --image_path $IMAGE_PATH
                            --max_batch_size $MAX_BATCH_SIZE
                            --max_input_length $MAX_INPUT_LENGTH
                            --max_output_length $MAX_OUTPUT_LENGTH"
if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.models.qwen3_vl.qwen3_vl $extra_param
else
    torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.models.qwen3_vl.qwen3_vl $extra_param
fi