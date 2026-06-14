#!/bin/bash
# Qwen2-VL model launch script
# For detailed parameter configuration and usage instructions, refer to README.md in the same directory

# Environment configuration
export BIND_CPU=1
export RESERVED_MEMORY_GB=0
export MASTER_PORT=20031
export ATB_LLM_BENCHMARK_ENABLE=1
export ATB_PROFILING_ENABLE=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Calculate TP_WORLD_SIZE (number of NPUs used)
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) + 1))

# Default parameters
MAX_BATCH_SIZE=1
MAX_INPUT_LENGTH=4096
MAX_OUTPUT_LENGTH=256
INPUT_TEXT="Explain the contents of the picture with more than 500 words and do not Answer the question using a single word or phrase."
SHM_NAME_SAVE_PATH="./shm_name.txt"
MODEL_PATH=""
INPUT_IMAGE=""
DATASET_PATH=""
MODE="single"  # Added mode selection: single or path

# Parameter parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      if [[ -n "$2" && "$2" != --* ]]; then
        MODEL_PATH="$2"
        shift 2
      else
        echo "Error: --model_path requires a valid non-empty value"
        exit 1
      fi
      ;;
    --input_image)
      if [[ -n "$2" && "$2" != --* ]]; then
        INPUT_IMAGE="$2"
        MODE="single"
        shift 2
      else
        echo "Error: --input_image requires a valid image path"
        exit 1
      fi
      ;;
    --dataset_path)
      if [[ -n "$2" && "$2" != --* ]]; then
        DATASET_PATH="$2"
        MODE="path"
        shift 2
      else
        echo "Error: --dataset_path requires a valid dataset path"
        exit 1
      fi
      ;;
    --max_batch_size)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        MAX_BATCH_SIZE="$2"
        shift 2
      else
        echo "Error: --max_batch_size must be a positive integer"
        exit 1 
      fi
      ;;
    --max_input_length)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        MAX_INPUT_LENGTH="$2"
        shift 2
      else
        echo "Error: --max_input_length must be a positive integer"
        exit 1
      fi
      ;;
    --max_output_length)
      if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
        MAX_OUTPUT_LENGTH="$2"
        shift 2
      else
        echo "Error: --max_output_length must be a positive integer"
        exit 1
      fi
      ;;
    --input_text)
      if [[ -n "$2" && "$2" != --* ]]; then
        INPUT_TEXT="$2"
        shift 2
      else
        echo "Error: --input_text requires valid text"
        exit 1
      fi
      ;;
    --shm_name_save_path)
      if [[ -n "$2" && "$2" != --* ]]; then
        SHM_NAME_SAVE_PATH="$2"
        shift 2
      else
        echo "Error: --shm_name_save_path requires a valid file path"
        exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Parameter validation
if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: --model_path parameter is required"
  exit 1
fi

if [[ "$MODE" == "single" && -z "$INPUT_IMAGE" ]]; then
  echo "Error: Single image mode requires --input_image parameter"
  exit 1
elif [[ "$MODE" == "path" && -z "$DATASET_PATH" ]]; then
  echo "Error: Path mode requires --dataset_path parameter"
  exit 1
fi

# Build base command
base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT \
    -m examples.models.qwen2_vl.run_pa \
    --model_path \"$MODEL_PATH\" \
    --shm_name_save_path \"$SHM_NAME_SAVE_PATH\" \
    --max_input_length $MAX_INPUT_LENGTH \
    --max_output_length $MAX_OUTPUT_LENGTH \
    --max_batch_size $MAX_BATCH_SIZE \
    --input_text \"${INPUT_TEXT}\""

# Add mode-specific parameters
if [[ "$MODE" == "single" ]]; then
  base_cmd+=" --input_image \"$INPUT_IMAGE\""
else
  base_cmd+=" --dataset_path \"$DATASET_PATH\""
fi


# Execute command
eval "$base_cmd"