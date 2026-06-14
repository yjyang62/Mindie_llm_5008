#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model type and quant type. Support qwen_w4a16 and qwen_w8a8 and qwencode_w8a8s."
    echo "-act_method LEVEL  act_method set level"
    echo "-trust_remote_code  Whether to trust local executable files. Default value is false."
}

#默认使用npu 如果需要使用cpu 传入-device_type cpu
device_type="npu"
w_bit="8"
a_bit="8"
disable_level="L0"
data_list_index="1"
use_devices="0,1,2,3,4,5,6,7"
trust_remote_code=False


# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -src)
            if [[ -n "$2" ]]; then
                src="$2"
                shift
            else
                echo "Error: -src requires a non-empty argument."
                exit 1
            fi
            ;;
        -dst)
            if [[ -n "$2" ]]; then
                dst="$2"
                shift
            else
                echo "Error: -dst requires a non-empty argument."
                exit 1
            fi
            ;;
        -type)
            if [[ -n "$2" ]]; then
                type="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -device_type)
            if [[ -n "$2" ]]; then
                device_type="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -data_list_index)
            if [[ -n "$2" ]]; then
                data_list_index="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -act_method)
            if [[ -n "$2" ]]; then
                act_method="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -w_bit)
            if [[ -n "$2" ]]; then
                w_bit="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -a_bit)
            if [[ -n "$2" ]]; then
                a_bit="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -disable_level)
            if [[ -n "$2" ]]; then
                disable_level="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -anti_method)
            if [[ -n "$2" ]]; then
                anti_method="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -w_sym)
            if [[ -n "$2" ]]; then
                w_sym="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -use_devices)
            if [[ -n "$2" ]]; then
                use_devices="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -msmodelslim_path)
            if [[ -n "$2" ]]; then
                msmodelslim_path="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -fa3_use_anti_calib)
            if [[ -n "$2" ]]; then
                fa3_use_anti_calib="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -trust_remote_code)
            trust_remote_code=True
            echo "[trust_remote_code]: true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 参数校验
if [[ -z "$src" ]]; then
    echo "Error: Missing required option: -src"
    show_help
    exit 1
fi

if [[ -z "$dst" ]]; then
    echo "Error: Missing required option: -dst"
    show_help
    exit 1
fi

if [[ -z "$type" ]]; then
    echo "Error: Missing required option: -type"
    show_help
    exit 1
fi

# 设置环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=$use_devices

param=""

case "$type" in
    qwen_w4a16)
        param="--w_bit 4 --a_bit 16 --disable_level L0 --device_type ${device_type} --act_method 2 --w_sym True --mm_tensor False"
        ;;
    qwen_w8a8)
        param="--w_bit 8 --a_bit 8 --disable_level L0 --device_type ${device_type}"
        ;;
    qwen_w8a8s)
        param="--w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse False --device_type ${device_type} --do_smooth False --use_sigma True --is_lowbit True"
        ;;
    qwencode_w8a8s)
        param="--w_bit 4 --a_bit 8 --fraction 0.02 --co_sparse True --device_type ${device_type} --do_smooth False --use_sigma True --is_lowbit False"
        ;;
    quant_w"$w_bit"a"$a_bit")
        param="--w_bit ${w_bit} --a_bit ${a_bit} --disable_level ${disable_level} --device_type ${device_type} \
        --data_list_index ${data_list_index}"
        param+=" ${act_method:+ --act_method $act_method}"
        param+=" ${anti_method:+ --anti_method $anti_method}"
        param+=" ${w_sym:+ --w_sym}"
        ;;
    qwen2_72b_w8a8c8)
        param="--w_bit 8 --a_bit 8 --disable_level L0 --device_type ${device_type} --calib_file examples/convert/model_slim/boolq.jsonl --use_kvcache_quant True"
        ;;
    qwen2_72b_w8a8)
        param="--w_bit 8 --a_bit 8 --disable_level L0 --device_type ${device_type} --calib_file examples/convert/model_slim/boolq.jsonl"
        ;;
    qwen2_72b_w8a8s)
        param="--w_bit 4 --a_bit 8 --calib_file atb_llm/models/qwen2/cn_en.jsonl --fraction 0.011 --co_sparse True --device_type ${device_type} --do_smooth False --use_sigma True --is_lowbit True"
        ;;
    qwen_w8a16)
        param=""
        ;;
    qwen2p5_14b_w8a8c8_pdmix)
        param="--device_type npu --anti_method m6 --act_method 3 --use_kvcache_quant True --disable_threshold 1 --pdmix True --trust_remote_code ${trust_remote_code}"
        ;;
    qwen2p5_72b_w8a8c8_pdmix)
        param="--device_type npu --anti_method m6 --act_method 3 --use_kvcache_quant True --disable_threshold 1 --pdmix True --trust_remote_code ${trust_remote_code}"
        ;;
    qwen2p5_fa3)
        param=""
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

# 初始化 disable_names 列表
disable_names=("lm_head")

# 根据模型类型生成 disable_names 列表
if [ "$type" == "qwen_w4a16" ] || [ "$type" == "qwen_w8a8" ] || [ "$type" == "qwen_w8a8s" ] || [ "$type" == "qwencode_w8a8s" ] || [ "$type" == "qwen2_72b_w8a8c8" ] || [ "$type" == "qwen2_72b_w8a8" ] || [ "$type" == "qwen2_72b_w8a8s" ]; then
    config_file="$src/config.json"
    num_layers=$(jq '.num_hidden_layers' "$config_file")
    for ((layer=0; layer<num_layers; layer++)); do
        disable_names+=("model.layers.$layer.mlp.down_proj")
    done
elif [ "$type" == "quant_w8a8" ]; then
    config_file="$src/config.json"
    num_layers=$(jq '.num_hidden_layers' "$config_file")
    for ((layer=0; layer<num_layers; layer++)); do
        disable_names+=("transformer.h.$layer.mlp.c_proj")
    done
fi

# 运行 Python 脚本
if [[ "$type" == "qwen_w4a16" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --disable_names "${disable_names[@]}"
elif [[ "$type" == "qwen_w8a8" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --calib_file "./examples/convert/model_slim/boolq.jsonl" --disable_names "${disable_names[@]}"
elif [[ "$type" == "qwen_w8a8s" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --calib_file "./atb_llm/models/qwen2/cn_en.jsonl" --disable_names "${disable_names[@]}"
elif [[ "$type" == "qwencode_w8a8s" ]]; then
    python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --calib_file "./atb_llm/models/qwen2/humaneval_x.jsonl" --disable_names "${disable_names[@]}"
elif [ "$type" == "qwen2_72b_w8a8c8" ] || [ "$type" == "qwen2_72b_w8a8" ] || [ "$type" == "qwen2_72b_w8a8s" ]; then
    python examples/models/qwen/convert_quant_weights_72b.py --model_path $src --save_directory $dst $param
elif [ "$type" == "qwen2p5_fa3" ] &&  [ ! $fa3_use_anti_calib ]; then
    python $msmodelslim_path/example/Qwen/quant_qwen.py --model_path $src --save_directory $dst --calib_file $msmodelslim_path/example/common/boolq.jsonl  --w_bit 8 --a_bit 8 --device_type npu  --use_fa_quant True
elif [ "$type" == "qwen2p5_fa3" ] && [ $fa3_use_anti_calib ]; then
    python $msmodelslim_path/example/Qwen/quant_qwen.py --model_path $src --save_directory $dst --calib_file $msmodelslim_path/example/common/boolq.jsonl --anti_calib_file $msmodelslim_path/example/Qwen/calib_data/anti_prompt.jsonl  --w_bit 8 --a_bit 8 --device_type npu  --use_fa_quant True
elif [ "$type" == "qwen_w8a16" ]; then
    python examples/models/qwen/quant_qwen2_w8a16_fast.py $src $dst
elif [ "$type" == "qwen2p5_14b_w8a8c8_pdmix" ] || [ "$type" == "qwen2p5_72b_w8a8c8_pdmix" ]; then
    python $msmodelslim_path/example/Qwen/quant_qwen.py --model_path $src --save_directory $dst --calib_file $msmodelslim_path/example/Qwen/calib_data/calib_prompt.jsonl --anti_calib_file $msmodelslim_path/example/Qwen/calib_data/anti_prompt.jsonl $param
fi
