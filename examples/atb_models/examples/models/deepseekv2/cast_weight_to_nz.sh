#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src QUANT_WEIGHT_PATH  Specify source quant weight path."
    echo "-dst CAST_QUANT_WEIGHT_PATH  Specify target casted-nz quant weight path."
    echo "-index INDEX_JSON_FILE_NAME  Specify the index file name of model weights, such as quant_model_weight_w8a8.index.json."
}

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
        -index)
            if [[ -n "$2" ]]; then
                index="$2"
                shift
            else
                echo "Error: -index requires a non-empty argument."
                exit 1
            fi
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

if [[ -z "$index" ]]; then
    echo "Error: Missing required option: -index"
    show_help
    exit 1
fi

# 路径校验
if [[ ! -d "${src}" ]]; then
    echo "Error: please check src directory exists."
    show_help
    exit 1
fi

if [[ ! -d "${dst}" ]];then
    echo "Error: please check dst directory exists."
    show_help
    exit 1
fi


# 设置环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 进入运行路径
cd ${ATB_SPEED_HOME_PATH}

# copy需要的json的文件
cp $src/*.json $dst/
cp $src/*.py $dst/

python -m examples.convert.weight_cast --model_path $src --save_directory $dst --index_file $index
