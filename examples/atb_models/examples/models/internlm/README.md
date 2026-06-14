# README

- InternLM 开源了 InternLM 系列的多个基础模型和为实际场景量身定制的聊天模型。该系列模型具有以下特点：

    - 它利用数万亿个高质量的代币进行培训，以建立强大的知识库。
    - internlm-20B 支持 8k 上下文窗口长度，InternLM2.5-7B-Chat-1M 有效支持百万字超长上下文，可实现更长的输入序列和更强的推理能力，Internlm3-8B 专为通用和高级推理而设计。
    - 它为用户提供了一个多功能的工具集，可以灵活地构建自己的工作流程。

- 此代码仓中实现了一套基于 NPU 硬件的 Internlm 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 特性矩阵

此矩阵罗列了各 Internlm 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化 | MOE 量化 | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|-----------------------------|------|------|-----------------|-----------------|-----------|------------|--------------|--------|---------|----------------|-----|--------|
| internlm2.5-1.8B | 支持 world size 1 | 支持 world size 1 | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| internlm2.5-7B | 支持 world size 1,2,4,8 | 支持 world size 2,4,8 | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| internlm2.5-20B | 支持 world size 2,4,8 | 支持 world size 2,4,8 | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## Paged Attention 推理使用说明

### 路径变量解释

| 变量名      | 含义                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；Internlm 系列模型的工作脚本所在路径为 `${llm_path}/examples/models/internlm`   |
| weight_path | 模型权重路径                                                                                 |
| chat        | 是否启用对话模式                                                                             |

### 权重

**权重下载**

- [internlm2.5-1.8B](https://huggingface.co/internlm/internlm2_5-1_8b-chat/tree/main)
- [internlm2.5-7B](https://huggingface.co/internlm/internlm2_5-7b-chat/tree/main)
- [internlm2.5-20B](https://huggingface.co/internlm/internlm2_5-20b-chat/tree/main)

**权重转换**

- Paged Attention 场景下需要.safetensors 格式的权重，如果没有，参考[此 README 文件](../../README.md)转换

**量化权重生成**

下载 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 量化工具，根据 README 进行相关操作。

> 注：安装完 CANN 后需要执行 `source set_env.sh` 声明 ASCEND_HOME_PATH 值，后续安装 msmodelslim 前需保证其不为空。

```shell
cd ${llm_path}
# 指定当前机器上可用的逻辑 NPU 核心，通过修改 convert_quant_weight.sh 文件中 export ASCEND_RT_VISIBLE_DEVICES 值，指定使用卡号及数量
vi examples/models/internlm/convert_quant_weight.sh
```

- 基于原始的 FP16 的权重，生成量化权重
- W8A8 量化权重请使用以下指令生成

  ```shell
  bash examples/models/internlm/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8 量化权重路径} -type w8a8
  ```

- W8A16 量化权重请使用以下指令生成

  ```shell
  bash examples/models/internlm/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A16 量化权重路径} -type w8a16
  ```

- W8A8C8 量化权重请使用以下指令生成

  ```shell
  bash examples/models/internlm/convert_quant_weight.sh -src {浮点权重路径} -dst {W8A8C8 量化权重路径} -type w8a8c8
  ```

- 稀疏量化权重请使用以下指令生成
  - 暂不支持

**基础环境变量**

- 参考[此 README 文件](../../../README.md)
- 检查 Python 依赖库中 transformers 版本的配置，Internlm3 要求 transformers 库版本为 4.47.1 及以上。

  ```shell
  pip show transformers
  # 请将 transformers 更新至对应版本
  # Internlm2/Internlm2.5
  pip install transformers==4.41.0
  # Internlm3
  pip install transformers==4.47.1
  ```

### 推理

#### 对话测试

**运行 Flash Attention FP16**

- 其余 Internlm 模型参考以下运行方式
    - 运行启动脚本
        - 在\${llm_path} 目录下执行以下指令

          ```shell
          bash ${script_path}/run_fa.sh ${weight_path}
          ```

    - 环境变量说明
        - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
            - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
            - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
            - 各模型支持的核心数参考“特性矩阵”
        - `export MASTER_PORT=20050`
            - 设置卡间通信端口
            - 默认使用 20050 端口
            - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
            - 设置时端口建议范围为：20000-20050
        - 以下环境变量与性能和内存优化相关，通常情况下无需修改

          ```shell
          export INF_NAN_MODE_ENABLE=0
          export ATB_OPERATION_EXECUTE_ASYNC=1
          export TASK_QUEUE_ENABLE=1
          export ATB_CONVERT_NCHW_TO_ND=1
          export HCCL_BUFFSIZE=120
          export HCCL_WHITELIST_DISABLE=1
          export ATB_LAUNCH_KERNEL_WITH_TILING=0
          export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
          export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0

          ```

**运行 Flash Attention BF16**

- 暂不支持

**运行 Paged Attention FP16**

- 运行启动脚本
    - 在\${llm_path} 目录下执行以下指令

      ```shell
      bash ${script_path}/run_pa.sh ${weight_path} chat
      ```

- 环境变量说明
    - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
        - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
        - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
        - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
        - 各模型支持的核心数参考“特性矩阵”
    - `export MASTER_PORT=20050`
        - 设置卡间通信端口
        - 默认使用 20050 端口
        - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
        - 设置时端口建议范围为：20000-20050
    - 以下环境变量与性能和内存优化相关，通常情况下无需修改

      ```shell
      export INF_NAN_MODE_ENABLE=0
      export ATB_OPERATION_EXECUTE_ASYNC=1
      export TASK_QUEUE_ENABLE=1
      export ATB_CONVERT_NCHW_TO_ND=1
      ```

**运行 Paged Attention BF16**

- 暂不支持

**运行 Paged Attention W8A8**

- bash ${script_path}/run_pa.sh ${weight_path} chat

**运行 Paged Attention W8A16**

- bash ${script_path}/run_pa.sh ${weight_path} chat

**运行 KV cache 量化**

- bash ${script_path}/run_pa.sh ${weight_path} chat

**200K 长序列**

- 修改模型权重下的 config.json

    ```json
    internlm2-7B 改为
    "rope_scaling": {
        "factor": 2.0,
        "type": "dynamic"
    },

    internlm2-20B 改为
    "rope_scaling": {
        "factor": 3.0,
        "type": "dynamic"
    },
    ```

- 修改 run_pa.py 文件 `parse_arguments()`函数的参数，max_input_length 必须大于文本 token 数。因为分词原因，文本长度不等于文本 token 数，通常文本字符数大于文本 token 数。
- --input_texts
      --input_file
      --max_input_length
      --max_output_length

    ```python

    parser.add_argument(
        '--input_texts',
        type=str,
        nargs='+',
        default="text_200K")
    parser.add_argument(
        '--input_file',
        type=str,
        help='CSV or Numpy file containing tokenized input. Alternative to text input.',
        default="./text_200K.jsonl")
    parser.add_argument('--max_input_length', type=int, default=210000)
    parser.add_argument('--max_output_length', type=int, default=256)
    ```

- 输入 32K/64K/128K/192K 长序列
    - 使用 --input_texts 参数或者 --input_file 参数。
    - `--input_texts`
      - 必须为 str 或 List[str] 格式的对话数据

    ```text
    """
    这里是 10 万字的小说内容 \n 总结以上文本内容。
    """
    ```

    - `--input_file`（推荐）
      - 目前仅支持 jsonl 格式文件，每一行必须为 List[Dict] 格式的按时间顺序排序对话数据
      - 每个 Dict 字典中需要至少包含 "role" 和 "content" 两个字段

    ```json
    [{"role": "user", "content": "这里是 10 万字的小说内容 \n 总结以上文本内容。"}]
    ```

- 运行启动脚本
    - 在\${llm_path} 目录下执行以下指令(后面加一个 chat 参数)

      ```shell
      bash ${script_path}/run_pa.sh ${weight_path} chat
      ```

### 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
    - 示例

      ```shell
      cd ${llm_path}/tests/modeltest
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      bash run.sh pa_fp16 full_BoolQ 1 internlm ${internlm 系列模型权重路径} trust_remote_code 8
      bash run.sh pa_fp16 full_CEval 5 1 internlm ${internlm 系列模型权重路径} trust_remote_code 8

      internlm_20b, internlm2_7b, internlm2_20b, internlm2.5_7b, 已合并为 internlm，模型名称都是 internlm，
      对应于 tests/modeltest/core/internlm_test.py。
      ```

### 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
    - 示例

      ```shell
      cd ${llm_path}/tests/modeltest
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export ATB_LLM_BENCHMARK_ENABLE=1
      bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 internlm ${internlm 系列模型权重路径} trust_remote_code 8

      bash run.sh pa_fp16 performance_maxbs [[256,256],[512,512]] [[1,2048],[1,2048]] 50 internlm ${internlm 系列模型权重路径} trust_remote_code 4
      ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_fa.py` 和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此 README 文件](../../README.md)
- 如果模型生成了 `[UNUSED_TOKEN_146]`、`[UNUSED_TOKEN_145]`等特殊字符，升级 transformers 版本到 4.37.1 以上。`</s>`是大模型常用结束词，`[UNUSED_TOKEN_146]`是旧版结束词，`<|im_end|>`是新版结束词。
- 如果遇到模型生成停不下来/没有按照预期停止，可以对生成的文本进行后处理，比如 `response = response.split("<|im_end|>")[0]`。
- transformers 不支持直接修改 `tokenizer.eos_token_id`，只能通过修改 `tokenizer.eos_token` 间接修改：确保 `router_internlm2.py` 文件中 `safe_get_tokenizer_from_pretrained` 的参数 `use_fast=True`，同时令 `tokenizer.eos_token = "<|im_end|>"`。
- Internlm2 模型使用 bfloat16 完成训练，使用 float16 进行推理会有一些精度波动，如果 logits 输出在 fp16 数据类型 1 个 ulp 的最小波动范围内，是正常现象。
