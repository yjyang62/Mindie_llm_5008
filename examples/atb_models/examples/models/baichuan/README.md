# README

- Baichuan 大模型，融合了意图理解、信息检索以及强化学习技术，结合有监督微调与人类意图对齐，在知识问答、文本创作领域表现突出。

- 此代码仓中实现了一套基于 NPU 硬件的 Baichuan 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 特性矩阵

此矩阵罗列了各 Baichuan 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化 | MOE 量化 | MindIE Service | TGI | 长序列 | Razor Attention |
|-------------|----------------------------|-----------------------------|------|------|-----------------|-----------------|-----------|------------|--------------|--------|---------|----------------|-----|--------|-----------------|
| Baichuan2-7B  | 支持 world size 1,2,4,8 | 支持 world size 2 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Baichuan2-13B | 支持 world size 2,4,8   | 支持 world size 2,4 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |

## 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；Baichuan 系列模型的工作脚本所在路径为 `${llm_path}/examples/models/baichuan`   |
| weight_path | 模型权重路径                                                                                 |

## 权重

**权重下载**

- [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/tree/main)
- [Baichuan2-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main)
- 注意事项：
  - 请下载全部权重文件

**权重转换**

- Paged Attention 场景下需要.safetensors 格式的权重，如果没有，参考[此 README 文件](../../README.md)转换

**量化权重生成**
基于原始的 FP16 的权重，生成量化权重。量化导出脚本使用参数介绍如下：

| 模型          | dst            | src                  | type                      | 备注 |
| ------------- | -------------- | -------------------- | ------------------------- | ---- |
| baichuan2_7b  | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_7b_w8a8         | -    |
| baichuan2_7b  | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_7b_w8a8_kvcache | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w8a8        | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w4a16       | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w8a16       | -    |

trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。

例如，对于 Baichuan2_7b 的 w8a8 格式量化而言，命令如下：

```bash
cd ${llm_path}
bash examples/models/baichuan/generate_baichuan2_quant_weights.sh -src "源权重路径" -dst "目标权重路径" -type baichuan2_7b_w8a8 -trust_remote_code
```

- 稀疏量化权重请使用以下指令生成
  - 暂不支持

**基础环境变量**

- 参考[此 README 文件](../../../README.md)

## 推理

### 对话测试

**运行 Flash Attention FP16**

- 其余 Baichuan 模型参考以下运行方式

  - 运行启动脚本
    - 在 `${llm_path}` 目录下执行以下指令

        ```shell
        bash examples/models/baichuan/run_fa.sh ${weight_path} -trust_remote_code
        ```

  - trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
  - 环境变量说明
    - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
      - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
      - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
      - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
      - 各模型支持的核心数参考"特性矩阵"
    - `export MASTER_PORT=20036`

    - 设置卡间通信端口

    - 默认使用 20036 端口

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
          export ATB_CONTEXT_WORKSPACE_RING=1
          export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
          export ATB_LAUNCH_KERNEL_WITH_TILING=0
          export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
          export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0

    ```

**运行 Flash Attention BF16**

- 暂不支持

**运行 Flash Attention W8A8**

- 暂不支持

**运行 Flash Attention W8A16**

- 暂不支持

**运行 Paged Attention FP16**

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    chat 模式（仅支持 baichuan2 系列）:
    bash examples/models/baichuan/run_pa.sh ${weight_path} chat -trust_remote_code

    非 chat 模式:
    bash examples/models/baichuan/run_pa.sh ${weight_path} -trust_remote_code
    ```

  - trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考"特性矩阵"
  - `export MASTER_PORT=20036`
    - 设置卡间通信端口
    - 默认使用 20036 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

**运行 Paged Attention BF16**

- 暂不支持

**运行 Paged Attention W8A8**

- 运行启动脚本
  - 与“运行 Paged Attention FP16”的启动方式相同
  - `${weight_path}`为 W8A8 量化权重的路径
- 环境变量说明
  - 参见“运行 Paged Attention FP16”中的环境变量说明
- 相比于 FP16，运行量化时需修改 W8A8 量化权重`${weight_path}/config.json` 中的 `quantize` 字段，将此字段对应的值修改为 `w8a8`
  - 若 config.json 中无此字段，则新增

**运行 Paged Attention W8A16**

- 运行启动脚本
  - 与“运行 Paged Attention FP16”的启动方式相同
  - `${weight_path}`为 W8A16 量化权重的路径
- 环境变量说明
  - 参见“运行 Paged Attention FP16”中的环境变量说明
- 相比于 FP16，运行量化时需修改 W8A16 量化权重`${weight_path}/config.json` 中的 `quantize` 字段，将此字段对应的值修改为 `w8a16`
  - 若 config.json 中无此字段，则新增

**运行 KV cache 量化**

- 暂不支持

**运行稀疏量化**

- 暂不支持

**运行 MOE 量化**

- 暂不支持

**运行 Razor Attention FP16**

- 开启环境变量

    ```bash
  export ATB_LLM_RAZOR_ATTENTION_ENABLE=1
    ```

- 运行启动脚本
  - 与“运行 Paged Attention FP16”的启动方式相同

### 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例

    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_7b ${baichuan-7b 权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_13b ${baichuan-13b 权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_7b ${baichuan2-7b 权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_13b ${baichuan2-13b 权重路径} trust_remote_code 4
    ```

- 注意：baichuan-7b 和 baichuan-13b 模型测试时复用 baichuan2_7b 和 baichuan2_13b 的 model_name
- 运行量化权重时需注意 `${weight_path}/config.json` 中的 `quantize` 字段和 `torch_dtype` 字段是否与权重匹配，参考[此 README 文件](../../README.md)
- 测试 longbench 时需开启环境变量 ALiBi Mask Free:

```bash
export IS_ALIBI_MASK_FREE=1
```

### 性能测试

- 支持 ALiBi Mask Free。默认关闭，如需开启，请修改当前目录下的 run_pa.sh 中环境变量如下：

```bash
export IS_ALIBI_MASK_FREE=1
```

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例

    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_7b ${baichuan2-7b 权重路径} trust_remote_code 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_13b ${baichuan2-13b 权重路径} trust_remote_code 8
    ```

- 运行量化权重时需注意 `${weight_path}/config.json` 中的 `quantize` 字段和 `torch_dtype` 字段是否与权重匹配，参考[此 README 文件](../../README.md)
- 特殊场景说明: 若在性能测试时发现有波动情况，可配置透明大页，提升内存访问性能。该功能请按需开启，对内存占用有一定影响。

```shell
# 性能测试时，可按需开启透明大页
echo always > /sys/kernel/mm/transparent_hugepage/enabled
# 关闭透明大页
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_fa.py` 和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此 README 文件](../../README.md)
