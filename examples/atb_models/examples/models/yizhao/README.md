# README

- 一招（YiZhao）是由招商银行联合华为、智谱 AI，在国产设备与训练框架下开发训练的金融领域大语言模型系列。该系列包含一个 120 亿参数版本，YiZhao-12B-Chat 具备自然语言理解、文本生成、舆情事件抽取、工具使用互动等多种功能，支持 32K 上下文长度。

- 此代码仓中实现了一套基于 NPU 硬件的 YiZhao-12B-Chat 推理模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

## 特性矩阵

此矩阵罗列了 YiZhao-12B-Chat 模型支持的特性。

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8 量化 | W8A16 量化 | KV cache 量化 | 稀疏量化 | MOE 量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|-----------|------------|--------------|--------|---------|----------------|-----|--------|
| YiZhao-12B-Chat | 支持 world size 1,2,4,8 | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

- 此模型仓已适配的模型版本：
  - [一招-12B-Chat-HF](https://modelscope.cn/models/CMB_AILab/YiZhao-12B-Chat-HF)
- **注意**：transformers 需要安装模型配置文件中指定的 4.46.1 版本。

## 使用说明

### 路径变量解释

| 变量名      | 含义                                                                                       |
|------------|-------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；YiZhao 的工作脚本所在路径为 `${llm_path}/examples/models/yizhao/`               |
| weight_path | 模型权重路径                                                                                 |

### 权重

#### 权重下载

- [一招-12B-Chat-HF](https://modelscope.cn/models/CMB_AILab/YiZhao-12B-Chat-HF)

#### 权重转换

- 若权重中不包含 safetensors 格式，则执行权重转换步骤，否则跳过
- 参考[此 README 文件](../../README.md)

### 对话测试

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} -trust_remote_code
    ```

  - trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定 CPU 核心开关
    - 默认进行绑核
    - 若当前机器未设置 NUMA 或绑核失败，可将 BIND_CPU 设为 0
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的 TP 数，即 world size
    - 默认为单卡
    - 各模型支持的 TP 数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入 Python 查询模块和包的搜索路径中
    - 将 `${llm_path}` 替换为实际路径
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改

    ```shell
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=110
    ```

### 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
- 示例

  ```shell
  cd ${llm_path}/tests/modeltest
  # 测试 BoolQ/CEval 双卡精度
  bash run.sh pa_bf16 full_BoolQ 1 chatglm ${weight_path} trust_remote_code 2
  bash run.sh pa_bf16 full_CEval 5 1 chatglm ${weight_path} trust_remote_code 2
  ```

- trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。

### 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
- 示例

  ```shell
  cd ${llm_path}/tests/modeltest
  # 测试双卡性能用例 batch size=1，输入 256，输出 256
  bash run.sh pa_bf16 performance [[256,256]] 1 chatglm ${weight_path} trust_remote_code 2
  ```

- trust_remote_code 为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；文件参数说明见[此 README 文件](../../README.md)
