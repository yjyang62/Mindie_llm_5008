# README

- [Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct)，Qwen2-Audio 可以识别讲话者的情绪，判断音乐的节奏和类型，分辨各种环境声音，甚至能理解混合音频的含义，例如从一段包含警报声、刹车声和引擎声的音频中，推测出可能是交通事故现场。Qwen2-Audio 能够接受各种音频信号输入，对语音指令进行音频分析或直接文本回复。

## 特性矩阵

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------ | -------------------------- | --------------------------- | ---- | ---- | -------------- | -------------- | -------------- |
| Qwen2-Audio-7B-Instruct | 支持 world size 1,2,4,8 | ❌ | ✅ | ✅ | ✅ | 文本、音频 | 文本、音频 |

## 路径变量解释

| 变量名      | 含义                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录（仅源码下载场景使用）                                         |
| llm_path    | 模型仓所在路径。若使用镜像，则路径为 `/usr/local/Ascend/atb-models`；若使用 gitcode 下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；qwen2_audio 的工作脚本所在路径为 `${llm_path}/examples/models/qwen2_audio`     |
| weight_path | 模型权重路径                                                                                 |
| audio_path  | 音频所在路径                                                                                 |
| max_batch_size  | 最大 batch 数                                                                                                                                        |
| max_input_length  | 多模态模型的最大 embedding 长度                                                                                                                      |
| max_output_length | 生成的最大 token 数                                                                                                                                  |

> **注意**：max_input_length 长度设置可参考模型权重路径下 config.json 里的 max_position_embeddings 参数值。

## 权重

**权重下载**

- [Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct)

**基础环境变量**

- Python 其他第三方库依赖，参考 [requirements_qwen2_audio.txt](../../../requirements/models/requirements_qwen2_audio.txt)
- 参考[此 README 文件](../../../README.md)
- 注意：保证先后顺序，否则 qwen2_audio 的其余三方依赖会重新安装 torch，导致出现别的错误
- transformers 版本等于 4.46.2
- 模型运行依赖 ffmpeg，需要确保安装后运行。安装可参考以下指令

    ```shell
    apt-get update -y && apt-get install ffmpeg
    ```

**服务化推理测试**

- Python 第三方库依赖安装完毕之后，在测试服务化之前，需要在\${script_path} 目录下执行以下指令

    ```shell
    python librosa_file_util.py --modify_librosa True
    ```

用于修改 librosa，保证服务化能正常推理。

- 在所有服务化推理结束之后，可在\${script_path} 目录下执行以下指令

    ```shell
    python librosa_file_util.py --restore_librosa True
    ```

将 librosa 改回至原文件状态。

- 执行以上指令，如果用户权限不够，请切换至 root 用户执行。

## 推理

### 对话测试

- 运行启动脚本
  - 在 `${llm_path}` 目录下执行以下指令

    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${audio_path}
    ```

- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
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

## 服务化推理

- 打开配置文件

  ```shell
  vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
  ```

- 修改 `MindIE-Service` 配置文件 `config.json`，以 `Qwen2-Audio-7B-Instruct` 为例，在 800I A2 环境下，推荐使用以下配置，请自行修改 `modelWeightPath` 为实际权重路径：

  - **Qwen2-Audio-7B-Instruct**

  ```json
  {
      "ServerConfig": {
          "port" : 1025,
          "managementPort" : 1026,
          "httpsEnabled": false,
      },
      "BackendConfig": {
          "npuDeviceIds": [
              [0,1]
          ],
          "ModelDeployConfig": {
              "maxSeqLen": 32768,
              "maxInputTokenLen": 32768,
              "ModelConfig": [
                  {
                      "modelInstanceType": "Standard",
                      "modelName": "qwen2_audio",
                      "modelWeightPath": "/data/datasets/Qwen2-Audio-7B-Instruct",
                      "worldSize": 2,
                      "npuMemSize": 10, #KV Cache 的显存大小，单位是 GB，请勿设置为-1，需要给 vit 预留空间。
                  }
              ]
          },
          "ScheduleConfig": {
              "maxPrefillTokens": 32768,
          }
      }
  }
  ```

- 部署服务化

  ```shell
  cd /usr/local/Ascend/mindie/latest/mindie-service/bin
  ./mindieservice_daemon
  ```

- 新建同一个 Docker 容器的终端会话，在任意路径下发送 curl 请求完成推理，以下分别以 OpenAI 接口与 vLLM 接口为例

  - **OpenAI 接口**

  ```shell
  curl http://localhost:${端口号，与起服务化时 config.json 中的'port'保持一致}/v1/chat/completions -d '{
      "model": "qwen2_audio",
      "messages": [{
          "role": "user",
          "content": [
                      {"type": "audio_url", "audio_url": ${音频路径}},
                      {"type": "text", "text": "What did the speaker say in the audio?"}
                  ]
      }],
      "max_tokens": 512,
      "stream": false
  }'
  ```

  - **vLLM 接口**

  ```shell
  curl localhost:${端口号，与起服务化时 config.json 中的'port'保持一致}/generate -d '{
      "prompt": [
          {"type": "audio_url", "audio_url": ${音频路径}},
          {"type": "text", "text": "What did the speaker say in the audio?"}
      ],
      "max_tokens": 512,
      "do_sample": false,
      "stream": false,
      "model": "qwen2_audio"
  }'
    ```
