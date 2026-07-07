# Tokenflow Docker 镜像运行

镜像内默认入口脚本是：

```bash
docker-entrypoint
```

它会执行：

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  $SGLANG_ARGS
```

因此运行镜像时，基础参数通过环境变量传入：

| 环境变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MODEL_PATH` | 容器内模型路径 | `/models/model` |
| `HOST` | SGLang 监听地址 | `0.0.0.0` |
| `PORT` | SGLang 监听端口 | `30000` |
| `SGLANG_ARGS` | 追加给 `python3 -m sglang.launch_server` 的所有额外参数 | 空 |

入口脚本还会设置：

```bash
PYTHONPATH=/workspace/sglang/python:$PYTHONPATH
```

这样即使 `sglang` 没有以 wheel/console-script 形式正确安装，也可以直接从镜像内源码目录导入 `sglang.launch_server`。

## 1. 加载已保存的镜像

```bash
gunzip -c tokenflow-sglang_cuda12.8.tar.gz | docker load
```

查看镜像名：

```bash
docker images | grep tokenflow
```

下面示例假设镜像名是：

```bash
tokenflow-sglang:cuda12.8
```

如果 `docker load` 后只有 image id，可以手动打 tag：

```bash
docker tag <IMAGE_ID> tokenflow-sglang:cuda12.8
```

## 2. 启用 KV Offload

启用 KV offload 时，需要通过 `SGLANG_ARGS` 额外传入 hierarchical cache 和 KV offload policy 参数。

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -v /path/to/model:/models/model:ro \
  -e MODEL_PATH=/models/model \
  -e HOST=0.0.0.0 \
  -e PORT=30000 \
  -e SGLANG_ARGS='--tp-size 8 --ep-size 8 --dtype auto --trust-remote-code --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think --max-running-requests 128 --mem-fraction-static 0.6 --enable-hierarchical-cache --hicache-size 40 --kv-offload-policy paper_v1' \
  tokenflow-sglang:cuda12.8
```

KV offload 的关键参数是：

| 参数 | 作用 |
| --- | --- |
| `--enable-hierarchical-cache` | 启用 host 侧分层 KV cache，让 KV block 可以在 GPU 显存和 host 内存之间迁移。 |
| `--hicache-size 40` | 分配 40 GB host KV cache。单位是 GB，应根据机器内存容量调整。 |
| `--kv-offload-policy paper_v1` | 启用 KV offload 感知的调度策略。默认值 `default` 表示普通 SGLang 行为。 |

建议保留：

```bash
--ulimit memlock=-1 --cap-add IPC_LOCK
```

这有助于 host KV cache 相关内存注册或锁页场景，尤其是大 cache 或高吞吐实验时。

## 4. KV Offload 额外参数怎么传

所有额外 server 参数都放进 `SGLANG_ARGS` 即可。比如：

```bash
-e SGLANG_ARGS='--tp-size 8 --ep-size 8 --enable-hierarchical-cache --hicache-size 40 --kv-offload-policy paper_v1 --kv-offload-reschedule-interval 1000 --kv-offload-default-output-speed 8 --kv-offload-buffer-conservativeness 2.5'
```

这些参数最终会被 `docker-entrypoint` 追加到：

```bash
python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT"
```

后面。

常用 KV offload 参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--kv-offload-policy` | `default` | 调度策略。设为 `paper_v1` 时启用 KV offload 感知重调度。 |
| `--kv-offload-reschedule-interval` | `5000` | 每隔多少个 scheduler tick 做一次 KV offload 重调度。值越小反应越快，但调度开销更高。 |
| `--kv-offload-token-value-alpha` | `1.0` | `paper_v1` 目标函数中 token value 项的权重。 |
| `--kv-offload-load-cost-beta` | `1.0` | 从 host 加载 KV 回 GPU 的 cost 权重。 |
| `--kv-offload-evict-cost-gamma` | `1.0` | 将 GPU KV 驱逐到 host 的 cost 权重。 |
| `--kv-offload-recompute-cost-delta` | `1.0` | recompute cost 项权重。 |
| `--kv-offload-default-output-speed` | `5.0` | 请求没有指定 `output_speed` 时使用的默认 token 消费速度，单位 tok/s。 |
| `--kv-offload-buffer-conservativeness` | `2.0` | request buffer 的保守系数。越大越保守，越不容易暂停或 offload 正在流式输出的请求。 |
| `--kv-offload-enable-emergency-eviction` | 关闭 | 显存压力下启用紧急驱逐路径。 |
| `--kv-offload-emergency-min-evict-tokens` | `256` | 每次紧急驱逐至少释放的 token 数。 |
| `--kv-offload-emergency-decode-retry` | `1` | decode 阶段紧急驱逐后的重试次数。 |
| `--kv-offload-emergency-prefill-retry` | `1` | prefill 阶段紧急驱逐后的重试次数。 |
| `--kv-offload-emergency-trigger-ratio` | `0.05` | 可用 token 比例低于该阈值时，允许触发紧急路径。 |

注意：

- `paper_v1` 需要配合 `--enable-hierarchical-cache` 使用。
- `paper_v1` 不兼容 PD disaggregation，启动时不要同时设置非 `null` 的 `--disaggregation-mode`。

## 5. 请求级 `output_speed` 的传递方式

KV offload 里有两个层级的 output speed：

| 层级 | 传递方式 | 作用 |
| --- | --- | --- |
| 全局默认值 | `--kv-offload-default-output-speed 5` | 请求没有单独指定速度时使用。 |
| 请求级值 | 请求 sampling 参数里的 `output_speed`，最终进入 `custom_params["output_speed"]` | 针对每个请求指定 token 消费速度。 |

服务端调度器会读取请求上的 `output_speed`，用于估计该请求的输出 buffer 会以多快速度被消费：

- `output_speed` 越大，buffer 衰减越快；
- buffer 越容易变低，调度器越倾向于保护该请求继续生成；
- `output_speed` 未提供时，使用 `--kv-offload-default-output-speed`。

如果你用 OpenAI-compatible 接口发请求，可以在请求 JSON 里带上额外 sampling 参数。例如：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/model",
    "messages": [
      {"role": "user", "content": "写一个简短的测试。"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "output_speed": 5
  }'
```

代码里会把请求级 `output_speed` 转入 `custom_params["output_speed"]`，KV offload 调度器再从请求的 `custom_params` 中读取它。

如果请求里不传：

```bash
--kv-offload-default-output-speed 5
```

就是所有请求的默认消费速度。

## 6. 完整 KV Offload 示例

下面是一个更完整的启动示例：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -v /storage/nas/dch/models/MiniMax-M2.7:/models/model:ro \
  -e MODEL_PATH=/models/model \
  -e HOST=0.0.0.0 \
  -e PORT=30000 \
  -e SGLANG_ARGS='--tp-size 8 --ep-size 8 --dtype auto --trust-remote-code --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think --max-running-requests 128 --mem-fraction-static 0.6 --enable-hierarchical-cache --hicache-size 40 --kv-offload-policy paper_v1 --kv-offload-reschedule-interval 1000 --kv-offload-default-output-speed 5 --kv-offload-buffer-conservativeness 2.0' \
  tokenflow-sglang:cuda12.8
```

服务起来后检查健康状态：

```bash
curl http://127.0.0.1:30000/health
```

如果要保存日志：

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -v /storage/nas/dch/models/MiniMax-M2.7:/models/model:ro \
  -e MODEL_PATH=/models/model \
  -e HOST=0.0.0.0 \
  -e PORT=30000 \
  -e SGLANG_ARGS='--tp-size 8 --ep-size 8 --dtype auto --trust-remote-code --enable-hierarchical-cache --hicache-size 40 --kv-offload-policy paper_v1' \
  tokenflow-sglang:cuda12.8 2>&1 | tee sglang-kvoffload.log
```
