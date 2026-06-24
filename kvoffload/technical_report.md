# SGLang KV Cache 卸载适配技术报告

**分支**：`kvoffload`（基于上游 commit `505eb312e`，2026-03-31）  
**报告生成日期**：2026-06-23  
**作者**：s7a9 (dch_7723@outlook.com)

---

## 1. 概述

本报告梳理了 `kvoffload` 分支相对于上游 SGLang fork 基点（commit `505eb312e`）所做的全部改动。总体目标是在 SGLang 推理服务中引入**分层 KV Cache 管理**能力，具体包括：

1. **增量同步缓存（SyncCache）**：在 decode 阶段周期性地将请求私有的 KV Cache 同步到 Radix Tree，从而在请求未完成时也能将其 KV 数据卸载到 Host 内存。
2. **KV 卸载感知的调度策略（PaperV1OffloadReschedulePolicy）**：在调度决策时考虑 Host/GPU 间的加载与写回开销、请求的 buffer 水位和饥饿状态，以最大化服务吞吐同时保证输出流畅性。
3. **基准测试与分析工具集**：专为 KV 卸载场景编写的评测脚本、服务指标仿真与可视化工具。

### 1.1 提交历史（时序）

| 日期 | Commit | 说明 |
|------|--------|------|
| 2026-03-31 | `73a581d` | **初始实现**：SyncCache、调度策略、服务器参数 |
| 2026-03-31 | `9fb7a11` | 修复 sync_batch 和 evict 逻辑 |
| 2026-04-01 | `39fef2c` | SyncCache 以 HiRadix 节点为同步粒度基础 |
| 2026-04-13 | `cd3e52f` | 修复 kvoffload 基准测试 |
| 2026-04-13 | `11bdd16` | 新增等待队列长度统计与时序图改进 |
| 2026-04-14 | `ba94d8a` | 修复调度器：reschedule 前必须先完成一轮 decode |
| 2026-04-16 | `52d2028` | 新增 TTFT 分布 CDF 绘图工具 |
| 2026-04-22 | `02e31964` | 修复 sync cache 若干边界情况 |
| 2026-04-24 | `15839d0` | 使 KV 卸载重调度确定性化（基于 tick 而非 wall clock） |

### 1.2 变更文件统计

```
新增文件（kvoffload 工程目录）：
  kvoffload/bench_analysis.py                         +328 行
  kvoffload/plot_service_metrics_timeline.py           +416 行
  kvoffload/plot_ttft_cdf.py                           +209 行
  kvoffload/scripts/bench_serving.sh                    +88 行
  kvoffload/scripts/start_server_default.sh             +48 行
  kvoffload/scripts/start_server_paper_v1.sh            +55 行

新增 Python 模块：
  python/sglang/srt/mem_cache/sync_cache.py            +207 行

修改现有模块（净增量）：
  python/sglang/srt/managers/schedule_policy.py        +386 行
  python/sglang/srt/managers/scheduler.py              +279 行
  python/sglang/srt/managers/cache_controller.py       +155 行
  python/sglang/srt/server_args.py                     +169 行
  python/sglang/bench_serving.py                        +97 行
  python/sglang/srt/managers/scheduler_output_processor_mixin.py   +6 行
  python/sglang/srt/managers/scheduler_runtime_checker_mixin.py   +33 行
  python/sglang/srt/managers/tokenizer_manager.py       +8 行
  python/sglang/srt/mem_cache/memory_pool_host.py       -3 行（注释 assert）
```

---

## 2. 核心新增模块

### 2.1 SyncCache（`python/sglang/srt/mem_cache/sync_cache.py`）

#### 2.1.1 设计动机

原生的 `HiRadixCache` 仅在请求**完成**时才将其 KV Cache 写入 Radix Tree，这意味着对于长序列请求，在整个生成过程中其 KV 数据都占用着 GPU 显存且无法被卸载。`SyncCache` 通过在 prefill 和 decode 阶段周期性地将未完成请求的 KV Cache 增量同步到 Radix Tree，使卸载得以在请求完成之前发生。

#### 2.1.2 类结构

```
SyncCache(HiRadixCache)
  ├── SyncCacheReqState  — 每个请求的同步状态
  │     ├── rid: str
  │     ├── synced_len: int          — 已同步到 radix 的 token 数
  │     └── decode_since_last_sync: int  — 距上次同步的 decode 步数
  ├── sync_batch(batch)              — 主入口，每轮 forward 后调用
  ├── _sync_req_to_radix(...)        — 条件性地将请求同步到 radix
  ├── evict_device(req, seq_len)     — 主动卸载请求的 GPU KV Cache
  └── cache_finished_req(req, ...)   — 完成时清理状态
```

初始化时强制启用 `write_through` 策略且不使用 storage 后端（Host 内存作为唯一的卸载目标）。

#### 2.1.3 同步触发条件

同步遵循节流逻辑，避免频繁的细粒度 I/O：

**Prefill（EXTEND）模式**：
```
unsynced_len >= prefill_sync_chunk_size (默认 2048)
```

**Decode 模式**：
```
decode_since_last_sync >= decode_sync_stride_steps (512)
OR unsynced_len >= decode_sync_min_tokens (max(512, page_size))
```

`force=True` 参数可绕过阈值，用于强制同步（如主动卸载前）。

#### 2.1.4 主动卸载流程（`evict_device`）

```
evict_device(req) 执行步骤：
1. 强制同步：将未同步的 KV 数据写入 Radix（force=True）
2. 等待 write-through ack，确保 GPU→Host 传输完成
3. 对请求私有节点（lock_ref==1）触发 write_backup，将其持久化
4. 等待 write_backup 完成，随后调用 _evict_backuped 释放 GPU 显存
5. 释放请求的 lock_ref，释放非 Radix 保护的 tail 槽位
6. 调用 req_to_token_pool.free(req) 归还资源
```

---

### 2.2 KV 卸载调度策略（`python/sglang/srt/managers/schedule_policy.py`）

#### 2.2.1 新增数据结构

**`OffloadPolicyRuntimeSnapshot`**：每次 reschedule 时的调度器快照，包含：
- `running_reqs` / `waiting_reqs`：当前运行中和等待中的请求列表
- `available_tokens` / `evictable_tokens` / `protected_tokens`：GPU 显存分布
- `workload_snapshot`：来自 cache_controller 的 I/O 负载指标（加载/写回 backlog 和速度）
- `request_buffer_tokens` / `request_output_speeds` / `request_rebuffer_times`：每请求 buffer 状态

**`OffloadRescheduleDecision`**：策略输出，包含：
- `keep_running_list`：继续运行的请求
- `new_resume_list`：从等待队列恢复的请求（按优先级排列）
- `request_objectives`：每请求的优先级数值（供抢占机制使用）
- `objective_terms`：汇总的目标函数各项（供日志分析）

#### 2.2.2 PaperV1OffloadReschedulePolicy 算法

该策略是本次适配的核心贡献，使用加权目标函数对所有候选请求（运行中 + 等待中）进行排序与筛选。

**目标函数（token value 项）**：

```python
ibt_urgency       = 1.0 / (1.0 + max(ibt, 0.0))   # 当前 buffer 时长的紧迫性
decode_progress   = len(req.output_ids) + 1          # 已产生输出的进度
ttft_proxy        = 1.0 / max(len(req.origin_input_ids), 1)  # 短 prompt 优先
wait_utility      = max(schedule_tick - entry_tick, 0)  # 等待时间
starvation_utility = max(-buffer_tokens, 0) + rebuffer_time  # 饥饿惩罚

token_value = α × (
    1.5 × ibt_urgency         # 最高权重：输出流畅性
  + 0.6 × decode_progress     # 进度奖励
  + 0.2 × ttft_proxy          # TTFT 亲和
  + 0.02 × wait_utility       # 防饥饿
  + 0.1 × starvation_utility  # 已发生饥饿的补偿
)
```

**代价项（等待请求恢复时）**：

```python
load_pressure    = 1.0 + (load_backlog / load_speed)
load_cost        = β × (host_hit_len / load_speed) × load_pressure
recompute_cost   = δ × uncached_len
buffer_penalty   = max(-predicted_buffer, 0.0)

# 恢复的目标分数
objective_resume = token_value - load_cost - recompute_cost - buffer_penalty
```

**代价项（运行中请求保留时）**：

```python
write_pressure   = 1.0 + (write_backlog / write_speed)
evict_cost       = γ × cache_protected_len × write_pressure

# 保留运行的目标分数
objective_keep   = token_value - buffer_penalty + evict_cost
```

**选择算法**：
1. 贪心选择：按目标分数降序排列所有候选
2. 平局处理：先比 `token_value`，再比总代价
3. 位置折扣：`0.985^position`（候选越靠后权重越低）
4. 可选的局部搜索（`enable_local_search=True`）：尝试相邻交换以进一步提升总目标

---

## 3. 修改的核心模块

### 3.1 调度器（`python/sglang/srt/managers/scheduler.py`）

新增约 279 行，主要集中在以下几个方面：

#### 3.1.1 初始化阶段（SyncCache 选择）

```python
# scheduler.py:776
if server_args.kv_offload_policy != "default":
    from sglang.srt.mem_cache.sync_cache import SyncCache
    self.tree_cache = SyncCache(params=params, server_args=server_args)
```

当策略为 `paper_v1` 时，用 `SyncCache` 替换默认的 `HiRadixCache`。

#### 3.1.2 请求 Buffer 状态管理

调度器维护三个字典，跟踪每个请求的 buffer 状态：

```python
self.offload_request_buffer_tokens: Dict[str, float]   # 当前 buffer token 数
self.offload_request_output_speeds: Dict[str, float]   # 消费速度（tok/s）
self.offload_request_rebuffer_times: Dict[str, float]  # 历史饥饿时长（s）
```

**Buffer 衰减**（`_decay_kv_offload_request_buffers`）：每轮调度器循环时，根据经过的 tick 数和请求的输出速度扣减 buffer，当 buffer 降为负值时累积 `rebuffer_time`（表示输出"断流"的时长）。

#### 3.1.3 Token 产出钩子

`_on_kv_offload_tokens_produced(req, token_count)` 在每产出 token 时调用，向对应请求的 buffer 中加入产出量。此方法通过 `scheduler_output_processor_mixin.py` 的钩子触发（见 3.2 节）。

#### 3.1.4 重调度主逻辑（`_maybe_apply_offload_reschedule_policy`）

触发条件：每隔 `kv_offload_reschedule_tick_interval` 个 schedule tick，且上一批不是 prefill 批次（防止卡死）。

执行流程：
1. GC 已结束请求的 buffer 状态
2. 对所有等待请求调用 `init_next_round_input`（计算 GPU hit / Host hit / uncached 分布）
3. 从 cache_controller 采集 I/O 负载快照
4. 构造 `OffloadPolicyRuntimeSnapshot` 并调用策略 `compute()`
5. 校验决策合法性后调用 `_apply_offload_reschedule_decision`

#### 3.1.5 决策应用（`_apply_offload_reschedule_decision`）

- 将 `request_objectives` 写入对应请求的 `priority` 字段，供原有的优先级抢占机制使用
- 重排等待队列：`new_resume_list` 中的请求优先，其余按原顺序追加
- 不直接驱逐运行中的请求，而是通过 priority 驱动 `PrefillAdder` 中原有的抢占逻辑

#### 3.1.6 增量同步调用

```python
# scheduler.py 每轮 step 结束后
if hasattr(tree_cache, "sync_batch"):
    tree_cache.sync_batch(self.last_batch)
```

### 3.2 SchedulerOutputProcessorMixin（`scheduler_output_processor_mixin.py`）

新增 6 行，在两处 token 产出点插入钩子：

1. 普通 decode 路径（非投机解码）：每产出 1 个 token 时
2. Spec v2 decode 路径：每轮接受 `new_accepted_len` 个 token 时

```python
if hasattr(self, "_on_kv_offload_tokens_produced"):
    self._on_kv_offload_tokens_produced(req, token_count)
```

使用 `hasattr` 保持向后兼容，非 KV 卸载场景无性能开销。

### 3.3 SchedulerRuntimeCheckerMixin（`scheduler_runtime_checker_mixin.py`）

新增 `check_kv_offload_policy_decision` 方法，对策略输出做完整性验证：

- `keep_running_list` 不得包含重复 rid
- `new_resume_list` 不得包含重复 rid
- `keep_running_list` 必须是当前运行请求的子集
- `new_resume_list` 必须是当前等待请求的子集

验证失败时记录 warning 并丢弃本次决策（保持现状），避免非法调度。

### 3.4 TokenizerManager（`tokenizer_manager.py`）

新增 8 行，向后兼容地将请求级 `output_speed` 参数透传至调度器：

```python
# 若 sampling_params 中包含 output_speed，将其移入 custom_params
if "output_speed" in sampling_kwargs:
    custom_params["output_speed"] = sampling_kwargs.pop("output_speed")
    sampling_kwargs["custom_params"] = custom_params
```

调度器通过 `_resolve_kv_offload_output_speed` 从 `custom_params` 读取该值，用于 buffer 衰减计算。未指定时使用全局默认值 `kv_offload_default_output_speed`（5.0 tok/s）。

### 3.5 MemoryPoolHost（`memory_pool_host.py`）

注释掉了一个限制性的 assert：

```python
# assert (
#     self.size > device_pool.size
# ), "The host memory should be larger than the device memory..."
```

原来的断言要求 Host 内存必须大于 GPU 显存，但在 KV 卸载场景下 Host 内存可以合理地小于 GPU 总容量（仅作为临时缓冲层），因此去除此约束。

### 3.6 server_args.py

新增 14 个 KV 卸载相关配置参数（附验证逻辑）：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `kv_offload_policy` | str | `"default"` | 调度策略：`"default"` 或 `"paper_v1"` |
| `kv_offload_reschedule_interval` | int | `5000` | 重调度间隔（tick 数） |
| `kv_offload_token_value_alpha` | float | `1.0` | token_value 项权重 α |
| `kv_offload_load_cost_beta` | float | `1.0` | load_cost 项权重 β |
| `kv_offload_evict_cost_gamma` | float | `1.0` | evict_cost 项权重 γ |
| `kv_offload_recompute_cost_delta` | float | `1.0` | recompute_cost 项权重 δ |
| `kv_offload_enable_local_search` | bool | `True` | 是否启用局部搜索优化 |
| `kv_offload_default_output_speed` | float | `5.0` | 默认 token 消费速度（tok/s） |
| `kv_offload_buffer_conservativeness` | float | `2.0` | buffer 衰减保守系数 |
| `kv_offload_enable_emergency_eviction` | bool | `False` | 是否启用紧急驱逐 |
| `kv_offload_emergency_min_evict_tokens` | int | `256` | 紧急驱逐最小 token 数 |
| `kv_offload_emergency_decode_retry` | int | `1` | 紧急驱逐后 decode 重试次数 |
| `kv_offload_emergency_prefill_retry` | int | `1` | 紧急驱逐后 prefill 重试次数 |
| `kv_offload_emergency_trigger_ratio` | float | `0.05` | 可用显存低于总量 5% 时触发紧急驱逐 |

同时增加了 `_handle_kv_offload_policy()` 验证函数：检查参数范围、策略合法性，并确认 KV 卸载策略与 disaggregation 模式不兼容（互斥）。

### 3.7 bench_serving.py

新增约 97 行，增强基准测试的输出能力：

- 新增 `--save-request-outputs` 选项，将每个请求的 TTFT/ITL/延迟详情保存至 JSONL
- 新增 `--tag` 选项，为每次测试打标签，便于多组实验对比
- 新增 `--output-speed` 选项，将 token 消费速度嵌入请求的 `custom_params` 传给服务器
- 每行 JSONL 记录中保存完整的 `ttfts`、`itls`、`output_lens`，供时序仿真使用

---

## 4. 基准测试与分析工具（`kvoffload/` 目录）

### 4.1 服务器启动脚本

**`scripts/start_server_default.sh`**：基线配置，不启用 HiCache，其他参数（TP=8、EP=8、max 128 并发、60% 静态显存占比）与实验组保持一致。

**`scripts/start_server_paper_v1.sh`**：启用 KV 卸载实验配置：
```bash
--enable-hierarchical-cache \
--hicache-size 40000         # 40GB Host 内存缓冲
--kv-offload-policy paper_v1
```

### 4.2 基准测试脚本（`scripts/bench_serving.sh`）

封装了对 `bench_serving.py` 的调用，支持：
- 随机数据集生成（可配置 input/output 长度范围）
- 自定义并发量和请求速率
- 结果输出为带 tag 的 JSONL，方便后续分析

### 4.3 结果分析工具

**`bench_analysis.py`**：将 JSONL 基准结果转换为汇总指标（P50/P90/P99 的 TTFT、ITL、E2E 延迟），导出为 CSV 并绘制分组 token 到达时间轴对比图。

**`plot_service_metrics_timeline.py`**：核心仿真与可视化工具。  
给定基准结果 JSONL 和 token 消费速度，在时间轴上仿真每个请求的 buffer 演化，输出以下指标随时间的变化曲线：
- `waiting_requests_num`：等待中的请求数量
- `queued_requests_num`：已到达但尚未收到第一个 token 的请求数量（2026-04-13 新增）
- `valid_throughput_tps`：有效吞吐量（token/s）

支持 `--time-range START END` 参数限制绘图时间窗口（2026-04-13 新增）。

**`plot_ttft_cdf.py`**：绘制各组实验的 TTFT CDF 曲线，对比 KV 卸载对首 token 延迟分布的影响（2026-04-16 新增）。

---

## 5. 技术架构总结

### 5.1 端到端数据流

```
请求到达
  ↓
TokenizerManager 提取 output_speed → custom_params
  ↓
Scheduler 收到请求，初始化 buffer_tokens=0, output_speed
  ↓
[每隔 reschedule_tick_interval 个 tick]
_maybe_apply_offload_reschedule_policy()
  ├── init_next_round_input() → 计算 GPU hit / Host hit / uncached
  ├── 采集 cache_controller I/O 负载快照
  ├── 构造 OffloadPolicyRuntimeSnapshot
  ├── PaperV1Policy.compute() → 目标函数排序
  └── _apply_offload_reschedule_decision() → 重排等待队列 + 设置 priority
  ↓
Scheduler 批次构造（PrefillAdder）
→ 高优先级请求抢占低优先级，低优先级请求卸载 GPU KV Cache
  ↓
[每轮 decode forward 后]
SyncCache.sync_batch() → 增量同步 KV 到 Radix Tree
  ↓
[每产出 token]
_on_kv_offload_tokens_produced() → buffer_tokens += 1
  ↓
[每轮 step 结束]
_decay_kv_offload_request_buffers()
→ buffer_tokens -= elapsed_s × output_speed
→ 若 buffer < 0：rebuffer_time 累积
```

### 5.2 设计特点

**确定性跨 TP Rank 一致性**：使用 `schedule_tick`（调度器内部计数器）而非 wall-clock time 触发重调度，确保多卡并行时所有 rank 在相同迭代触发同样的调度决策（`15839d0` 修复）。

**最小侵入性**：所有 KV 卸载逻辑通过 `hasattr` 守卫和可选初始化嵌入，默认路径（`kv_offload_policy="default"`）无任何运行时开销。

**渐进式卸载**：`SyncCache` 的分块同步避免了大批量 I/O，与 HiCache 的异步 write-through 机制配合，最小化对 decode 延迟的影响。

**不直接抢占**：重调度策略不直接中断运行中的请求，而是通过优先级调整利用已有的 `PrefillAdder` 抢占逻辑，降低了实现复杂度并复用了已有的调度安全保障。

---

## 6. 已知限制与未完成项

1. **紧急驱逐（Emergency Eviction）**：服务器参数中定义了相关配置（`kv_offload_enable_emergency_eviction` 等），但调度器中尚未看到完整的实现路径，可能仅作为预留接口。

2. **与 Disaggregation 模式不兼容**：`server_args.py` 中明确验证了两者互斥，P/D 分离模式下不支持 KV 卸载重调度策略。

3. **Host 内存大小约束放松**：`memory_pool_host.py` 中注释掉的 assert 意味着 Host 内存可以小于 GPU 内存，但调度器目前并未针对 Host 内存不足的情况做降级处理。

4. **benchmark 数据依赖**：`bench_results.jsonl` 文件已提交到仓库，不适合长期保留在版本控制中。
