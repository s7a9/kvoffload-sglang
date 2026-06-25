# C2KV 实现报告

**提交：** `aa33d1efb`  
**分支：** `c2kv-v0.5.10`  
**日期：** 2026-05-18  
**范围：** SGLang + Qwen3

---

## 1. 什么是 C2KV

C2KV（Concatenable and Compressible KV Cache，可拼接且可压缩的 KV Cache）是一种用于摊销长篇、重复文档 prefill 成本的技术。文档会先通过一次专用的提取流程，被压缩成少量 *gist tokens*。这些 gist tokens 的 KV cache 会存储在 GPU 上。当后续请求引用同一份文档时，gist KV 会被直接注入到当前请求的实时 KV cache 中，从而完全跳过完整文档的 prefill。

这里的取舍是：每个唯一文档只支付一次昂贵的提取成本，换取之后每次重复查询时大幅降低的 prefill 成本。

---

## 2. 高层架构

```
Client
  │
  ├─ POST /v1/c2kv/extract  ──► TokenizerManager.c2kv_extract()
  │                               └─► Scheduler.handle_extract_request()
  │                                     └─► ModelRunner.forward_c2kv_extract()
  │                                           └─► Qwen3ForCausalLM.generate_gist()
  │                                     └─► C2KVPool.store()  ← GPU RAM, LRU
  │
  └─ POST /v1/chat/completions
       (messages with c2kv_key_hash)
         │
         ├─ serving_chat._compute_c2kv_segments()
         │    builds C2KVSegmentInfo list (injection points in token stream)
         │
         ├─ Scheduler._build_c2kv_prefill_rounds()
         │    splits request into normal-token rounds + post-round injection lists
         │
         └─ [prefill round N completes]
              scheduler_output_processor_mixin
                ├─ Scheduler._inject_c2kv_gist_segment()  ← RoPE + KV write
                ├─ if more rounds: requeue at front of waiting_queue
                └─ if done: patch seq_lens, hand off to decode
```

---

## 3. 新增文件

| 文件 | 用途 |
|------|------|
| `python/sglang/srt/mem_cache/c2kv_pool.py` | 常驻 GPU 的 gist KV 条目 LRU 存储 |
| `python/sglang/srt/mem_cache/c2kv_injection.py` | RoPE 重定位 + KV pool 写入 |
| `python/sglang/srt/models/gist_utils.py` | Attention mask 和 position ID 构造工具 |

---

## 4. 修改文件

| 文件 | 变更 |
|------|------|
| `server_args.py` | 新增 5 个参数：`enable_c2kv`、`c2kv_gist_type`、`c2kv_gist_param`、`c2kv_pool_fraction`、`c2kv_max_tokens` |
| `models/qwen3.py` | `gist_qkv_proj`、`forward_with_gist`、`generate_gist`、权重加载映射 |
| `model_executor/model_runner.py` | `forward_c2kv_extract` 透传 |
| `model_executor/forward_batch_info.py` | `c2kv_position_corrections` tensor + position 修正 |
| `managers/io_struct.py` | `C2KVSegmentInfo`、`TokenizedExtractReqInput`、`C2KVExtractReqOutput`，以及 generate 类型上的 `c2kv_segments` |
| `managers/schedule_batch.py` | `C2KVPrefillRound`、`Req` 上的 6 个 c2kv 字段、`ModelWorkerBatch` 上的 `c2kv_position_corrections` |
| `managers/scheduler.py` | Pool 初始化、`handle_extract_request`、`_build_c2kv_prefill_rounds`、`_inject_c2kv_gist_segment`、early-process 标记 |
| `managers/scheduler_output_processor_mixin.py` | 轮次后注入 hook、重新入队逻辑、seq_lens 修正 |
| `managers/tokenizer_manager.py` | `c2kv_extract_communicator`、`c2kv_extract()` 异步方法 |
| `entrypoints/openai/protocol.py` | message 类型上的 `c2kv_key_hash`，`C2KVExtractRequest/Response` |
| `entrypoints/openai/serving_chat.py` | `_compute_c2kv_segments()`，以及 `_process_messages` 之前的 hook |
| `entrypoints/http_server.py` | `POST /v1/c2kv/extract` 端点 |

---

## 5. 组件细节

### 5.1 C2KV Pool（`c2kv_pool.py`）

一个内存中的、常驻 GPU 的 LRU cache，通过 `max_total_tokens` 进行容量限制（即所有条目的 `gist_len` 之和）。使用 `collections.OrderedDict` 实现 O(1) 的 LRU 提升。

**关键设计选择——基于 token ID 计算 hash，而不是基于文本：**

```python
@staticmethod
def compute_hash(token_ids: List[int]) -> str:
    raw = struct.pack(f"{len(token_ids)}i", *token_ids)
    return hashlib.sha256(raw).hexdigest()
```

对 token ID（而不是原始文本）做 hash，可以保证 hash 在空白符、编码差异等情况下保持稳定，并且与 tokenizer 实际产出的内容完全一致。

**用于 radix cache 兼容性的合成 token ID：**

```python
C2KV_GIST_TOKEN_BASE = 1 << 60

def c2kv_gist_token_ids(key_hash: str, gist_len: int) -> List[int]:
    base_hash = struct.unpack(">Q", bytes.fromhex(key_hash[:16]))[0]
    base = C2KV_GIST_TOKEN_BASE + (base_hash % (1 << 59))
    return [base + j for j in range(gist_len)]
```

这些 ID 位于远高于任何真实词表 token 的高位范围，因此可以与普通 token 一起存在于 radix cache 的 token 序列追踪中，而不会发生冲突。

### 5.2 Gist 工具（`gist_utils.py`）

`prepare_gist_input(input_ids, attention_mask, ratio=4)` 返回：

- `block_mask`：为 `flex_attention` 预编译的 `BlockMask`（只创建一次，并在所有 layer 中复用）
- `gist_mask`：全 True 的 `(1, gist_len)` bool tensor
- `position_ids`：`(1, total_len)` int64；输入 token 位于 `[0, seq_len)`，第 j 个 gist token 位于 `min((j+1)*ratio-1, seq_len-1)`（放置在其所摘要 chunk 的末尾）

`block_mask` 编码了一个 2×2 block attention 模式：

- input→input：causal
- input→gist：blocked（避免未来信息泄漏）
- gist→input：full（gist token 可以关注所有源 token）
- gist→gist：causal

`get_apply_gist_residual_func` 在当前实现中返回 identity（无 residual connection）。

### 5.3 Gist 注入（`c2kv_injection.py`）

`inject_c2kv_gist(entry, position_cursor, loc, token_to_kv_pool, attn_layers, cos_sin_cache)`

对于每一层：

1. 计算 `abs_pos[j] = position_cursor + gist_position_ids[0, j]`（并 clamp 到 cos_sin_cache 大小范围内）。
2. 对 K 应用 RoPE：`k_rotated = apply_rotary_emb(k, cos[abs_pos], sin[abs_pos])`。
3. 写入：`token_to_kv_pool.set_kv_buffer(attn_layers[i], loc, cache_k=k_rotated, cache_v=v)`。

V 会在**不经过** RoPE 的情况下写入（V 不参与 Q·K attention 点积，因此不需要 RoPE）。

### 5.4 Qwen3 模型变更（`qwen3.py`）

**`Qwen3Attention`：**

- 新增 `self.gist_qkv_proj`（形状和并行方式与 `qkv_proj` 相同，但使用单独训练的 gist token projection 权重）。
- 新增 `self.flex_attention = torch.compile(flex_attention)`，在初始化时编译一次，并在所有 `forward_with_gist` 调用中复用。
- 新方法 `forward_with_gist`：切分 hidden states，分别对 input/gist 做 projection，拼接 Q/K/V，保存 RoPE 之前的 gist KV，应用 RoPE，然后使用预构建的 `block_mask` 运行 `flex_attention`。

**`Qwen3Model`：**

- `_init_c2kv`：构造 `GistConfig`，创建 `gist_embed_tokens` embedding，保存 `prepare_gist_input` 和 `apply_gist_residual` 闭包。

**`Qwen3ForCausalLM`：**

- `generate_gist(input_ids, attention_mask, ratio)`：构造 gist embeddings（所有 gist 位置都使用 `gist_embed_tokens` 的 index 0），对每一层调用 `layer.forward_with_gist`，返回 `(gist_key_values, gist_mask, gist_position_ids)`。
- `load_weights`：新增三组 stacked-param 映射，将 `gist_q/k/v_proj` 映射到 `gist_qkv_proj`。

### 5.5 多轮 prefill 编排

当 generate 请求携带 `c2kv_segments` 到达时：

**`_build_c2kv_prefill_rounds`** 会将 token stream 拆分为交替的（普通 token，注入）阶段：

```
Input token stream (virtual):
  [normal tokens] [gist_A injection point] [normal tokens] [gist_B injection point] [normal tokens]

Rounds:
  Round 0: prefill normal tokens before A → inject A → round done
  Round 1: prefill normal tokens between A and B → inject B → round done
  Round 2: prefill remaining normal tokens → decode begins
```

每个 `C2KVPrefillRound` 包含：

- `tokens`：本轮需要 prefill 的普通 token ID
- `post_inject_seg_indices`：本轮 prefill 完成后需要注入的 segment 索引

完整的虚拟 token ID 序列（普通 token + 合成 gist ID）会被预先构造到 `req.c2kv_full_origin_input_ids` 中，并在所有轮次结束后恢复到 `req.origin_input_ids`。这样，output processor 和 decode 路径看到的是完整的逻辑序列。

**`_inject_c2kv_gist_segment`** 会在每轮结束后由 output processor 调用：

1. 裁剪当前请求 KV 范围内的任何陈旧 KV。
2. 分配 `gist_len` 个新的 KV slot。
3. 将 slot 索引写入 `req_to_token_pool`。
4. 调用 `inject_c2kv_gist` 完成 RoPE 重定位并写入 KV。
5. 更新 `req.kv_committed_len` 和 `req.c2kv_position_correction`。

`c2kv_position_correction` 累积每个已处理 segment 的虚拟 token 跨度与压缩后 gist 跨度之间的差值：`correction += original_seq_len - gist_len`。这个 correction 会通过 `ForwardBatch` 应用到后续所有 position ID（包括普通 prefill 和 decode）。

**Scheduler early-process 标记：** 当任何飞行中的请求仍存在待处理的 c2kv 轮次时，scheduler 会禁用 prefill–decode overlap，并强制同步处理输出，以保证注入发生在下一个 batch 构造之前。

### 5.6 `ForwardBatch` 中的 position correction

```python
if batch.c2kv_position_corrections is not None:
    corr = torch.tensor(batch.c2kv_position_corrections, dtype=torch.int64, device=device)
    if is_decode:
        ret.positions = ret.positions + corr          # scalar per request
    else:
        ext_lens = torch.tensor(batch.extend_seq_lens, ...)
        per_token_corr = torch.repeat_interleave(corr, ext_lens)
        ret.positions = ret.positions + per_token_corr  # broadcast per token
```

这可以确保压缩 segment 之后的 token 看到的绝对位置，就像完整原始文档已经被 prefill 过一样，从而为后续序列保持 RoPE 连续性。

### 5.7 HTTP API

**`POST /v1/c2kv/extract`**

请求：

```json
{"text": "<document>", "compression_ratio": 4, "role": "user"}
```

响应：

```json
{"key_hash": "sha256hex...", "gist_len": 128, "original_seq_len": 512, "success": true}
```

`role` 字段控制文本如何被 tokenized。设置该字段时，端点会使用一个 dummy prefix 应用 `apply_chat_template`，并切分出目标 message 对应的 token ID——这与 `serving_chat` 中 `_compute_c2kv_segments` 的行为完全一致。这样可以确保提取阶段计算出的 hash，与生成阶段期望的 hash 保持一致。

**`POST /v1/chat/completions`（携带 c2kv 注解）**

任意 message 都可以携带 `"c2kv_key_hash": "..."`。在 tokenization 之前，`serving_chat._compute_c2kv_segments` 会从 prompt 中移除带注解的 message，并在这些 message 原本出现的位置记录零长度注入点。随后普通 tokenizer 会处理剩余 prompt，scheduler 会在记录的位置注入 gist KV。

---

## 6. 数据流总结

```
extract phase:
  input_ids (1, L) → generate_gist() → gist_key_values (num_layers × (gist_len, kv_size)), pre-RoPE
  → C2KVPool.store(key_hash, gist_key_values, gist_mask, gist_position_ids, original_seq_len)

generate phase:
  request.messages (some annotated with c2kv_key_hash)
  → _compute_c2kv_segments()         # injection point list
  → _build_c2kv_prefill_rounds()     # split into normal-token rounds
  → [for each round]
      prefill normal tokens
      → [output processor]
          _inject_c2kv_gist_segment() # alloc slots, RoPE, write KV
          → requeue or fall through to decode
  → decode with corrected positions
```

---

## 7. 关键设计决策与取舍

| 决策 | 理由 |
|------|------|
| 基于 token ID 计算 hash，而不是基于文本 | 与 tokenizer 对齐；在文本编码差异下保持稳定 |
| 合成 gist token ID（`1<<60` 范围） | 允许 radix cache 在不发生 ID 冲突的情况下，将 gist slot 与普通 token 一起追踪 |
| 在 `prepare_gist_input` 中构造 `block_mask`，而不是每层构造 | 避免每层重复产生 `create_block_mask` 开销；单个预编译 mask 可在所有层复用 |
| 在初始化时按层保存 `torch.compile(flex_attention)` | 避免每次 forward 重新编译；编译后的 kernel 可在所有 extraction 请求中复用 |
| 存储 RoPE 之前的 gist KV | RoPE 依赖绝对位置，而绝对位置会随请求变化。存储 pre-RoPE KV 并在注入时重新定位，是唯一正确的方式 |
| 零长度 segment（`token_start == token_end`） | 带注解的 message 内容会从 prompt 中移除；gist KV 注入在间隙处，而不是与周围文本 token 合并 |
| `c2kv_requeued` 标记 | 防止 scheduler 在 chunked-prefill 状态检查时，将重新入队的多轮请求误判为新请求 |
| 使用 early-process 标记禁用 overlap | 当 gist 注入必须发生在轮次之间时，overlap（同一步中同时 prefill 和 decode）是不安全的；该标记会清晰地串行化这些步骤 |
| 专用的 `c2kv_extract_communicator` | 避免将 extraction 响应混入 generate 输出流，因为二者的响应类型和生命周期不同 |

---

## 8. 限制与未来工作

| 项目 | 说明 |
|------|------|
| 仅支持 Qwen3 | `forward_with_gist` 和 `generate_gist` 是 Qwen3 专用的。其他模型需要等价实现。 |
| extraction 仅 batch-size-1 | `generate_gist` 一次处理一个文档。批量 extraction 需要 ragged attention 或 padding。 |
| Pool 不持久化 | `C2KVPool` 存在于进程内存中。服务重启会导致整个 pool 丢失，客户端必须重新提取。 |
| `flex_attention` / PyTorch ≥ 2.5 | extraction 的硬性要求。旧版本 PyTorch 需要手写 masked SDPA fallback。 |
| overlap 串行化 | 只要 batch 中任一请求存在待处理轮次，`c2kv_early_process` 就会禁用整个 batch 的 prefill–decode overlap，在混合负载下会降低吞吐。 |
| role-hash 耦合 | `/v1/c2kv/extract` 中的 `role` 必须与 chat message 的 role 匹配，hash 才能保持一致。不匹配会静默地产生 cache miss。 |
| generation 中没有多文档批处理优化 | 每个请求一次最多处理一个活跃 c2kv 轮次；尚未优化多个请求各自独立轮次调度下的跨请求 batching。 |
