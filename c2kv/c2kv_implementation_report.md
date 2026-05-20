# C2KV Implementation Report

**Commit:** `aa33d1efb`  
**Branch:** `c2kv-v0.5.10`  
**Date:** 2026-05-18  
**Scope:** SGLang + Qwen3

---

## 1. What is C2KV

C2KV (Concatenable and Compressible KV Cache) is a technique for amortising the prefill cost of
long, repeated documents. A document is first compressed into a small number of *gist tokens* via
a dedicated extraction pass. The KV cache of these gist tokens is stored on GPU. When a subsequent
request references the same document, the gist KV is injected directly into the live KV cache pool
skipping the full document prefill entirely.

The trade-off: one expensive extraction pass per unique document, paid once, in exchange for
dramatically cheaper prefill on every repeated query.

---

## 2. High-level architecture

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

## 3. New files

| File | Purpose |
|------|---------|
| `python/sglang/srt/mem_cache/c2kv_pool.py` | GPU-resident LRU store for gist KV entries |
| `python/sglang/srt/mem_cache/c2kv_injection.py` | RoPE reposition + KV pool write |
| `python/sglang/srt/models/gist_utils.py` | Attention mask and position ID builders |

---

## 4. Modified files

| File | Changes |
|------|---------|
| `server_args.py` | 4 new flags: `enable_c2kv`, `c2kv_gist_type`, `c2kv_gist_param`, `c2kv_pool_size` |
| `models/qwen3.py` | `gist_qkv_proj`, `forward_with_gist`, `generate_gist`, weight loader mappings |
| `model_executor/model_runner.py` | `forward_c2kv_extract` pass-through |
| `model_executor/forward_batch_info.py` | `c2kv_position_corrections` tensor + position patching |
| `managers/io_struct.py` | `C2KVSegmentInfo`, `TokenizedExtractReqInput`, `C2KVExtractReqOutput`, `c2kv_segments` on generate types |
| `managers/schedule_batch.py` | `C2KVPrefillRound`, 6 c2kv fields on `Req`, `c2kv_position_corrections` on `ModelWorkerBatch` |
| `managers/scheduler.py` | Pool init, `handle_extract_request`, `_build_c2kv_prefill_rounds`, `_inject_c2kv_gist_segment`, early-process flag |
| `managers/scheduler_output_processor_mixin.py` | Post-round injection hook, requeue logic, seq_lens patch |
| `managers/tokenizer_manager.py` | `c2kv_extract_communicator`, `c2kv_extract()` async method |
| `entrypoints/openai/protocol.py` | `c2kv_key_hash` on message types, `C2KVExtractRequest/Response` |
| `entrypoints/openai/serving_chat.py` | `_compute_c2kv_segments()`, hook before `_process_messages` |
| `entrypoints/http_server.py` | `POST /v1/c2kv/extract` endpoint |

---

## 5. Component details

### 5.1 C2KV Pool (`c2kv_pool.py`)

An in-memory, GPU-resident LRU cache bounded by `max_total_tokens` (sum of `gist_len` across all
entries). Uses `collections.OrderedDict` for O(1) LRU promotion.

**Key design choice — hash over token IDs, not text:**

```python
@staticmethod
def compute_hash(token_ids: List[int]) -> str:
    raw = struct.pack(f"{len(token_ids)}i", *token_ids)
    return hashlib.sha256(raw).hexdigest()
```

Hashing token IDs (not raw text) ensures the hash is stable across whitespace/encoding
differences and matches exactly what the tokenizer produces.

**Synthetic token IDs for radix cache compatibility:**

```python
C2KV_GIST_TOKEN_BASE = 1 << 60

def c2kv_gist_token_ids(key_hash: str, gist_len: int) -> List[int]:
    base_hash = struct.unpack(">Q", bytes.fromhex(key_hash[:16]))[0]
    base = C2KV_GIST_TOKEN_BASE + (base_hash % (1 << 59))
    return [base + j for j in range(gist_len)]
```

These IDs occupy the high bit range far above any real vocabulary token, so they can coexist
with normal tokens in the radix cache's token sequence tracking without collisions.

### 5.2 Gist utilities (`gist_utils.py`)

`prepare_gist_input(input_ids, attention_mask, ratio=4)` returns:
- `block_mask` — pre-compiled `BlockMask` for `flex_attention` (created once, reused across all layers)
- `gist_mask` — all-True `(1, gist_len)` bool tensor
- `position_ids` — `(1, total_len)` int64; input tokens at positions `[0, seq_len)`, gist token j
  at `min((j+1)*ratio-1, seq_len-1)` (positioned at the end of the chunk it summarises)

The `block_mask` encodes a 2×2 block attention pattern:
- input→input: causal
- input→gist: blocked (no future leakage)
- gist→input: full (gist tokens attend all source tokens)
- gist→gist: causal

`get_apply_gist_residual_func` returns identity in the current implementation (no residual connection).

### 5.3 Gist injection (`c2kv_injection.py`)

`inject_c2kv_gist(entry, position_cursor, loc, token_to_kv_pool, attn_layers, cos_sin_cache)`

For each layer:
1. Compute `abs_pos[j] = position_cursor + gist_position_ids[0, j]` (clamped to cos_sin_cache size).
2. Apply RoPE to K: `k_rotated = apply_rotary_emb(k, cos[abs_pos], sin[abs_pos])`.
3. Write: `token_to_kv_pool.set_kv_buffer(attn_layers[i], loc, cache_k=k_rotated, cache_v=v)`.

V is written **without** RoPE (V does not participate in the Q·K attention dot product and does
not need RoPE).

### 5.4 Qwen3 model changes (`qwen3.py`)

**`Qwen3Attention`:**
- Adds `self.gist_qkv_proj` (same shape/parallelism as `qkv_proj`, separate weights trained for
  gist token projection).
- Adds `self.flex_attention = torch.compile(flex_attention)` — compiled once at init, reused
  across all `forward_with_gist` calls.
- New method `forward_with_gist`: splits hidden states, projects input/gist separately, concatenates
  Q/K/V, saves pre-RoPE gist KV, applies RoPE, runs `flex_attention` with the prebuilt `block_mask`.

**`Qwen3Model`:**
- `_init_c2kv`: builds `GistConfig`, creates `gist_embed_tokens` embedding, stores `prepare_gist_input`
  and `apply_gist_residual` closures.

**`Qwen3ForCausalLM`:**
- `generate_gist(input_ids, attention_mask, ratio)`: builds gist embeddings (using index 0 of
  `gist_embed_tokens` for all gist positions), calls `layer.forward_with_gist` for each layer,
  returns `(gist_key_values, gist_mask, gist_position_ids)`.
- `load_weights`: three extra stacked-param mappings for `gist_q/k/v_proj → gist_qkv_proj`.

### 5.5 Multi-round prefill orchestration

When a generate request arrives with `c2kv_segments`:

**`_build_c2kv_prefill_rounds`** splits the token stream into alternating (normal tokens, inject)
phases:

```
Input token stream (virtual):
  [normal tokens] [gist_A injection point] [normal tokens] [gist_B injection point] [normal tokens]

Rounds:
  Round 0: prefill normal tokens before A → inject A → round done
  Round 1: prefill normal tokens between A and B → inject B → round done
  Round 2: prefill remaining normal tokens → decode begins
```

Each `C2KVPrefillRound` holds:
- `tokens`: normal token IDs to prefill in this round
- `post_inject_seg_indices`: segment indices to inject after this round's prefill completes

The full virtual token ID sequence (normal + synthetic gist IDs) is pre-built into
`req.c2kv_full_origin_input_ids` and restored to `req.origin_input_ids` after all rounds, so the
output processor and decode path see the complete logical sequence.

**`_inject_c2kv_gist_segment`** is called by the output processor after each round:
1. Trims any stale KV from the current request's KV range.
2. Allocates `gist_len` fresh KV slots.
3. Writes slot indices into `req_to_token_pool`.
4. Calls `inject_c2kv_gist` to RoPE-reposition and write KV.
5. Updates `req.kv_committed_len` and `req.c2kv_position_correction`.

`c2kv_position_correction` accumulates the gap between each prior segment's virtual token span
and its compressed gist span: `correction += original_seq_len - gist_len`. This correction is
applied to all subsequent position IDs (both normal prefill and decode) via `ForwardBatch`.

**Scheduler early-process flag:** when any in-flight request has pending c2kv rounds, the
scheduler disables prefill–decode overlap and forces synchronous output processing to guarantee
injections happen before the next batch is constructed.

### 5.6 Position correction in `ForwardBatch`

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

This ensures that tokens after the compressed segment see absolute positions as if the full
original document had been prefilled, preserving RoPE continuity for the rest of the sequence.

### 5.7 HTTP API

**`POST /v1/c2kv/extract`**

Request:
```json
{"text": "<document>", "compression_ratio": 4, "role": "user"}
```

Response:
```json
{"key_hash": "sha256hex...", "gist_len": 128, "original_seq_len": 512, "success": true}
```

The `role` field controls how the text is tokenized. When set, the endpoint applies
`apply_chat_template` with a dummy prefix and slices out the token IDs for the target message —
exactly mirroring what `_compute_c2kv_segments` does in serving_chat. This ensures the hash
computed at extraction time is consistent with the hash expected during generation.

**`POST /v1/chat/completions` (with c2kv annotations)**

Any message can carry `"c2kv_key_hash": "..."`. Before tokenization, `serving_chat._compute_c2kv_segments` removes annotated messages from the prompt and records zero-length injection points at the positions where those messages would have appeared. The normal tokenizer then processes the remaining prompt, and the scheduler injects gist KV at the recorded positions.

---

## 6. Data flow summary

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

## 7. Key design decisions and trade-offs

| Decision | Rationale |
|----------|-----------|
| Hash over token IDs, not text | Tokenizer-aligned; stable across text encoding differences |
| Synthetic gist token IDs (`1<<60` range) | Allows radix cache to track gist slots alongside normal tokens without ID collision |
| `block_mask` built in `prepare_gist_input`, not in each layer | Avoids `create_block_mask` overhead per layer; single precompiled mask reused across all layers |
| `torch.compile(flex_attention)` stored per-layer at init | Avoids recompilation per forward call; the compiled kernel is reused across all extraction requests |
| Pre-RoPE gist KV storage | RoPE depends on absolute position, which changes per request. Storing pre-RoPE KV and repositioning at injection time is the only correct approach |
| Zero-length segments (`token_start == token_end`) | Annotated message content is removed from the prompt; gist KV is injected at the gap, not merged with surrounding text tokens |
| `c2kv_requeued` flag | Prevents the scheduler from treating a requeued multi-round request as a new request during chunked-prefill state checks |
| Early-process flag to disable overlap | Overlap (prefill and decode in the same step) is unsafe when gist injections must happen between rounds; the flag serialises these steps cleanly |
| Dedicated `c2kv_extract_communicator` | Avoids multiplexing extraction responses through the generate output stream, which has different response types and lifecycle |

---

## 8. Limitations and future work

| Item | Notes |
|------|-------|
| Qwen3-only | `forward_with_gist` and `generate_gist` are Qwen3-specific. Other models need equivalent implementations. |
| Batch-size-1 extraction | `generate_gist` processes one document at a time. Batched extraction would need ragged attention or padding. |
| Pool not persistent | `C2KVPool` lives in process memory. Server restart = full pool loss; clients must re-extract. |
| `flex_attention` / PyTorch ≥ 2.5 | Hard requirement for extraction. Older PyTorch needs a manual masked SDPA fallback. |
| Overlap serialisation | `c2kv_early_process` disables prefill–decode overlap for the entire batch when any request has pending rounds, reducing throughput under mixed workloads. |
| Role-hash coupling | The `role` in `/v1/c2kv/extract` must match the chat message role for the hash to be consistent. Mismatches silently produce cache misses. |
| No multi-document batching in generation | Each request processes at most one active c2kv round at a time; batching across multiple requests with independent round schedules is not optimised. |
