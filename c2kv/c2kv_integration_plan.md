# C2KV Integration Plan (Revised)

**Purpose:** Step-by-step instructions for integrating C2KV (Concatenable and Compressible KV
Cache) into SGLang. Written for a coding agent; each task is independently executable and specifies
exactly what files to touch, what to add, and how to verify.

> **Note:** This is the revised plan reflecting the actual implementation in commit `aa33d1efb`.
> Deviations from the original plan are annotated with **[Changed]** or **[Added]**.

**What C2KV does:** Compresses a long document into a small number of "gist tokens". The gist
tokens' KV cache is computed once during an *extraction* pass and stored on GPU. Subsequent
requests that reference the same document inject the stored gist KV directly into the live KV
cache, skipping the full prefill of the original document. The tradeoff is one expensive extraction
pass per unique document in exchange for dramatically smaller prefill on every repeated query.

---

## Prerequisites

Before starting, confirm:

1. The target model has standard transformer decoder layers with separate Q, K, V projections (or a
   fused QKV projection) and rotary position embeddings (RoPE).
2. The engine exposes a paged or contiguous KV cache where individual token slots can be written
   programmatically (`token_to_kv_pool.set_kv_buffer(layer, loc, K, V)`).
3. The engine has a mechanism to allocate KV pool slots independently of running a forward pass
   (`token_to_kv_pool_allocator.alloc(n)`).
4. PyTorch >= 2.5 is available (needed for `torch.nn.attention.flex_attention`).
5. The model checkpoint must include `gist_qkv_proj` and `gist_embed_tokens` weights trained for
   C2KV. If these weights are absent, the extraction pass will produce random output.

**Abbreviations:**
- *KV pool* = the GPU tensor holding all token KV states
- *Req* = the engine's per-request state object
- *kv_size* = `num_kv_heads_per_rank × head_dim` (TP-local shape per token per layer)

---

## Task 0 — Add server flags

**File:** `python/sglang/srt/server_args.py`

Add these fields to `ServerArgs` with defaults:

```python
# C2KV (Concatenable and Compressible KV Cache)
enable_c2kv: bool = False
c2kv_gist_type: str = "dynamic-interleave"
c2kv_gist_param: str = "qkv"
c2kv_pool_fraction: float = 0.01
c2kv_max_tokens: int = 4096
```

Add corresponding CLI flags in the argument parser section:

```
--enable-c2kv           (store_true)
--c2kv-gist-type STR    (default: "dynamic-interleave")
--c2kv-gist-param STR   (default: "qkv")
--c2kv-pool-fraction FLOAT    (default: 0.01)
--c2kv-max-tokens INT         (default: 4096)
```

**Verification:** `python -m sglang.launch_server --help` prints the four flags without error.

---

## Task 1 — Add the gist utility module

**File:** `python/sglang/srt/models/gist_utils.py`

This module builds the gist attention mask, position IDs, and optional residual connections. It
exports:

- `GistConfig` — dataclass holding all gist hyperparameters.
- `get_prepare_gist_input_func(gist_cfg)` — factory returning a closure.
- `get_apply_gist_residual_func(gist_cfg)` — factory returning identity (no residual in current impl).

### `GistConfig` fields

```python
@dataclass
class GistConfig:
    gist_type: str = "dynamic-interleave"
    gist_param: str = "qkv"
    gist_extra_embed_num: int = 1
    gist_token_id: Optional[int] = None
    gist_residual_type: str = "none"
    hidden_size: int = 4096
    attention_bias: bool = False
```

### `prepare_gist_input(input_ids, attention_mask, ratio=4)`

**[Changed]** The function returns `(block_mask, gist_mask, position_ids)` where `block_mask` is
already a compiled `BlockMask` object (from `flex_attention.create_block_mask`), **not** a raw
boolean tensor. This means `create_block_mask` is called once inside this utility, not inside the
model forward.

Shapes:

| Tensor | Shape | Dtype | Meaning |
|--------|-------|-------|---------|
| `block_mask` | `BlockMask` | — | Compiled flex-attention mask |
| `gist_mask` | `(1, gist_len)` | bool | All True (valid gist tokens) |
| `position_ids` | `(1, total_len)` | int64 | Absolute RoPE positions for every token |

Where `gist_len = ceil(seq_len / ratio)` and `total_len = seq_len + gist_len`.

Mask logic (encoded in the `block_mask`):
- input→input: causal lower-triangular
- input→gist: never (no future leak)
- gist→input: always (full attention to source)
- gist→gist: causal lower-triangular

Position IDs:
- Input token i → position i (0-indexed).
- Gist token j → `min((j+1)*ratio - 1, seq_len - 1)`.

**Verification:** Import and call `get_prepare_gist_input_func` with a `GistConfig`; confirm the
returned block_mask has type `BlockMask` and `gist_mask`/`position_ids` have correct shapes.

---

## Task 2 — Add the C2KV pool module

**File:** `python/sglang/srt/mem_cache/c2kv_pool.py`

GPU-resident LRU cache for compressed gist KV entries.

### Helper: `c2kv_gist_token_ids(key_hash, gist_len)`

**[Added]** Generates deterministic synthetic token IDs for gist KV slots so they can be
registered in the radix cache without colliding with real vocabulary tokens:

```python
C2KV_GIST_TOKEN_BASE = 1 << 60

def c2kv_gist_token_ids(key_hash: str, gist_len: int) -> List[int]:
    base_hash = struct.unpack(">Q", bytes.fromhex(key_hash[:16]))[0]
    base = C2KV_GIST_TOKEN_BASE + (base_hash % (1 << 59))
    return [base + j for j in range(gist_len)]
```

### `C2KVEntry` (dataclass)

```python
@dataclass
class C2KVEntry:
    key_hash: str
    gist_key_values: List[Tuple[torch.Tensor, torch.Tensor]]  # per-layer (K, V), pre-RoPE
    gist_mask: torch.Tensor          # (1, gist_len) bool
    gist_position_ids: torch.Tensor  # (1, gist_len) int64
    gist_len: int
    original_seq_len: int
```

### `C2KVPool`

```python
class C2KVPool:
    def __init__(self, max_total_tokens: int): ...

    @staticmethod
    def compute_hash(token_ids: List[int]) -> str:
        # SHA-256 of struct.pack("{n}i", *token_ids) — hashes over token IDs, NOT text
        raw = struct.pack(f"{len(token_ids)}i", *token_ids)
        return hashlib.sha256(raw).hexdigest()

    def store(self, key_hash, gist_key_values, gist_mask,
              gist_position_ids, original_seq_len) -> C2KVEntry: ...

    def get(self, key_hash) -> Optional[C2KVEntry]: ...
    def current_tokens(self) -> int: ...
    def num_entries(self) -> int: ...
```

**[Changed]** `compute_hash` takes `token_ids: List[int]`, not `text: str`. The hash is derived
from the exact token sequence, making it tokenizer-aligned and independent of text encoding
ambiguities.

Capacity invariant: `sum(entry.gist_len for entry in pool) <= max_total_tokens`.

**Verification:** Construct a pool, store a dummy entry, retrieve it, verify LRU eviction triggers
when capacity is exceeded.

---

## Task 3 — Add the C2KV injection module

**File:** `python/sglang/srt/mem_cache/c2kv_injection.py`

Reads a stored gist entry, applies RoPE at the correct absolute position, and writes K/V into the
engine's KV pool.

### `inject_c2kv_gist`

```python
def inject_c2kv_gist(
    entry: C2KVEntry,
    position_cursor: int,         # absolute position offset for this gist block
    loc: torch.Tensor,            # (gist_len,) int64 — flat slot indices in the KV pool
    token_to_kv_pool,             # engine's KV pool with set_kv_buffer()
    attn_layers: List,            # list of attention layer objects (one per decoder layer)
    cos_sin_cache: torch.Tensor,  # (max_pos, rotary_dim) float — RoPE lookup table
    is_neox_style: bool = True,
) -> None:
```

Steps:
1. Compute `abs_pos[j] = position_cursor + entry.gist_position_ids[0, j]`, clamped to
   `[0, cos_sin_cache.shape[0] - 1]`.
2. Gather `cos = cos_sin_cache[abs_pos, :head_dim//2]`, `sin = cos_sin_cache[abs_pos, head_dim//2:]`.
3. For each layer, reshape stored K from `(gist_len, kv_size)` to `(gist_len, num_kv_heads,
   head_dim)`, apply `apply_rotary_emb(k, cos, sin, is_neox_style)`.
4. Call `token_to_kv_pool.set_kv_buffer(layer, loc, cache_k=k_rotated, cache_v=v_pre_rope)`.

V is stored pre-linear and written **without** RoPE (RoPE only applies to K and Q in standard
transformer attention).

**Verification:** Create a mock KV pool, call `inject_c2kv_gist` with a synthetic `C2KVEntry`, and
verify the KV pool received correctly shaped tensors.

---

## Task 4 — Modify the model class (Qwen3)

**File:** `python/sglang/srt/models/qwen3.py`

### 4a — Imports

At the top of the file, add:

```python
from sglang.srt.models.gist_utils import (
    GistConfig, get_apply_gist_residual_func, get_prepare_gist_input_func,
)
```

### 4b — `Qwen3Attention.__init__`: add `gist_qkv_proj` and compiled flex_attention

```python
if server_args.enable_c2kv:
    self.gist_qkv_proj = QKVParallelLinear(
        hidden_size, head_dim, total_num_heads, total_num_kv_heads,
        bias=attention_bias, quant_config=quant_config,
        tp_rank=attn_tp_rank, tp_size=attn_tp_size,
        prefix=f"{prefix}.gist_qkv_proj",
    )
    self.flex_attention = torch.compile(flex_attention)  # [Changed] compiled once here
```

### 4c — `Qwen3Attention.forward_with_gist`

Signature: `(self, hidden_states, gist_len, positions, attention_mask, apply_gist_residual, **kwargs)`

**[Changed]** `attention_mask` here is already a `BlockMask` (pre-built by `gist_utils`). The
method does **not** call `create_block_mask` itself.

Steps:
1. Split `hidden_states (1, total_len, H)` into `input_hidden` and `gist_hidden`.
2. Optionally call `apply_gist_residual(input_hidden, gist_hidden)`.
3. Project input with `self.qkv_proj` → `(q_i, k_i, v_i)`.
4. Project gist with `self.gist_qkv_proj` → `(q_g, k_g, v_g)`.
5. Concatenate: `q = cat([q_i, q_g])`, same for k, v.
6. Apply QK norm on full concatenated `q, k`.
7. Save pre-RoPE gist KV:
   ```python
   gist_key_values = (k[0, -gist_len:].clone(), v[0, -gist_len:].clone())
   ```
8. Apply RoPE (`self.rotary_emb`) to full `(q, k)` using `positions`.
9. Reshape to `(1, num_heads, total_len, head_dim)` and call `self.flex_attention(q, k, v,
   block_mask=attention_mask, scale=self.scaling)`. GQA repeat-kv is applied before flex_attention.
10. Apply `o_proj`, return `(output, gist_key_values)`.

### 4d — `Qwen3DecoderLayer.forward_with_gist`

Wraps attention with pre/post layernorm and MLP:

```python
def forward_with_gist(self, hidden_states, gist_len, positions, attention_mask,
                      apply_gist_residual, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, gist_key_values = self.self_attn.forward_with_gist(
        hidden_states, gist_len, positions, attention_mask, apply_gist_residual, **kwargs,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, gist_key_values
```

### 4e — `Qwen3Model._init_c2kv` **[Changed — now on Qwen3Model, not Qwen3ForCausalLM]**

```python
def _init_c2kv(self, config, server_args):
    gist_cfg = GistConfig(
        gist_type=server_args.c2kv_gist_type,
        gist_param=server_args.c2kv_gist_param,
        gist_extra_embed_num=getattr(config, "gist_extra_embed_num", 1),
        gist_token_id=getattr(config, "gist_token_id", None),
        gist_residual_type=getattr(config, "gist_residual_type", "none"),
        hidden_size=config.hidden_size,
        attention_bias=getattr(config, "attention_bias", False),
    )
    self.gist_embed_tokens = nn.Embedding(gist_cfg.gist_extra_embed_num, config.hidden_size)
    self.prepare_gist_input = get_prepare_gist_input_func(gist_cfg)
    self.apply_gist_residual = get_apply_gist_residual_func(gist_cfg)
```

`Qwen3Model.__init__` calls `self._init_c2kv(config, server_args)` when `enable_c2kv` is set.

### 4f — `Qwen3ForCausalLM.__init__` and `generate_gist`

```python
self.enable_c2kv = _server_args and _server_args.enable_c2kv
```

`generate_gist` lives on `Qwen3ForCausalLM`:

```python
@torch.no_grad()
def generate_gist(self, input_ids, attention_mask, ratio=4):
    block_mask, gist_mask, position_ids = self.model.prepare_gist_input(
        input_ids, attention_mask, ratio=ratio
    )
    gist_len = gist_mask.shape[1]
    gist_embed = self.model.gist_embed_tokens(
        torch.zeros((1, gist_len), dtype=torch.long, device=input_ids.device)
    )
    inputs_embeds = torch.cat([self.model.embed_tokens(input_ids), gist_embed], dim=1)
    hidden_states = inputs_embeds
    gist_key_values = []
    for layer in self.model.layers:
        hidden_states, layer_kv = layer.forward_with_gist(
            hidden_states, gist_len, positions=position_ids,
            attention_mask=block_mask,
            apply_gist_residual=self.model.apply_gist_residual,
            ratio=ratio,
        )
        gist_key_values.append(layer_kv)
    gist_position_ids = position_ids[:, -gist_len:].contiguous()
    return gist_key_values, gist_mask, gist_position_ids
```

**[Changed]** `block_mask` is passed in from `prepare_gist_input` directly; no `create_block_mask`
call inside `generate_gist`.

### 4g — `load_weights`: gist weight mappings

Add alongside the existing `qkv_proj` merge rule:

```python
("gist_qkv_proj", "gist_q_proj", "q"),
("gist_qkv_proj", "gist_k_proj", "k"),
("gist_qkv_proj", "gist_v_proj", "v"),
```

Also add: `if name.startswith("gist_embed_tokens."): ...` to the weight-skipping logic so this
weight is not silently dropped.

**Verification:** With `--enable-c2kv` and a loaded model, call `model.generate_gist(input_ids,
attention_mask)` on a short sequence. Verify the list has `num_layers` elements, each is a tuple
of two tensors of shape `(gist_len, kv_size)`.

---

## Task 5 — Add `forward_c2kv_extract` to the model runner

**File:** `python/sglang/srt/model_executor/model_runner.py`

```python
def forward_c2kv_extract(
    self,
    input_ids: torch.Tensor,       # (1, seq_len) int64
    attention_mask: torch.Tensor,  # (1, seq_len) bool
    compression_ratio: int,
):
    return self.model.generate_gist(input_ids, attention_mask, ratio=compression_ratio)
```

Pure pass-through to `Qwen3ForCausalLM.generate_gist`.

---

## Task 6 — Add data structures for C2KV requests and segments

**File:** `python/sglang/srt/managers/io_struct.py`

### `C2KVSegmentInfo`

```python
class C2KVSegmentInfo:
    def __init__(self, key_hash: str = "", token_start: int = 0, token_end: int = 0):
        self.key_hash = key_hash
        self.token_start = token_start
        self.token_end = token_end  # exclusive; equals token_start when content is removed
```

**[Changed]** In the serving_chat implementation, annotated message content is **removed** from the
prompt before tokenization. As a result, `token_start == token_end` — the segment represents a
zero-length injection point in the prompt token stream.

### `TokenizedExtractReqInput(BaseReq)`

```python
@dataclass
class TokenizedExtractReqInput(BaseReq):
    input_ids: List[int] = field(default_factory=list)
    input_text: str = ""
    compression_ratio: int = 4
```

**[Changed]** Extends `BaseReq` (which already provides `rid`).

### `C2KVExtractReqOutput(BaseReq)`

```python
@dataclass
class C2KVExtractReqOutput(BaseReq):
    key_hash: str = ""
    gist_len: int = 0
    original_seq_len: int = 0
    error: str = ""
    success: bool = True
```

### Extend `GenerateReqInput` and `TokenizedGenerateReqInput`

Add `c2kv_segments: Optional[List] = None` to both.

---

## Task 7 — Add `C2KVPrefillRound` and C2KV state to the request object

**File:** `python/sglang/srt/managers/schedule_batch.py`

### `C2KVPrefillRound`

```python
class C2KVPrefillRound:
    __slots__ = ("tokens", "post_inject_seg_indices")

    def __init__(self, tokens: List[int], post_inject_seg_indices: List[int]):
        self.tokens = tokens
        self.post_inject_seg_indices = post_inject_seg_indices
```

### `Req` c2kv fields **[Changed — 6 fields, not 4]**

```python
self.c2kv_segments = None           # List[C2KVSegmentInfo], sorted by token_start
self.c2kv_rounds = None             # List[C2KVPrefillRound]
self.c2kv_round_idx = 0
self.c2kv_position_correction = 0  # running sum: original_seq_len - gist_len
self.c2kv_full_origin_input_ids = None  # [Added] virtual IDs including synthetic gist IDs
self.c2kv_requeued = False              # [Added] True while waiting for the next round
```

`c2kv_full_origin_input_ids` holds the final virtual token ID sequence (normal tokens +
`c2kv_gist_token_ids(...)` for each injected segment) that is written back to `req.origin_input_ids`
after all rounds complete. `c2kv_requeued` prevents stale checks during requeue.

### `ModelWorkerBatch` **[Added]**

Add to the `ModelWorkerBatch` dataclass:

```python
c2kv_position_corrections: Optional[List[int]] = None  # per-request correction, one per req
```

---

## Task 8 — Initialise the C2KV pool in the scheduler

**File:** `python/sglang/srt/managers/scheduler.py`

```python
self.c2kv_pool = None
if server_args.enable_c2kv:
    from sglang.srt.mem_cache.c2kv_pool import C2KVPool, calculate_c2kv_pool_size
    max_total_tokens = calculate_c2kv_pool_size(...)
    self.c2kv_pool = C2KVPool(
        max_total_tokens=max_total_tokens,
        max_entry_tokens=server_args.c2kv_max_tokens,
    )
    logger.info(f"C2KV pool initialised (max_total_tokens={max_total_tokens})")
```

---

## Task 9 — Register `TokenizedExtractReqInput` in the scheduler dispatcher

**File:** `python/sglang/srt/managers/scheduler.py`

Add alongside the existing `TokenizedGenerateReqInput` handler in the type-based dispatcher:

```python
(TokenizedExtractReqInput, self.handle_extract_request),
```

---

## Task 10 — Implement `handle_extract_request` in the scheduler

**File:** `python/sglang/srt/managers/scheduler.py`

```python
def handle_extract_request(self, recv_req: TokenizedExtractReqInput):
    if self.c2kv_pool is None:
        return C2KVExtractReqOutput(rid=recv_req.rid, error="C2KV not enabled.", success=False)
    if not recv_req.input_ids:
        return C2KVExtractReqOutput(rid=recv_req.rid, error="Empty input_ids.", success=False)

    key_hash = self.c2kv_pool.compute_hash(recv_req.input_ids)  # [Changed] hashes token IDs

    existing = self.c2kv_pool.get(key_hash)
    if existing is not None:
        return C2KVExtractReqOutput(
            rid=recv_req.rid, key_hash=key_hash,
            gist_len=existing.gist_len, original_seq_len=existing.original_seq_len,
        )

    input_ids = torch.tensor([recv_req.input_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    try:
        gist_key_values, gist_mask, gist_position_ids = (
            self.tp_worker.model_runner.forward_c2kv_extract(
                input_ids, attention_mask, recv_req.compression_ratio,
            )
        )
    except Exception as e:
        logger.error(f"C2KV extract failed: {e}")
        return C2KVExtractReqOutput(rid=recv_req.rid, error=str(e), success=False)

    entry = self.c2kv_pool.store(
        key_hash=key_hash, gist_key_values=gist_key_values, gist_mask=gist_mask,
        gist_position_ids=gist_position_ids, original_seq_len=len(recv_req.input_ids),
    )
    return C2KVExtractReqOutput(
        rid=recv_req.rid, key_hash=key_hash,
        gist_len=entry.gist_len, original_seq_len=len(recv_req.input_ids),
    )
```

---

## Task 11 — Build the multi-round prefill schedule on incoming generate requests

**File:** `python/sglang/srt/managers/scheduler.py`

### 11a — Hook in `handle_generate_request`

After building the `Req` from the tokenized input:

```python
if getattr(recv_req, "c2kv_segments", None):
    req.c2kv_segments = recv_req.c2kv_segments
    self._build_c2kv_prefill_rounds(req)
```

### 11b — `_build_c2kv_prefill_rounds` **[Changed — more robust than original plan]**

```python
def _build_c2kv_prefill_rounds(self, req: Req) -> None:
```

Key changes vs original plan:
1. **Validation:** asserts segments are non-overlapping and within token bounds.
2. **Pool lookup:** for each segment, looks up `entry.gist_len` in `self.c2kv_pool` to compute
   exact virtual token ID sequences via `c2kv_gist_token_ids`.
3. **`c2kv_full_origin_input_ids`:** builds the final virtual token ID sequence (normal tokens +
   synthetic gist IDs concatenated at each injection point) and stores it in
   `req.c2kv_full_origin_input_ids`. This is used after all rounds complete to restore
   `req.origin_input_ids` so the output processor sees the full logical sequence.
4. Algorithm: iterates segments in sorted order, emits `C2KVPrefillRound(normal_tokens, [])` for
   the normal token slice before each segment, appends `seg_idx` to the last round's
   `post_inject_seg_indices`. Remaining tokens after the last segment form the final round.
5. Resets `req.origin_input_ids` and `req.fill_ids` to round-0 tokens.

**Design invariant:** After this function, `req.origin_input_ids` contains only normal tokens for
round 0. Gist regions are injected separately after each round's prefill completes.

---

## Task 12 — Implement `_inject_c2kv_gist_segment`

**File:** `python/sglang/srt/managers/scheduler.py`

Signature: **[Changed]** `_inject_c2kv_gist_segment(self, req, seg_idx, logical_kv_start) -> bool`

```python
def _inject_c2kv_gist_segment(self, req: Req, seg_idx: int, logical_kv_start: int) -> bool:
```

Steps (more detailed than original plan):
1. Validate `logical_kv_start <= req.kv_committed_len`.
2. **Trim stale KV:** if `req.kv_committed_len > kv_start`, free the excess slots via
   `token_to_kv_pool_allocator.free()` and update `req.kv_committed_len`.
3. Look up `entry` in `self.c2kv_pool`; return False on miss.
4. Check that `kv_start + gist_len <= max_context_len`.
5. Allocate `gist_len` slots: `loc = token_to_kv_pool_allocator.alloc(gist_len)`.
6. Write slot indices into `req_to_token_pool.req_to_token[req_pool_idx, kv_start:kv_start+gist_len]`.
7. Compute `position_cursor = seg.token_start + req.c2kv_position_correction`.
8. Call `inject_c2kv_gist(entry, position_cursor, loc, token_to_kv_pool, attn_layers,
   cos_sin_cache, is_neox_style=True)`.
9. Update `req.kv_committed_len += gist_len`, `req.kv_allocated_len += gist_len`,
   `req.c2kv_position_correction += entry.original_seq_len - gist_len`.
10. Return True.

**Engine adapter notes:**
- `cos_sin_cache` path for Qwen3: `model.model.layers[0].self_attn.rotary_emb.cos_sin_cache`.
- `attn_layers`: `[layer.self_attn.attn for layer in model.model.layers]`.
- `is_neox_style=True` for Qwen3.

---

## Task 13 — Hook injection into the prefill output processor

**File:** `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

In the loop that processes finished prefill requests, at the `is_chunked <= 0` branch:

1. Read `cur_round = req.c2kv_rounds[req.c2kv_round_idx]`.
2. Compute `logical_kv_start = req.kv_committed_len`.
3. For each `seg_idx` in `cur_round.post_inject_seg_indices`, call
   `_inject_c2kv_gist_segment(req, seg_idx, logical_kv_start)`. On failure, abort the request.
4. Increment `req.c2kv_round_idx`.
5. If more rounds remain:
   - Set `req.prefix_indices` and `req.already_computed` to reflect the current KV state.
   - Set `req.origin_input_ids` and `req.fill_ids` to the next round's tokens.
   - Release the tree cache lock on `last_node` (decref).
   - Set `req.c2kv_requeued = True`.
   - Insert req at front of `waiting_queue`.
   - `continue` (skip normal decode output).
6. If last round done:
   - Restore `req.origin_input_ids = req.c2kv_full_origin_input_ids`.
   - **[Added] Patch `batch.seq_lens`:** compute `gist_delta = req.kv_committed_len -
     batch.seq_lens_cpu[i]` and update `batch.seq_lens[i]` so the decode kernel sees the correct
     sequence length including injected gist tokens.
   - Fall through to normal decode output.

**[Added] Scheduler main loop early-process flag:** When any running batch has requests with
incomplete c2kv rounds (`req.c2kv_rounds is not None and req.c2kv_round_idx < len(req.c2kv_rounds)`),
the scheduler sets `c2kv_early_process=True`, disabling overlap and forcing synchronous
`process_batch_result` before the next `get_next_batch_to_run`. This prevents the overlap
scheduler from picking up the next batch before injections are applied.

---

## Task 14 — Add `c2kv_position_corrections` to `ForwardBatch`

**File:** `python/sglang/srt/model_executor/forward_batch_info.py`

Add to `ForwardBatch`:

```python
c2kv_position_corrections: Optional[torch.Tensor] = None  # (batch_size,) int64
```

In `ForwardBatch.init_new`, when `batch.c2kv_position_corrections is not None`:

```python
corr = torch.tensor(batch.c2kv_position_corrections, dtype=torch.int64, device=device)
if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
    ret.positions = ret.positions + corr
elif ret.positions is not None and batch.extend_seq_lens is not None:
    ext_lens = torch.tensor(batch.extend_seq_lens, dtype=torch.int32, device=device)
    per_token_corr = torch.repeat_interleave(corr, ext_lens)
    ret.positions = ret.positions + per_token_corr
```

For decode mode, the correction is a scalar shift per request. For prefill/extend mode, it is
broadcast per-token using `repeat_interleave` against `extend_seq_lens`.

**[Changed]** The original plan proposed a `C2KV_EXTRACT = auto()` enum value in `ForwardMode`.
This was **not implemented**. Instead, position correction is propagated through `ForwardBatch`
directly.

---

## Task 15 — Add the `TokenizerManager.c2kv_extract` method **[Added — not in original plan]**

**File:** `python/sglang/srt/managers/tokenizer_manager.py` (or `tokenizer_communicator_mixin.py`)

Add a dedicated `_Communicator` for C2KV extract responses and an async method:

```python
self.c2kv_extract_communicator = _Communicator(self.send_to_scheduler, dp_size)

async def c2kv_extract(self, input_ids, input_text, compression_ratio=4, rid=None):
    req = TokenizedExtractReqInput(
        rid=rid or uuid4().hex,
        input_ids=input_ids,
        input_text=input_text,
        compression_ratio=compression_ratio,
    )
    return (await self.c2kv_extract_communicator(req))[0]
```

The communicator dispatches `C2KVExtractReqOutput` responses back to awaiting callers, separate
from the normal generate output stream.

---

## Task 16 — Add OpenAI-compatible API protocol types

**File:** `python/sglang/srt/entrypoints/openai/protocol.py`

### Annotate messages with `c2kv_key_hash`

Add to `ChatCompletionMessageGenericParam` and `ChatCompletionMessageUserParam`:

```python
c2kv_key_hash: Optional[str] = None
```

### Extract request/response

```python
class C2KVExtractRequest(BaseModel):
    text: str
    compression_ratio: int = Field(default=4)
    role: Optional[str] = None  # used for role-aware tokenization in http_server

class C2KVExtractResponse(BaseModel):
    key_hash: str
    gist_len: int
    original_seq_len: int
    success: bool = True
    error: Optional[str] = None
```

---

## Task 17 — Compute C2KV segment boundaries in the chat serving layer

**File:** `python/sglang/srt/entrypoints/openai/serving_chat.py`

**[Changed significantly vs original plan]**

The actual implementation removes annotated messages from the prompt entirely rather than
keeping their content and computing boundaries around it. As a result:
- `token_start == token_end` for every segment (zero-length injection point).
- The `token_start` is computed as the length of the chat-template token sequence of the
  compressed prefix (all messages up to but not including the annotated message, with the
  annotated message removed).

```python
def _compute_c2kv_segments(self, request) -> Optional[List[C2KVSegmentInfo]]:
    annotated = [i for i, m in enumerate(request.messages)
                 if getattr(m, "c2kv_key_hash", None)]
    if not annotated:
        return None

    tokenizer = self.tokenizer_manager.tokenizer
    segments = []
    for i in annotated:
        msg = request.messages[i]
        # Compressed prefix: all messages before i, after removing msg i
        prefix = [m.model_dump() for m in request.messages if m is not request.messages[i]]
        compressed_ids = tokenizer.apply_chat_template(
            prefix[:i], tokenize=True, add_generation_prompt=False
        )
        insertion_point = len(compressed_ids)
        segments.append(C2KVSegmentInfo(
            key_hash=msg.c2kv_key_hash,
            token_start=insertion_point,
            token_end=insertion_point,  # zero-length: content is injected, not prefilled
        ))

    # Remove annotated messages from the prompt (reversed to preserve indices)
    for i in reversed(annotated):
        request.messages.pop(i)

    return segments
```

Called at the start of `_convert_to_internal_request`, before `_process_messages`. The result is
passed as `c2kv_segments=c2kv_segments` to `GenerateReqInput`.

---

## Task 18 — Add the HTTP extract endpoint

**File:** `python/sglang/srt/entrypoints/http_server.py`

**[Changed vs original plan]**

```python
@app.post("/v1/c2kv/extract")
async def v1_c2kv_extract(request: C2KVExtractRequest, raw_request: Request) -> C2KVExtractResponse:
    tokenizer_manager = app.state.tokenizer_manager
    tokenizer = tokenizer_manager.tokenizer

    if request.role:
        # Role-aware tokenization: slice tokens the same way serving_chat does
        dummy = [{"role": "system", "content": ""}]
        target = [{"role": request.role, "content": request.text}]
        prev_ids = tokenizer.apply_chat_template(dummy, tokenize=True, add_generation_prompt=False)
        full_ids = tokenizer.apply_chat_template(dummy + target, tokenize=True, add_generation_prompt=False)
        input_ids = full_ids[len(prev_ids):]
    else:
        input_ids = tokenizer.encode(request.text)

    result = await tokenizer_manager.c2kv_extract(
        input_ids=input_ids,
        input_text=request.text,
        compression_ratio=request.compression_ratio,
    )
    return C2KVExtractResponse(
        key_hash=result.key_hash,
        gist_len=result.gist_len,
        original_seq_len=result.original_seq_len,
        success=result.success,
        error=result.error or None,
    )
```

**Key change:** The endpoint uses `tokenizer_manager.c2kv_extract()` (Task 15), **not** the
`generate_request` pipeline. The `role` field enables token-boundary-consistent tokenization so
that the hash computed here matches what `_compute_c2kv_segments` will compute on the same message
text in a future chat completion.

---

## End-to-end workflow (for validation)

```
1. Start server:
   python -m sglang.launch_server <model> --enable-c2kv \
       --c2kv-pool-fraction 0.01 --c2kv-max-tokens 4096

2. Extract a document:
   POST /v1/c2kv/extract
   {"text": "<document content>", "compression_ratio": 4, "role": "user"}
   → {"key_hash": "abc123...", "gist_len": 128, "original_seq_len": 512, "success": true}

3. Use in a chat completion:
   POST /v1/chat/completions
   {
     "messages": [
       {"role": "user", "content": "<document content>", "c2kv_key_hash": "abc123..."},
       {"role": "user", "content": "Summarise the document."}
     ]
   }
   → Normal chat completion response; the document was injected from KV cache, not re-prefilled.

4. Same key_hash again: pool hit, extraction step is skipped, injection is instant.
```

---

## Testing checklist

Run `work/test_c2kv_endpoint.py --base-url http://localhost:30000` which covers:

- [ ] Single document extraction → `success=true`, `gist_len > 0`, stable `key_hash`.
- [ ] Duplicate extraction → same `key_hash`, no error.
- [ ] Baseline chat (no C2KV annotation) → normal response, no regression.
- [ ] Chat with one C2KV-annotated message → coherent response.
- [ ] Chat with multiple C2KV-annotated messages → coherent response.
- [ ] Chat with invalid `key_hash` → graceful abort, no server crash.

Additional unit tests:

- `C2KVPool`: store/get/evict under capacity constraints.
- `_build_c2kv_prefill_rounds`: various segment layouts (leading, trailing, multiple adjacent).
- `inject_c2kv_gist`: mock KV pool; verify correct slot writes and RoPE application.

---

## Known limitations and follow-up work

| Limitation | Notes |
|------------|-------|
| TP sync for extraction | `generate_gist` must run on all TP ranks simultaneously. The scheduler broadcasts the extract request to all ranks via the existing TP worker mechanism. |
| Batch-size-1 extraction | `generate_gist` only handles `batch_size=1`. Batched extraction would require padding or ragged attention. |
| Model specificity | `gist_qkv_proj` (Task 4), `cos_sin_cache` accessor (Task 12), and `attn_layers` accessor (Task 12) are Qwen3-specific. Other models need their own `forward_with_gist` and `generate_gist` implementations. |
| `flex_attention` requirement | PyTorch >= 2.5 required. For older PyTorch, use a manual masked SDPA. |
| Pool not persistent | `C2KVPool` is in-process GPU RAM. Server restarts clear the pool; clients must re-extract. |
| `token_start == token_end` segments | Because annotated message content is removed before tokenization, every segment is a zero-length injection point. Injected gist tokens are not visible to the tokenizer at all. |
| `c2kv_early_process` serialisation | The early-process flag disables prefill–decode overlap whenever any request has pending c2kv rounds. This may reduce throughput under mixed workloads. |
| Role-based tokenization in extract | The `role` field in `C2KVExtractRequest` must match the role used in the chat message for the hash to be consistent. Mismatched roles produce different token sequences and different hashes. |
