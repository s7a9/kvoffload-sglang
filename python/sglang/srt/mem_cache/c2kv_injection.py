"""
C2KV KV injection helper.

Reads a stored gist entry, applies RoPE at the correct absolute positions,
and writes K/V tensors into the engine's KV pool.
"""

from typing import List

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.c2kv_pool import C2KVEntry, C2KVPool


def inject_c2kv_gist(
    entry: C2KVEntry,
    c2kv_pool: C2KVPool,
    position_cursor: int,
    loc: torch.Tensor,
    token_to_kv_pool,
    attn_layers: List,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
) -> None:
    """
    Apply RoPE at absolute positions and write gist K/V into the KV pool.

    Args:
        entry:           C2KVEntry containing C2KV pool slot indices.
        c2kv_pool:       Preallocated C2KV K/V and position storage.
        position_cursor: Absolute position offset for this gist block.
        loc:             (gist_len,) int64 slot indices in the KV pool.
        token_to_kv_pool: Engine KV pool with set_kv_buffer().
        attn_layers:     List of attention layer objects (one per decoder layer).
        cos_sin_cache:   (max_pos, rotary_dim) float RoPE lookup table.
        is_neox_style:   True for Neox-style rotation (Qwen3 and most modern models).
    """
    gist_len = entry.gist_len
    if loc.numel() != gist_len:
        raise ValueError(
            f"C2KV loc length mismatch: loc.numel()={loc.numel()} != {gist_len=}"
        )
    if c2kv_pool.num_layers != len(attn_layers):
        raise ValueError(
            "C2KV layer count mismatch: "
            f"{c2kv_pool.num_layers=} != {len(attn_layers)=}"
        )

    gist_pos = c2kv_pool.get_position_ids(entry)

    # Absolute positions for each gist token
    abs_pos = (position_cursor + gist_pos).clamp(0, cos_sin_cache.shape[0] - 1)

    rotary_dim = cos_sin_cache.shape[1]
    half_dim = rotary_dim // 2
    cos = cos_sin_cache[abs_pos, :half_dim]   # (gist_len, half_dim)
    sin = cos_sin_cache[abs_pos, half_dim:]   # (gist_len, half_dim)
    head_dim = half_dim * 2

    for layer_idx in range(c2kv_pool.num_layers):
        k_pre, v_pre = c2kv_pool.get_layer_kv(entry, layer_idx)
        if k_pre.shape[2] != head_dim:
            raise ValueError(
                f"C2KV head_dim mismatch at layer {layer_idx}: "
                f"{k_pre.shape[2]} != {head_dim}"
            )

        k_rotated = apply_rotary_emb(k_pre, cos, sin, is_neox_style)

        token_to_kv_pool.set_kv_buffer(
            layer=attn_layers[layer_idx],
            loc=loc,
            cache_k=k_rotated,
            cache_v=v_pre,
        )
