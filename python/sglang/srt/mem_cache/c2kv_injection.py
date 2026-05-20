"""
C2KV KV injection helper.

Reads a stored gist entry, applies RoPE at the correct absolute positions,
and writes K/V tensors into the engine's KV pool.
"""

from typing import List

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.c2kv_pool import C2KVEntry


def inject_c2kv_gist(
    entry: C2KVEntry,
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
        entry:           C2KVEntry with pre-RoPE K/V tensors per layer.
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
    if entry.gist_position_ids.shape != (1, gist_len):
        raise ValueError(
            "C2KV gist_position_ids shape mismatch: "
            f"{tuple(entry.gist_position_ids.shape)} != {(1, gist_len)}"
        )
    if len(entry.gist_key_values) != len(attn_layers):
        raise ValueError(
            "C2KV layer count mismatch: "
            f"{len(entry.gist_key_values)=} != {len(attn_layers)=}"
        )

    # gist_position_ids: (1, gist_len)
    gist_pos = entry.gist_position_ids[0]  # (gist_len,)

    # Absolute positions for each gist token
    abs_pos = (position_cursor + gist_pos).clamp(0, cos_sin_cache.shape[0] - 1)

    rotary_dim = cos_sin_cache.shape[1]
    half_dim = rotary_dim // 2
    cos = cos_sin_cache[abs_pos, :half_dim]   # (gist_len, half_dim)
    sin = cos_sin_cache[abs_pos, half_dim:]   # (gist_len, half_dim)
    head_dim = half_dim * 2

    for layer_idx, (k_pre, v_pre) in enumerate(entry.gist_key_values):
        # k_pre, v_pre: (gist_len, kv_size) where kv_size = num_kv_heads * head_dim
        if k_pre.ndim != 2:
            raise ValueError(
                f"C2KV K tensor at layer {layer_idx} must be 2D, got {k_pre.ndim}D"
            )
        if k_pre.shape[0] != gist_len:
            raise ValueError(
                f"C2KV K length mismatch at layer {layer_idx}: "
                f"{k_pre.shape[0]} != {gist_len}"
            )
        if v_pre.shape != k_pre.shape:
            raise ValueError(
                f"C2KV K/V shape mismatch at layer {layer_idx}: "
                f"{tuple(k_pre.shape)} != {tuple(v_pre.shape)}"
            )

        kv_size = k_pre.shape[1]
        if kv_size % head_dim != 0:
            raise ValueError(
                f"C2KV kv_size is not divisible by head_dim at layer {layer_idx}: "
                f"{kv_size=} {head_dim=}"
            )

        # Infer num_kv_heads and head_dim from kv_size and cos dimension
        num_kv_heads = kv_size // head_dim

        # Reshape for apply_rotary_emb: (gist_len, num_kv_heads, head_dim)
        k_3d = k_pre.view(gist_len, num_kv_heads, head_dim)

        k_rotated = apply_rotary_emb(k_3d, cos, sin, is_neox_style)

        # Flatten back to (gist_len, kv_size)
        k_rotated = k_rotated.view(gist_len, kv_size)

        token_to_kv_pool.set_kv_buffer(
            layer=attn_layers[layer_idx],
            loc=loc,
            cache_k=k_rotated,
            cache_v=v_pre,
        )
