"""
Gist utility functions for C2KV (Concatenable and Compressible KV Cache).

Builds the custom attention mask, position IDs, and optional residual
connections used during the gist extraction forward pass.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask


@dataclass
class GistConfig:
    gist_type: str = "dynamic-interleave"
    gist_param: str = "qkv"
    gist_extra_embed_num: int = 1
    gist_token_id: Optional[int] = None
    gist_residual_type: str = "none"
    hidden_size: int = 4096
    attention_bias: bool = False


def get_prepare_gist_input_func(gist_cfg: GistConfig) -> Callable:
    """
    Returns prepare_gist_input(input_ids, attention_mask, ratio)
    -> (new_attn_mask, gist_mask, position_ids).

    Attention mask layout (True = attend):
        input tokens see each other causally; cannot see gist tokens.
        gist tokens attend all input tokens; see each other causally.

    Position IDs:
        input token i  -> position i
        gist token j   -> min((j+1)*ratio - 1, seq_len - 1)
    """

    def prepare_gist_input(input_ids, attention_mask, ratio=4):
        device = input_ids.device
        seq_len = input_ids.shape[1]
        gist_len = math.ceil(seq_len / ratio)
        total_len = seq_len + gist_len

        # --- block_mask for flex_attention ---
        # Mask logic (True = attend):
        #   input-to-input: causal (q_idx >= kv_idx)
        #   input-to-gist: never (input tokens cannot see gist tokens)
        #   gist-to-input: full attention
        #   gist-to-gist: causal (q_idx >= kv_idx)
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            is_q_input = q_idx < seq_len
            is_kv_input = kv_idx < seq_len

            # input query attending input key: causal
            input_to_input = is_q_input & is_kv_input & (q_idx >= kv_idx)
            # input query attending gist key: never
            # gist query attending input key: always
            gist_to_input = (~is_q_input) & is_kv_input
            # gist query attending gist key: causal
            gist_to_gist = (~is_q_input) & (~is_kv_input) & (q_idx >= kv_idx)

            return input_to_input | gist_to_input | gist_to_gist

        block_mask = create_block_mask(
            mask_mod, B=1, H=None, Q_LEN=total_len, KV_LEN=total_len, device=device
        )

        # --- gist_mask (1, gist_len) ---
        gist_mask = torch.ones((1, gist_len), dtype=torch.bool, device=device)

        # --- position_ids (1, total_len) ---
        input_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        gist_pos = torch.tensor(
            [min((j + 1) * ratio - 1, seq_len - 1) for j in range(gist_len)],
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.cat([input_pos, gist_pos], dim=0).unsqueeze(0)

        return block_mask, gist_mask, position_ids

    return prepare_gist_input


def get_apply_gist_residual_func(gist_cfg: GistConfig) -> Callable:
    """
    Returns apply_gist_residual(input_hidden, gist_hidden, **kwargs) -> gist_hidden.

    For gist_residual_type == "none" this is an identity on gist_hidden.
    """

    def apply_gist_residual(input_hidden, gist_hidden, **kwargs):
        # Identity: no residual connection by default.
        return gist_hidden

    return apply_gist_residual
