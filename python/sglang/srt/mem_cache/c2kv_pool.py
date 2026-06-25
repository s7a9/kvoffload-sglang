"""
C2KV GPU-resident LRU pool.

Stores compressed gist KV entries per unique document hash.
Capacity is bounded by max_total_tokens (sum of gist_len across all entries).
"""

import hashlib
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

C2KV_GIST_TOKEN_BASE = 1 << 60
C2KV_METADATA_BYTES_PER_TOKEN = (
    torch.tensor([], dtype=torch.bool).element_size()
    + torch.tensor([], dtype=torch.int64).element_size()
)


def calculate_c2kv_pool_size(
    *,
    total_memory_bytes: int,
    pool_fraction: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
) -> Tuple[int, int]:
    """Return (max gist tokens, bytes per token) for a TP/PP-local C2KV pool."""
    kv_bytes_per_token = (
        num_layers
        * num_kv_heads
        * (head_dim + value_head_dim)
        * torch.tensor([], dtype=dtype).element_size()
    )
    bytes_per_token = kv_bytes_per_token + C2KV_METADATA_BYTES_PER_TOKEN
    memory_budget_bytes = int(total_memory_bytes * pool_fraction)
    return memory_budget_bytes // bytes_per_token, bytes_per_token


def c2kv_gist_token_ids(key_hash: str, gist_len: int) -> List[int]:
    """Deterministic synthetic token IDs for gist KV slots in the radix cache."""
    base_hash = struct.unpack(">Q", bytes.fromhex(key_hash[:16]))[0]
    base = C2KV_GIST_TOKEN_BASE + (base_hash % (1 << 59))
    return [base + j for j in range(gist_len)]


@dataclass
class C2KVEntry:
    key_hash: str
    gist_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
    gist_mask: torch.Tensor         # (1, gist_len) bool
    gist_position_ids: torch.Tensor # (1, gist_len) int64
    gist_len: int
    original_seq_len: int


class C2KVPool:
    def __init__(
        self, max_total_tokens: int, max_entry_tokens: Optional[int] = None
    ):
        self.max_total_tokens = max_total_tokens
        self.max_entry_tokens = (
            max_total_tokens if max_entry_tokens is None else max_entry_tokens
        )
        self._current_tokens = 0
        # OrderedDict: LRU order (MRU at end)
        self._cache: OrderedDict[str, C2KVEntry] = OrderedDict()

    @staticmethod
    def compute_hash(token_ids: List[int]) -> str:
        import struct
        raw = struct.pack(f"{len(token_ids)}i", *token_ids)
        return hashlib.sha256(raw).hexdigest()

    def store(
        self,
        key_hash: str,
        gist_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        gist_mask: torch.Tensor,
        gist_position_ids: torch.Tensor,
        original_seq_len: int,
    ) -> C2KVEntry:
        gist_len = gist_mask.shape[1]
        if gist_len > self.max_entry_tokens:
            raise ValueError(
                f"C2KV entry has {gist_len} gist tokens, exceeding the per-entry "
                f"limit of {self.max_entry_tokens} tokens."
            )
        if gist_len > self.max_total_tokens:
            raise ValueError(
                f"C2KV entry has {gist_len} gist tokens, exceeding the pool "
                f"capacity of {self.max_total_tokens} tokens."
            )

        # Remove existing entry with same key first
        if key_hash in self._cache:
            old = self._cache.pop(key_hash)
            self._current_tokens -= old.gist_len

        # Evict LRU entries until the new entry fits
        while self._current_tokens + gist_len > self.max_total_tokens and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._current_tokens -= evicted.gist_len

        entry = C2KVEntry(
            key_hash=key_hash,
            gist_key_values=gist_key_values,
            gist_mask=gist_mask,
            gist_position_ids=gist_position_ids,
            gist_len=gist_len,
            original_seq_len=original_seq_len,
        )
        self._cache[key_hash] = entry
        self._current_tokens += gist_len
        return entry

    def get(self, key_hash: str) -> Optional[C2KVEntry]:
        entry = self._cache.get(key_hash)
        if entry is None:
            return None
        # Promote to MRU
        self._cache.move_to_end(key_hash)
        return entry

    def current_tokens(self) -> int:
        return self._current_tokens

    def num_entries(self) -> int:
        return len(self._cache)
