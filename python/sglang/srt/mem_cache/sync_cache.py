from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SyncCacheReqState:
    rid: str
    synced_len: int = 0
    decode_since_last_sync: int = 0


class SyncCache(HiRadixCache):
    """HiRadix-based sync cache with periodic decode-time synchronization.

    This mode reuses HiRadixCache offload/load-back implementation while forcing
    write-through behavior and triggering incremental radix synchronization during
    decode to avoid waiting until request completion.
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        # Sync cache must maintain radix nodes even if disable_radix_cache is set.
        # It also uses write-through without storage backend.
        sync_params = dataclasses.replace(params, disable=False)
        sync_server_args = dataclasses.replace(
            server_args,
            hicache_write_policy="write_through",
            hicache_storage_backend=None,
        )
        super().__init__(params=sync_params, server_args=sync_server_args)

        self._req_states: Dict[str, SyncCacheReqState] = {}

        # Keep the knobs close to the old SyncCache behavior.
        self.prefill_sync_chunk_size = max(1, int(sync_params.chunked_prefill_size or 256))
        self.decode_sync_stride_steps = 4
        self.decode_sync_min_tokens = max(8, sync_params.page_size)

    def _infer_seq_len(self, req: Req) -> int:
        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is not None and len(fill_ids) > 0:
            return len(fill_ids)
        return max(len(req.origin_input_ids) + len(req.output_ids) - 1, 0)

    def _ensure_fill_ids(self, req: Req, seq_len: int) -> None:
        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is None or len(fill_ids) != seq_len:
            req.fill_ids = req.origin_input_ids + req.output_ids

    def _sync_req_to_radix(
        self,
        req: Req,
        seq_len: int,
        *,
        force: bool,
        is_decode: bool,
        chunked: bool,
    ) -> None:
        if req.req_pool_idx is None or seq_len <= 0:
            return

        state = self._req_states.setdefault(req.rid, SyncCacheReqState(rid=req.rid))
        unsynced_len = seq_len - state.synced_len
        if unsynced_len <= 0:
            return

        if is_decode and not force:
            state.decode_since_last_sync += 1
            if (
                state.decode_since_last_sync < self.decode_sync_stride_steps
                and unsynced_len < self.decode_sync_min_tokens
            ):
                return

        # For long prefill in sync mode, avoid very tiny synchronization steps.
        if (not is_decode) and (not force) and unsynced_len < self.prefill_sync_chunk_size:
            return

        self._ensure_fill_ids(req, seq_len)
        super().cache_unfinished_req(req, chunked=chunked)
        state.synced_len = seq_len
        state.decode_since_last_sync = 0

    def sync_batch(self, batch: ScheduleBatch) -> None:
        self.flush_write_through_acks()
        is_extend = batch.forward_mode.is_extend()

        for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
            seq_len = int(seq_len)
            if is_extend:
                self._sync_req_to_radix(
                    req,
                    seq_len,
                    force=False,
                    is_decode=False,
                    chunked=getattr(req, "is_chunked", 0) > 0,
                )
            else:
                self._sync_req_to_radix(
                    req,
                    seq_len,
                    force=False,
                    is_decode=True,
                    chunked=False,
                )

    def _collect_req_path_nodes(self, req: Req) -> List[TreeNode]:
        nodes: List[TreeNode] = []
        node = getattr(req, "last_node", None)
        while node is not None and node is not self.root_node:
            nodes.append(node)
            node = node.parent
        return nodes

    def _req_has_ongoing_write(self, req: Req) -> bool:
        for node in self._collect_req_path_nodes(req):
            if node.id in self.ongoing_write_through:
                return True
        return False

    def _free_unprotected_tail(self, req: Req, target_len: int) -> None:
        if req.req_pool_idx is None or target_len <= 0:
            return
        protected_len = min(getattr(req, "cache_protected_len", 0), target_len)
        if protected_len >= target_len:
            return
        tail_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, protected_len:target_len
        ]
        self.token_to_kv_pool_allocator.free(tail_indices)

    def evict_device(self, req: Req, seq_len: Optional[int] = None) -> None:
        if req.req_pool_idx is None:
            self._req_states.pop(req.rid, None)
            return

        inferred_seq_len = self._infer_seq_len(req)
        target_len = min(seq_len, inferred_seq_len) if seq_len is not None else inferred_seq_len
        if target_len <= 0:
            self._req_states.pop(req.rid, None)
            return

        # Ensure latest KV is represented in radix before offloading.
        self._sync_req_to_radix(
            req,
            inferred_seq_len,
            force=True,
            is_decode=False,
            chunked=False,
        )
        self.flush_write_through_acks()

        # If this request is still being written, block until write-through ack arrives.
        if self._req_has_ongoing_write(req):
            self.writing_check(write_back=True)

        path_nodes = self._collect_req_path_nodes(req)

        # Write and evict request-private nodes first (lock_ref == 1).
        requested_write_back = False
        for node in path_nodes:
            if node.evicted or node.lock_ref > 1:
                continue
            if not node.backuped:
                if self.write_backup(node, write_back=True) > 0:
                    requested_write_back = True

        if requested_write_back:
            self.writing_check(write_back=True)

        for node in path_nodes:
            if node.evicted or node.lock_ref > 1:
                continue
            if node.backuped:
                self._evict_backuped(node)

        # Release request lock refs after path eviction.
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

        # Free non-radix-protected tail slots.
        self._free_unprotected_tail(req, target_len)

        self._req_states.pop(req.rid, None)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        self._req_states.pop(req.rid, None)
        super().cache_finished_req(req, is_insert=is_insert, **kwargs)

    def reset(self):
        self._req_states.clear()
        super().reset()
