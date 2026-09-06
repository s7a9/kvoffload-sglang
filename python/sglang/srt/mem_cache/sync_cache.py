from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache, TreeNode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class SyncCacheReqState:
    rid: str
    synced_len: int = 0


class SyncCache(HiRadixCache):
    """HiRadix cache that preserves a request only when it is preempted.

    Running requests stay exclusively on the device. On memory-pressure
    preemption, their committed KV is inserted into the radix tree, written to
    host memory, and then removed from the device before the request is queued
    again.
    """

    supports_request_offload = True

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        # Request offload needs radix metadata even when the regular radix cache
        # was disabled. Write-back avoids copying active requests continuously.
        sync_params = dataclasses.replace(params, disable=False)
        sync_server_args = dataclasses.replace(
            server_args,
            hicache_write_policy="write_back",
            hicache_storage_backend=None,
        )
        self._req_states: Dict[str, SyncCacheReqState] = {}
        self._before_device_offload: Optional[Callable[[], None]] = None

        super().__init__(params=sync_params, server_args=sync_server_args)

    def set_before_device_offload(self, callback: Callable[[], None]) -> None:
        self._before_device_offload = callback

    def _infer_seq_len(self, req: Req) -> int:
        committed_len = int(getattr(req, "kv_committed_len", 0))
        if committed_len > 0:
            return committed_len
        return max(len(req.origin_input_ids) + len(req.output_ids) - 1, 0)

    def _ensure_fill_ids(self, req: Req, seq_len: int) -> None:
        token_ids = req.origin_input_ids + req.output_ids
        # EAGLE radix keys are bigrams, so N committed KV entries require
        # N + 1 token ids. Regular radix keys require exactly N token ids.
        fill_len = seq_len + 1 if self.is_eagle else seq_len
        req.fill_ids = token_ids[:fill_len]

    def _sync_req_to_radix(self, req: Req, seq_len: int) -> None:
        if req.req_pool_idx is None or seq_len <= 0:
            return

        state = self._req_states.setdefault(req.rid, SyncCacheReqState(rid=req.rid))
        if seq_len <= state.synced_len:
            return

        self._ensure_fill_ids(req, seq_len)
        super().cache_unfinished_req(req, chunked=False)
        state.synced_len = seq_len

    def sync_batch(self, batch: ScheduleBatch) -> None:
        # Deliberately do no per-step synchronization. The full committed
        # request is synchronized once in evict_device when pressure requires it.
        return

    def _collect_req_path_nodes(self, req: Req) -> List[TreeNode]:
        nodes: List[TreeNode] = []
        node = getattr(req, "last_node", None)
        while node is not None and node is not self.root_node:
            nodes.append(node)
            node = node.parent
        return nodes

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
        target_len = (
            min(seq_len, inferred_seq_len)
            if seq_len is not None
            else inferred_seq_len
        )
        if target_len <= 0:
            self._req_states.pop(req.rid, None)
            return

        # With overlap scheduling, the latest committed slots may still be
        # produced by the previous forward (including a CUDA Graph replay).
        # Order all radix/indexer reads and the HiCache write stream after it.
        if self._before_device_offload is not None:
            self._before_device_offload()

        # Ensure the latest committed KV is represented in radix before offloading.
        self._sync_req_to_radix(req, target_len)

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

        # Free the unaligned committed tail and speculative over-allocation.
        allocated_len = max(
            int(getattr(req, "kv_allocated_len", target_len)), target_len
        )
        self._free_unprotected_tail(req, allocated_len)

        self._req_states.pop(req.rid, None)

        self.req_to_token_pool.free(req)

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        self._req_states.pop(req.rid, None)
        super().cache_finished_req(req, is_insert=is_insert, **kwargs)

    def reset(self):
        self._req_states.clear()
        super().reset()
