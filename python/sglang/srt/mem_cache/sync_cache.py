from __future__ import annotations

import dataclasses
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, Optional, Tuple

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SyncCacheEntry:
    rid: str
    req: Optional[Req] = None
    synced_len: int = 0
    written_len: int = 0
    pending_write_len: int = 0
    decode_since_last_write: int = 0
    is_synced: bool = True
    evicted_len: int = 0
    evicted: bool = False
    host_segments: list[torch.Tensor] = dataclasses.field(default_factory=list)
    write_chunks: Deque[int] = dataclasses.field(default_factory=deque)


class SyncCache(ChunkCache):
    """Chunk-cache style write-through manager.

    This cache keeps ChunkCache scheduling behavior while adding CPU mirror writes for
    newly produced KV blocks. The write completion is ack-driven through HiCacheController.
    """

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        super().__init__(params)

        kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        if isinstance(kv_cache, MHATokenToKVPool):
            token_to_kv_pool_host = MHATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                params.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend or "default",
            )
        elif isinstance(kv_cache, NSATokenToKVPool):
            token_to_kv_pool_host = NSATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                params.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend or "default",
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            token_to_kv_pool_host = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                params.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend or "default",
            )
        else:
            raise ValueError(f"SyncCache unsupported kvcache type: {type(kv_cache)}")

        self.cache_controller = HiCacheController(
            params.token_to_kv_pool_allocator,
            token_to_kv_pool_host,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=threading.Event(),
            write_policy="write_through",
            io_backend=server_args.hicache_io_backend,
            storage_backend=None,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            enable_storage_metrics=False,
        )

        self.entries: Dict[str, SyncCacheEntry] = {}
        self._lock = threading.Lock()
        self.write_token_num = 0
        self.wrote_token_num = 0
        self.req_to_evict: Dict[str, int] = {}
        # Old SyncChunkCache-style knobs.
        self.prefill_write_chunk_size = max(1, int(params.chunked_prefill_size or 256))
        self.decode_write_stride_steps = 4
        self.decode_write_min_tokens = max(8, params.page_size)

    def _is_writing_overloaded(self, threshold: float = 1.0) -> bool:
        _, write_speed = self.cache_controller.get_writing_workload()
        if write_speed <= 0:
            return False
        backlog = self.write_token_num - self.wrote_token_num
        return backlog > threshold * write_speed

    def _infer_seq_len(self, req: Req) -> int:
        fill_ids = getattr(req, "fill_ids", None)
        if fill_ids is not None and len(fill_ids) > 0:
            return len(fill_ids)
        return max(len(req.origin_input_ids) + len(req.output_ids) - 1, 0)

    def _drain_write_acks(self, wait_for_rid: Optional[str] = None) -> None:
        while True:
            progressed = False
            for ack in list(self.cache_controller.ack_write_queue):
                if (wait_for_rid is None) and (not ack.finish_event.query()):
                    continue

                if wait_for_rid is not None:
                    node_ids = {str(x) for x in ack.node_ids}
                    if wait_for_rid not in node_ids:
                        continue
                    ack.finish_event.synchronize()

                if not ack.finish_event.query():
                    continue

                for rid in (str(x) for x in ack.node_ids):
                    entry = self.entries.get(rid)
                    if entry is None or not entry.write_chunks:
                        continue
                    chunk_len = entry.write_chunks.popleft()
                    entry.written_len += chunk_len
                    entry.pending_write_len = max(entry.pending_write_len - chunk_len, 0)
                    self.wrote_token_num += chunk_len
                    if rid in self.req_to_evict:
                        req = entry.req
                        if req is not None and req.req_pool_idx is not None:
                            target = self.req_to_evict[rid]
                            newly_evictable = max(min(entry.written_len, target) - entry.evicted_len, 0)
                            if newly_evictable > 0:
                                self._evict_device(req, newly_evictable)
                                entry.evicted_len += newly_evictable
                            if entry.evicted_len >= target:
                                del self.req_to_evict[rid]
                self.cache_controller.ack_write_queue.remove(ack)
                progressed = True

            if wait_for_rid is None:
                if not progressed:
                    break
            else:
                pending = self.entries.get(wait_for_rid)
                if pending is None or pending.pending_write_len == 0:
                    break

    def _write_through_req(
        self,
        req: Req,
        max_tokens: Optional[int] = None,
        force: bool = False,
        is_decode: bool = False,
    ) -> None:
        seq_len = self._infer_seq_len(req)
        if req.req_pool_idx is None or seq_len <= 0:
            return

        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len]
        entry = self.entries.setdefault(req.rid, SyncCacheEntry(rid=req.rid, req=req))
        entry.req = req

        unsynced_len = seq_len - entry.synced_len
        if unsynced_len <= 0:
            entry.decode_since_last_write = 0
            return

        if is_decode and not force:
            entry.decode_since_last_write += 1
            if (
                entry.decode_since_last_write < self.decode_write_stride_steps
                and unsynced_len < self.decode_write_min_tokens
            ):
                return

        if max_tokens is not None:
            unsynced_len = min(unsynced_len, max_tokens)
        if unsynced_len <= 0:
            return

        device_tail = kv_indices[entry.synced_len : entry.synced_len + unsynced_len]
        host_indices = self.cache_controller.write(device_tail, node_id=req.rid)
        if host_indices is None:
            raise RuntimeError(f"SyncCache failed host alloc for rid={req.rid}")

        entry.host_segments.append(host_indices)
        entry.synced_len += unsynced_len
        entry.pending_write_len += unsynced_len
        entry.write_chunks.append(unsynced_len)
        entry.is_synced = entry.synced_len >= seq_len
        entry.decode_since_last_write = 0
        self.write_token_num += unsynced_len
        self.cache_controller.start_writing()

    def _sync_prefill(self, req: Req, seq_len: int) -> None:
        if req.req_pool_idx is None or seq_len <= 0:
            return
        entry = self.entries.setdefault(req.rid, SyncCacheEntry(rid=req.rid, req=req))
        entry.req = req
        if self._is_writing_overloaded(5.0):
            entry.is_synced = False
            return

        while entry.synced_len < seq_len:
            before = entry.synced_len
            self._write_through_req(
                req,
                max_tokens=self.prefill_write_chunk_size,
                force=True,
                is_decode=False,
            )
            if entry.synced_len == before:
                break
        entry.is_synced = entry.synced_len >= seq_len

    def _sync_decode(self, req: Req, seq_len: int) -> None:
        if req.req_pool_idx is None or seq_len <= 0:
            return
        entry = self.entries.setdefault(req.rid, SyncCacheEntry(rid=req.rid, req=req))
        entry.req = req
        self._write_through_req(req, is_decode=True)

    def sync_unsynced_reqs(self) -> None:
        unsynced_entries = [
            x for x in self.entries.values() if x.req is not None and not x.is_synced
        ]
        unsynced_entries.sort(key=lambda x: self._infer_seq_len(x.req))
        for entry in unsynced_entries:
            if self._is_writing_overloaded():
                break
            seq_len = self._infer_seq_len(entry.req)
            self._sync_prefill(entry.req, seq_len)

    def sync_batch(self, batch: ScheduleBatch) -> None:
        with self._lock:
            self._drain_write_acks()
            is_extend = batch.forward_mode.is_extend()
            for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
                if is_extend:
                    self._sync_prefill(req, int(seq_len))
                else:
                    self._sync_decode(req, int(seq_len))
            if not self._is_writing_overloaded():
                self.sync_unsynced_reqs()

    def _evict_device(self, req: Req, evict_len: int) -> None:
        if evict_len <= 0 or req.req_pool_idx is None:
            return
        current_len = self._infer_seq_len(req)
        if current_len <= 0:
            return
        evict_len = min(evict_len, current_len)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :evict_len]
        self.token_to_kv_pool_allocator.free(kv_indices)

    def evict_device(self, req: Req, seq_len: int) -> None:
        with self._lock:
            self._drain_write_acks()
            entry = self.entries.get(req.rid)
            if entry is None:
                return
            # Push all remaining unsynced tokens to write queue first.
            self._write_through_req(req, force=True)
            target = min(seq_len, self._infer_seq_len(req))
            if target <= 0:
                return

            if entry.pending_write_len > 0:
                self.req_to_evict[req.rid] = target
                ready = max(min(entry.written_len, target) - entry.evicted_len, 0)
                if ready > 0:
                    self._evict_device(req, ready)
                    entry.evicted_len += ready
            else:
                to_evict = max(target - entry.evicted_len, 0)
                if to_evict > 0:
                    self._evict_device(req, to_evict)
                    entry.evicted_len += to_evict

    def evict_nowait(self, reqs: list[Req], num_tokens: int) -> Tuple[list[int], list[int]]:
        with self._lock:
            self._drain_write_acks()
            keep_indices: list[int] = []
            removed_indices: list[int] = []

            for i, req in enumerate(reqs):
                if self.token_to_kv_pool_allocator.available_size() >= num_tokens:
                    keep_indices.extend(range(i, len(reqs)))
                    break

                seq_len = self._infer_seq_len(req)
                if seq_len <= 0 or req.req_pool_idx is None:
                    removed_indices.append(i)
                    continue

                self._evict_device(req, seq_len)
                removed_indices.append(i)

            if self.token_to_kv_pool_allocator.available_size() < num_tokens:
                return [], list(range(len(reqs)))

            if not keep_indices:
                keep_set = set(removed_indices)
                keep_indices = [i for i in range(len(reqs)) if i not in keep_set]
            return keep_indices, removed_indices

    def cache_unfinished_req(self, req: Req, chunked: bool = False):
        super().cache_unfinished_req(req, chunked=chunked)
        with self._lock:
            self._drain_write_acks()
            self._sync_prefill(req, self._infer_seq_len(req))

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        with self._lock:
            self._drain_write_acks(wait_for_rid=req.rid)
            entry = self.entries.pop(req.rid, None)
            if entry is not None:
                for host_indices in entry.host_segments:
                    self.cache_controller.evict_host(host_indices)
        super().cache_finished_req(req, is_insert=is_insert, **kwargs)

    def reset(self):
        with self._lock:
            self.entries.clear()
            self.write_token_num = 0
            self.wrote_token_num = 0
            self.cache_controller.reset()
            self.cache_controller.mem_pool_host.clear()

    def flush_write_through_acks(self) -> None:
        with self._lock:
            self._drain_write_acks()

    def check_hicache_events(self):
        with self._lock:
            self._drain_write_acks()
            self.sync_unsynced_reqs()

    def get_loading_workload(self) -> Tuple[int, float]:
        return self.cache_controller.get_loading_workload()

    def get_writing_workload(self, rid: Optional[str] = None) -> Tuple[int, float]:
        if rid is None:
            return self.cache_controller.get_writing_workload()
        entry = self.entries.get(rid)
        if entry is None:
            return 0, self.cache_controller.get_writing_workload()[1]
        return entry.pending_write_len, self.cache_controller.get_writing_workload()[1]

    def get_write_through_backlog(self) -> int:
        backlog, _ = self.cache_controller.get_writing_workload()
        return backlog

    def is_writing(self, rid: str) -> bool:
        return self.cache_controller.is_writing(rid)
