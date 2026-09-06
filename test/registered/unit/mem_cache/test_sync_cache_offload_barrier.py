import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.sync_cache import SyncCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class _StopOffload(Exception):
    pass


class TestSyncCacheOffloadBarrier(unittest.TestCase):
    def test_sync_cache_runs_barrier_before_radix_sync(self):
        cache = SyncCache.__new__(SyncCache)
        order = []
        cache._req_states = {}
        cache._before_device_offload = lambda: order.append("barrier")
        cache._infer_seq_len = MagicMock(return_value=8)

        def stop_after_radix_sync(req, seq_len):
            order.append("radix_sync")
            raise _StopOffload

        cache._sync_req_to_radix = stop_after_radix_sync
        req = SimpleNamespace(req_pool_idx=0, rid="req")

        with self.assertRaises(_StopOffload):
            cache.evict_device(req)

        self.assertEqual(order, ["barrier", "radix_sync"])

    def test_scheduler_orders_offload_after_forward_when_overlapping(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_overlap = True
        current_stream = MagicMock()
        scheduler.device_module = MagicMock()
        scheduler.device_module.current_stream.return_value = current_stream
        scheduler.forward_stream = MagicMock()

        scheduler._wait_for_inflight_forward_before_device_offload()

        current_stream.wait_stream.assert_called_once_with(scheduler.forward_stream)

    def test_scheduler_skips_barrier_without_overlap(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_overlap = False
        scheduler.device_module = MagicMock()
        scheduler.forward_stream = MagicMock()

        scheduler._wait_for_inflight_forward_before_device_offload()

        scheduler.device_module.current_stream.assert_not_called()

    def test_iteration_barrier_is_limited_to_request_offload_cache(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = SimpleNamespace(supports_request_offload=True)
        scheduler._wait_for_inflight_forward_before_device_offload = MagicMock()

        scheduler._apply_request_offload_overlap_barrier()

        scheduler._wait_for_inflight_forward_before_device_offload.assert_called_once()

        scheduler.tree_cache.supports_request_offload = False
        scheduler._wait_for_inflight_forward_before_device_offload.reset_mock()

        scheduler._apply_request_offload_overlap_barrier()

        scheduler._wait_for_inflight_forward_before_device_offload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
