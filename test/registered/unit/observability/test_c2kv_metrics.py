import unittest
from types import SimpleNamespace

from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.srt.observability.scheduler_metrics_mixin import SchedulerMetricsMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class FakeC2KVPool:
    max_total_tokens = 100

    @staticmethod
    def current_tokens():
        return 25

    @staticmethod
    def num_entries():
        return 3


class TestC2KVMetrics(unittest.TestCase):
    def test_log_c2kv_pool_stats(self):
        scheduler = SimpleNamespace(
            c2kv_pool=FakeC2KVPool(),
            stats=SchedulerStats(),
        )

        SchedulerMetricsMixin._log_c2kv_pool_stats(scheduler)

        self.assertEqual(scheduler.stats.c2kv_pool_used_tokens, 25)
        self.assertEqual(scheduler.stats.c2kv_pool_total_tokens, 100)
        self.assertEqual(scheduler.stats.c2kv_pool_utilization, 0.25)
        self.assertEqual(scheduler.stats.c2kv_pool_num_entries, 3)

    def test_disabled_c2kv_pool_keeps_zero_stats(self):
        scheduler = SimpleNamespace(c2kv_pool=None, stats=SchedulerStats())

        SchedulerMetricsMixin._log_c2kv_pool_stats(scheduler)

        self.assertEqual(scheduler.stats.c2kv_pool_used_tokens, 0)
        self.assertEqual(scheduler.stats.c2kv_pool_total_tokens, 0)
        self.assertEqual(scheduler.stats.c2kv_pool_utilization, 0.0)
        self.assertEqual(scheduler.stats.c2kv_pool_num_entries, 0)

    def test_c2kv_pool_log_msg(self):
        scheduler = SimpleNamespace(c2kv_pool=FakeC2KVPool())

        msg = SchedulerMetricsMixin._get_c2kv_pool_log_msg(scheduler)

        self.assertEqual(
            msg,
            "c2kv pool: 25/100 tokens, c2kv usage: 0.25, #c2kv-entry: 3, ",
        )

    def test_disabled_c2kv_pool_log_msg(self):
        scheduler = SimpleNamespace(c2kv_pool=None)

        self.assertEqual(
            SchedulerMetricsMixin._get_c2kv_pool_log_msg(scheduler), ""
        )


if __name__ == "__main__":
    unittest.main()
