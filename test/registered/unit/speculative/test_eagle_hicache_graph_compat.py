import unittest
from types import SimpleNamespace

from sglang.srt.speculative.eagle_worker import (
    _LARGE_HICACHE_GRAPH_LIMIT_BYTES,
    _should_disable_draft_extend_graph_for_hicache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class TestEagleHiCacheGraphCompat(unittest.TestCase):
    def test_large_explicit_hicache_size_crosses_graph_limit(self):
        self.assertLess(4 * 1_000_000_000, _LARGE_HICACHE_GRAPH_LIMIT_BYTES)
        self.assertGreater(5 * 1_000_000_000, _LARGE_HICACHE_GRAPH_LIMIT_BYTES)

    def test_large_hicache_disables_only_draft_extend_graph(self):
        server_args = SimpleNamespace(
            enable_hierarchical_cache=True,
            hicache_size=5,
        )

        self.assertTrue(_should_disable_draft_extend_graph_for_hicache(server_args))

    def test_small_or_disabled_hicache_keeps_draft_extend_graph(self):
        for enabled, size in [(True, 4), (False, 40), (True, 0)]:
            with self.subTest(enabled=enabled, size=size):
                server_args = SimpleNamespace(
                    enable_hierarchical_cache=enabled,
                    hicache_size=size,
                )
                self.assertFalse(
                    _should_disable_draft_extend_graph_for_hicache(server_args)
                )


if __name__ == "__main__":
    unittest.main()
