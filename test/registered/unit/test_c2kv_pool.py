import unittest

import torch

from sglang.srt.mem_cache.c2kv_pool import C2KVPool, calculate_c2kv_pool_size


class TestC2KVPoolSizing(unittest.TestCase):
    def test_calculate_c2kv_pool_size(self):
        max_tokens, bytes_per_token = calculate_c2kv_pool_size(
            total_memory_bytes=80 * (1 << 30),
            pool_fraction=0.01,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            value_head_dim=128,
            dtype=torch.bfloat16,
        )

        self.assertEqual(bytes_per_token, 32 * 8 * (128 + 128) * 2 + 9)
        self.assertEqual(max_tokens, int(80 * (1 << 30) * 0.01) // bytes_per_token)

    def test_reject_entry_larger_than_pool(self):
        pool = C2KVPool(max_total_tokens=1, max_entry_tokens=2)
        gist_mask = torch.ones((1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "exceeding the pool capacity"):
            pool.store(
                key_hash="key",
                gist_key_values=[],
                gist_mask=gist_mask,
                gist_position_ids=torch.arange(2).unsqueeze(0),
                original_seq_len=8,
            )

    def test_reject_entry_larger_than_per_entry_limit(self):
        pool = C2KVPool(max_total_tokens=8, max_entry_tokens=1)
        gist_mask = torch.ones((1, 2), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "per-entry limit"):
            pool.store(
                key_hash="key",
                gist_key_values=[],
                gist_mask=gist_mask,
                gist_position_ids=torch.arange(2).unsqueeze(0),
                original_seq_len=8,
            )


if __name__ == "__main__":
    unittest.main()
