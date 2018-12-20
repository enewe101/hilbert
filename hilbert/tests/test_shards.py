from unittest import TestCase
import hilbert as h

try:
    import torch
except ImportError:
    torch = None


class TestShards(TestCase):

    def test_shards_iteration(self):
        shard_factor = 4
        shards = h.shards.Shards(shard_factor)
        M = torch.arange(64, dtype=torch.float32).view(8,8)
        self.assertTrue(len(list(shards)))
        for i, shard in enumerate(shards):
            if i == 0:
                expected = torch.Tensor([[0,4],[32,36]])
                self.assertTrue(torch.allclose(M[shard], expected))
            elif i == 1:
                expected = torch.Tensor([[1,5],[33,37]])
                self.assertTrue(torch.allclose(M[shard], expected))
            else:
                expected_shard = torch.Tensor([
                    j for j in range(64)
                    # If the row is among allowed rows
                    if (j // 8) % shard_factor == i // shard_factor
                    # If the column is among alowed columns
                    and (j % 8) % shard_factor == i % shard_factor
                ]).view(2,2)
                self.assertTrue(torch.allclose(
                    M[shard], expected_shard
                ))



