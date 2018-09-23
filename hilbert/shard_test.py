import torch

def test_shard_1(vocab, d):
    num_shards = 4
    v = torch.rand((vocab, d), device='cpu')
    w = torch.rand((vocab, d), device='cpu')
    m = torch.rand((vocab, vocab), device='cpu')
    c = torch.zeros((vocab, d), device='cpu')

    for i_shard in range(num_shards):
        for j_shard in range(num_shards):
            a_shard = a[i_shard::num_shards, j_shard::num_shards].to('cuda')
            b_shard = b[i_shard::num_shards, j_shard::num_shards].to('cuda')
            c[i_shard::num_shards, j_shard::num_shards] = a_shard * b_shard
    return c


def test_shard_1(size):

    num_shards = 4
    a = torch.rand((size, size), device='cpu')
    b = torch.rand((size, size), device='cpu')
    c = torch.zeros((size, size), device='cpu')

    for i_shard in range(num_shards):
        for j_shard in range(num_shards):
            a_shard = a[i_shard::num_shards, j_shard::num_shards].to('cuda')
            b_shard = b[i_shard::num_shards, j_shard::num_shards].to('cuda')
            c[i_shard::num_shards, j_shard::num_shards] = a_shard * b_shard
    return c
