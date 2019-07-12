from unittest import TestCase
import hilbert as h
try:
    import torch
except ImportError:
    torch = None


class TestDependencySampler(TestCase):

    def test_dependency_sampler(self):
        V = torch.tensor([
            [0,0,0,0],
            [1,1,1,1],
            [-1,1,-1,1],
            [2,0,2,0],
            [-1,-1,1,1],
            [-1,-1,-1,-1],
        ], dtype=torch.float32)
        W = torch.tensor([
            [-1,1,-1,1],
            [0.5,0.5,0.5,0.5],
            [-1,-1,1,1],
            [1,1,1,1],
            [-1,-1,-1,-1],
            [2,0,2,0],
        ], dtype=torch.float32)
        sampler = h.loader.DependencySampler(V=V, W=W)
        positives = torch.tensor([
            [[0,1,2,3,0], [3,2,1,0,0]],
            [[1,2,3,0,0], [3,2,0,1,0]],
            [[2,3,4,5,0], [5,4,2,3,0]],
            [[1,2,0,0,0], [2,1,0,0,0]]
        ], dtype=torch.int64)
        mask = torch.tensor([
            [1,1,1,1,0],
            [1,1,1,0,0],
            [1,1,1,1,0],
            [1,1,0,0,0],
        ], dtype=torch.uint8)

        negatives = sampler.sample(positives, mask)

        # draw negative samples
        # validate shape and range of negative samples
        # check that negative samples are distributed as expected



