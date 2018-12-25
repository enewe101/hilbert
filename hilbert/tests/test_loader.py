import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class MockLoader(h.loader.Loader):

    def __init__(
        self, shard_schedule, num_loaders, bigram_path,
        queue_size=1, device=None
    ):
        self.bigram_path = bigram_path
        super(MockLoader, self).__init__(
            shard_schedule=shard_schedule, num_loaders=num_loaders, 
            queue_size=queue_size, device=device)

    def _setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def _preload(self, shard_spec):
        return self.bigram.load_shard(shard=shard_spec, device='cpu')

    def _load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        Nxx, Nx, Nxt, N = cpu_data
        return Nxx.to(device), Nx.to(device), Nxt.to(device), N.to(device)


class MockMultiLoader(h.loader.MultiLoader):

    def __init__(
        self, shard_schedule, num_loaders, bigram_path,
        queue_size=1, device=None
    ):
        self.bigram_path = bigram_path
        super(MockMultiLoader, self).__init__(
            shard_schedule=shard_schedule, num_loaders=num_loaders, 
            queue_size=queue_size, device=device)

    def _setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def _preload(self, shard_spec):
        return self.bigram.load_shard(shard=shard_spec, device='cpu')

    def _load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        Nxx, Nx, Nxt, N = cpu_data
        return Nxx.to(device), Nx.to(device), Nxt.to(device), N.to(device)



class TestCPULoader(TestCase):


    def test_loader(self):

        shard_schedule = h.shards.Shards(5)
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        num_loaders = 2
        loader = MockLoader(shard_schedule, num_loaders, bigram_path)

        expected_bigram = h.bigram.Bigram.load(bigram_path)
        expected_shards = list(shard_schedule)
        num_shards_iterated = 0
        for found_shard_spec, found_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_spec in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_spec, device=h.CONSTANTS.MATRIX_DEVICE)
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


    def test_multi_loader(self):

        shard_schedule = h.shards.Shards(5)
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        num_loaders = 2
        loader = MockMultiLoader(shard_schedule, num_loaders, bigram_path)

        expected_bigram = h.bigram.Bigram.load(bigram_path)
        expected_shards = list(shard_schedule)
        num_shards_iterated = 0
        for found_shard_spec, found_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_spec in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_spec, device=h.CONSTANTS.MATRIX_DEVICE)
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


    def test_bigram_loader(self):

        shard_schedule = h.shards.Shards(5)
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        num_loaders = 2
        loader = h.loader.BigramLoader(
            shard_schedule, num_loaders, bigram_path)

        expected_bigram = h.bigram.Bigram.load(bigram_path)
        expected_shards = list(shard_schedule)
        num_shards_iterated = 0
        for found_shard_spec, found_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_spec in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_spec, device=h.CONSTANTS.MATRIX_DEVICE)
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


