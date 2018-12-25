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


class MockLoader(h.cpu_loader.MultiLoader):

    def __init__(self, shard_schedule, num_loaders, bigram_path, device=None):
        self.bigram_path = bigram_path
        super(MockLoader, self).__init__(shard_schedule, num_loaders, device)

    def setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def preload(self, shard_spec):
        return self.bigram.load_shard(shard=shard_spec, device='cpu')

    def load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        Nxx, Nx, Nxt, N = cpu_data
        return Nxx.to(device), Nx.to(device), Nxt.to(device), N.to(device)



class TestCPULoader(TestCase):

    #def test_multi_cpu_loader(self):

    #    bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
    #    preloaders = [
    #        h.preloader.BigramPreloader(bigram_path),
    #        h.preloader.BigramPreloader(bigram_path)
    #    ]
    #    shard_schedule = h.shards.Shards(5)
    #    cpu_loader = h.cpu_loader.MockLoader(
    #        shard_schedule, preloaders, bigram_path)

    #    expected_shards = list(shard_schedule)
    #    expected_worker_ids = set(range(len(preloaders)))

    #    for found_shard_spec, found_data in cpu_loader:
    #        self.assertTrue(found_shard_spec in expected_shards)
    #        worker_id, shard_spec_in_data = found_data
    #        self.assertEqual(found_shard_spec, shard_spec_in_data)
    #        self.assertTrue(worker_id in expected_worker_ids)


    def test_multi_cpu_loader(self):

        shard_schedule = h.shards.Shards(5)
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        num_loaders = 2
        cpu_loader = MockLoader(shard_schedule, num_loaders, bigram_path)

        expected_bigram = h.bigram.Bigram.load(bigram_path)
        expected_shards = list(shard_schedule)
        num_shards_iterated = 0
        for found_shard_spec, found_data in cpu_loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_spec in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_spec, device=h.CONSTANTS.MATRIX_DEVICE)
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


#    def test_bigram_cpu_loader(self):
#
#        shard_schedule = h.shards.Shards(5)
#        num_workers = 2
#        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
#        cpu_loader = h.cpu_loader.construct_bigram_cpu_loader(
#            shard_schedule, bigram_path, num_workers)
#
#        expected_bigram = h.bigram.Bigram.load(bigram_path)
#        expected_shards = list(shard_schedule)
#
#        num_shards_iterated = 0
#        for found_shard_spec, found_data in cpu_loader:
#            num_shards_iterated += 1
#            self.assertTrue(found_shard_spec in expected_shards)
#            expected_data = expected_bigram.load_shard(
#                found_shard_spec, device='cpu')
#            for found_tensor, expected_tensor in zip(found_data, expected_data):
#                self.assertTrue(torch.allclose(found_tensor, expected_tensor))
#
#        self.assertEqual(num_shards_iterated, len(expected_shards))


