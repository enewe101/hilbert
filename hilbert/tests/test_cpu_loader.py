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


class PreloaderMock(h.preloader.Preloader):
    def setup(self, loader_id):
        self.loader_id = loader_id

    def __getitem__(self, shard_spec):
        return self.loader_id, shard_spec


class TestCPULoader(TestCase):

    def test_multi_cpu_loader(self):

        preloaders = [PreloaderMock(), PreloaderMock()]
        shard_schedule = h.shards.Shards(5)
        cpu_loader = h.cpu_loader.MultiCPULoader(shard_schedule, preloaders)

        expected_shards = list(shard_schedule)
        expected_worker_ids = set(range(len(preloaders)))

        for found_shard_spec, found_data in cpu_loader:
            self.assertTrue(found_shard_spec in expected_shards)
            worker_id, shard_spec_in_data = found_data
            self.assertEqual(found_shard_spec, shard_spec_in_data)
            self.assertTrue(worker_id in expected_worker_ids)


    def test_bigram_cpu_loader(self):

        shard_schedule = h.shards.Shards(5)
        num_workers = 2
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        cpu_loader = h.cpu_loader.construct_bigram_cpu_loader(
            shard_schedule, bigram_path, num_workers)

        expected_bigram = h.bigram.Bigram.load(bigram_path)
        expected_shards = list(shard_schedule)

        num_shards_iterated = 0
        for found_shard_spec, found_data in cpu_loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_spec in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_spec, device='cpu')
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


