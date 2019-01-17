import time
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


class MockLoader():

    def __init__(self, bigram_path, shard_factor, num_loaders, queue_size=1):
        self.bigram = h.bigram.BigramBase.load(bigram_path)
        self.shard_factor = shard_factor
        super(MockLoader, self).__init__(
            num_loaders=num_loaders, queue_size=queue_size)

    def _preload_iter(self, worker_id):
        for i, shard_id in enumerate(h.shards.Shards(self.shard_factor)):
            if i % self.num_loaders == worker_id:
                cpu_data = self.bigram.load_shard(shard=shard_id, device='cpu')
                yield shard_id, cpu_data

    def _load(self, preloaded):
        shard_id, cpu_data = preloaded
        device = h.CONSTANTS.MATRIX_DEVICE
        return shard_id, tuple(tensor.to(device) for tensor in cpu_data)


class SimpleMockLoader():

    def _preload_iter(self, worker_id):
        for i in range(10):
            yield worker_id, i

    def _load(self, preloaded):
        return preloaded



class TestLoader(TestCase):


    def test_loader(self):

        class MockSequentialLoader(MockLoader, h.loader.Loader):
            pass

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        shard_factor = 3
        num_loaders = 2
        loader = MockSequentialLoader(bigram_path, shard_factor, num_loaders)
        expected_bigram = h.bigram.BigramBase.load(bigram_path)
        expected_shards = list(h.shards.Shards(shard_factor))
        num_shards_iterated = 0
        for found_shard_id, found_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_id in expected_shards)
            expected_data = expected_bigram.load_shard(
                found_shard_id, device=h.CONSTANTS.MATRIX_DEVICE)
            for found_tensor, expected_tensor in zip(found_data, expected_data):
                self.assertTrue(torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


class TestMultiLoaderKeep(TestCase):

    def test_multi_loader(self):
        shard_factor = 2
        num_loaders = 9
        sector_factor = 3
        num_epochs = 3
        bigram_path = os.path.join(
            '/home', 'ndg', 'projects', 'shared_datasets', 'hilbert',
            'cooccurrence', '5w-dynamic-10k-giga-wiki'
        )
        loader = h.bigram_loader.BigramMultiLoader(
            bigram_path, sector_factor, shard_factor, num_loaders=num_loaders,
            queue_size=32
        )
        for epoch in range(num_epochs):
            print('+')
            first = True
            start = time.time()
            for item in loader:
                if first:
                    first = False
                    print(time.time() - start)
                print('.')


class TestMultiLoader(TestCase):

    def test_multi_loader(self):

        class MockMultiLoader(MockLoader, h.loader.MultiLoader):
            pass

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        shard_factor = 5
        num_loaders = 2
        num_epochs = 3

        loader = MockMultiLoader(bigram_path, shard_factor, num_loaders)

        expected_bigram = h.bigram.BigramBase.load(bigram_path)
        expected_shards = list(h.shards.Shards(shard_factor))
        num_shards_iterated = 0
        for epoch in range(num_epochs):
            print
            for found_shard_id, found_data in loader:
                num_shards_iterated += 1
                self.assertTrue(found_shard_id in expected_shards)
                expected_data = expected_bigram.load_shard(
                    found_shard_id, device=h.CONSTANTS.MATRIX_DEVICE)
                comparisons = zip(found_data, expected_data)
                for found_tensor, expected_tensor in comparisons:
                    self.assertTrue(
                        torch.allclose(found_tensor, expected_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards)*num_epochs)


class TestMultiLoaderKeepalive(TestCase):

    def test_multi_loader(self):
        class SimpleMockMultiLoader(SimpleMockLoader, h.loader.MultiLoader):
            pass
        num_loaders = 2
        loader = SimpleMockMultiLoader(num_loaders)
        for worker_id, item_num in loader:
            print(worker_id, item_num)
        print('\n'*3)
        for worker_id, item_num in loader:
            print(worker_id, item_num)





