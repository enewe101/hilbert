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


class TestBigramPreloader(TestCase):


    def test_bigram_preloader(self):

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
        preloader = h.preloader.BigramPreloader(bigram_path)
        bigram = h.bigram.Bigram.load(bigram_path)

        shards = h.shards.Shards(5)

        # Trying to load a shard before running preloader.setup causes an
        # error because bigram is not yet loaded
        with self.assertRaises(AttributeError):
            preloader[shards[0]]

        worker_id = 0
        preloader.setup(worker_id)
        for shard_spec in shards: 
            expected = bigram.load_shard(shard_spec, device='cpu')
            found = preloader[shard_spec]
            self.assertEqual(len(expected), len(found))
            for found_tensor, expected_tensor in zip(found, expected):
                self.assertTrue(torch.allclose(expected_tensor, found_tensor))
                self.assertEqual(expected_tensor.device.type, 'cpu')

