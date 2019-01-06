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


class TestBigramLoader(TestCase):

    def test_bigram_loader(self):

        # Make a bigram loader
        t = 1e-5
        alpha = 0.6
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        sector_factor = 3
        shard_factor = 4
        num_loaders = sector_factor**2
        loader = h.bigram_loader.BigramLoader(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t, 
            alpha_unigram_smoothing=alpha
        )

        expected_bigram = h.bigram.BigramSector.load(
            bigram_path, h.shards.whole)
        expected_bigram.apply_w2v_undersampling(t)
        expected_bigram.apply_unigram_smoothing(alpha)

        expected_shards = list(h.shards.Shards(shard_factor * sector_factor))
        num_shards_iterated = 0
        for found_shard_id, bigram_data, unigram_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_id in expected_shards)
            expected_bigram_data = expected_bigram.load_shard(found_shard_id)
            expected_unigram_data = expected_bigram.load_unigram_shard(
                found_shard_id)

            comparisons = [
                (bigram_data, expected_bigram_data),
                (unigram_data, expected_unigram_data)]

            for found_data, expected_data in comparisons:
                for f_tensor, ex_tensor in zip(found_data, expected_data):
                    self.assertTrue(torch.allclose(f_tensor, ex_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


class TestBigramMultiLoader(TestCase):

    def test_bigram_multi_loader(self):

        # Make a bigram loader
        t = 1e-5
        alpha = 0.6
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        sector_factor = 3
        shard_factor = 4
        num_loaders = sector_factor**2
        loader = h.bigram_loader.BigramMultiLoader(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t, 
            alpha_unigram_smoothing=alpha
        )

        expected_bigram = h.bigram.BigramSector.load(
            bigram_path, h.shards.whole)
        expected_bigram.apply_w2v_undersampling(t)
        expected_bigram.apply_unigram_smoothing(alpha)

        expected_shards = list(h.shards.Shards(sector_factor * shard_factor))
        num_shards_iterated = 0
        for found_shard_id, bigram_data, unigram_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_id in expected_shards)
            expected_bigram_data = expected_bigram.load_shard(found_shard_id)
            expected_unigram_data = expected_bigram.load_unigram_shard(
                found_shard_id)

            comparisons = [
                (bigram_data, expected_bigram_data),
                (unigram_data, expected_unigram_data)]

            for found_data, expected_data in comparisons:
                for f_tensor, ex_tensor in zip(found_data, expected_data):
                    self.assertTrue(torch.allclose(f_tensor, ex_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))
