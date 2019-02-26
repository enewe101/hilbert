import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h
import random

try:
    import numpy as np
    import torch
    from scipy import sparse
except ImportError:
    np = None
    torch = None
    sparse = None


class TestBigramBase(TestCase):

    def get_test_bigram_base(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramBase(unigram, array, verbose=False)
        return bigram


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    def test_bigram_base(self):
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
        array = torch.tensor(Nxx.toarray(), dtype=dtype)

        # BigramSector's length and shape are correct.
        self.assertTrue(len(bigram), len(unigram))
        self.assertEqual(bigram.shape, array.shape)

        # Except for the cooccurrence matrix Nxx, which is in sparse
        # matrix form, the other statistics are `torch.Tensor`s.
        self.assertTrue(isinstance(bigram.Nxx, sparse.lil_matrix))
        self.assertTrue(isinstance(bigram.Nx, torch.Tensor))
        self.assertTrue(isinstance(bigram.Nxt, torch.Tensor))
        self.assertTrue(isinstance(bigram.uNx, torch.Tensor))
        self.assertTrue(isinstance(bigram.uNxt, torch.Tensor))
        self.assertTrue(isinstance(bigram.uN, torch.Tensor))
        self.assertTrue(isinstance(bigram.N, torch.Tensor))

        
        # Bigram posesses full unigram data
        self.assertEqual(bigram.unigram, unigram)
        self.assertEqual(bigram.dictionary, unigram.dictionary)
        self.assertTrue(torch.allclose(
            bigram.uNx, torch.tensor(unigram.Nx, dtype=dtype).reshape(-1,1)))
        self.assertTrue(torch.allclose(
            bigram.uNxt, torch.tensor(unigram.Nx, dtype=dtype).reshape(1,-1)))
        self.assertTrue(torch.allclose(
            bigram.uN, torch.sum(torch.tensor(unigram.Nx, dtype=dtype))))


        # Bigram posesses full bigram data
        self.assertTrue(np.allclose(bigram.Nxx.toarray(), array.numpy()))
        self.assertTrue(torch.allclose(
            bigram.Nx, torch.sum(array, dim=1, keepdim=True)))
        self.assertTrue(torch.allclose(
            bigram.Nxt, torch.sum(array, dim=0, keepdim=True)))
        self.assertTrue(torch.allclose(
            bigram.N, torch.tensor(np.sum(Nxx), dtype=dtype)))



    def test_invalid_arguments(self):

        random.seed(0)

        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()

        # BigramBases should generally be made by passing a unigram and Nxx
        h.bigram.BigramBase(unigram, Nxx)

        # BigramBases need a sorted unigram instance
        unsorted_unigram = deepcopy(unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            h.bigram.BigramBase(unsorted_unigram, Nxx)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            h.bigram.BigramBase(truncated_unigram, Nxx)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            h.bigram.BigramBase(truncated_unigram, Nxx)



    def test_load_shard(self):
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        shards = h.shards.Shards(3)

        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramBase(unigram, Nxx, verbose=False, device=device)

        Nxx = torch.tensor(Nxx, device=device, dtype=dtype)
        Nx = torch.sum(Nxx, dim=1, keepdim=True)
        Nxt = torch.sum(Nxx, dim=0, keepdim=True)
        N = torch.sum(Nxx)

        uNx = torch.tensor(
            np.array(unigram.Nx).reshape(-1, 1), device=device, dtype=dtype)
        uNxt = torch.tensor(
            np.array(unigram.Nx).reshape(1, -1), device=device, dtype=dtype)
        uN = torch.tensor(np.sum(unigram.Nx), device=device, dtype=dtype)

        for shard in shards:

            # Check that the shard is correctly loaded
            sNxx, sNx, sNxt, sN = bigram.load_shard(shard, device)
            self.assertTrue(torch.allclose(Nxx[shard], sNxx))
            self.assertTrue(torch.allclose(Nx[shard[0]], sNx))
            self.assertTrue(torch.allclose(Nxt[:,shard[1]], sNxt))
            self.assertTrue(torch.allclose(N, sN))

            # Shards can also be loaded using __getitem__.
            sNxx, sNx, sNxt, sN = bigram[shard]
            self.assertTrue(torch.allclose(Nxx[shard], sNxx))
            self.assertTrue(torch.allclose(Nx[shard[0]], sNx))
            self.assertTrue(torch.allclose(Nxt[:,shard[1]], sNxt))
            self.assertTrue(torch.allclose(N, sN))

            # Unigram statistics for the shard can be loaded.
            suNx, suNxt, suN = bigram.load_unigram_shard(shard, device)
            self.assertTrue(torch.allclose(uNx[shard[0]], suNx))
            self.assertTrue(torch.allclose(uNxt[:,shard[1]], suNxt))
            self.assertTrue(torch.allclose(uN, suN))


    def test_merge(self):

        # Make a bigram
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramBase(unigram, array)

        # Make a similar bigram, but change some of the bigram statistics
        decremented_array = array - 1
        decremented_array[decremented_array<0] = 0
        decremented_bigram = h.bigram.BigramBase(unigram,decremented_array)

        # Merge the two bigram instances
        bigram.merge(decremented_bigram)

        # The merged bigram should have the sum of the individual bigrams'
        # statistics.
        self.assertTrue(np.allclose(
            bigram.Nxx.toarray(), array + decremented_array))


    def test_apply_unigram_smoothing(self):
        alpha = 0.6
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()

        expected_uNx = bigram.uNx**alpha
        expected_uNxt = bigram.uNxt**alpha
        expected_uN = torch.sum(expected_uNx)

        bigram.apply_unigram_smoothing(alpha)
        self.assertTrue(torch.allclose(expected_uNx, bigram.uNx))
        self.assertTrue(torch.allclose(expected_uNxt, bigram.uNxt))
        self.assertTrue(torch.allclose(expected_uN, bigram.uN))



    def test_apply_w2v_undersampling(self):

        t = 1e-5
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()

        # Initially the counts reflect the provided cooccurrence matrix
        Nxx, Nx, Nxt, N = bigram.load_shard()
        uNx, uNxt, uN = bigram.unigram
        self.assertTrue(np.allclose(Nxx, bigram.Nxx.toarray()))

        # Now apply undersampling
        p_i = h.corpus_stats.w2v_prob_keep(uNx, uN, t)
        p_j = h.corpus_stats.w2v_prob_keep(uNxt, uN, t)
        expected_Nxx = Nxx * p_i * p_j

        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = torch.sum(expected_Nxx, dim=0, keepdim=True)
        expected_N = torch.sum(expected_Nxx)

        #pre_PMI = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        bigram.apply_w2v_undersampling(t)
        #nNxx, nNx, nNxt, nN = bigram.load_shard()
        #post_PMI = h.corpus_stats.calc_PMI((nNxx, nNx, nNxt, nN))
        #diff = torch.sum((pre_PMI - post_PMI) / pre_PMI) / (500*500)

        found_Nxx, found_Nx, found_Nxt, found_N = bigram.load_shard()

        self.assertTrue(torch.allclose(found_Nxx, expected_Nxx))
        self.assertTrue(torch.allclose(found_Nx, expected_Nx))
        self.assertTrue(torch.allclose(found_Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(found_N, expected_N))

        # attempting to call apply_undersampling twice is an error
        with self.assertRaises(ValueError):
            bigram.apply_w2v_undersampling(t)

        # Attempting to call apply_undersampling when in posession of a 
        # smoothed unigram would produce incorrect results, and is an error.
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
        alpha = 0.6
        unigram.apply_smoothing(alpha)
        with self.assertRaises(ValueError):
            bigram.apply_w2v_undersampling(t)




    def test_count(self):
        bigram = self.get_test_bigram_base()
        self.assertTrue(bigram.count('banana', 'socks'), 3)
        self.assertTrue(bigram.count('socks', 'car'), 1)


    def test_density(self):
        bigram = self.get_test_bigram_base()
        self.assertEqual(bigram.density(), 0.5)
        self.assertEqual(bigram.density(2), 0.125)


    def test_get_sector(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram_base, unigram, Nxx = h.corpus_stats.get_test_bigram_base()

        for sector in h.shards.Shards(3):
            bigram_sector = bigram_base.get_sector(sector)
            self.assertTrue(isinstance(
                bigram_sector, h.bigram.BigramSector))
            self.assertTrue(np.allclose(
                bigram_sector.Nxx.toarray(),
                bigram_base.Nxx[sector].toarray()
            ))


