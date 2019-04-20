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




def get_test_cooccurrence(device=None, verbose=True):
    """
    For testing purposes, builds a cooccurrence from constituents (not using
    it's own load function) and returns the cooccurrence along with the
    constituents used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
    unigram = h.unigram.Unigram.load(path, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    cooccurrence = h.cooccurrence.Cooccurrence(unigram, Nxx, verbose=verbose)

    return cooccurrence, unigram, Nxx


class TestCooccurrence(TestCase):

    def get_test_cooccurrence(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooccurrence.Cooccurrence(unigram, array, verbose=False)
        return cooccurrence


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    def test_cooccurrence(self):
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        cooccurrence, unigram, Nxx = get_test_cooccurrence()
        array = torch.tensor(Nxx.toarray(), dtype=dtype)

        # CooccurrenceSector's length and shape are correct.
        #self.assertTrue(len(cooccurrence), len(unigram))
        self.assertEqual(cooccurrence.shape, array.shape)

        # Except for the cooccurrence matrix Nxx, which is in sparse
        # matrix form, the other statistics are `torch.Tensor`s.
        self.assertTrue(isinstance(cooccurrence.Nxx, sparse.lil_matrix))
        self.assertTrue(isinstance(cooccurrence.Nx, torch.Tensor))
        self.assertTrue(isinstance(cooccurrence.Nxt, torch.Tensor))
        self.assertTrue(isinstance(cooccurrence.uNx, torch.Tensor))
        self.assertTrue(isinstance(cooccurrence.uNxt, torch.Tensor))
        self.assertTrue(isinstance(cooccurrence.uN, torch.Tensor))
        self.assertTrue(isinstance(cooccurrence.N, torch.Tensor))

        
        # Cooccurrence posesses full unigram data
        self.assertEqual(cooccurrence.unigram, unigram)
        self.assertEqual(cooccurrence.dictionary, unigram.dictionary)
        self.assertTrue(torch.allclose(
            cooccurrence.uNx, 
            torch.tensor(unigram.Nx, dtype=dtype).reshape(-1,1)
        ))
        self.assertTrue(torch.allclose(
            cooccurrence.uNxt,
            torch.tensor(unigram.Nx, dtype=dtype).reshape(1,-1)
        ))
        self.assertTrue(torch.allclose(
            cooccurrence.uN, torch.sum(torch.tensor(unigram.Nx, dtype=dtype))))

        # Cooccurrence posesses full cooccurrence data
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), array.numpy()))
        self.assertTrue(torch.allclose(
            cooccurrence.Nx, torch.sum(array, dim=1, keepdim=True)))
        self.assertTrue(torch.allclose(
            cooccurrence.Nxt, torch.sum(array, dim=0, keepdim=True)))
        self.assertTrue(torch.allclose(
            cooccurrence.N, torch.tensor(np.sum(Nxx), dtype=dtype)))



    def test_invalid_arguments(self):

        random.seed(0)

        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()

        # Cooccurrence should generally be made by passing a unigram and Nxx
        h.cooccurrence.Cooccurrence(unigram, Nxx)

        # Cooccurrences need a sorted unigram instance
        unsorted_unigram = deepcopy(unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            h.cooccurrence.Cooccurrence(unsorted_unigram, Nxx)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            h.cooccurrence.Cooccurrence(truncated_unigram, Nxx)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            h.cooccurrence.Cooccurrence(truncated_unigram, Nxx)



    def test_load_shard(self):
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        shards = h.shards.Shards(3)

        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooccurrence.Cooccurrence(unigram, Nxx, verbose=False)

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
            sNxx, sNx, sNxt, sN = cooccurrence.load_shard(shard, device)
            self.assertTrue(torch.allclose(Nxx[shard], sNxx))
            self.assertTrue(torch.allclose(Nx[shard[0]], sNx))
            self.assertTrue(torch.allclose(Nxt[:,shard[1]], sNxt))
            self.assertTrue(torch.allclose(N, sN))

            # Shards can also be loaded using __getitem__.
            sNxx, sNx, sNxt, sN = cooccurrence[shard]
            self.assertTrue(torch.allclose(Nxx[shard], sNxx))
            self.assertTrue(torch.allclose(Nx[shard[0]], sNx))
            self.assertTrue(torch.allclose(Nxt[:,shard[1]], sNxt))
            self.assertTrue(torch.allclose(N, sN))

            # Unigram statistics for the shard can be loaded.
            suNx, suNxt, suN = cooccurrence.load_unigram_shard(shard, device)
            self.assertTrue(torch.allclose(uNx[shard[0]], suNx))
            self.assertTrue(torch.allclose(uNxt[:,shard[1]], suNxt))
            self.assertTrue(torch.allclose(uN, suN))


    def test_merge(self):

        # Make a cooccurrence
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooccurrence.Cooccurrence(unigram, array)

        # Make a similar cooccurrence, but change some of the cooccurrence statistics
        decremented_array = array - 1
        decremented_array[decremented_array<0] = 0
        decremented_cooccurrence = h.cooccurrence.Cooccurrence(unigram,decremented_array)

        # Merge the two cooccurrence instances
        cooccurrence.merge(decremented_cooccurrence)

        # The merged cooccurrence should have the sum of the individual cooccurrences'
        # statistics.
        self.assertTrue(np.allclose(
            cooccurrence.Nxx.toarray(), array + decremented_array))


    def test_apply_unigram_smoothing(self):
        alpha = 0.6
        cooccurrence, unigram, Nxx = get_test_cooccurrence()

        expected_uNx = cooccurrence.uNx**alpha
        expected_uNxt = cooccurrence.uNxt**alpha
        expected_uN = torch.sum(expected_uNx)

        cooccurrence.apply_unigram_smoothing(alpha)
        self.assertTrue(torch.allclose(expected_uNx, cooccurrence.uNx))
        self.assertTrue(torch.allclose(expected_uNxt, cooccurrence.uNxt))
        self.assertTrue(torch.allclose(expected_uN, cooccurrence.uN))



    def test_apply_w2v_undersampling(self):

        t = 1e-5
        cooccurrence, unigram, Nxx = get_test_cooccurrence()

        # Initially the counts reflect the provided cooccurrence matrix
        Nxx, Nx, Nxt, N = cooccurrence.load_shard()
        uNx, uNxt, uN = cooccurrence.unigram
        self.assertTrue(np.allclose(Nxx, cooccurrence.Nxx.toarray()))

        # Now apply undersampling
        p_i = h.cooccurrence.cooccurrence.w2v_prob_keep(uNx, uN, t)
        p_j = h.cooccurrence.cooccurrence.w2v_prob_keep(uNxt, uN, t)
        expected_Nxx = Nxx * p_i * p_j

        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = torch.sum(expected_Nxx, dim=0, keepdim=True)
        expected_N = torch.sum(expected_Nxx)

        cooccurrence.apply_w2v_undersampling(t)

        found_Nxx, found_Nx, found_Nxt, found_N = cooccurrence.load_shard()

        self.assertTrue(torch.allclose(found_Nxx, expected_Nxx))
        self.assertTrue(torch.allclose(found_Nx, expected_Nx))
        self.assertTrue(torch.allclose(found_Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(found_N, expected_N))

        # attempting to call apply_undersampling twice is an error
        with self.assertRaises(ValueError):
            cooccurrence.apply_w2v_undersampling(t)

        # Attempting to call apply_undersampling when in posession of a 
        # smoothed unigram would produce incorrect results, and is an error.
        cooccurrence, unigram, Nxx = get_test_cooccurrence()
        alpha = 0.6
        unigram.apply_smoothing(alpha)
        with self.assertRaises(ValueError):
            cooccurrence.apply_w2v_undersampling(t)




    def test_count(self):
        cooccurrence = self.get_test_cooccurrence()
        self.assertTrue(cooccurrence.count('banana', 'socks'), 3)
        self.assertTrue(cooccurrence.count('socks', 'car'), 1)


    def test_get_sector(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence, unigram, Nxx = get_test_cooccurrence()

        for sector in h.shards.Shards(3):
            cooccurrence_sector = cooccurrence.get_sector(sector)
            self.assertTrue(isinstance(
                cooccurrence_sector, h.cooccurrence.CooccurrenceSector))
            self.assertTrue(np.allclose(
                cooccurrence_sector.Nxx.toarray(),
                cooccurrence.Nxx[sector].toarray()
            ))


