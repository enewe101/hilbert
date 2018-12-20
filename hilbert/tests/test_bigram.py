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


class TestBigram(TestCase):


    def test_unpacking(self):
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        bigram = h.corpus_stats.get_test_bigram(2)

        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(torch.allclose(Nxx, torch.tensor(
            bigram.Nxx.toarray(), device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(Nx, torch.tensor(
            bigram.Nx, device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(Nxt, torch.tensor(
            bigram.Nxt, device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(N, torch.tensor(
            bigram.N, device=device, dtype=dtype)))


    def test_load_shard(self):
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        bigram = h.corpus_stats.get_test_bigram(2)

        shards = h.shards.Shards(2)
        Nxx, Nx, Nxt, N = bigram.load_shard(shards[1])
        self.assertTrue(torch.allclose(Nxx, torch.tensor(
            bigram.Nxx.toarray()[shards[1]], device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(Nx, torch.tensor(
            bigram.Nx[shards[1][0]], device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(Nxt, torch.tensor(
            bigram.Nxt[:,shards[1][1]], device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(N, torch.tensor(
            bigram.N, device=device, dtype=dtype)))


    def get_test_cooccurrence_stats(self):
        #COUNTS = {
        #    (0,1):3, (1,0):3,
        #    (0,3):1, (3,0):1,
        #    (2,1):1, (1,2):1,
        #    (0,2):1, (2,0):1
        #}
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    def test_apply_w2v_undersampling(self):

        t = 1e-5
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.Bigram(unigram, array)

        # Initially the counts reflect the provided cooccurrence matrix
        self.assertTrue(np.allclose(np.asarray(bigram.Nxx.todense()), array))
        Nxx, Nx, Nxt, N = bigram
        uNx, uNxt, uN = unigram
        self.assertTrue(np.allclose(Nxx, array))

        # Now apply undersampling
        freq_x = uNx / uN
        freq_xt = uNxt / uN
        expected_Nxx = torch.zeros_like(Nxx)
        for i in range(Nxx.shape[0]):
            for j in range(Nxx.shape[1]):
                freq_i = torch.clamp(
                    t/freq_x[i,0] + torch.sqrt(t/freq_x[i,0]), 0, 1)
                freq_j = torch.clamp(
                    t/freq_xt[0,j] + torch.sqrt(t/freq_xt[0,j]), 0, 1)
                expected_Nxx[i,j] = Nxx[i,j]  * freq_i * freq_j

        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = torch.sum(expected_Nxx, dim=0, keepdim=True)
        expected_N = torch.sum(expected_Nxx)

        bigram.apply_w2v_undersampling(t)
        found_Nxx, found_Nx, found_Nxt, found_N = bigram

        self.assertTrue(np.allclose(found_Nxx, expected_Nxx))
        self.assertTrue(np.allclose(found_Nx, expected_Nx))
        self.assertTrue(np.allclose(found_Nxt, expected_Nxt))
        self.assertTrue(np.allclose(found_N, expected_N))





    def test_invalid_arguments(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Can make an empty Bigram instance, if provided a unigram instance
        h.bigram.Bigram(unigram)
        with self.assertRaises(Exception):
            h.bigram.Bigram(None)

        # Can make a non-empty Bigram when provided a proper unigram instance.
        h.bigram.Bigram(unigram, array)

        # Cannot make a non-empty Bigram when not provided a unigram instance.
        h.unigram.Unigram(dictionary, array.sum(axis=1))
        with self.assertRaises(Exception):
            h.bigram.Bigram(None, Nxx=array)

        # Cannot make a non-empty Bigram if unigram instance does not have
        # the same vocabulary size
        small_unigram = h.unigram.Unigram(dictionary, array[:3,:3].sum(axis=1))
        with self.assertRaises(ValueError):
            bigram = h.bigram.Bigram(small_unigram, array)


    def test_count(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.Bigram(unigram, Nxx=array, verbose=False)
        self.assertTrue(bigram.count('banana', 'socks'), 3)
        self.assertTrue(bigram.count('socks', 'car'), 1)


    def test_add(self):

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        Nxx = array
        Nx = np.sum(Nxx, axis=1).reshape(-1,1)
        Nxt = np.sum(Nxx, axis=0).reshape(1,-1)
        N = np.sum(Nxx)

        # Create a bigram instance using counts
        bigram = h.bigram.Bigram(unigram, Nxx=Nxx, verbose=False)

        # We can add tokens if they are in the unigram vocabulary
        bigram.add('banana', 'socks')
        expected_Nxx = Nxx.copy()
        expected_Nxx[0,1] += 1
        expected_Nx = expected_Nxx.sum(axis=1, keepdims=True)
        expected_Nxt = expected_Nxx.sum(axis=0, keepdims=True)
        expected_N = expected_Nxx.sum()

        self.assertTrue(np.allclose(bigram.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(bigram.Nx, expected_Nx))
        self.assertTrue(np.allclose(bigram.Nxt, expected_Nxt))
        self.assertEqual(bigram.N, expected_N)

        # We cannot add tokens if they are outside of the unigram vocabulary
        with self.assertRaises(ValueError):
            bigram.add('archaeopteryx', 'socks')

        # If skip_unk is True, then don't raise an error when attempting to
        # add tokens outside vocabulary, just skip
        bigram.add('archaeopteryx', 'socks', skip_unk=True)


    # Sorting should be based on unigram frequencies.
    def test_sort(self):

        unsorted_dictionary = h.dictionary.Dictionary([
            'car', 'banana', 'socks',  'field'
        ])
        unsorted_Nx = [2, 3, 2, 1]
        unsorted_Nxx = np.array([
            [2,2,2,2],
            [2,0,2,1],
            [2,2,0,0],
            [2,1,0,0],
        ])

        sorted_dictionary = h.dictionary.Dictionary([
            'banana', 'car', 'socks', 'field'])
        sorted_Nx = [3,2,2,1]
        sorted_Nxx = np.array([
            [0,2,2,1],
            [2,2,2,2],
            [2,2,0,0],
            [1,2,0,0]
        ])

        unsorted_unigram = h.unigram.Unigram(unsorted_dictionary, unsorted_Nx)
        bigram = h.bigram.Bigram(unsorted_unigram, unsorted_Nxx, verbose=False)

        # Bigram is unsorted
        self.assertFalse(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
        self.assertTrue(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))
        self.assertFalse(bigram.sorted)

        # Unigram is unsorted
        self.assertEqual(bigram.unigram.Nx, unsorted_Nx)
        self.assertFalse(bigram.unigram.sorted)

        # Sorting bigram works.
        bigram.sort()
        self.assertTrue(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
        self.assertFalse(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))
        self.assertTrue(bigram.sorted)

        # The unigram is also sorted
        self.assertTrue(np.allclose(bigram.unigram.Nx, sorted_Nx))
        self.assertFalse(np.allclose(bigram.unigram.Nx, unsorted_Nx))
        self.assertEqual(bigram.dictionary.tokens, sorted_dictionary.tokens)
        self.assertTrue(bigram.unigram.sorted)


    def test_save_load(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-bigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Create a bigram instance.
        bigram = h.bigram.Bigram(unigram, array, verbose=False)
        Nxx, Nx, Nxt, N = bigram

        # Save it, then load it
        bigram.save(write_path)
        bigram2 = h.bigram.Bigram.load(write_path, verbose=False)

        Nxx2, Nx2, Nxt2, N2 = bigram2

        self.assertEqual(
            bigram2.dictionary.tokens,
            bigram.dictionary.tokens
        )
        self.assertTrue(np.allclose(Nxx2, Nxx))
        self.assertTrue(np.allclose(Nx2, Nx))
        self.assertTrue(np.allclose(Nxt2, Nxt))
        self.assertTrue(np.allclose(N2, N))
        self.assertTrue(np.allclose(bigram.unigram.Nx, bigram2.unigram.Nx))
        self.assertTrue(np.allclose(bigram.unigram.N, bigram2.unigram.N))

        shutil.rmtree(write_path)


    def test_density(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.Bigram(unigram, array, verbose=False)
        self.assertEqual(bigram.density(), 0.5)
        self.assertEqual(bigram.density(2), 0.125)


    def test_truncate(self):

        unsorted_tokens = ['car', 'banana', 'socks',  'field']
        unsorted_dictionary = h.dictionary.Dictionary(unsorted_tokens)
        unsorted_Nx = [1, 3, 2, 1]
        unsorted_Nxx = np.array([
            [2,2,2,2],
            [2,0,2,1],
            [2,2,0,0],
            [2,1,0,0],
        ])

        trunc_tokens = ['banana', 'socks']
        trunc_Nx = [3, 2]
        trunc_Nxx = np.array([[0,2],[2,0]])
        trunc_Nx_bigram = [[2],[2]]
        trunc_Nxt_bigram = [[2,2]]
        trunc_N_bigram = 4

        unigram = h.unigram.Unigram(unsorted_dictionary, unsorted_Nx)
        bigram = h.bigram.Bigram(unigram, unsorted_Nxx, verbose=False)

        self.assertFalse(bigram.sorted)
        self.assertFalse(bigram.unigram.sorted)
        self.assertEqual(bigram.dictionary.tokens, unsorted_tokens)
        self.assertEqual(bigram.unigram.dictionary.tokens, unsorted_tokens)
        self.assertTrue(np.allclose(
            np.asarray(bigram.Nxx.todense()), unsorted_Nxx))
        self.assertEqual(bigram.unigram.Nx, unsorted_Nx)

        bigram.truncate(2)
        # The top two tokens by unigram frequency are 'banana' and 'socks',
        # but the top two tokens by bigram frequency are 'car', and 'banana'
        self.assertTrue(bigram.sorted)
        self.assertTrue(bigram.unigram.sorted)
        self.assertEqual(bigram.dictionary.tokens, ['banana', 'socks'])
        self.assertEqual(bigram.unigram.dictionary.tokens, ['banana', 'socks'])
        self.assertTrue(np.allclose(
            np.asarray(bigram.Nxx.todense()), trunc_Nxx))
        self.assertEqual(bigram.unigram.Nx, trunc_Nx)
        self.assertTrue(np.allclose(bigram.Nx, trunc_Nx_bigram))
        self.assertTrue(np.allclose(bigram.Nxt, trunc_Nxt_bigram))
        self.assertEqual(bigram.N, trunc_N_bigram)


    def test_deepcopy(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram1 = h.bigram.Bigram(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = bigram1

        bigram2 = deepcopy(bigram1)

        self.assertTrue(bigram2 is not bigram1)
        self.assertTrue(bigram2.dictionary is not bigram1.dictionary)
        self.assertTrue(bigram2.unigram is not bigram1.unigram)
        self.assertTrue(bigram2.Nxx is not bigram1.Nxx)
        self.assertTrue(bigram2.Nx is not bigram1.Nx)
        self.assertTrue(bigram2.Nxt is not bigram1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = bigram2
        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertTrue(np.allclose(Nxt2, Nxt1))
        self.assertEqual(N2, N1)
        self.assertEqual(bigram2.dictionary.tokens, bigram1.dictionary.tokens)
        self.assertEqual(bigram2.unigram.Nx, bigram1.unigram.Nx)
        self.assertEqual(bigram2.unigram.N, bigram1.unigram.N)
        self.assertEqual(bigram2.verbose, bigram1.verbose)
        self.assertEqual(bigram2.verbose, bigram1.verbose)


    def test_copy(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram1 = h.bigram.Bigram(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = bigram1

        bigram2 = copy(bigram1)

        self.assertTrue(bigram2 is not bigram1)
        self.assertTrue(bigram2.dictionary is not bigram1.dictionary)
        self.assertTrue(bigram2.unigram is not bigram1.unigram)
        self.assertTrue(bigram2.Nxx is not bigram1.Nxx)
        self.assertTrue(bigram2.Nx is not bigram1.Nx)
        self.assertTrue(bigram2.Nxt is not bigram1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = bigram2
        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertTrue(np.allclose(Nxt2, Nxt1))
        self.assertEqual(N2, N1)
        self.assertEqual(bigram2.dictionary.tokens, bigram1.dictionary.tokens)
        self.assertEqual(bigram2.unigram.Nx, bigram1.unigram.Nx)
        self.assertEqual(bigram2.unigram.N, bigram1.unigram.N)
        self.assertEqual(bigram2.verbose, bigram1.verbose)
        self.assertEqual(bigram2.verbose, bigram1.verbose)


    def test_plus(self):
        """
        When Bigrams add, their counts add.
        """

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE

        # Make one CoocStat instance to be added.
        dictionary, array, unigram1 = self.get_test_cooccurrence_stats()
        bigram1 = h.bigram.Bigram(unigram1, array, verbose=False)

        # Make another CoocStat instance to be added.
        token_pairs2 = [
            ('banana', 'banana'),
            ('banana','car'), ('banana','car'),
            ('banana','socks'), ('cave','car'), ('cave','socks')
        ]
        dictionary2 = h.dictionary.Dictionary([
            'banana', 'car', 'socks', 'cave'])
        counts2 = {
            (0,0):2,
            (0,1):2, (0,2):1, (3,1):1, (3,2):1,
            (1,0):2, (2,0):1, (1,3):1, (2,3):1
        }
        array2 = np.array([
            [2,2,1,0],
            [2,0,0,1],
            [1,0,0,1],
            [0,1,1,0],
        ])
        unigram2 = h.unigram.Unigram(dictionary2, array2.sum(axis=1))

        bigram2 = h.bigram.Bigram(unigram2, verbose=False)
        for tok1, tok2 in token_pairs2:
            bigram2.add(tok1, tok2)
            bigram2.add(tok2, tok1)

        bigram_sum = bigram1 + bigram2

        # Ensure that bigram1 was not changed
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        array = torch.tensor(array, device=device, dtype=dtype)
        Nxx1, Nx1, Nxt1, N1 = bigram1
        self.assertTrue(np.allclose(Nxx1, array))
        expected_Nx = torch.sum(array, dim=1).reshape(-1,1)
        expected_Nxt = torch.sum(array, dim=0).reshape(1,-1)
        self.assertTrue(np.allclose(Nx1, expected_Nx))
        self.assertTrue(np.allclose(Nxt1, expected_Nxt))
        self.assertTrue(torch.allclose(N1[0], torch.sum(array)))
        self.assertEqual(bigram1.dictionary.tokens, dictionary.tokens)
        self.assertEqual(
            bigram1.dictionary.token_ids, dictionary.token_ids)
        self.assertEqual(bigram1.verbose, False)

        # Ensure that bigram2 was not changed
        Nxx2, Nx2, Nxt2, N2 = bigram2
        array2 = torch.tensor(array2, dtype=dtype, device=device)
        self.assertTrue(np.allclose(Nxx2, array2))
        expected_Nx2 = torch.sum(array2, dim=1).reshape(-1,1)
        expected_Nxt2 = torch.sum(array2, dim=0).reshape(1,-1)
        self.assertTrue(torch.allclose(Nx2, expected_Nx2))
        self.assertTrue(torch.allclose(Nxt2, expected_Nxt2))
        self.assertEqual(N2, torch.sum(array2))
        self.assertEqual(bigram2.dictionary.tokens, dictionary2.tokens)
        self.assertEqual(
            bigram2.dictionary.token_ids, dictionary2.token_ids)
        self.assertEqual(bigram2.verbose, False)


        # Ensure that bigram_sum is as desired.  Sort to make comparison
        # easier.  Double check that sort flag is False to begin with.
        self.assertFalse(bigram_sum.sorted)
        bigram_sum.sort()
        dictionary_sum = h.dictionary.Dictionary([
            'banana', 'socks', 'car', 'cave', 'field'])
        expected_Nxx_sum = torch.tensor([
            [2, 4, 3, 0, 1],
            [4, 0, 1, 1, 0],
            [3, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ], dtype=dtype, device=device)
        expected_Nx_sum = torch.sum(expected_Nxx_sum, dim=1).reshape(-1,1)
        expected_Nxt_sum = torch.sum(expected_Nxx_sum, dim=0).reshape(1,-1)
        expected_N_sum = torch.tensor(
            bigram1.N + bigram2.N, dtype=dtype, device=device)
        Nxx_sum, Nx_sum, Nxt_sum, N_sum = bigram_sum

        self.assertEqual(dictionary_sum.tokens, bigram_sum.dictionary.tokens)
        self.assertTrue(torch.allclose(Nxx_sum, expected_Nxx_sum))
        self.assertTrue(torch.allclose(Nx_sum, expected_Nx_sum))
        self.assertTrue(torch.allclose(Nxt_sum, expected_Nxt_sum))
        self.assertEqual(N_sum, expected_N_sum)


    # should be sorted only if unigram is sorted.
    def test_load_unigram(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-bigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        unigram.save(write_path)

        bigram = h.bigram.Bigram.load_unigram(write_path)

        self.assertTrue(np.allclose(bigram.unigram.Nx, unigram.Nx))
        self.assertTrue(np.allclose(bigram.unigram.N, unigram.N))
        self.assertEqual(bigram.dictionary.tokens, unigram.dictionary.tokens)

        # Ensure that we can add any pairs of tokens found in the unigram
        # vocabulary.  As long as this runs without errors everything is fine.
        for tok1 in dictionary.tokens:
            for tok2 in dictionary.tokens:
                bigram.add(tok1, tok2)


