import sys
import os
import shutil
from unittest import main, TestCase
from copy import copy, deepcopy
from collections import Counter
import hilbert as h

try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    sparse = None
    torch = None

class TestCoocStats(TestCase):


    def test_cooc_stats_unpacking(self):
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        cooc_stats = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = cooc_stats
        self.assertTrue(torch.allclose(Nxx, torch.tensor(
            cooc_stats.Nxx.toarray(), device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(Nx, torch.tensor(
            cooc_stats.Nx, device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(N, torch.tensor(
            cooc_stats.N, device=device, dtype=dtype)))



    def get_test_cooccurrence_stats(self):
        DICTIONARY = h.dictionary.Dictionary([
            'banana', 'socks', 'car', 'field'])
        COUNTS = {
            (0,1):3, (1,0):3,
            (0,3):1, (3,0):1,
            (2,1):1, (1,2):1,
            (0,2):1, (2,0):1
        }
        DIJ = ([3,1,1,1,3,1,1,1], ([0,0,2,0,1,3,1,2], [1,3,1,2,0,0,2,0]))
        ARRAY = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        return DICTIONARY, COUNTS, DIJ, ARRAY


    def test_invalid_arguments(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Can make an empty CoocStats instance.
        h.cooc_stats.CoocStats()

        # Can make a non-empty CoocStats instance using counts and
        # a matching dictionary.
        h.cooc_stats.CoocStats(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CoocStats
        # instance when using counts.
        with self.assertRaises(ValueError):
            h.cooc_stats.CoocStats(
                counts=counts)

        # Can make a non-empty CoocStats instance using Nxx and
        # a matching dictionary.
        Nxx = sparse.coo_matrix(dij).tocsr()
        h.cooc_stats.CoocStats(dictionary, counts)

        # Must supply a dictionary to make a  non-empty CoocStats
        # instance when using Nxx.
        with self.assertRaises(ValueError):
            h.cooc_stats.CoocStats(Nxx=Nxx)

        # Cannot provide both an Nxx and counts
        with self.assertRaises(ValueError):
            h.cooc_stats.CoocStats(
                dictionary, counts, Nxx=Nxx)


    def test_add_when_basis_is_counts(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        cooccurrence.add('banana', 'rice')
        self.assertEqual(cooccurrence.dictionary.get_id('rice'), 4)
        expected_counts = Counter(counts)
        expected_counts[0,4] += 1
        self.assertEqual(cooccurrence.counts, expected_counts)


    def test_add_when_basis_is_Nxx(self):

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        Nxx = array
        Nx = np.sum(Nxx, axis=1).reshape(-1,1)
        Nxt = np.sum(Nxx, axis=0).reshape(1,-1)
        N = np.sum(Nxx)

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, Nxx=Nxx, verbose=False)

        # Currently the cooccurrence instance has no internal counter for
        # cooccurrences, because it is based on the cooccurrence_array
        self.assertTrue(cooccurrence._counts is None)
        self.assertTrue(np.allclose(cooccurrence._Nxx.toarray(), Nxx))
        self.assertTrue(np.allclose(cooccurrence._Nx, Nx))
        self.assertTrue(np.allclose(cooccurrence._Nxt, Nxt))
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, Nxt))
        self.assertTrue(np.allclose(cooccurrence.N, N))

        # Adding more cooccurrence statistics will force it to "decompile" into
        # a counter, then add to the counter.  This will cause the stale Nxx
        # arrays to be dropped.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        expected_counts = Counter(counts)
        expected_counts[4,0] += 1
        expected_counts[0,4] += 1
        self.assertEqual(cooccurrence._counts, expected_counts)
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nxt, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._N, None)

        # Asking for Nxx forces it to sync itself.  
        # Ensure it it obtains the correct cooccurrence matrix
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        expected_Nx = np.sum(expected_Nxx, axis=1).reshape(-1,1)
        expected_Nxt = np.sum(expected_Nxx, axis=0).reshape(1,-1)
        expected_N = np.sum(expected_Nx)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, expected_Nxt))
        self.assertEqual(cooccurrence.N, expected_N)


    def test_uncompile(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        Nxx = sparse.coo_matrix(dij)
        Nx = np.array(np.sum(Nxx, axis=1)).reshape(-1)

        # Create a cooccurrence instance using Nxx
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, Nxx=Nxx, verbose=False)
        self.assertEqual(cooccurrence._counts, None)

        cooccurrence.decompile()
        self.assertEqual(cooccurrence._counts, counts)


    def test_compile(self):

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)

        # The cooccurrence instance has no Nxx array, but it will be calculated
        # when we try to access it directly.
        expected_Nx = np.sum(array, axis=1).reshape(-1,1)
        expected_Nxt = np.sum(array, axis=0).reshape(1,-1)
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._Nxt, None)
        self.assertEqual(cooccurrence._N, None)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), array))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, expected_Nxt))
        self.assertEqual(cooccurrence.N, np.sum(array))

        # We can still add more counts.  This causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._Nxt, None)
        self.assertEqual(cooccurrence._N, None)

        # Asking for an array forces it to sync itself.
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        expected_Nx = np.sum(expected_Nxx, axis=1).reshape(-1,1)
        expected_Nxt = np.sum(expected_Nxx, axis=0).reshape(1,-1)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, expected_Nxt))
        self.assertEqual(cooccurrence.N, np.sum(expected_Nxx))

        # Adding more counts once again causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'field')
        cooccurrence.add('field', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._Nxt, None)
        self.assertEqual(cooccurrence._N, None)

        # Asking for an array forces it to sync itself.  This time start with
        # Nx.
        expected_Nxx[0,3] += 1
        expected_Nxx[3,0] += 1
        expected_N = np.sum(expected_Nxx)
        expected_Nx = np.sum(expected_Nxx, axis=1).reshape(-1,1)
        expected_Nxt = np.sum(expected_Nxx, axis=0).reshape(1,-1)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, expected_Nxt))
        self.assertEqual(cooccurrence.N, expected_N)



    def test_sort(self):
        unsorted_dictionary = h.dictionary.Dictionary([
            'field', 'car', 'socks', 'banana'
        ])
        unsorted_counts = {
            (0,3): 1, (3,0): 1,
            (1,2): 1, (2,1): 1,
            (1,3): 1, (3,1): 1,
            (2,3): 3, (3,2): 3
        }
        unsorted_Nxx = np.array([
            [0,0,0,1],
            [0,0,1,1],
            [0,1,0,3],
            [1,1,3,0],
        ])
        sorted_dictionary = h.dictionary.Dictionary([
            'banana', 'socks', 'car', 'field'])
        sorted_counts = {
            (0,1):3, (1,0):3,
            (0,3):1, (3,0):1,
            (2,1):1, (1,2):1,
            (0,2):1, (2,0):1
        }
        sorted_array = np.array([
            [0,3,1,1],
            [3,0,1,0],
            [1,1,0,0],
            [1,0,0,0]
        ])
        cooccurrence = h.cooc_stats.CoocStats(
            unsorted_dictionary, unsorted_counts, verbose=False
        )
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), sorted_array))
        self.assertEqual(cooccurrence.counts, sorted_counts)
        self.assertEqual(
            cooccurrence.dictionary.tokens, sorted_dictionary.tokens)


    def test_save_load(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-cooccurrences')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        Nxx, Nx, Nxt, N = cooccurrence

        # Save it, then load it
        cooccurrence.save(write_path)
        cooccurrence2 = h.cooc_stats.CoocStats.load(
            write_path, verbose=False)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2
        self.assertEqual(
            cooccurrence2.dictionary.tokens, 
            cooccurrence.dictionary.tokens
        )

        self.assertEqual(cooccurrence2.counts, cooccurrence.counts)
        self.assertTrue(np.allclose(Nxx2, Nxx))
        self.assertTrue(np.allclose(Nx2, Nx))
        self.assertTrue(np.allclose(Nxt2, Nxt))
        self.assertTrue(np.allclose(N2, N))

        shutil.rmtree(write_path)


    def test_density(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        self.assertEqual(cooccurrence.density(), 0.5)
        self.assertEqual(cooccurrence.density(2), 0.125)


    def test_truncate(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        cooccurrence.truncate(3)
        trunc_Nxx = np.array([
            [0,3,1],
            [3,0,1],
            [1,1,0],
        ])
        trunc_Nx = np.sum(trunc_Nxx, axis=1, keepdims=True)
        trunc_Nxt = np.sum(trunc_Nxx, axis=0, keepdims=True)
        trunc_N = np.sum(trunc_Nx)

        Nxx, Nx, Nxt, N = cooccurrence
        self.assertTrue(np.allclose(Nxx, trunc_Nxx))
        self.assertTrue(np.allclose(Nx, trunc_Nx))
        self.assertTrue(np.allclose(Nxt, trunc_Nxt))
        self.assertTrue(np.allclose(N, trunc_N))


    def test_dict_to_sparse(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        csr_matrix = h.cooc_stats.dict_to_sparse(counts)
        self.assertTrue(isinstance(csr_matrix, sparse.csr_matrix))
        self.assertTrue(np.allclose(csr_matrix.todense(), array))


    def test_deepcopy(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = cooccurrence1

        cooccurrence2 = deepcopy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(
            cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.counts is not cooccurrence1.counts)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2
        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertEqual(N2, N1)
        self.assertEqual(cooccurrence2.counts, cooccurrence1.counts)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    def test_copy(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = cooccurrence1

        cooccurrence2 = copy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(
            cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.counts is not cooccurrence1.counts)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2
        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertEqual(N2, N1)
        self.assertEqual(cooccurrence2.counts, cooccurrence1.counts)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    def test_add(self):
        """
        When CoocStats add, their counts add.
        """

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE

        # Make one CoocStat instance to be added.
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)

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

        cooccurrence2 = h.cooc_stats.CoocStats(verbose=False)
        for tok1, tok2 in token_pairs2:
            cooccurrence2.add(tok1, tok2)
            cooccurrence2.add(tok2, tok1)

        cooccurrence_sum = cooccurrence2 + cooccurrence1

        # Ensure that cooccurrence1 was not changed
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        array = torch.tensor(array, device=device, dtype=dtype)
        self.assertEqual(cooccurrence1.counts, counts)
        Nxx1, Nx1, Nxt1, N1 = cooccurrence1
        self.assertTrue(np.allclose(Nxx1, array))
        expected_Nx = torch.sum(array, dim=1).reshape(-1,1)
        expected_Nxt = torch.sum(array, dim=0).reshape(1,-1)
        self.assertTrue(np.allclose(Nx1, expected_Nx))
        self.assertTrue(np.allclose(Nxt1, expected_Nxt))
        self.assertTrue(torch.allclose(N1[0], torch.sum(array)))
        self.assertEqual(cooccurrence1.dictionary.tokens, dictionary.tokens)
        self.assertEqual(
            cooccurrence1.dictionary.token_ids, dictionary.token_ids)
        self.assertEqual(cooccurrence1.verbose, False)

        # Ensure that cooccurrence2 was not changed
        self.assertEqual(cooccurrence2.counts, counts2)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2
        array2 = torch.tensor(array2, dtype=dtype, device=device)
        self.assertTrue(np.allclose(Nxx2, array2))
        expected_Nx2 = torch.sum(array2, dim=1).reshape(-1,1)
        expected_Nxt2 = torch.sum(array2, dim=0).reshape(1,-1)
        self.assertTrue(torch.allclose(Nx2, expected_Nx2))
        self.assertTrue(torch.allclose(Nxt2, expected_Nxt2))
        self.assertEqual(N2[0], torch.sum(array2))
        self.assertEqual(cooccurrence2.dictionary.tokens, dictionary2.tokens)
        self.assertEqual(
            cooccurrence2.dictionary.token_ids, dictionary2.token_ids)
        self.assertEqual(cooccurrence2.verbose, False)
        

        # Ensure that cooccurrence_sum is as desired
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
        counts_sum = Counter({
            (0, 0): 2, 
            (0, 1): 4, (1, 0): 4, (2, 0): 3, (0, 2): 3, (1, 2): 1, (3, 2): 1,
            (3, 1): 1, (2, 1): 1, (1, 3): 1, (2, 3): 1, (0, 4): 1, (4, 0): 1
        })
        expected_N_sum = torch.tensor(
            cooccurrence1.N + cooccurrence2.N, dtype=dtype, device=device)
        Nxx_sum, Nx_sum, Nxt_sum, N_sum = cooccurrence_sum
        self.assertTrue(torch.allclose(Nxx_sum, expected_Nxx_sum))
        self.assertTrue(torch.allclose(Nx_sum, expected_Nx_sum))
        self.assertTrue(torch.allclose(Nxt_sum, expected_Nxt_sum))
        self.assertEqual(
            N_sum, expected_N_sum)
        self.assertEqual(cooccurrence_sum.counts, counts_sum)



if __name__ == '__main__':

    if '--cpu' in sys.argv:
        print('\nTESTING DEVICE: CPU\n')
        sys.argv.remove('--cpu')
        h.CONSTANTS.MATRIX_DEVICE = 'cpu'
    else:
        print('\nTESTING DEVICE: CUDA.  Use --cpu to test on cpu.\n')

    main()

