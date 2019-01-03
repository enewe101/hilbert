import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h
import random

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class TestBigramMutable(TestCase):

    def get_test_bigram(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramMutable(unigram, array, verbose=False)
        return bigram


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    def test_invalid_arguments(self):

        random.seed(0)
        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()

        # Make a bigram by passing in a unigram, and optionally some 
        # cooccurrence data (Nxx)
        h.bigram.BigramMutable(unigram)
        h.bigram.BigramMutable(unigram, Nxx=Nxx)

        # Unigram is required
        with self.assertRaises(TypeError):
            h.bigram.BigramMutable()
        with self.assertRaises(TypeError):
            h.bigram.BigramMutable(Nxx=Nxx)

        # BigramBases need a sorted unigram instance
        unsorted_unigram = deepcopy(unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            h.bigram.BigramMutable(unsorted_unigram, Nxx)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            h.bigram.BigramMutable(truncated_unigram, Nxx)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            h.bigram.BigramMutable(truncated_unigram, Nxx)


    def test_deepcopy(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram1 = h.bigram.BigramMutable(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = bigram1.load_shard()

        bigram2 = deepcopy(bigram1)

        self.assertTrue(bigram2 is not bigram1)
        self.assertTrue(bigram2.dictionary is not bigram1.dictionary)
        self.assertTrue(bigram2.unigram is not bigram1.unigram)
        self.assertTrue(bigram2.Nxx is not bigram1.Nxx)
        self.assertTrue(bigram2.Nx is not bigram1.Nx)
        self.assertTrue(bigram2.Nxt is not bigram1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = bigram2.load_shard()

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
        bigram1 = h.bigram.BigramMutable(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = bigram1.load_shard()

        bigram2 = copy(bigram1)

        self.assertTrue(bigram2 is not bigram1)
        self.assertTrue(bigram2.dictionary is not bigram1.dictionary)
        self.assertTrue(bigram2.unigram is not bigram1.unigram)
        self.assertTrue(bigram2.Nxx is not bigram1.Nxx)
        self.assertTrue(bigram2.Nx is not bigram1.Nx)
        self.assertTrue(bigram2.Nxt is not bigram1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = bigram2.load_shard()

        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertTrue(np.allclose(Nxt2, Nxt1))
        self.assertEqual(N2, N1)
        self.assertEqual(bigram2.dictionary.tokens, bigram1.dictionary.tokens)
        self.assertEqual(bigram2.unigram.Nx, bigram1.unigram.Nx)
        self.assertEqual(bigram2.unigram.N, bigram1.unigram.N)
        self.assertEqual(bigram2.verbose, bigram1.verbose)
        self.assertEqual(bigram2.verbose, bigram1.verbose)


    #
    #   Only the functionality of self.merge(), provided by BigramBase is needed
    #   and it is much simpler!
    #
    #def test_plus(self):
    #    """
    #    When Bigrams add, their counts add.
    #    """

    #    dtype=h.CONSTANTS.DEFAULT_DTYPE
    #    device=h.CONSTANTS.MATRIX_DEVICE

    #    # Make one CoocStat instance to be added.
    #    dictionary, array, unigram1 = self.get_test_cooccurrence_stats()
    #    bigram1 = h.bigram.BigramMutable(unigram1, array, verbose=False)

    #    # Make another CoocStat instance to be added.
    #    token_pairs2 = [
    #        ('banana', 'banana'),
    #        ('banana','car'), ('banana','car'),
    #        ('banana','socks'), ('cave','car'), ('cave','socks')
    #    ]
    #    dictionary2 = h.dictionary.Dictionary([
    #        'banana', 'car', 'socks', 'cave'])
    #    counts2 = {
    #        (0,0):2,
    #        (0,1):2, (0,2):1, (3,1):1, (3,2):1,
    #        (1,0):2, (2,0):1, (1,3):1, (2,3):1
    #    }
    #    array2 = np.array([
    #        [2,2,1,0],
    #        [2,0,0,1],
    #        [1,0,0,1],
    #        [0,1,1,0],
    #    ])
    #    unigram2 = h.unigram.Unigram(dictionary2, array2.sum(axis=1))

    #    bigram2 = h.bigram.BigramMutable(unigram2, verbose=False)
    #    for tok1, tok2 in token_pairs2:
    #        bigram2.add(tok1, tok2)
    #        bigram2.add(tok2, tok1)

    #    bigram_sum = bigram1 + bigram2

    #    # Ensure that bigram1 was not changed
    #    dictionary, array, unigram = self.get_test_cooccurrence_stats()
    #    array = torch.tensor(array, device=device, dtype=dtype)
    #    Nxx1, Nx1, Nxt1, N1 = bigram1.load_shard()
    #    self.assertTrue(np.allclose(Nxx1, array))
    #    expected_Nx = torch.sum(array, dim=1).reshape(-1,1)
    #    expected_Nxt = torch.sum(array, dim=0).reshape(1,-1)
    #    self.assertTrue(np.allclose(Nx1, expected_Nx))
    #    self.assertTrue(np.allclose(Nxt1, expected_Nxt))
    #    self.assertTrue(torch.allclose(N1, torch.sum(array)))
    #    self.assertEqual(bigram1.dictionary.tokens, dictionary.tokens)
    #    self.assertEqual(
    #        bigram1.dictionary.token_ids, dictionary.token_ids)
    #    self.assertEqual(bigram1.verbose, False)

    #    # Ensure that bigram2 was not changed
    #    Nxx2, Nx2, Nxt2, N2 = bigram2.load_shard()
    #    array2 = torch.tensor(array2, dtype=dtype, device=device)
    #    self.assertTrue(np.allclose(Nxx2, array2))
    #    expected_Nx2 = torch.sum(array2, dim=1).reshape(-1,1)
    #    expected_Nxt2 = torch.sum(array2, dim=0).reshape(1,-1)
    #    self.assertTrue(torch.allclose(Nx2, expected_Nx2))
    #    self.assertTrue(torch.allclose(Nxt2, expected_Nxt2))
    #    self.assertEqual(N2, torch.sum(array2))
    #    self.assertEqual(bigram2.dictionary.tokens, dictionary2.tokens)
    #    self.assertEqual(
    #        bigram2.dictionary.token_ids, dictionary2.token_ids)
    #    self.assertEqual(bigram2.verbose, False)


    #    # Ensure that bigram_sum is as desired.  Sort to make comparison
    #    # easier.  Double check that sort flag is False to begin with.
    #    self.assertFalse(bigram_sum.sorted)
    #    bigram_sum.sort()
    #    dictionary_sum = h.dictionary.Dictionary([
    #        'banana', 'socks', 'car', 'cave', 'field'])
    #    expected_Nxx_sum = torch.tensor([
    #        [2, 4, 3, 0, 1],
    #        [4, 0, 1, 1, 0],
    #        [3, 1, 0, 1, 0],
    #        [0, 1, 1, 0, 0],
    #        [1, 0, 0, 0, 0],
    #    ], dtype=dtype, device=device)
    #    expected_Nx_sum = torch.sum(expected_Nxx_sum, dim=1).reshape(-1,1)
    #    expected_Nxt_sum = torch.sum(expected_Nxx_sum, dim=0).reshape(1,-1)
    #    expected_N_sum = torch.tensor(
    #        bigram1.N + bigram2.N, dtype=dtype, device=device)
    #    Nxx_sum, Nx_sum, Nxt_sum, N_sum = bigram_sum.load_shard()

    #    self.assertEqual(dictionary_sum.tokens, bigram_sum.dictionary.tokens)
    #    self.assertTrue(torch.allclose(Nxx_sum, expected_Nxx_sum))
    #    self.assertTrue(torch.allclose(Nx_sum, expected_Nx_sum))
    #    self.assertTrue(torch.allclose(Nxt_sum, expected_Nxt_sum))
    #    self.assertEqual(N_sum, expected_N_sum)


    def test_add(self):

        # Create a `BigramMutable` instance using counts
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram_mutable = h.bigram.BigramMutable(
            unigram, Nxx=array, verbose=False)
        Nxx = torch.tensor(array, dtype=h.CONSTANTS.DEFAULT_DTYPE)

        # We can add tokens if they are in the unigram vocabulary
        add_count = 3
        bigram_mutable.add('banana', 'socks', add_count)
        expected_Nxx = Nxx.clone()
        expected_Nxx[0,1] += add_count
        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = torch.sum(expected_Nxx, dim=0, keepdim=True)
        expected_N = torch.sum(expected_Nxx)

        # Check that adding occurred
        self.assertTrue(np.allclose(bigram_mutable.Nxx.toarray(), expected_Nxx))
        self.assertTrue(torch.allclose(bigram_mutable.Nx, expected_Nx))
        self.assertTrue(torch.allclose(bigram_mutable.Nxt, expected_Nxt))
        self.assertEqual(bigram_mutable.N, expected_N)

        # We cannot add tokens if they are outside of the unigram vocabulary
        with self.assertRaises(ValueError):
            bigram_mutable.add('archaeopteryx', 'socks')

        # If skip_unk is True, then don't raise an error when attempting to
        # add tokens outside vocabulary, just skip
        bigram_mutable.add('archaeopteryx', 'socks', skip_unk=True)


    #
    #   BigramMutable is guaranteed to be sorted because they are made from
    #   sorted unigrams.
    #
    #def test_sort(self):

    #    unsorted_dictionary = h.dictionary.Dictionary([
    #        'car', 'banana', 'socks',  'field'
    #    ])
    #    unsorted_Nx = [2, 3, 2, 1]
    #    unsorted_Nxx = np.array([
    #        [2,2,2,2],
    #        [2,0,2,1],
    #        [2,2,0,0],
    #        [2,1,0,0],
    #    ])

    #    sorted_dictionary = h.dictionary.Dictionary([
    #        'banana', 'car', 'socks', 'field'])
    #    sorted_Nx = [3,2,2,1]
    #    sorted_Nxx = np.array([
    #        [0,2,2,1],
    #        [2,2,2,2],
    #        [2,2,0,0],
    #        [1,2,0,0]
    #    ])

    #    unsorted_unigram = h.unigram.Unigram(unsorted_dictionary, unsorted_Nx)
    #    bigram = h.bigram.BigramMutable(unsorted_unigram, unsorted_Nxx, verbose=False)

    #    # Bigram is unsorted
    #    self.assertFalse(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
    #    self.assertTrue(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))
    #    self.assertFalse(bigram.sorted)

    #    # Unigram is unsorted
    #    self.assertEqual(bigram.unigram.Nx, unsorted_Nx)
    #    self.assertFalse(bigram.unigram.sorted)

    #    # Sorting bigram works.
    #    bigram.sort()
    #    self.assertTrue(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
    #    self.assertFalse(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))
    #    self.assertTrue(bigram.sorted)

    #    # The unigram is also sorted
    #    self.assertTrue(np.allclose(bigram.unigram.Nx, sorted_Nx))
    #    self.assertFalse(np.allclose(bigram.unigram.Nx, unsorted_Nx))
    #    self.assertEqual(bigram.dictionary.tokens, sorted_dictionary.tokens)
    #    self.assertTrue(bigram.unigram.sorted)

    def test_truncate(self):

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramMutable(unigram, array, verbose=False)

        trunc_tokens = ['banana', 'socks']
        trunc_uNx = [5, 4]
        trunc_Nxx = np.array([[0,3],[3,0]])
        trunc_Nx = [[3],[3]]
        trunc_Nxt = [[3,3]]
        trunc_N = 6

        bigram.truncate(2)
        # The top two tokens by unigram frequency are 'banana' and 'socks',
        # but the top two tokens by bigram frequency are 'car', and 'banana'
        self.assertTrue(bigram.sorted)
        self.assertTrue(bigram.unigram.sorted)
        self.assertEqual(bigram.dictionary.tokens, ['banana', 'socks'])
        self.assertEqual(bigram.unigram.dictionary.tokens, ['banana', 'socks'])

        self.assertTrue(np.allclose(
            np.asarray(bigram.Nxx.todense()), trunc_Nxx))
        self.assertEqual(bigram.unigram.Nx, trunc_uNx)
        self.assertTrue(np.allclose(bigram.Nx, trunc_Nx))
        self.assertTrue(np.allclose(bigram.Nxt, trunc_Nxt))
        self.assertEqual(bigram.N, trunc_N)


    def test_save_load(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-bigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Create a bigram instance.
        bigram = h.bigram.BigramMutable(unigram, array, verbose=False)
        Nxx, Nx, Nxt, N = bigram.load_shard()

        # Save it, then load it
        bigram.save(write_path)
        bigram2 = h.bigram.BigramMutable.load(write_path, verbose=False)

        Nxx2, Nx2, Nxt2, N2 = bigram2.load_shard()

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


    def test_save_load_sector(self):
        """
        We should be able to save just a sector of the bigram, which is
        logically like a shard, except that shards are smaller in practice.
        On disk, the data is separated into sectors, and then, to load onto 
        GPU, it is further separated into shards.  A key thing to ensure is
        that the indexing of Unigram and Dictionary are concordant with the
        indexing of the loaded sector.
        """

        shard_factor = 3
        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-bigram-sector')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Create a bigram instance.
        bigram = h.bigram.BigramMutable(unigram, array, verbose=False)
        Nxx, Nx, Nxt, N = bigram.load_shard()

        # Save all the sectors, but don't save the unigram (and dictionary)
        sectors = h.shards.Shards(shard_factor)
        for sector in sectors:
            bigram.save_sector(
                write_path, sector, save_marginal=False, save_unigram=False)

        # Check that the sectors were in fact saved, and that the unigram and
        # dictionary were not.
        paths = set(os.listdir(write_path))
        expected_paths = {
            'Nxx-{}-{}-{}.npz'.format(i,j,shard_factor)
            for i in range(shard_factor) for j in range(shard_factor)
        }
        self.assertEqual(paths, expected_paths)

        # Finally, re-save the 0th shard, this time with the dictionary and 
        # unigram being saved
        bigram.save_sector(
            write_path, sector=sectors[0], save_marginal=True, 
            save_unigram=True
        )
        paths = set(os.listdir(write_path))
        expected_paths.update(['dictionary', 'Nx.txt', 'Nx.npy', 'Nxt.npy'])
        self.assertEqual(paths, expected_paths)

        # Now go sector by sector, and load the sector, and check that
        # everything lines up.
        for sector in sectors:

            bigram_sector = h.bigram.BigramSector.load(
                write_path, sector=sector, verbose=False)

            row_dict = bigram_sector.row_dictionary
            col_dict = bigram_sector.column_dictionary
            for row_token in row_dict.tokens:
                for col_token in col_dict.tokens:
                    r_id_sector = row_dict.get_id(row_token)
                    c_id_sector = col_dict.get_id(col_token)
                    r_id = bigram.dictionary.get_id(row_token)
                    c_id = bigram.dictionary.get_id(col_token)

                    self.assertEqual(
                        bigram_sector.Nxx[r_id_sector, c_id_sector],
                        bigram.Nxx[r_id, c_id]
                    )
                    self.assertEqual(
                        bigram_sector.Nx[r_id_sector,0], bigram.Nx[r_id,0])
                    self.assertEqual(
                        bigram_sector.Nxt[0,c_id_sector], bigram.Nxt[0,c_id])
                    self.assertEqual(
                        bigram_sector.uNx[r_id_sector,0], bigram.uNx[r_id,0])
                    self.assertEqual(
                        bigram_sector.uNxt[0,c_id_sector], bigram.uNxt[0,c_id])
                    self.assertEqual(bigram_sector.uN, bigram.uN)

                    self.assertTrue(np.allclose(
                        bigram_sector.Nxx.toarray(),
                        bigram.Nxx.toarray()[sector]
                    ))
                    self.assertTrue(np.allclose(
                        bigram_sector.Nx, bigram.Nx[sector[0]]))
                    self.assertTrue(np.allclose(
                        bigram_sector.Nxt, bigram.Nxt[:,sector[1]]))
                    self.assertTrue(np.allclose(
                        bigram_sector.uNx, bigram.uNx[sector[0]]))
                    self.assertTrue(np.allclose(
                        bigram_sector.uNxt, bigram.uNxt[:,sector[1]]))

        shutil.rmtree(write_path)


    # should be sorted only if unigram is sorted.
    def test_load_unigram(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-bigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        unigram.save(write_path)

        bigram = h.bigram.BigramMutable.load_unigram(write_path)

        self.assertTrue(np.allclose(bigram.unigram.Nx, unigram.Nx))
        self.assertTrue(np.allclose(bigram.unigram.N, unigram.N))
        self.assertEqual(bigram.dictionary.tokens, unigram.dictionary.tokens)

        # Ensure that we can add any pairs of tokens found in the unigram
        # vocabulary.  As long as this runs without errors everything is fine.
        for tok1 in dictionary.tokens:
            for tok2 in dictionary.tokens:
                bigram.add(tok1, tok2)


