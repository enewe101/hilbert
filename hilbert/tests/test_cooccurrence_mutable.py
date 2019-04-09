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


class TestSectorize(TestCase):

    def test_sectorize(self):
        sector_factor = 3
        path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
        out_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-sectorize')
        sector_fnames = {
            'Nxx-{}-{}-{}.npz'.format(i,j,sector_factor)
            for i in range(sector_factor) for j in range(sector_factor)
        }

        # Ensure that there are not sector files already at path.
        for sector_fname in sector_fnames:
            sector_path = os.path.join(path, sector_fname)
            if os.path.exists(sector_path):
                os.remove(sector_path)

        h.cooccurrence.sectorize(path, sector_factor, verbose=False)

        # Check that the sectors were in fact saved
        found_paths = set(os.listdir(path))
        extra_paths = {'dictionary', 'Nxx.npz', 'Nx.txt', 'Nx.npy', 'Nxt.npy'}
        self.assertEqual(found_paths, sector_fnames | extra_paths)

        # Clean up.
        # Ensure that there are not sector files already at path.
        for sector_fname in sector_fnames:
            os.remove(os.path.join(path, sector_fname))

        # Ensure out_path doesn't exist
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        h.cooccurrence.sectorize(path, sector_factor, out_path, verbose=False)

        found_paths = set(os.listdir(out_path))
        extra_paths -= {'Nxx.npz'}
        self.assertEqual(found_paths, sector_fnames | extra_paths)

        # Cleanup
        shutil.rmtree(out_path)




class TestCooccurrenceMutable(TestCase):

    def get_test_cooccurrence(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooccurrence.CooccurrenceMutable(
            unigram, array, verbose=False)
        return cooccurrence


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary(['banana','socks','car','field'])
        array = np.array([[0,3,1,1],[3,0,1,0],[1,1,0,0],[1,0,0,0]])
        unigram = h.unigram.Unigram(dictionary, array.sum(axis=1))
        return dictionary, array, unigram


    def test_invalid_arguments(self):

        random.seed(0)
        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()

        # Make a cooccurrence by passing in a unigram, and optionally some 
        # cooccurrence data (Nxx)
        h.cooccurrence.CooccurrenceMutable(unigram)
        h.cooccurrence.CooccurrenceMutable(unigram, Nxx=Nxx)

        # Unigram is required
        with self.assertRaises(TypeError):
            h.cooccurrence.CooccurrenceMutable()
        with self.assertRaises(TypeError):
            h.cooccurrence.CooccurrenceMutable(Nxx=Nxx)

        # Cooccurrence instances need a sorted unigram instance
        unsorted_unigram = deepcopy(unigram)
        random.shuffle(unsorted_unigram.Nx)
        self.assertFalse(unsorted_unigram.check_sorted())
        with self.assertRaises(ValueError):
            h.cooccurrence.CooccurrenceMutable(unsorted_unigram, Nxx)

        # Truncated unigram leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.Nx = truncated_unigram.Nx[:-1]
        with self.assertRaises(ValueError):
            h.cooccurrence.CooccurrenceMutable(truncated_unigram, Nxx)

        # Truncated unigram dictionary leads to ValueError
        truncated_unigram = deepcopy(unigram)
        truncated_unigram.dictionary = h.dictionary.Dictionary(
            unigram.dictionary.tokens[:-1])
        with self.assertRaises(ValueError):
            h.cooccurrence.CooccurrenceMutable(truncated_unigram, Nxx)


    def test_deepcopy(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooccurrence.CooccurrenceMutable(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = cooccurrence1.load_shard()

        cooccurrence2 = deepcopy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.unigram is not cooccurrence1.unigram)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)
        self.assertTrue(cooccurrence2.Nxt is not cooccurrence1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2.load_shard()

        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertTrue(np.allclose(Nxt2, Nxt1))
        self.assertEqual(N2, N1)
        self.assertEqual(cooccurrence2.dictionary.tokens, cooccurrence1.dictionary.tokens)
        self.assertEqual(cooccurrence2.unigram.Nx, cooccurrence1.unigram.Nx)
        self.assertEqual(cooccurrence2.unigram.N, cooccurrence1.unigram.N)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    def test_copy(self):
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooccurrence.CooccurrenceMutable(unigram, array, verbose=False)
        Nxx1, Nx1, Nxt1, N1 = cooccurrence1.load_shard()

        cooccurrence2 = copy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.unigram is not cooccurrence1.unigram)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)
        self.assertTrue(cooccurrence2.Nxt is not cooccurrence1.Nxt)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2.load_shard()

        self.assertTrue(np.allclose(Nxx2, Nxx1))
        self.assertTrue(np.allclose(Nx2, Nx1))
        self.assertTrue(np.allclose(Nxt2, Nxt1))
        self.assertEqual(N2, N1)
        self.assertEqual(cooccurrence2.dictionary.tokens, cooccurrence1.dictionary.tokens)
        self.assertEqual(cooccurrence2.unigram.Nx, cooccurrence1.unigram.Nx)
        self.assertEqual(cooccurrence2.unigram.N, cooccurrence1.unigram.N)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    #
    #   Only the functionality of self.merge(), provided by Cooccurrence is needed
    #   and it is much simpler!
    #
    #def test_plus(self):
    #    """
    #    When Cooccurrences add, their counts add.
    #    """

    #    dtype=h.CONSTANTS.DEFAULT_DTYPE
    #    device=h.CONSTANTS.MATRIX_DEVICE

    #    # Make one CoocStat instance to be added.
    #    dictionary, array, unigram1 = self.get_test_cooccurrence_stats()
    #    cooccurrence1 = h.cooccurrence.CooccurrenceMutable(unigram1, array, verbose=False)

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

    #    cooccurrence2 = h.cooccurrence.CooccurrenceMutable(unigram2, verbose=False)
    #    for tok1, tok2 in token_pairs2:
    #        cooccurrence2.add(tok1, tok2)
    #        cooccurrence2.add(tok2, tok1)

    #    cooccurrence_sum = cooccurrence1 + cooccurrence2

    #    # Ensure that cooccurrence1 was not changed
    #    dictionary, array, unigram = self.get_test_cooccurrence_stats()
    #    array = torch.tensor(array, device=device, dtype=dtype)
    #    Nxx1, Nx1, Nxt1, N1 = cooccurrence1.load_shard()
    #    self.assertTrue(np.allclose(Nxx1, array))
    #    expected_Nx = torch.sum(array, dim=1).reshape(-1,1)
    #    expected_Nxt = torch.sum(array, dim=0).reshape(1,-1)
    #    self.assertTrue(np.allclose(Nx1, expected_Nx))
    #    self.assertTrue(np.allclose(Nxt1, expected_Nxt))
    #    self.assertTrue(torch.allclose(N1, torch.sum(array)))
    #    self.assertEqual(cooccurrence1.dictionary.tokens, dictionary.tokens)
    #    self.assertEqual(
    #        cooccurrence1.dictionary.token_ids, dictionary.token_ids)
    #    self.assertEqual(cooccurrence1.verbose, False)

    #    # Ensure that cooccurrence2 was not changed
    #    Nxx2, Nx2, Nxt2, N2 = cooccurrence2.load_shard()
    #    array2 = torch.tensor(array2, dtype=dtype, device=device)
    #    self.assertTrue(np.allclose(Nxx2, array2))
    #    expected_Nx2 = torch.sum(array2, dim=1).reshape(-1,1)
    #    expected_Nxt2 = torch.sum(array2, dim=0).reshape(1,-1)
    #    self.assertTrue(torch.allclose(Nx2, expected_Nx2))
    #    self.assertTrue(torch.allclose(Nxt2, expected_Nxt2))
    #    self.assertEqual(N2, torch.sum(array2))
    #    self.assertEqual(cooccurrence2.dictionary.tokens, dictionary2.tokens)
    #    self.assertEqual(
    #        cooccurrence2.dictionary.token_ids, dictionary2.token_ids)
    #    self.assertEqual(cooccurrence2.verbose, False)


    #    # Ensure that cooccurrence_sum is as desired.  Sort to make comparison
    #    # easier.  Double check that sort flag is False to begin with.
    #    self.assertFalse(cooccurrence_sum.sorted)
    #    cooccurrence_sum.sort()
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
    #        cooccurrence1.N + cooccurrence2.N, dtype=dtype, device=device)
    #    Nxx_sum, Nx_sum, Nxt_sum, N_sum = cooccurrence_sum.load_shard()

    #    self.assertEqual(dictionary_sum.tokens, cooccurrence_sum.dictionary.tokens)
    #    self.assertTrue(torch.allclose(Nxx_sum, expected_Nxx_sum))
    #    self.assertTrue(torch.allclose(Nx_sum, expected_Nx_sum))
    #    self.assertTrue(torch.allclose(Nxt_sum, expected_Nxt_sum))
    #    self.assertEqual(N_sum, expected_N_sum)


    def test_add(self):

        # Create a `CooccurrenceMutable` instance using counts
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence_mutable = h.cooccurrence.CooccurrenceMutable(
            unigram, Nxx=array, verbose=False)
        Nxx = torch.tensor(array, dtype=h.CONSTANTS.DEFAULT_DTYPE)

        # We can add tokens if they are in the unigram vocabulary
        add_count = 3
        cooccurrence_mutable.add('banana', 'socks', add_count)
        expected_Nxx = Nxx.clone()
        expected_Nxx[0,1] += add_count
        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = torch.sum(expected_Nxx, dim=0, keepdim=True)
        expected_N = torch.sum(expected_Nxx)

        # Check that adding occurred
        self.assertTrue(np.allclose(cooccurrence_mutable.Nxx.toarray(), expected_Nxx))
        self.assertTrue(torch.allclose(torch.tensor(cooccurrence_mutable.Nx, dtype=torch.float32), expected_Nx))
        self.assertTrue(torch.allclose(torch.tensor(cooccurrence_mutable.Nxt, dtype=torch.float32), expected_Nxt))
        self.assertEqual(cooccurrence_mutable.N, expected_N)

        # We cannot add tokens if they are outside of the unigram vocabulary
        with self.assertRaises(ValueError):
            cooccurrence_mutable.add('archaeopteryx', 'socks')

        # If skip_unk is True, then don't raise an error when attempting to
        # add tokens outside vocabulary, just skip
        cooccurrence_mutable.add('archaeopteryx', 'socks', skip_unk=True)


    #
    #   CooccurrenceMutable is guaranteed to be sorted because they are made from
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
    #    cooccurrence = h.cooccurrence.CooccurrenceMutable(unsorted_unigram, unsorted_Nxx, verbose=False)

    #    # Cooccurrence is unsorted
    #    self.assertFalse(np.allclose(cooccurrence.Nxx.toarray(), sorted_Nxx))
    #    self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), unsorted_Nxx))
    #    self.assertFalse(cooccurrence.sorted)

    #    # Unigram is unsorted
    #    self.assertEqual(cooccurrence.unigram.Nx, unsorted_Nx)
    #    self.assertFalse(cooccurrence.unigram.sorted)

    #    # Sorting cooccurrence works.
    #    cooccurrence.sort()
    #    self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), sorted_Nxx))
    #    self.assertFalse(np.allclose(cooccurrence.Nxx.toarray(), unsorted_Nxx))
    #    self.assertTrue(cooccurrence.sorted)

    #    # The unigram is also sorted
    #    self.assertTrue(np.allclose(cooccurrence.unigram.Nx, sorted_Nx))
    #    self.assertFalse(np.allclose(cooccurrence.unigram.Nx, unsorted_Nx))
    #    self.assertEqual(cooccurrence.dictionary.tokens, sorted_dictionary.tokens)
    #    self.assertTrue(cooccurrence.unigram.sorted)

    def test_truncate(self):

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        cooccurrence = h.cooccurrence.CooccurrenceMutable(unigram, array, verbose=False)

        trunc_tokens = ['banana', 'socks']
        trunc_uNx = [5, 4]
        trunc_Nxx = np.array([[0,3],[3,0]])
        trunc_Nx = [[3],[3]]
        trunc_Nxt = [[3,3]]
        trunc_N = 6

        cooccurrence.truncate(2)
        # The top two tokens by unigram frequency are 'banana' and 'socks',
        # but the top two tokens by cooccurrence frequency are 'car', and 'banana'
        self.assertTrue(cooccurrence.sorted)
        self.assertTrue(cooccurrence.unigram.sorted)
        self.assertEqual(cooccurrence.dictionary.tokens, ['banana', 'socks'])
        self.assertEqual(cooccurrence.unigram.dictionary.tokens, ['banana', 'socks'])

        self.assertTrue(np.allclose(
            np.asarray(cooccurrence.Nxx.todense()), trunc_Nxx))
        self.assertEqual(cooccurrence.unigram.Nx, trunc_uNx)
        self.assertTrue(np.allclose(cooccurrence.Nx, trunc_Nx))
        self.assertTrue(np.allclose(cooccurrence.Nxt, trunc_Nxt))
        self.assertEqual(cooccurrence.N, trunc_N)


    def test_save_load(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-cooccurrence')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance.
        cooccurrence = h.cooccurrence.CooccurrenceMutable(unigram, array, verbose=False)
        Nxx, Nx, Nxt, N = cooccurrence.load_shard()

        # Save it, then load it
        cooccurrence.save(write_path)
        cooccurrence2 = h.cooccurrence.CooccurrenceMutable.load(write_path, verbose=False)

        Nxx2, Nx2, Nxt2, N2 = cooccurrence2.load_shard()

        self.assertEqual(
            cooccurrence2.dictionary.tokens,
            cooccurrence.dictionary.tokens
        )
        self.assertTrue(np.allclose(Nxx2, Nxx))
        self.assertTrue(np.allclose(Nx2, Nx))
        self.assertTrue(np.allclose(Nxt2, Nxt))
        self.assertTrue(np.allclose(N2, N))
        self.assertTrue(np.allclose(cooccurrence.unigram.Nx, cooccurrence2.unigram.Nx))
        self.assertTrue(np.allclose(cooccurrence.unigram.N, cooccurrence2.unigram.N))

        shutil.rmtree(write_path)


    def test_save_load_sector(self):
        """
        We should be able to save just a sector of the cooccurrence, which is
        logically like a shard, except that shards are smaller in practice.
        On disk, the data is separated into sectors, and then, to load onto 
        GPU, it is further separated into shards.  A key thing to ensure is
        that the indexing of Unigram and Dictionary are concordant with the
        indexing of the loaded sector.
        """

        sector_factor = 3
        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-cooccurrence-sector')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()

        # Create a cooccurrence instance.
        cooccurrence = h.cooccurrence.CooccurrenceMutable(unigram, array, verbose=False)
        Nxx, Nx, Nxt, N = cooccurrence.load_shard()

        # Save all the sectors, but don't save the unigram (and dictionary)
        sectors = h.shards.Shards(sector_factor)
        for sector in sectors:
            cooccurrence.save_sector(
                write_path, sector, save_marginal=False, save_unigram=False)

        # Check that the sectors were in fact saved, and that the unigram and
        # dictionary were not.
        paths = set(os.listdir(write_path))
        expected_paths = {
            'Nxx-{}-{}-{}.npz'.format(i,j,sector_factor)
            for i in range(sector_factor) for j in range(sector_factor)
        }
        self.assertEqual(paths, expected_paths)

        # Finally, re-save the 0th shard, this time with the dictionary and 
        # unigram being saved
        cooccurrence.save_sector(
            write_path, sector=sectors[0], save_marginal=True, 
            save_unigram=True
        )
        paths = set(os.listdir(write_path))
        expected_paths.update(['dictionary', 'Nx.txt', 'Nx.npy', 'Nxt.npy'])
        self.assertEqual(paths, expected_paths)

        # Now go sector by sector, and load the sector, and check that
        # everything lines up.
        for sector in sectors:

            cooccurrence_sector = h.cooccurrence.CooccurrenceSector.load(
                write_path, sector=sector, verbose=False)

            row_dict = cooccurrence_sector.row_dictionary
            col_dict = cooccurrence_sector.column_dictionary
            for row_token in row_dict.tokens:
                for col_token in col_dict.tokens:
                    r_id_sector = row_dict.get_id(row_token)
                    c_id_sector = col_dict.get_id(col_token)
                    r_id = cooccurrence.dictionary.get_id(row_token)
                    c_id = cooccurrence.dictionary.get_id(col_token)

                    self.assertEqual(
                        cooccurrence_sector.Nxx[r_id_sector, c_id_sector],
                        cooccurrence.Nxx[r_id, c_id]
                    )
                    self.assertEqual(
                        cooccurrence_sector.Nx[r_id_sector,0], cooccurrence.Nx[r_id,0])
                    self.assertEqual(
                        cooccurrence_sector.Nxt[0,c_id_sector], cooccurrence.Nxt[0,c_id])
                    self.assertEqual(
                        cooccurrence_sector.uNx[r_id_sector,0], cooccurrence.uNx[r_id,0])
                    self.assertEqual(
                        cooccurrence_sector.uNxt[0,c_id_sector], cooccurrence.uNxt[0,c_id])
                    self.assertEqual(cooccurrence_sector.uN, cooccurrence.uN)

                    self.assertTrue(np.allclose(
                        cooccurrence_sector.Nxx.toarray(),
                        cooccurrence.Nxx.toarray()[sector]
                    ))
                    self.assertTrue(np.allclose(
                        cooccurrence_sector.Nx, cooccurrence.Nx[sector[0]]))
                    self.assertTrue(np.allclose(
                        cooccurrence_sector.Nxt, cooccurrence.Nxt[:,sector[1]]))
                    self.assertTrue(np.allclose(
                        cooccurrence_sector.uNx, cooccurrence.uNx[sector[0]]))
                    self.assertTrue(np.allclose(
                        cooccurrence_sector.uNxt, cooccurrence.uNxt[:,sector[1]]))

        shutil.rmtree(write_path)


    # should be sorted only if unigram is sorted.
    def test_load_unigram(self):

        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-cooccurrence')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        unigram.save(write_path)

        cooccurrence = h.cooccurrence.CooccurrenceMutable.load_unigram(write_path)

        self.assertTrue(np.allclose(cooccurrence.unigram.Nx, unigram.Nx))
        self.assertTrue(np.allclose(cooccurrence.unigram.N, unigram.N))
        self.assertEqual(cooccurrence.dictionary.tokens, unigram.dictionary.tokens)

        # Ensure that we can add any pairs of tokens found in the unigram
        # vocabulary.  As long as this runs without errors everything is fine.
        for tok1 in dictionary.tokens:
            for tok2 in dictionary.tokens:
                cooccurrence.add(tok1, tok2)


