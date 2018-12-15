import sys
import os
import shutil
from unittest import main, TestCase
from copy import copy, deepcopy
from collections import Counter
from matplotlib import pyplot as plt
import hilbert as h
import random

try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    sparse = None
    torch = None



class TestCorpusStats(TestCase):

    UNIQUE_TOKENS = {
    '.': 5, 'Drive': 3, 'Eat': 7, 'The': 10, 'bread': 0, 'car': 6,
    'has': 8, 'sandwich': 9, 'spin': 4, 'the': 1, 'wheels': 2
    }
    N_XX_2 = np.array([
        [ 0., 23., 12.,  7.,  8.,  8.,  8.,  8., 12.,  4.,  4.],
        [23.,  0.,  0.,  8.,  4.,  0.,  4.,  4.,  0.,  4.,  0.],
        [12.,  0.,  0.,  0.,  8.,  8.,  0.,  8.,  8.,  0.,  4.],
        [ 7.,  8.,  0.,  0.,  4.,  0., 11.,  0.,  0.,  0.,  0.],
        [ 8.,  4.,  8.,  4.,  0.,  4.,  4.,  0.,  0.,  0.,  0.],
        [ 8.,  0.,  8.,  0.,  4.,  0.,  4.,  4.,  4.,  0.,  0.],
        [ 8.,  4.,  0., 11.,  4.,  4.,  0.,  0.,  0.,  1.,  0.],
        [ 8.,  4.,  8.,  0.,  0.,  4.,  0.,  0.,  4.,  4.,  0.],
        [12.,  0.,  8.,  0.,  0.,  4.,  0.,  4.,  0.,  0.,  4.],
        [ 4.,  4.,  0.,  0.,  0.,  0.,  1.,  4.,  0.,  0.,  3.],
        [ 4.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  4.,  3.,  0.]
    ])
    N_XX_3 = np.array([
        [ 0., 11.,  4., 15.,  0.,  4., 11.,  0.,  0.,  0.,  0.],
        [11.,  0.,  4., 23.,  8.,  0., 12.,  5.,  4.,  0.,  3.],
        [ 4.,  4.,  8., 15.,  8.,  4.,  4.,  0.,  0.,  0.,  0.],
        [15., 23., 15.,  0., 16., 16., 12.,  8., 16., 12.,  8.],
        [ 0.,  8.,  8., 16.,  0., 12.,  4.,  0.,  8., 12.,  4.],
        [ 4.,  0.,  4., 16., 12.,  0.,  4.,  0.,  4.,  4.,  0.],
        [11., 12.,  4., 12.,  4.,  4.,  0.,  1.,  0.,  0.,  0.],
        [ 0.,  5.,  0.,  8.,  0.,  0.,  1.,  0.,  4.,  3.,  3.],
        [ 0.,  4.,  0., 16.,  8.,  4.,  0.,  4.,  8.,  4.,  0.],
        [ 0.,  0.,  0., 12., 12.,  4.,  0.,  3.,  4.,  8.,  4.],
        [ 0.,  3.,  0.,  8.,  4.,  0.,  0.,  3.,  0.,  4.,  0.]
    ])


    def test_PMI(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram
        expected_PMI = torch.log(Nxx*N / (Nx * Nxt))
        found_PMI = h.corpus_stats.calc_PMI(bigram)
        self.assertTrue(np.allclose(found_PMI, expected_PMI))


    def test_sparse_PMI(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram
        expected_PMI = torch.log(Nxx*N / (Nx * Nxt))
        expected_PMI[expected_PMI==-np.inf] = 0
        pmi_data, I, J = h.corpus_stats.calc_PMI_sparse(bigram)
        self.assertTrue(len(pmi_data) < np.product(bigram.Nxx.shape))
        found_PMI = sparse.coo_matrix((pmi_data,(I,J)),bigram.Nxx.shape)
        self.assertTrue(np.allclose(found_PMI.toarray(), expected_PMI))


    def test_PMI_star(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram
        Nxx = Nxx.clone()
        Nxx[Nxx==0] = 1
        expected_PMI_star = torch.log(Nxx * N / (Nx * Nxt))
        found_PMI_star = h.corpus_stats.calc_PMI_star(bigram)
        self.assertTrue(np.allclose(found_PMI_star, expected_PMI_star))


    def test_get_stats(self):
        # Next, test with a cooccurrence window of +/-2
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        bigram = h.corpus_stats.get_test_bigram(2)

        # Sort to make comparison easier
        bigram.sort()

        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(torch.allclose(
            Nxx,
            torch.tensor(self.N_XX_2, dtype=dtype, device=device)
        ))

        # Next, test with a cooccurrence window of +/-3
        bigram = h.corpus_stats.get_test_bigram(3)
        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(np.allclose(
            Nxx,
            torch.tensor(
                self.N_XX_3, dtype=dtype, device=device)
        ))


    def test_calc_exp_pmi_stats(self):
        bigram = h.corpus_stats.get_test_bigram(2)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        PMI = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        PMI = PMI[Nxx>0]
        exp_PMI = torch.exp(PMI)
        expected_mean, expected_std = torch.mean(exp_PMI), torch.std(exp_PMI)

        found_mean, found_std = h.corpus_stats.calc_exp_pmi_stats(bigram)

        self.assertTrue(torch.allclose(found_mean, expected_mean))
        self.assertTrue(torch.allclose(found_std, expected_std))


    def test_calc_prior_beta_params(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        exp_mean, exp_std = h.corpus_stats.calc_exp_pmi_stats(bigram)
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_independent = (Nx / N) * (Nxt / N)

        means = exp_mean * Pxx_independent
        stds = exp_std * Pxx_independent
        expected_alpha = means * (means * (1-means) / stds**2 - 1)
        expected_beta = (1-means) / means * expected_alpha

        found_alpha, found_beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)

        self.assertTrue(torch.allclose(found_alpha, expected_alpha))
        self.assertTrue(torch.allclose(found_beta, expected_beta))




class TestM(TestCase):


    def test_calc_M_pmi(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram

        # First calculate using no options
        found_M = h.M.M_pmi(bigram).load_all()

        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        self.assertTrue(np.allclose(found_M, expected_M))

        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE,
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N)) + shift_by
        found_M = h.M.M_pmi(bigram, shift_by=shift_by).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI(bigram)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M_pmi(bigram, clip_thresh=clip_thresh).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        diag = 5
        expected_M = h.corpus_stats.calc_PMI(bigram)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M_pmi(bigram, diag=diag).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))


    def test_calc_M_logNxx(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram

        # First calculate using no options.
        M = h.M.M_logNxx(bigram).load_all()
        expected_M = torch.log(Nxx)
        self.assertTrue(torch.allclose(M, expected_M))

        # Test shift option.
        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE,
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = torch.log(Nxx) + shift_by
        found_M = h.M.M_logNxx(bigram, shift_by=shift_by).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))

        # Test setting a clip threshold.
        clip_thresh = -0.1
        expected_M = torch.log(Nxx)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M_logNxx(
            bigram, clip_thresh=clip_thresh).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))

        # Test setting diagonal values to a given constant.
        diag = 5
        expected_M = torch.log(Nxx)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M_logNxx(bigram, diag=diag).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))


    def test_calc_M_pmi_star(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram

        # First calculate using no options
        M = h.M.M_pmi_star(bigram).load_all()
        expected_M = h.corpus_stats.calc_PMI_star(bigram)
        self.assertTrue(np.allclose(M, expected_M))

        # Test shift option.
        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE,
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = h.corpus_stats.calc_PMI_star(bigram) + shift_by
        found_M = h.M.M_pmi_star(bigram, shift_by=shift_by).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        # Test setting a clip threshold.
        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI_star(bigram)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M_pmi_star(
            bigram, clip_thresh=clip_thresh).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        # Test setting diagonal values to a given constant.
        diag = 5
        expected_M = h.corpus_stats.calc_PMI_star(bigram)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M_pmi_star(bigram, diag=diag).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))


    def test_sharding(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        bigram.truncate(6)
        Nxx, Nx, Nxt, N = bigram

        # First calculate using no options
        shards = h.shards.Shards(2)
        M = h.M.M_pmi(bigram)
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]

        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        self.assertTrue(np.allclose(found_M, expected_M))

        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE,
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N)) + shift_by

        M = h.M.M_pmi(bigram, shift_by=shift_by).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]

        self.assertTrue(np.allclose(found_M, expected_M))

        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI(bigram)
        expected_M[expected_M<clip_thresh] = clip_thresh
        M = h.M.M_pmi(bigram, clip_thresh=clip_thresh).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]
        self.assertTrue(np.allclose(found_M, expected_M))

        diag = 5
        expected_M = h.corpus_stats.calc_PMI(bigram)
        h.utils.fill_diagonal(expected_M, diag)
        M = h.M.M_pmi(bigram, diag=diag).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]
        self.assertTrue(np.allclose(found_M, expected_M))




# These functions came from hilbert-experiments, where they were only being
# used to support testing.  Now that the Dictionary and it's testing have moved
# here, I have copied these helper functions and changed them minimally.
def iter_test_fnames():
    for path in os.listdir(h.CONSTANTS.TEST_DOCS_DIR):
        if not skip_file(path):
            yield os.path.basename(path)
def iter_test_paths():
    for fname in iter_test_fnames():
        yield get_test_path(fname)
def get_test_tokens():
    paths = iter_test_paths()
    return read_tokens(paths)
def read_tokens(paths):
    tokens = []
    for path in paths:
        with open(path) as f:
            tokens.extend([token for token in f.read().split()])
    return tokens
def skip_file(fname):
    if fname.startswith('.'):
        return True
    if fname.endswith('.swp') or fname.endswith('.swo'):
        return True
    return False
def get_test_path(fname):
    return os.path.join(h.CONSTANTS.TEST_DOCS_DIR, fname)


class TestDictionary(TestCase):

    def get_test_dictionary(self):

        tokens = get_test_tokens()
        return tokens, h.dictionary.Dictionary(tokens)


    def test_copy(self):

        # NOTE: currently implementation of copy is simply deferred to deepcopy

        tokens, dictionary1 = self.get_test_dictionary()
        dictionary2 = copy(dictionary1)

        # None of the obejects are the same
        self.assertTrue(dictionary2 is not dictionary1)
        self.assertTrue(dictionary2.tokens is not dictionary1.tokens)
        self.assertTrue(dictionary2.token_ids is not dictionary1.token_ids)

        # But they are equal
        self.assertEqual(dictionary2.tokens, dictionary1.tokens)
        self.assertEqual(dictionary2.token_ids, dictionary1.token_ids)


    def test_deepcopy(self):

        # NOTE: currently implementation of copy is simply deferred to deepcopy

        tokens, dictionary1 = self.get_test_dictionary()
        dictionary2 = deepcopy(dictionary1)

        # None of the obejects are the same
        self.assertTrue(dictionary2 is not dictionary1)
        self.assertTrue(dictionary2.tokens is not dictionary1.tokens)
        self.assertTrue(dictionary2.token_ids is not dictionary1.token_ids)

        # But they are equal
        self.assertEqual(dictionary2.tokens, dictionary1.tokens)
        self.assertEqual(dictionary2.token_ids, dictionary1.token_ids)


    def test_dictionary(self):
        tokens, dictionary = self.get_test_dictionary()
        for token in tokens:
            dictionary.add_token(token)

        self.assertEqual(set(tokens), set(dictionary.tokens))
        expected_token_ids = {
            token:idx for idx, token in enumerate(dictionary.tokens)}
        self.assertEqual(expected_token_ids, dictionary.token_ids)


    def test_save_load_dictionary(self):
        write_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test.dictionary')

        # Remove files that could be left from a previous test.
        if os.path.exists(write_path):
            os.remove(write_path)

        tokens, dictionary = self.get_test_dictionary()
        dictionary.save(write_path)
        loaded_dictionary = h.dictionary.Dictionary.load(
            write_path)

        self.assertEqual(loaded_dictionary.tokens, dictionary.tokens)
        self.assertEqual(loaded_dictionary.token_ids, dictionary.token_ids)

        # Cleanup
        os.remove(write_path)



class TestUnigram(TestCase):

    def test_unigram_creation_from_corpus(self):
        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()

        # An empty unigram is by definition sorted.
        self.assertTrue(unigram.sorted)

        # Add counts
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        # Adding counts disrupts the sorting
        self.assertFalse(unigram.sorted)

        # The correct number of counts are registered for each token
        counts = Counter(h.corpus_stats.load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])

        # Test sorting.
        unigram.sort()
        for i in range(len(unigram.Nx)-1):
            self.assertTrue(unigram.Nx[i] >= unigram.Nx[i+1])


    def test_unigram_creation_from_Nx(self):
        tokens = h.corpus_stats.load_test_tokens()
        dictionary = h.dictionary.Dictionary(tokens)
        Nx = [0] * len(dictionary)
        for token in tokens:
            Nx[dictionary.get_id(token)] += 1

        # Must supply a dictionary to create a non-empty Unigram.
        with self.assertRaises(ValueError):
            unigram = h.unigram.Unigram(Nx=Nx)

        unigram = h.unigram.Unigram(dictionary=dictionary, Nx=Nx)

        # Check that unigram knows it is not yet sorted
        self.assertFalse(unigram.sorted)

        # The correct number of counts are registered for each token
        counts = Counter(h.corpus_stats.load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])

        # Ensure that if adding counts undoes sorting, then the sorting flag
        # of unigram becomes false.
        unigram.sort()
        self.assertTrue(unigram.sorted)
        for i in range(5):
            unigram.add('Eat')
        self.assertFalse(unigram.sorted)


    def test_load_shard(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        # To simplify the test, sor the unigram
        unigram.sort()

        # Ensure that all shards are correct.
        shards = h.shards.Shards(3)
        for i, shard in enumerate(shards):
            expected_Nxs = [
                torch.tensor([24,8,8,4], dtype=dtype, device=device),
                torch.tensor([12,8,8,4], dtype=dtype, device=device),
                torch.tensor([12,8,8], dtype=dtype, device=device),
            ]
            expected_N = torch.tensor(104, dtype=dtype, device=device)
            Nx, Nxt, N = unigram.load_shard(shard)

            if i // 3 ==0:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[0].view(-1,1)))
            elif i // 3 == 1:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[1].view(-1,1)))
            elif i // 3 == 2:
                self.assertTrue(torch.allclose(Nx, expected_Nxs[2].view(-1,1)))

            if i % 3 == 0:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[0].view(1,-1)))
            elif i % 3 == 1:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[1].view(1,-1)))
            elif i % 3 == 2:
                self.assertTrue(torch.allclose(Nxt,expected_Nxs[2].view(1,-1)))

            self.assertEqual(Nxt.dtype, h.CONSTANTS.DEFAULT_DTYPE)
            self.assertEqual(Nx.dtype, h.CONSTANTS.DEFAULT_DTYPE)
            self.assertTrue(
                str(Nxt.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
            self.assertTrue(
                str(Nx.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
            self.assertTrue(isinstance(Nxt, torch.Tensor))
            self.assertTrue(isinstance(Nx, torch.Tensor))

        self.assertTrue(torch.allclose(N, expected_N))
        self.assertTrue(str(N.device).startswith(h.CONSTANTS.MATRIX_DEVICE))
        self.assertEqual(N.dtype, h.CONSTANTS.DEFAULT_DTYPE)


    def test_copy(self):

        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram1.add(token)

        unigram2 = copy(unigram1)

        # Objects are distinct.
        self.assertFalse(unigram2 is unigram1)
        self.assertFalse(unigram2.Nx is unigram1.Nx)
        self.assertFalse(unigram2.dictionary is unigram1.dictionary)

        # Objects are equal.
        self.assertEqual(unigram2.N, unigram1.N)
        self.assertEqual(unigram2.Nx, unigram1.Nx)
        self.assertEqual(
            unigram2.dictionary.tokens, unigram2.dictionary.tokens)
        self.assertEqual(
            unigram2.dictionary.token_ids, unigram1.dictionary.token_ids)


    def test_unpacking(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        # To simplify the test, sor the unigram
        unigram.sort()

        expected_Nx = torch.tensor(
            [24, 12, 12, 8, 8, 8, 8, 8, 8, 4, 4], dtype=dtype, device=device
        ).view(-1,1)
        expected_Nxt = torch.tensor(
            [24, 12, 12, 8, 8, 8, 8, 8, 8, 4, 4], dtype=dtype, device=device
        ).view(1,-1)
        expected_N = torch.tensor(104, dtype=dtype, device=device)
        Nx, Nxt, N = unigram
        self.assertTrue(torch.allclose(Nx, expected_Nx))
        self.assertTrue(torch.allclose(Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(N, expected_N))


    def test_sort_by_tokens(self):

        # Get tokens in alphabetical order
        tokens = list(set(h.corpus_stats.load_test_tokens()))
        tokens.sort()

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        # Unigram has same tokens, but is not in alphabetical order.
        self.assertCountEqual(unigram.dictionary.tokens, tokens)
        self.assertNotEqual(unigram.dictionary.tokens, tokens)

        # Sort by provided token list.  Now token lists have same order.
        unigram.sort_by_tokens(tokens)
        self.assertEqual(unigram.dictionary.tokens, tokens)

        # Sort by unigram order, to verify that calling sort_by_tokens resets
        # the sorted flag
        self.assertFalse(unigram.sorted)
        unigram.sort()
        self.assertTrue(unigram.sorted)
        unigram.sort_by_tokens(tokens)
        self.assertFalse(unigram.sorted)



    def test_add(self):
        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram1.add(token)

        unigram2 = h.unigram.Unigram()
        additional_tokens = 'the car is green .'.split()
        for token in additional_tokens:
            unigram2.add(token)

        expected_counts = Counter(
            h.corpus_stats.load_test_tokens()+additional_tokens)

        # Orginary addition
        unigram4 = unigram1 + unigram2
        for token in expected_counts:
            token_idx = unigram4.dictionary.get_id(token)
            self.assertEqual(unigram4.Nx[token_idx], expected_counts[token])

        # In-place addition
        unigram3 = h.unigram.Unigram()
        unigram3 += unigram1    # adds larger into smaller.
        unigram3 += unigram2    # adds smaller into larger.
        for token in expected_counts:
            token_idx = unigram3.dictionary.get_id(token)
            self.assertEqual(unigram3.Nx[token_idx], expected_counts[token])


    def test_save_load(self):

        # Work out the path, and clear away anything that is currently there.
        write_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-save-load-unigram')
        if os.path.exists(write_path):
            shutil.rmtree(write_path)

        # Make a unigram and fill it with tokens and counts.
        unigram1 = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram1.add(token)

        # Do a save and load cycle
        unigram1.save(write_path)
        unigram2 = h.unigram.Unigram.load(write_path)

        # Objects are distinct.
        self.assertFalse(unigram2 is unigram1)
        self.assertFalse(unigram2.Nx is unigram1.Nx)
        self.assertFalse(unigram2.dictionary is unigram1.dictionary)

        # Objects are equal.
        self.assertEqual(unigram2.N, unigram1.N)
        self.assertEqual(unigram2.Nx, unigram1.Nx)
        self.assertEqual(
            unigram2.dictionary.tokens, unigram2.dictionary.tokens)
        self.assertEqual(
            unigram2.dictionary.token_ids, unigram1.dictionary.token_ids)

        # Cleanup.
        shutil.rmtree(write_path)


    def test_truncate(self):

        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        expected_tokens = ['.', 'the', 'The', 'Eat', 'sandwich']
        expected_Nx = [24, 12, 12, 8, 8]
        expected_N = sum(expected_Nx)
        unigram.truncate(5)

        self.assertEqual(unigram.Nx, expected_Nx)
        self.assertEqual(unigram.N, expected_N)
        self.assertEqual(unigram.dictionary.tokens, expected_tokens)
        self.assertEqual(
            set(unigram.dictionary.token_ids.keys()), set(expected_tokens))




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



def get_test_dictionary():
    return h.dictionary.Dictionary.load(
        os.path.join(h.CONSTANTS.TEST_DIR, 'dictionary'))



class TestEmbeddings(TestCase):

    def test_creating_embeddings(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        shared = False
        V = torch.rand(vocab, d)

        # Try creating a embeddings that lack W
        embeddings = h.embeddings.Embeddings(V, dictionary=dictionary)
        self.assertTrue(embeddings.W is None)

        # Try creating one-sided embeddings
        embeddings = h.embeddings.Embeddings(
            V, dictionary=dictionary, shared=True)
        self.assertTrue(embeddings.W is embeddings.V)

        # Try making embeddings using an incorrectly-lengthed dictionary.
        V = V[:-5]
        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(V, dictionary=dictionary)


    def test_random(self):
        d = 300
        vocab = 5000
        shared = False
        dictionary = get_test_dictionary()

        # Can make random embeddings and provide a dictionary to use.
        embeddings = h.embeddings.random(vocab, d, dictionary, shared)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can have random embeddings with shared parameters.
        embeddings = h.embeddings.random(vocab, d, dictionary, shared=True)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertTrue(embeddings.W is embeddings.V)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can omit the dictionary
        embeddings = h.embeddings.random(
            vocab, d, dictionary=None, shared=False)
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is None)

        # Uses torch.
        embeddings = h.embeddings.random(vocab, d, dictionary, shared=False)
        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))


    def test_random_distribution(self):
        d = 300
        vocab = 5000
        shared = False

        dictionary = get_test_dictionary()

        # Can make numpy random embeddings with uniform distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, distribution='uniform', scale=0.2,
            seed=0
        )

        np.random.seed(0)
        expected_uniform_V = np.random.uniform(-0.2, 0.2, (vocab, d))
        expected_uniform_W = np.random.uniform(-0.2, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_W))


        # Can make numpy random embeddings with normal distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, distribution='normal', scale=0.2,
            seed=0
        )

        np.random.seed(0)
        expected_normal_V = np.random.normal(0, 0.2, (vocab, d))
        expected_normal_W = np.random.normal(0, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, distribution='uniform', scale=1,
            seed=0
        )

        np.random.seed(0)
        expected_uniform_scale_V = np.random.uniform(-1, 1, (vocab, d))
        expected_uniform_scale_W = np.random.uniform(-1, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_scale_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, distribution='normal', scale=1,
            seed=0
        )

        np.random.seed(0)
        expected_normal_scale_V = np.random.normal(0, 1, (vocab, d))
        expected_normal_scale_W = np.random.normal(0, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_scale_W))




    def test_unk(self):

        d = 300
        vocab = 5000
        shared = False
        dictionary = get_test_dictionary()

        embeddings = h.embeddings.random(vocab, d, dictionary, shared)
        self.assertTrue(np.allclose(embeddings.unk, embeddings.V.mean(0)))
        self.assertTrue(np.allclose(embeddings.unkV, embeddings.V.mean(0)))
        self.assertTrue(np.allclose(embeddings.unkW, embeddings.W.mean(0)))

        embeddings = h.embeddings.random(vocab, d, dictionary, shared)
        self.assertTrue(torch.allclose(embeddings.unk, embeddings.V.mean(0)))
        self.assertTrue(torch.allclose(embeddings.unkV, embeddings.V.mean(0)))
        self.assertTrue(torch.allclose(embeddings.unkW, embeddings.W.mean(0)))

        with self.assertRaises(KeyError):
            embeddings.get_vec('archaeopteryx')
        with self.assertRaises(KeyError):
            embeddings.get_covec('archaeopteryx')
        with self.assertRaises(KeyError):
            embeddings['archaeopteryx']

        self.assertTrue(torch.allclose(
            embeddings.get_vec('archaeopteryx', 'unk'),
            embeddings.V.mean(0)
        ))
        self.assertTrue(torch.allclose(
            embeddings.get_covec('archaeopteryx', 'unk'),
            embeddings.W.mean(0)
        ))


    def test_embedding_access(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(V, W, dictionary)

        self.assertTrue(np.allclose(embeddings.get_vec(1000), V[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_vec('apple'),
            V[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings.get_covec(1000), W[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_covec('apple'),
            W[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings[1000], V[1000]))
        self.assertTrue(np.allclose(
            embeddings['apple'],
            V[dictionary.tokens.index('apple')]
        ))

        # KeyErrors are trigerred when trying to access embeddings that are
        # out-of-vocabulary.
        with self.assertRaises(KeyError):
            embeddings.get_vec('archaeopteryx')

        with self.assertRaises(KeyError):
            embeddings.get_covec('archaeopteryx')

        with self.assertRaises(KeyError):
            embeddings['archaeopteryx']

        # IndexErrors are raised for trying to access non-existent embedding
        # indices
        with self.assertRaises(IndexError):
            embeddings.get_vec(5000)

        with self.assertRaises(IndexError):
            embeddings.get_vec((0,300))

        with self.assertRaises(IndexError):
            embeddings.get_covec(5000)

        with self.assertRaises(IndexError):
            embeddings.get_covec((0,300))

        with self.assertRaises(IndexError):
            embeddings[5000]

        with self.assertRaises(IndexError):
            embeddings[0,300]

        embeddings = h.embeddings.Embeddings(V, W, dictionary=None)
        with self.assertRaises(ValueError):
            embeddings['apple']

        embeddings = h.embeddings.Embeddings(V, W=None, dictionary=dictionary)
        self.assertTrue(np.allclose(embeddings.V, V))
        self.assertTrue(embeddings.W is None)
        self.assertTrue(embeddings.dictionary is dictionary)

        self.assertTrue(np.allclose(embeddings.get_vec(1000), V[1000]))
        self.assertTrue(np.allclose(
            embeddings.get_vec('apple'),
            V[dictionary.tokens.index('apple')]
        ))

        self.assertTrue(np.allclose(embeddings[1000], V[1000]))
        self.assertTrue(np.allclose(
            embeddings['apple'],
            V[dictionary.tokens.index('apple')]
        ))

        with self.assertRaises(ValueError):
            embeddings.get_covec(1000)

        with self.assertRaises(ValueError):
            embeddings.get_covec('apple'),


    def test_save_load(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        out_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-embeddings')

        if os.path.exists(out_path):
            shutil.rmtree(out_path)


        # Create vectors using the numpy implementation, save them, then
        # reload them alternately using either numpy or torch implementation.
        embeddings1 = h.embeddings.Embeddings(V, W, dictionary)
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)

        # We can do the same save and load cycle, this time starting from
        # torch embeddings.
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        embeddings1 = h.embeddings.Embeddings(V, W, dictionary)
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(out_path)
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)


    def test_embeddings_recognize_loading_normalized(self):

        in_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'normalized-test-embeddings')
        embeddings = h.embeddings.Embeddings.load(in_path)
        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())


    def test_normalize_embeddings(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(V, W, dictionary)

        self.assertFalse(embeddings.normed)
        self.assertFalse(embeddings.check_normalized())
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))

        embeddings.normalize()

        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))


    def test_normalize_embeddings_shared(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(
            V, dictionary=dictionary, shared=True)

        self.assertFalse(embeddings.normed)
        self.assertFalse(embeddings.check_normalized())
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertFalse(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))

        embeddings.normalize()

        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.V, axis=1), 1.0))
        self.assertTrue(
            np.allclose(h.utils.norm(embeddings.W, axis=1), 1.0))

        self.assertTrue(np.allclose(embeddings.V, embeddings.W))


    def test_cannot_provide_W_if_shared(self):
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000
        V = torch.rand((vocab, d), device=device, dtype=dtype)
        W = torch.rand((vocab, d), device=device, dtype=dtype)

        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(V, W, shared=True)


    def test_greatest_product(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Some products are tied, and their sorting isn't stable.  But, when
        # we set fix the seed, the top and bottom ten are stably ranked.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=False, seed=0)

        # Given a query vector, verify that we can find the other vector having
        # the greatest dot product.
        query = embeddings['dog']
        products = embeddings.V @ query
        ranks = sorted(
            [(p, idx) for idx, p in enumerate(products)], reverse=True)
        expected_ranked_tokens = [
            dictionary.get_token(idx) for p, idx in ranks
            if dictionary.get_token(idx) != 'dog'
        ]
        expected_ranked_ids = [
            idx for p, idx in ranks if dictionary.get_token(idx) != 'dog']

        found_ranked_tokens = embeddings.greatest_product('dog')
        self.assertEqual(found_ranked_tokens[:10], expected_ranked_tokens[:10])
        self.assertEqual(
            found_ranked_tokens[-10:], expected_ranked_tokens[-10:])

        # If we provide an id as a query, the matches are returned as ids.
        found_ranked_ids = embeddings.greatest_product(
            dictionary.get_id('dog'))
        self.assertEqual(
            list(found_ranked_ids[:10]), expected_ranked_ids[:10])
        self.assertEqual(
            list(found_ranked_ids[-10:]), expected_ranked_ids[-10:])

        # Verify that we can get the single best match:
        found_best_match = embeddings.greatest_product_one('dog')
        self.assertEqual(found_best_match, expected_ranked_tokens[0])

        # Again, if we provide an id as the query, the best match is returned
        # as an id.
        found_best_match = embeddings.greatest_product_one(
            dictionary.get_id('dog'))
        self.assertEqual(found_best_match, expected_ranked_ids[0])



    def test_greatest_cosine(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Some products are tied, and their sorting isn't stable.  But, when
        # we set fix the seed, the top and bottom ten are stably ranked.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=False, seed=0)

        # Given a query vector, verify that we can find the other vector having
        # the greatest dot product.  We want to test cosine similarity, so we
        # should take the dot product of normalized vectors.

        normed_query = h.utils.normalize(embeddings['dog'], axis=0)
        normed_V = h.utils.normalize(embeddings.V, axis=1)
        products = normed_V @ normed_query
        ranks = sorted(
            [(p, idx) for idx, p in enumerate(products)], reverse=True)
        expected_ranked_tokens = [
            dictionary.get_token(idx) for p, idx in ranks
            if dictionary.get_token(idx) != 'dog'
        ]
        expected_ranked_ids = [
            idx for p, idx in ranks if dictionary.get_token(idx) != 'dog']


        found_ranked_tokens = embeddings.greatest_cosine('dog')

        self.assertEqual(found_ranked_tokens[:10], expected_ranked_tokens[:10])
        self.assertEqual(
            found_ranked_tokens[-10:], expected_ranked_tokens[-10:])

        # If we provide an id as a query, the matches are returned as ids.
        found_ranked_ids = embeddings.greatest_cosine(
            dictionary.get_id('dog'))
        self.assertEqual(
            list(found_ranked_ids[:10]), expected_ranked_ids[:10])
        self.assertEqual(
            list(found_ranked_ids[-10:]), expected_ranked_ids[-10:])

        # Verify that we can get the single best match:
        found_best_match = embeddings.greatest_cosine_one('dog')
        self.assertEqual(found_best_match, expected_ranked_tokens[0])

        # Again, if we provide an id as the query, the best match is returned
        # as an id.
        found_best_match = embeddings.greatest_cosine_one(
            dictionary.get_id('dog'))
        self.assertEqual(found_best_match, expected_ranked_ids[0])


    def test_slicing(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = torch.rand((vocab, d), device=device, dtype=dtype)
        W = torch.rand((vocab, d), device=device, dtype=dtype)

        embeddings = h.embeddings.Embeddings(V, W, dictionary)

        self.assertTrue(torch.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(torch.allclose(
            embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(torch.allclose(
            embeddings.get_covec((slice(0,5000,1),slice(0,300,1))),
            W
        ))

        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(V, W, dictionary)

        self.assertTrue(
            np.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(np.allclose(
                embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(np.allclose(
                embeddings.get_covec((slice(0,5000,1),slice(0,300,1))), W))


    def test_sort_like(self):
        random.seed(0)
        in_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'normalized-test-embeddings')

        embeddings_pristine = h.embeddings.Embeddings.load(in_path)
        embeddings_to_be_sorted = h.embeddings.Embeddings.load(in_path)
        embeddings_to_sort_by = h.embeddings.Embeddings.load(in_path)
        sort_tokens = embeddings_to_sort_by.dictionary.tokens
        random.shuffle(sort_tokens)

        # Alter some of the embeddings so that we can test the handling of
        # sorting against embeddings that don't have the same set of tokens.
        extraneous_tokens = [
            'archeaopteryx', 'Calabi-Yau', 'snails-pace-maker',
            'xxxxxxx', 'yyyyyyy'
        ]
        ommited_tokens = sort_tokens[5:10]
        sort_tokens[5:10] = extraneous_tokens

        # Sort the embeddings according to a new shuffled token order.
        with self.assertRaises(ValueError):
            embeddings_to_be_sorted.sort_like(embeddings_to_sort_by)

        embeddings_to_be_sorted.sort_like(
            embeddings_to_sort_by, allow_mismatch=True
        )

        # The number of embeddings is reduced, because we lost ommitted tokens
        # and because extraneous tokens are ignored.
        self.assertEqual(
            embeddings_to_be_sorted.V.shape[0],
            embeddings_pristine.V.shape[0] - 5
        )

        # The embeddings are reordered but still bound to the same tokens.
        for i, token in enumerate(sort_tokens):

            # Extraneous tokens are left out though.
            if i >= 5 and i < 10:
                with self.assertRaises(KeyError):
                    embeddings_to_be_sorted.get_vec(token)
                continue

            # Adjust indices after extraneous tokens, which were dropped
            if i >= 10:
                i = i - 5

            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.V[i],
                embeddings_to_be_sorted.get_vec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_to_be_sorted.W[i],
                embeddings_to_be_sorted.get_covec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_pristine.get_vec(token),
                embeddings_to_be_sorted.get_vec(token)
            ))
            self.assertTrue(torch.allclose(
                embeddings_pristine.get_covec(token),
                embeddings_to_be_sorted.get_covec(token)
            ))



class TestUtils(TestCase):

    def test_normalize(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000

        V_numpy = np.random.random((vocab, d))

        norm = np.linalg.norm(V_numpy, ord=2, axis=1, keepdims=True)
        expected = V_numpy / norm
        found = h.utils.normalize(V_numpy, ord=2, axis=1)
        self.assertTrue(np.allclose(found, expected))
        self.assertTrue(V_numpy.shape, found.shape)
        self.assertTrue(np.allclose(
            np.linalg.norm(found, ord=2, axis=1), np.ones(vocab)))

        V_numpy = np.random.random((vocab, d))

        norm = np.linalg.norm(V_numpy, ord=2, axis=0, keepdims=True)
        expected = V_numpy / norm
        found = h.utils.normalize(V_numpy, ord=2, axis=0)
        self.assertTrue(np.allclose(found, expected))
        self.assertTrue(V_numpy.shape, found.shape)
        self.assertTrue(np.allclose(
            np.linalg.norm(found, ord=2, axis=0), np.ones(d)))

        V_torch = torch.rand((vocab, d), device=device, dtype=dtype)

        norm = torch.norm(V_torch, p=2, dim=1, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=1)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=1),
            torch.ones(vocab, device=device, dtype=dtype)
        ))


        norm = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=0)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=0),
            torch.ones(d, dtype=dtype, device=device)
        ))


    def test_norm(self):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        d = 300
        vocab = 5000

        V_numpy = np.random.random((vocab, d))

        expected = np.linalg.norm(V_numpy, ord=2, axis=0, keepdims=True)
        found = h.utils.norm(V_numpy, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = np.linalg.norm(V_numpy, ord=3, axis=1, keepdims=False)
        found = h.utils.norm(V_numpy, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))

        V_torch = torch.rand((vocab, d), device=device, dtype=dtype)

        expected = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        found = h.utils.norm(V_torch, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = torch.norm(V_torch, p=3, dim=1, keepdim=False)
        found = h.utils.norm(V_torch, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))


    def test_load_shard(self):

        shards = h.shards.Shards(5)
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE

        # Handles Numpy arrays properly.
        source = np.arange(100).reshape(10,10)
        expected = torch.tensor(
            [[0,5],[50,55]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0])
        self.assertTrue(torch.allclose(found, expected))

        # Handles Scipy CSR sparse matrices properly.
        source = sparse.random(10,10,0.3).tocsr()
        expected = torch.tensor(
            source.toarray()[shards[0]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0])
        self.assertTrue(torch.allclose(found, expected))

        # Handles Numpy matrices properly.
        source = np.matrix(range(100)).reshape(10,10)
        expected = torch.tensor(
            np.asarray(source)[shards[0]], device=device, dtype=dtype)
        found = h.utils.load_shard(source, shards[0])
        self.assertTrue(torch.allclose(found, expected))



class TestShards(TestCase):

    def test_shards_iteration(self):
        shard_factor = 4
        shards = h.shards.Shards(shard_factor)
        M = torch.arange(64, dtype=torch.float32).view(8,8)
        self.assertTrue(len(list(shards)))
        for i, shard in enumerate(shards):
            if i == 0:
                expected = torch.Tensor([[0,4],[32,36]])
                self.assertTrue(torch.allclose(M[shard], expected))
            elif i == 1:
                expected = torch.Tensor([[1,5],[33,37]])
                self.assertTrue(torch.allclose(M[shard], expected))
            else:
                expected_shard = torch.Tensor([
                    j for j in range(64)
                    # If the row is among allowed rows
                    if (j // 8) % shard_factor == i // shard_factor
                    # If the column is among alowed columns
                    and (j % 8) % shard_factor == i % shard_factor
                ]).view(2,2)
                self.assertTrue(torch.allclose(
                    M[shard], expected_shard
                ))






if __name__ == '__main__':

    if '--cpu' in sys.argv:
        print('\nTESTING DEVICE: CPU\n')
        sys.argv.remove('--cpu')
        h.CONSTANTS.MATRIX_DEVICE = 'cpu'
    else:
        print('\nTESTING DEVICE: CUDA.  Use --cpu to test on cpu.\n')

    main()
