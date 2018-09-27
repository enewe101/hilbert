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


class TestGetEmbedder(TestCase):

    def test_get_glove_embedder(self):

        cooc_stats = h.corpus_stats.get_test_stats(2)
        d=300
        learning_rate = 1e-6
        one_sided = False
        device = 'cpu'
        constrainer = h.constrainer.glove_constrainer
        X_max = 100.0
        base = 'logNxx'
        neg_inf_val=0
        solver='sgd'

        np.random.seed(0)
        torch.manual_seed(0)
        M = h.M.M(
            cooc_stats=cooc_stats, 
            base=base,
            #t_undersample=None,
            neg_inf_val=neg_inf_val,
            device=device,
        ).load_all()
        f_delta_str = 'glove'
        f_delta = h.f_delta.get_f_glove(
            cooc_stats=cooc_stats,
            M=M,
            X_max=X_max,
            device=device,
        )
        expected_embedder = h.torch_embedder.TorchHilbertEmbedder(
            M=M,
            f_delta=f_delta,
            d=d,
            learning_rate=learning_rate,
            one_sided=one_sided,
            constrainer=constrainer,
            device=device,
        )

        np.random.seed(0)
        torch.manual_seed(0)
        found_embedder = h.embedder.get_embedder(
            cooc_stats=cooc_stats,
            f_delta=f_delta_str,
            base=base,
            solver=solver,
            X_max=X_max,

            # vvv Defaults
            k=None,
            t_undersample=None,
            shift_by=None,
            # ^^^ Defaults

            neg_inf_val=neg_inf_val,

            # vvv Defaults
            clip_thresh=None,
            diag=None,
            k_samples=1,
            k_weight=None,
            alpha=1.0,
            # ^^^ Defaults

            d=d,
            learning_rate=learning_rate,
            one_sided=one_sided,
            constrainer=constrainer,

            # vvv Defaults
            momentum_decay=0.9,
            implementation='torch',
            # ^^^ Defaults

            device=device
        )

        expected_embedder.cycle(times=10, print_badness=False)
        found_embedder.cycle(times=10, print_badness=False)

        self.assertTrue(torch.allclose(expected_embedder.V, found_embedder.V))
        self.assertTrue(expected_embedder.badness, found_embedder.badness)



class TestCorpusStats(TestCase):

    UNIQUE_TOKENS = {
    '.': 5, 'Drive': 3, 'Eat': 7, 'The': 10, 'bread': 0, 'car': 6,
    'has': 8, 'sandwich': 9, 'spin': 4, 'the': 1, 'wheels': 2
    }
    N_XX_2 = np.array([
		[0, 12, 23, 8, 8, 8, 8, 12, 7, 4, 4], 
		[12, 0, 0, 8, 8, 0, 8, 8, 0, 0, 4], 
		[23, 0, 0, 4, 0, 4, 4, 0, 8, 4, 0], 
		[8, 8, 4, 0, 4, 4, 0, 0, 4, 0, 0], 
		[8, 8, 0, 4, 0, 4, 4, 4, 0, 0, 0], 
		[8, 0, 4, 4, 4, 0, 0, 0, 11, 1, 0], 
		[8, 8, 4, 0, 4, 0, 0, 4, 0, 4, 0], 
		[12, 8, 0, 0, 4, 0, 4, 0, 0, 0, 4], 
		[7, 0, 8, 4, 0, 11, 0, 0, 0, 0, 0], 
		[4, 0, 4, 0, 0, 1, 4, 0, 0, 0, 3], 
		[4, 4, 0, 0, 0, 0, 0, 4, 0, 3, 0]
    ]) 

    N_XX_3 = np.array([
        [0, 16, 23, 16, 12, 16, 15, 12, 15, 8, 8],
        [16, 0, 8, 12, 4, 8, 8, 12, 0, 0, 4],
        [23, 8, 0, 0, 12, 4, 4, 0, 11, 5, 3],
        [16, 12, 0, 0, 4, 4, 4, 4, 4, 0, 0],
        [12, 4, 12, 4, 0, 0, 4, 0, 11, 1, 0],
        [16, 8, 4, 4, 0, 8, 0, 4, 0, 4, 0],
        [15, 8, 4, 4, 4, 0, 8, 0, 4, 0, 0],
        [12, 12, 0, 4, 0, 4, 0, 8, 0, 3, 4],
        [15, 0, 11, 4, 11, 0, 4, 0, 0, 0, 0],
        [8, 0, 5, 0, 1, 4, 0, 3, 0, 0, 3],
        [8, 4, 3, 0, 0, 0, 0, 4, 0, 3, 0],
    ])


    def test_PMI(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        expected_PMI = np.load('test-data/expected_PMI.npz')['arr_0']
        found_PMI = h.corpus_stats.calc_PMI(cooc_stats)
        self.assertTrue(np.allclose(found_PMI, expected_PMI))

    def test_sparse_PMI(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        expected_PMI = np.load('test-data/expected_PMI.npz')['arr_0']
        # PMI sparse treats all negative infinite values as zero
        expected_PMI[expected_PMI==-np.inf] = 0
        pmi_data, I, J = h.corpus_stats.calc_PMI_sparse(cooc_stats)
        self.assertTrue(len(pmi_data) < np.product(cooc_stats.Nxx.shape))
        found_PMI = sparse.coo_matrix((pmi_data,(I,J)),cooc_stats.Nxx.shape)
        self.assertTrue(np.allclose(found_PMI.toarray(), expected_PMI))


    def test_calc_positive_PMI(self):
        expected_positive_PMI = np.load('test-data/expected_PMI.npz')['arr_0']
        expected_positive_PMI[expected_positive_PMI < 0] = 0
        cooc_stats = h.corpus_stats.get_test_stats(2)
        found_positive_PMI = h.corpus_stats.calc_positive_PMI(cooc_stats)
        self.assertTrue(np.allclose(found_positive_PMI, expected_positive_PMI))


    def test_calc_shifted_PMI(self):
        k = 15.0
        cooc_stats = h.corpus_stats.get_test_stats(2)
        expected_PMI_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'expected_PMI.npz')
        expected_PMI = np.load(expected_PMI_path)['arr_0']
        expected_shifted_PMI = expected_PMI - np.log(k)
        found = h.corpus_stats.calc_shifted_PMI(
            cooc_stats, torch.tensor(k, device='cuda'))
        self.assertTrue(np.allclose(found, expected_shifted_PMI))


    def test_PMI_star(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        expected_PMI_star_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'expected_PMI_star.npz')
        expected_PMI_star = np.load(expected_PMI_star_path)['arr_0']
        found_PMI_star = h.corpus_stats.calc_PMI_star(cooc_stats)
        self.assertTrue(np.allclose(found_PMI_star, expected_PMI_star))


    def test_get_stats(self):
        # Next, test with a cooccurrence window of +/-2
        cooc_stats = h.corpus_stats.get_test_stats(2)
        self.assertTrue(np.allclose(cooc_stats.denseNxx,self.N_XX_2))

        # Next, test with a cooccurrence window of +/-3
        cooc_stats = h.corpus_stats.get_test_stats(3)
        self.assertTrue(np.allclose(cooc_stats.denseNxx,self.N_XX_3))




class TestM(TestCase):

    def test_undersample(self):

        # Fix the seed for a replicable test.  In this case, numpy is the
        # source of randomness.
        np.random.seed(0)

        # Setup
        t_undersample = 0.1
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats
        Nx_ = Nx.reshape(-1)

        # Manually calculate probability that a token is kept.
        p_xx = torch.tensor([
            [
                min(1, np.sqrt(t_undersample * N / Nx_[i])) 
                    * min(1, np.sqrt(t_undersample * N / Nx_[j]))
                for i in range(Nx_.shape[0])
            ]
            for j in range(Nx_.shape[0])
        ], device=h.CONSTANTS.MATRIX_DEVICE)


        # Expectation counts given undersampling
        expectedNxx = Nxx * p_xx
        expectedNx = torch.sum(expectedNxx, dim=1, keepdim=True)
        expectedNxt = torch.sum(expectedNxx, dim=0, keepdim=True)
        expectedN = torch.sum(expectedNx)

        # Do the undersampling many times.  Then we can expect the average
        # count to become close to expectation
        foundNxx = torch.zeros(Nxx.shape, device=h.CONSTANTS.MATRIX_DEVICE)
        foundNx = torch.zeros(Nx.shape, device=h.CONSTANTS.MATRIX_DEVICE)
        foundNxt = torch.zeros(Nx.t().shape, device=h.CONSTANTS.MATRIX_DEVICE)
        foundN = torch.zeros(N.shape, device=h.CONSTANTS.MATRIX_DEVICE)
        num_replicates = 100
        for i in range(num_replicates):
            cooc_stats_undersamp = h.M.undersample(cooc_stats, t_undersample)
            one_Nxx, one_Nx, one_Nxt, one_N = cooc_stats_undersamp
            foundNxx += one_Nxx
            foundNx += one_Nx
            foundNxt += one_Nxt
            foundN += one_N
        foundNxx /= num_replicates
        foundNx /= num_replicates
        foundNxt /= num_replicates
        foundN /= num_replicates

        self.assertTrue(np.allclose(foundNxx, expectedNxx, atol=0.5))
        self.assertTrue(np.allclose(foundNx, expectedNx, atol=0.7))
        self.assertTrue(np.allclose(foundNxt, expectedNxt, atol=0.7))
        self.assertTrue(np.allclose(foundN, expectedN, atol=1.5))

        # Look more closely at the last instance returned by undersamp
        # It should be a CoocStats instance, and it's Nxx array should be
        # a sparse csr matrix with dtype int64.
        self.assertTrue(isinstance(
            cooc_stats_undersamp, h.cooc_stats.CoocStats))
        self.assertTrue(isinstance(
            cooc_stats_undersamp.Nxx, sparse.csr_matrix))
        self.assertEqual(cooc_stats_undersamp.Nxx.dtype, np.dtype('int64'))

        



    def test_calc_M_pmi(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats

        # First calculate using no options
        found_M = h.M.M(cooc_stats, 'pmi').load_all()

        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        self.assertTrue(np.allclose(found_M, expected_M))

        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE, 
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N)) + shift_by
        found_M = h.M.M(cooc_stats, 'pmi', shift_by=shift_by).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))


        # Test undersample option.  Fix randomness so that samples come out
        # the same when calling undersample directly, and when called within
        # the routine being tested
        t_undersample = 0.1
        np.random.seed(0)
        undersamp_cooc_stats = h.M.undersample(cooc_stats, t_undersample)
        expected_M = h.corpus_stats.calc_PMI(undersamp_cooc_stats)
        np.random.seed(0)
        found_M = h.M.M(
            cooc_stats, 'pmi', t_undersample=t_undersample).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI(cooc_stats)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M(cooc_stats, 'pmi', clip_thresh=clip_thresh).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        diag = 5
        expected_M = h.corpus_stats.calc_PMI(cooc_stats)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M(cooc_stats, 'pmi', diag=diag).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))


    def test_calc_M_logNxx(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats

        # First calculate using no options.
        M = h.M.M(cooc_stats, 'logNxx').load_all()
        expected_M = torch.log(Nxx)
        self.assertTrue(torch.allclose(M, expected_M))

        # Test shift option.
        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE, 
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = torch.log(Nxx) + shift_by
        found_M = h.M.M(cooc_stats, 'logNxx', shift_by=shift_by).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))

        # Test undersample option.  Fix randomness so that samples come out
        # the same when calling undersample directly, and when called within
        # the routine being tested
        t_undersample = 0.1
        np.random.seed(0)
        usampNxx, usampNx, usampNxt, usampN = h.M.undersample(
            cooc_stats, t_undersample)
        expected_M = torch.log(usampNxx)
        np.random.seed(0)
        found_M = h.M.M(
            cooc_stats, 'logNxx', t_undersample=t_undersample).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))

        # Test setting a clip threshold.
        clip_thresh = -0.1
        expected_M = torch.log(Nxx)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M(
            cooc_stats, 'logNxx', clip_thresh=clip_thresh).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))

        # Test setting diagonal values to a given constant.
        diag = 5
        expected_M = torch.log(Nxx)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M(cooc_stats, 'logNxx', diag=diag).load_all()
        self.assertTrue(torch.allclose(found_M, expected_M))


    def test_calc_M_pmi_star(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats

        # First calculate using no options
        M = h.M.M(cooc_stats, 'pmi-star').load_all()
        expected_M = h.corpus_stats.calc_PMI_star(cooc_stats)
        self.assertTrue(np.allclose(M, expected_M))

        # Test shift option.
        shift_by = -torch.log(torch.tensor(
            15, dtype=h.CONSTANTS.DEFAULT_DTYPE, 
            device=h.CONSTANTS.MATRIX_DEVICE
        ))
        expected_M = h.corpus_stats.calc_PMI_star(cooc_stats) + shift_by
        found_M = h.M.M(cooc_stats, 'pmi-star', shift_by=shift_by).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        # Test undersample option.  Fix randomness so that samples come out
        # the same when calling undersample directly, and when called within
        # the routine being tested
        t_undersample = 0.1
        np.random.seed(0)
        usamp_cooc_stats = h.M.undersample(cooc_stats, t_undersample)
        expected_M = h.corpus_stats.calc_PMI_star(usamp_cooc_stats)
        np.random.seed(0)
        found_M = h.M.M(
            cooc_stats, 'pmi-star', t_undersample=t_undersample).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        # Test setting a clip threshold.
        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI_star(cooc_stats)
        expected_M[expected_M<clip_thresh] = clip_thresh
        found_M = h.M.M(
            cooc_stats, 'pmi-star', clip_thresh=clip_thresh).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))

        # Test setting diagonal values to a given constant.
        diag = 5
        expected_M = h.corpus_stats.calc_PMI_star(cooc_stats)
        h.utils.fill_diagonal(expected_M, diag)
        found_M = h.M.M(cooc_stats, 'pmi-star', diag=diag).load_all()
        self.assertTrue(np.allclose(found_M, expected_M))


    def test_sharding(self):

        cooc_stats = h.corpus_stats.get_test_stats(2)
        cooc_stats.truncate(6)
        Nxx, Nx, Nxt, N = cooc_stats

        # First calculate using no options
        shards = h.shards.Shards(2)
        M = h.M.M(cooc_stats, 'pmi')
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

        M = h.M.M(cooc_stats, 'pmi', shift_by=shift_by).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]

        self.assertTrue(np.allclose(found_M, expected_M))


        # Test undersample option.  Fix randomness so that samples come out
        # the same when calling undersample directly, and when called within
        # the routine being tested
        t_undersample = 0.1
        np.random.seed(0)
        undersamp_cooc_stats = h.M.undersample(cooc_stats, t_undersample)
        expected_M = h.corpus_stats.calc_PMI(undersamp_cooc_stats)
        np.random.seed(0)
        M = h.M.M(cooc_stats, 'pmi', t_undersample=t_undersample).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]
        self.assertTrue(np.allclose(found_M, expected_M))

        clip_thresh = -0.1
        expected_M = h.corpus_stats.calc_PMI(cooc_stats)
        expected_M[expected_M<clip_thresh] = clip_thresh
        M = h.M.M(cooc_stats, 'pmi', clip_thresh=clip_thresh).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]
        self.assertTrue(np.allclose(found_M, expected_M))

        diag = 5
        expected_M = h.corpus_stats.calc_PMI(cooc_stats)
        h.utils.fill_diagonal(expected_M, diag)
        M = h.M.M(cooc_stats, 'pmi', diag=diag).load_all()
        found_M = np.zeros(Nxx.shape)
        for shard_num, shard in enumerate(shards):
            found_M[shard] = M[shard]
        self.assertTrue(np.allclose(found_M, expected_M))



    ####################################################################
    #
    #   Deactivate all these tests until I find an approach with good enough
    #   performance for this to even be worth it.
    #
    #def test_sample_multi_multinomial(self):
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    # Set k really high, so that sample statistics approach their
    #    # limiting values.
    #    k = 150000
    #    np.random.seed(0)
    #    Nxx, Nx, Nxt, N = cooc_stats

    #    kNx = k * Nx
    #    px = Nx / N
    #    sample = h.M._sample_multi_multinomial(kNx, px)

    #    # The number of samples in each row is exactly equal to the unigram
    #    # frequency of the corresponding token, times k
    #    self.assertTrue(np.allclose(
    #        np.sum(sample, axis=1) / float(k), Nx.reshape(-1)))

    #    # Given the very high value of k, the number of samples in each column
    #    # is approximately equal to the unigram frequency of the corresponding
    #    # token, times k.
    #    self.assertTrue(np.allclose(
    #        np.sum(sample, axis=0) / float(k), Nx.reshape(-1), atol=0.1))


    #def test_sample_multi_multinomial_torch(self):
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    # Set k really high, so that sample statistics approach their
    #    # limiting values.
    #    k = 150000
    #    torch.random.manual_seed(0)
    #    Nxx, Nx, Nxt, N = cooc_stats

    #    kNx = k * Nx
    #    px = Nx / N
    #    sample = h.M._sample_multi_multinomial_torch(kNx, px)

    #    # The number of samples in each row is exactly equal to the unigram
    #    # frequency of the corresponding token, times k
    #    self.assertTrue(torch.allclose(
    #        torch.sum(sample, dim=1) / float(k), Nx.view(-1)))

    #    # Given the very high value of k, the number of samples in each column
    #    # is approximately equal to the unigram frequency of the corresponding
    #    # token, times k.
    #    self.assertTrue(np.allclose(
    #        torch.sum(sample, dim=0) / float(k), Nx.view(-1), atol=0.1))



    #def test_calc_M_neg_samp(self):

    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    Nxx, Nx, Nxt, N = cooc_stats
    #    np.random.seed(0)
    #    atol = 0.1
    #    usamp_atol = 0.3
    #    k_samples = 1000

    #    # If we take enough samples, then negative sampling simulates PMI
    #    k_weight = 1.
    #    alpha = 1.
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, 
    #        k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        device='cpu'
    #    )
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats)
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))

    #    # Now try using a k_weight not equal to 1.
    #    k_weight = 15.
    #    alpha = 1.
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        device='cpu'
    #    )
    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats) - np.log(k_weight)
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))

    #    # Now we will use an alpha value not equal to 1
    #    k_weight = 15.
    #    alpha = 0.75
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        device='cpu'
    #    )
    #    distorted_unigram = Nx**alpha
    #    distorted_unigram = distorted_unigram / np.sum(distorted_unigram)
    #    with np.errstate(divide='ignore'):
    #        expected_M = (
    #            np.log(Nxx) - np.log(Nx) - np.log(distorted_unigram.T) 
    #            - np.log(k_weight)
    #        )
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))

    #    # Test shift option.
    #    shift_by = -np.log(15)
    #    k_weight = 1.
    #    alpha = 1.
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        shift_by=-np.log(15), device='cpu'
    #    )
    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats) - np.log(15)
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))


    #    # If we take enough samples, then negative sampling simulates PMI
    #    k_weight = 1.
    #    alpha = 1.
    #    t_undersample = 0.1
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, 
    #        k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        t_undersample=t_undersample,
    #        device='cpu'
    #    )
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    Nxx, Nx, Nxt, N = h.M.undersample(cooc_stats, t_undersample)
    #    expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=usamp_atol))


    #    # Test setting a clip threshold.
    #    clip_thresh = -0.1
    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats)
    #    expected_M[expected_M<clip_thresh] = clip_thresh
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        clip_thresh=clip_thresh, device='cpu'
    #    )
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))


    #    # Test setting diagonal values to a given constant.
    #    diag = 5
    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats)
    #    np.fill_diagonal(expected_M, diag)
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, k_samples=k_samples, k_weight=k_weight, alpha=alpha,
    #        diag=diag, device='cpu'
    #    )
    #    self.assertTrue(np.allclose(found_M, expected_M, atol=atol))


    #    # Test explicitly choosing implementation.
    #    implementation='numpy'
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, implementation=implementation)
    #    self.assertTrue(isinstance(found_M, np.ndarray))


    #    # Test explicitly choosing implementation.
    #    implementation='torch'
    #    device='cpu'
    #    found_M = h.M.calc_M_neg_samp(
    #        cooc_stats, implementation=implementation, device=device)
    #    self.assertTrue(isinstance(found_M, torch.Tensor))
    #    self.assertEqual(str(found_M.device), 'cpu')


    ########################################################
    #
    #   I'm not supporting this method anymore.  Use the class M.
    #
    #def test_calc_M(self):

    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    Nxx, Nx, Nxt, N = cooc_stats
    #    t_undersample=0.1
    #    shift_by=-np.log(15)
    #    neg_inf_val=1
    #    clip_thresh=-1
    #    diag=2
    #    implementation='torch'
    #    device='cpu'

    #    M = h.M.calc_M_pmi(
    #        cooc_stats,
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    M_ = h.M.calc_M(
    #        cooc_stats,
    #        base='pmi',
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    self.assertTrue(torch.allclose(M, M_))

    #    M = h.M.calc_M_logNxx(
    #        cooc_stats,
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    M_ = h.M.calc_M(
    #        cooc_stats,
    #        base='logNxx',
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    self.assertTrue(torch.allclose(M, M_))

    #    M = h.M.calc_M_pmi_star(
    #        cooc_stats,
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    M_ = h.M.calc_M(
    #        cooc_stats,
    #        base='pmi-star',
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    self.assertTrue(torch.allclose(M, M_))

    #    M = h.M.calc_M_neg_samp(
    #        cooc_stats,
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    M_ = h.M.calc_M(
    #        cooc_stats,
    #        base='neg-samp',
    #        t_undersample=t_undersample,
    #        shift_by=shift_by,
    #        neg_inf_val=neg_inf_val,
    #        clip_thresh=clip_thresh,
    #        diag=diag,
    #        implementation=implementation,
    #        device=device
    #    )
    #    self.assertTrue(torch.allclose(M, M_))



class TestFDeltas(TestCase):


    def test_sigmoid(self):
        # This should work for np.array and torch.Tensor
        cooc_stats = h.corpus_stats.get_test_stats(2)
        PMI = h.corpus_stats.calc_PMI(cooc_stats)
        expected = np.array([
            [1/(1+np.e**(-pmi)) for pmi in row]
            for row in PMI
        ])
        result = h.f_delta.sigmoid(PMI)
        self.assertTrue(np.allclose(expected, result))

        PMI = torch.tensor(PMI, dtype=torch.float32)
        expected = torch.tensor([
            [1/(1+np.e**(-pmi)) for pmi in row]
            for row in PMI
        ], dtype=torch.float32)
        result = h.f_delta.sigmoid(PMI)
        self.assertTrue(torch.allclose(expected, result))


    def test_N_neg(self):
        k = 15.0
        cooc_stats = h.corpus_stats.get_test_stats(2)
        expected = k * cooc_stats.Nx * cooc_stats.Nx.T / cooc_stats.N
        found = h.f_delta.calc_N_neg(cooc_stats, k)
        self.assertTrue(np.allclose(expected, found))


    def test_f_w2v(self):
        k = 15
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats
        expected_M = h.corpus_stats.calc_PMI(cooc_stats) - np.log(k)
        expected_M_hat = expected_M + 1
        N_neg = h.f_delta.calc_N_neg(cooc_stats, k)
        expected_difference = (
            h.f_delta.sigmoid(expected_M) - h.f_delta.sigmoid(expected_M_hat))
        expected_multiplier = N_neg + Nxx
        expected = expected_multiplier * expected_difference

        M = h.M.M(cooc_stats, 'pmi', shift_by=-np.log(k))
        M_ = M.load_all()
        M_hat = M_ + 1

        delta_w2v = h.f_delta.DeltaW2V(cooc_stats, M, k)
        found = torch.zeros(cooc_stats.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_w2v.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(expected, found))


    #
    #   Now that np is deprecated, this test is redundnat.  We *only* test
    #   torch.
    #
    #def test_f_w2v_torch(self):

    #    k = 15
    #    device = 'cuda'
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    Nxx, Nx, Nxt, N = cooc_stats

    #    expected_M = h.corpus_stats.calc_PMI(cooc_stats) - np.log(k)
    #    expected_M_hat = expected_M + 1
    #    N_neg = h.f_delta.calc_N_neg(cooc_stats, k)
    #    expected_difference = (
    #        h.f_delta.sigmoid(expected_M) - h.f_delta.sigmoid(expected_M_hat))
    #    expected_multiplier = N_neg + Nxx
    #    expected = torch.tensor(
    #        expected_multiplier * expected_difference,
    #        dtype=torch.float32, device=device
    #    )

    #    M = h.M.M(cooc_stats,'pmi', shift_by=-np.log(k), device=device)
    #    M_ = M.load_all()
    #    M_hat = M_ + 1
    #    delta_w2v = h.f_delta.DeltaW2V(cooc_stats, M, k, device=device)
    #    found = torch.zeros(cooc_stats.Nxx.shape)
    #    shards = h.shards.Shards(2)
    #    for shard in shards:
    #        found[shard] = delta_w2v(M_hat[shard], shard)

    #    self.assertTrue(torch.allclose(expected, found))



    def test_f_glove(self):

        cooc_stats = h.corpus_stats.get_test_stats(2)
        cooc_stats.truncate(10)

        Nxx, Nx, Nxt, N = cooc_stats
        expected_M = torch.log(Nxx)
        # Zero out cells containing negative infinity, which are ignored
        # by glove.  We still need to zero them out to avoid nans.
        expected_M[expected_M==-np.inf] = 0
        expected_M_hat = expected_M + 1
        multiplier = torch.tensor([[
                2 * min(1, (cooc_stats.Nxx[i,j] / 100.0)**0.75) 
                for j in range(cooc_stats.Nxx.shape[1])
            ] for i in range(cooc_stats.Nxx.shape[0])
        ])
        difference = torch.tensor([[
                expected_M[i,j] - expected_M_hat[i,j]
                if cooc_stats.Nxx[i,j] > 0 else 0 
                for j in range(cooc_stats.Nxx.shape[1])
            ] for i in range(cooc_stats.Nxx.shape[0])
        ])
        expected = multiplier * difference

        M = h.M.M(cooc_stats, 'logNxx', neg_inf_val=0)
        M_ = M.load_all()
        M_hat = M_ + 1
        delta_glove = h.f_delta.DeltaGlove(cooc_stats, M)
        found = torch.zeros(cooc_stats.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_glove.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(expected, found))

        # Try varying the X_max and alpha settings.
        alpha = 0.8
        X_max = 10
        delta_glove = h.f_delta.DeltaGlove(
            cooc_stats, M, X_max=X_max, alpha=alpha)
        found2 = torch.zeros(cooc_stats.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found2[shard] = delta_glove.calc_shard(M_hat[shard], shard)
        expected2 = torch.tensor([
            [
                2 * min(1, (cooc_stats.Nxx[i,j] / X_max)**alpha) 
                    * (expected_M[i,j] - expected_M_hat[i,j])
                if cooc_stats.Nxx[i,j] > 0 else 0 
                for j in range(cooc_stats.Nxx.shape[1])
            ]
            for i in range(cooc_stats.Nxx.shape[0])
        ])
        # The X_max setting has an effect, and matches a different expectation
        self.assertTrue(np.allclose(expected2, found2))
        self.assertFalse(np.allclose(expected2, expected))


    #
    #   Now that np is deprecated, this test is redundnat.  We *only* test
    #   torch.
    #
    #def test_f_glove_torch(self):

    #    device = 'cuda'
    #    cooc_stats = h.corpus_stats.get_test_stats(2)

    #    with np.errstate(divide='ignore'):
    #        expected_M = np.log(cooc_stats.denseNxx)
    #    # Zero out cells containing negative infinity, which are ignored
    #    # by glove.  We still need to zero them out to avoid nans.
    #    expected_M[expected_M==-np.inf] = 0
    #    expected_M_hat = expected_M + 1
    #    expected = torch.tensor(np.array([
    #        [
    #            2 * min(1, (cooc_stats.Nxx[i,j] / 100.0)**0.75) 
    #                * (expected_M[i,j] - expected_M_hat[i,j])
    #            if cooc_stats.Nxx[i,j] > 0 else 0 
    #            for j in range(cooc_stats.Nxx.shape[1])
    #        ]
    #        for i in range(cooc_stats.Nxx.shape[0])
    #    ]), dtype=torch.float32, device=device)

    #    M = h.M.M(cooc_stats, 'logNxx', neg_inf_val=0).load_all()

    #    M_hat = M_hat = M + 1
    #    f_glove = h.f_delta.get_f_glove(
    #        cooc_stats, M, implementation='torch', device=device)
    #    found = f_glove(M_hat)
    #    self.assertTrue(np.allclose(expected, found))

    #    # Verify that x_max takes effect, by passing in a non-default value.
    #    x_max_non_default = 10
    #    f_glove = h.f_delta.get_f_glove(
    #        cooc_stats, M, x_max_non_default,
    #        implementation='torch', device=device
    #    )
    #    found2 = f_glove(M_hat)

    #    expected2 = torch.tensor(np.array([
    #        [
    #            2 * min(1, (cooc_stats.Nxx[i,j] / 10.0)**0.75) 
    #                * (expected_M[i,j] - expected_M_hat[i,j])
    #            if cooc_stats.Nxx[i,j] > 0 else 0 
    #            for j in range(cooc_stats.Nxx.shape[1])
    #        ]
    #        for i in range(cooc_stats.Nxx.shape[0])
    #    ]), dtype=torch.float32, device=device)

    #    # We should find what we expected.
    #    self.assertTrue(np.allclose(expected2, found2))
    #    # It should be different from last time (hence x_max takes effect.)
    #    self.assertFalse(np.allclose(expected2, expected))


    def test_f_MSE(self):

        cooc_stats = h.corpus_stats.get_test_stats(2)
        cooc_stats.truncate(10)  # Need a compound number for sharding

        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device='cpu')
        M_ = M.load_all()
        M_hat = M_ + 1
        expected = M_ - M_hat
        delta_mse = h.f_delta.DeltaMSE(cooc_stats, M, device='cpu')
        found = delta_mse.calc_shard(M_hat)
        self.assertTrue(torch.allclose(expected, found))

        shards = h.shards.Shards(5)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device='cpu')
        M_ = M.load_all()
        M_hat = M_ + 1
        expected = M_ - M_hat
        delta_mse = h.f_delta.DeltaMSE(cooc_stats, M, device='cpu')
        found = torch.zeros(cooc_stats.Nxx.shape)
        for shard in shards:
            found[shard] = delta_mse.calc_shard(M_hat[shard], shard)
        self.assertTrue(torch.allclose(expected, found))


    def test_f_swivel(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)

        M = h.M.M(cooc_stats, 'pmi-star')
        M_ = M.load_all()
        M_hat = M_ + 1

        expected = np.array([
            [
                np.sqrt(cooc_stats.Nxx[i,j]) * (M_[i,j] - M_hat[i,j]) 
                if cooc_stats.Nxx[i,j] > 0 else
                (np.e**(M_[i,j] - M_hat[i,j]) /
                    (1 + np.e**(M_[i,j] - M_hat[i,j])))
                for j in range(M_.shape[1])
            ]
            for i in range(M_.shape[0])
        ])

        delta_swivel = h.f_delta.DeltaSwivel(cooc_stats, M)
        found = torch.zeros(cooc_stats.Nxx.shape)
        shards = h.shards.Shards(5)
        for shard in shards:
            found[shard] = delta_swivel.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(found, expected))


    #
    #   Now that np is deprecated, this test is redundnat.  We *only* test
    #   torch.
    #
    #def test_f_swivel_torch(self):
    #    cooc_stats = h.corpus_stats.get_test_stats(2)
    #    device = 'cpu'
    #    M = h.M.M(cooc_stats, 'pmi-star', device=device).load_all()
    #    M_hat = M + 1
    #    expected = torch.tensor(np.array([
    #        [
    #            np.sqrt(cooc_stats.Nxx[i,j]) * (M[i,j] - M_hat[i,j]) 

    #            if cooc_stats.Nxx[i,j] > 0 else

    #            (np.e**(M[i,j] - M_hat[i,j]) /
    #                (1 + np.e**(M[i,j] - M_hat[i,j])))
    #            for j in range(M.shape[1])
    #        ]
    #        for i in range(M.shape[0])
    #    ]), dtype=torch.float32, device=device)
    #    f_swivel = h.f_delta.get_f_swivel(
    #        cooc_stats, M, implementation='torch', device=device)
    #    found = f_swivel(M_hat)
    #    self.assertTrue(torch.allclose(found, expected))
    #    self.assertTrue(isinstance(found, torch.Tensor))


    def test_f_MLE(self):

        cooc_stats = h.corpus_stats.get_test_stats(2)
        cooc_stats.truncate(10)
        Nxx, Nx, Nxt, N = cooc_stats

        expected_M = h.corpus_stats.calc_PMI(cooc_stats)
        expected_M_hat = expected_M + 1
        N_indep_xx = cooc_stats.Nx * cooc_stats.Nx.T
        N_indep_max = np.max(N_indep_xx)
        multiplier = N_indep_xx / N_indep_max
        difference = np.e**expected_M - np.e**expected_M_hat
        expected = multiplier * difference

        M = h.M.M(cooc_stats, 'pmi')
        M_ = M.load_all()
        M_hat = M_ + 1
        delta_mle = h.f_delta.DeltaMLE(cooc_stats, M)
        found = torch.zeros(Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_mle.calc_shard(M_hat[shard], shard)
        self.assertTrue(np.allclose(found, expected))

        # Now test with a different setting for temperature (t).
        t = 10
        expected = (N_indep_xx / N_indep_max)**(1.0/t) * (
            np.e**expected_M - np.e**expected_M_hat)

        found = torch.zeros(Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_mle.calc_shard(M_hat[shard], shard, t=t)
        self.assertTrue(np.allclose(found, expected))



    #
    #   Now that np is deprecated, this test is redundnat.  We *only* test
    #   torch.
    #
    #def test_torch_f_MLE(self):

    #    device = 'cpu'
    #    cooc_stats = h.corpus_stats.get_test_stats(2)

    #    M = torch.tensor(
    #        h.corpus_stats.calc_PMI(cooc_stats), dtype=torch.float32)
    #    M_hat = M + 1
    #    N_indep_xx = torch.tensor(
    #        cooc_stats.Nx * cooc_stats.Nx.T, dtype=torch.float32)
    #    N_indep_max = torch.max(N_indep_xx)
    #    expected = N_indep_xx / N_indep_max * (np.e**M - np.e**M_hat)

    #    delta = np.zeros(M.shape)
    #    f_MLE = h.f_delta.get_f_MLE(cooc_stats, M, device=device)
    #    found = f_MLE(M_hat)

    #    self.assertTrue(np.allclose(found, expected))

    #    t = 10
    #    expected = (N_indep_xx / N_indep_max)**(1.0/t) * (
    #        np.e**M - np.e**M_hat)
    #    found = f_MLE(M_hat, t=t)
    #    self.assertTrue(np.allclose(found, expected))




class TestConstrainer(TestCase):

    def test_glove_constrainer(self):
        W, V = np.zeros((3,3)), np.zeros((3,3))
        h.constrainer.glove_constrainer(W, V)
        self.assertTrue(np.allclose(W, np.array([[0,1,0]]*3)))
        self.assertTrue(np.allclose(V, np.array([[1,0,0]]*3).T))



class TestHilbertEmbedder(TestCase):

    def test_integration_with_constrainer(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate,
            constrainer=h.constrainer.glove_constrainer
        )

        old_V = embedder.V.copy()
        old_W = embedder.W.copy()

        embedder.cycle(print_badness=False)

        # Check that the update was performed, and constraints applied.
        new_V = old_V + learning_rate * np.dot(old_W.T, M - embedder.M_hat)
        new_W = old_W + learning_rate * np.dot(M - embedder.M_hat, old_V.T)
        h.constrainer.glove_constrainer(new_W, new_V)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_W, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)



    def test_get_gradient(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate)

        W, V = embedder.W.copy(), embedder.V.copy()
        M_hat = np.dot(W,V)
        delta = M - M_hat
        expected_nabla_W = np.dot(delta, V.T)
        expected_nabla_V = np.dot(W.T, delta)

        nabla_V, nabla_W = embedder.get_gradient()

        self.assertTrue(np.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))


    def test_get_gradient_with_offsets(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        offset_W = np.random.random(cooc_stats.Nxx.shape)
        offset_V = np.random.random(cooc_stats.Nxx.shape)
        
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(M, f_MSE, d, learning_rate)

        original_W, original_V = embedder.W.copy(), embedder.V.copy()
        W, V =  original_W + offset_W,  original_V + offset_V
        M_hat = np.dot(W,V)
        delta = M - M_hat
        expected_nabla_W = np.dot(delta, V.T)
        expected_nabla_V = np.dot(W.T, delta)

        nabla_V, nabla_W = embedder.get_gradient(offsets=(offset_V, offset_W))

        self.assertTrue(np.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_W, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())
        
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True)

        original_V = embedder.V.copy()
        V =  original_V
        M_hat = np.dot(V.T,V)
        delta = M - M_hat
        expected_nabla_V = np.dot(V, delta)

        nabla_V = embedder.get_gradient()

        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_V.T, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided_with_offset(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())
        offset_V = np.random.random(cooc_stats.Nxx.shape)

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True)

        original_V = embedder.V.copy()
        V =  original_V + offset_V
        M_hat = np.dot(V.T,V)
        delta = M - M_hat
        expected_nabla_V = np.dot(V, delta)

        nabla_V = embedder.get_gradient(offsets=offset_V)

        self.assertTrue(np.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(np.allclose(original_V.T, embedder.W))
        self.assertTrue(np.allclose(original_V, embedder.V))



    def test_integration_with_f_delta(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())
        pass_args = {'a':True, 'b':False}

        def get_mock_f_delta(cooc_stats, M_):
            def mock_f_delta(M_hat_, **kwargs):
                self.assertTrue(M_ is M)
                self.assertEqual(kwargs, {'a':True, 'b':False})
                return M_ - M_hat_
            return mock_f_delta

        f_delta = get_mock_f_delta(cooc_stats, M)
        embedder = h.embedder.HilbertEmbedder(M, f_delta, d, learning_rate)

        self.assertEqual(embedder.learning_rate, learning_rate)
        self.assertEqual(embedder.d, d)
        self.assertTrue(embedder.M is M)
        self.assertEqual(embedder.f_delta, f_delta)

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        embedder.cycle(pass_args=pass_args, print_badness=False)

        # Check that the update was performed
        new_V = old_V + learning_rate * np.dot(old_W.T, embedder.delta)
        new_W = old_W + learning_rate * np.dot(embedder.delta, old_V.T)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_W, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)


    def test_arbitrary_f_delta(self):
        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        # Define an arbitrary f_delta
        delta_amount = 0.1
        delta_always = np.zeros(M.shape) + delta_amount
        def get_f_delta(cooc_stats, M, implementation):
            def f_delta(M_hat):
                return np.ones(M_hat.shape) * delta_amount
            return f_delta

        f_delta = get_f_delta(cooc_stats, M, implementation='numpy')
        # First make a non-one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(M, f_delta, d, learning_rate)

        old_V = embedder.V.copy()
        old_W = embedder.W.copy()

        embedder.cycle(print_badness=False)

        # Check that the update was performed.  Notice that the update of W
        # uses the old value of V, hence a synchronous update.
        new_V = old_V + learning_rate * np.dot(old_W.T, delta_always)
        new_W = old_W + learning_rate * np.dot(delta_always, old_V.T)
        self.assertTrue(np.allclose(embedder.V, new_V))
        self.assertTrue(np.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(delta_always) / (d*d)
        self.assertEqual(badness, embedder.badness)


    def test_one_sided(self):
        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        # First make a non-one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate
        )

        # The covectors and vectors are not the same.
        self.assertFalse(np.allclose(embedder.W, embedder.V.T))

        # Now make a one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True
        )

        # The covectors and vectors are the same.
        self.assertTrue(np.allclose(embedder.W, embedder.V.T))

        old_V = embedder.V.copy()
        embedder.cycle(print_badness=False)

        # Check that the update was performed.
        new_V = old_V + learning_rate * np.dot(old_V, embedder.delta)
        self.assertTrue(np.allclose(embedder.V, new_V))

        # Check that the vectors and covectors are still identical after the
        # update.
        self.assertTrue(np.allclose(embedder.W, embedder.V.T))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        embedder.calc_badness()
        badness = np.sum(abs(M - np.dot(old_V.T, old_V))) / (d*d)
        self.assertEqual(badness, embedder.badness)


    def test_mse_embedder(self):
        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', clip_thresh=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        mse_embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate)
        mse_embedder.cycle(100000, print_badness=False)

        self.assertEqual(mse_embedder.V.shape, (M.shape[1],d))
        self.assertEqual(mse_embedder.W.shape, (d,M.shape[0]))

        delta = np.zeros(M.shape, dtype='float64')
        residual = f_MSE(mse_embedder.M_hat)

        self.assertTrue(np.allclose(residual, delta))
        

    def test_update(self):

        d = 11
        learning_rate = 0.01
        cooc_stats= h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', neg_inf_val=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate)

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        delta_V = np.random.random(M.shape)
        delta_W = np.random.random(M.shape)
        updates = delta_V, delta_W
        embedder.update(*updates)
        self.assertTrue(np.allclose(old_W + delta_W, embedder.W))
        self.assertTrue(np.allclose(old_V + delta_V, embedder.V))


    def test_update_with_constraints(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', neg_inf_val=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate,
            constrainer=h.constrainer.glove_constrainer
        )

        old_W, old_V = embedder.W.copy(), embedder.V.copy()

        delta_V = np.random.random(M.shape)
        delta_W = np.random.random(M.shape)
        updates = delta_V, delta_W
        embedder.update(*updates)

        expected_updated_W = old_W + delta_W
        expected_updated_V = old_V + delta_V
        h.constrainer.glove_constrainer(expected_updated_W, expected_updated_V)

        self.assertTrue(np.allclose(expected_updated_W, embedder.W))
        self.assertTrue(np.allclose(expected_updated_V, embedder.V))



    def test_update_one_sided_rejects_delta_W(self):

        d = 11
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi', neg_inf_val=0).load_all())

        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate)

        # Show that we can update covector embeddings for a non-one-sided model
        delta_W = np.ones(M.shape)
        embedder.update(delta_W=delta_W)

        # Now show that a one-sided embedder rejects updates to covectors
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True)
        delta_W = np.ones(M.shape)
        with self.assertRaises(ValueError):
            embedder.update(delta_W=delta_W)



class TestTorchHilbertEmbedder(TestCase):


    def test_one_sided(self):
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)

        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device='cpu').load_all()

        # First make a non-one-sided embedder.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))
        self.assertTrue(isinstance(embedder.M, torch.Tensor))

        # The covectors and vectors are not the same.
        self.assertFalse(torch.allclose(embedder.W, embedder.V.t()))

        # Now make a one-sided embedder.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True, device=device
        )

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))
        self.assertTrue(isinstance(embedder.M, torch.Tensor))

        # Now, the covectors and vectors are the same.
        self.assertTrue(torch.allclose(embedder.W, embedder.V.t()))

        old_V = embedder.V.clone()
        embedder.cycle(print_badness=False)

        self.assertTrue(isinstance(old_V, torch.Tensor))

        # Check that the update was performed.
        M_hat = torch.mm(old_V.t(), old_V)
        M = torch.tensor(M, dtype=torch.float32)
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        delta = f_MSE(M_hat)
        nabla_V = torch.mm(old_V, delta)
        new_V = old_V + learning_rate * nabla_V
        self.assertTrue(torch.allclose(embedder.V, new_V))

        # Check that the vectors and covectors are still identical after the
        # update.
        self.assertTrue(torch.allclose(embedder.W, embedder.V.t()))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        delta = abs(M - M_hat)
        badness = torch.sum(delta) / (d*d)
        self.assertEqual(badness, embedder.badness)



    def test_get_gradient(self):

        # Set up conditions for the test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        # Make the embedder, whose method we are testing.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Take the random starting embeddings.  We will compute the gradient
        # manually here to see if it matches what the embedder's method
        # returns.
        W, V = embedder.W.clone(), embedder.V.clone()

        # Since we are not doing one-sided, W and V should be unrelated.
        self.assertFalse(torch.allclose(W.t(), V))

        # Calculate the expected gradient.
        M_hat = torch.mm(W,V)
        delta = M - M_hat
        expected_nabla_W = torch.mm(delta, V.t())
        expected_nabla_V = torch.mm(W.t(), delta)

        # Get the gradient according to the embedder.
        nabla_V, nabla_W = embedder.get_gradient()

        # Embedder's gradients should match manually calculated expectation.
        self.assertTrue(torch.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))


    def test_get_gradient_with_offsets(self):

        # Set up conditions for the test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        offset_W = torch.rand(cooc_stats.Nxx.shape)
        offset_V = torch.rand(cooc_stats.Nxx.shape)

        # Create an embedder, whose get_gradient method we are testing.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Manually calculate the gradients we expect, applying offsets to the
        # current embeddings first.
        original_W, original_V = embedder.W.clone(), embedder.V.clone()
        W, V =  original_W + offset_W,  original_V + offset_V
        M_hat = torch.mm(W,V)
        delta = M - M_hat
        expected_nabla_W = torch.mm(delta, V.t())
        expected_nabla_V = torch.mm(W.t(), delta)

        # Get the gradient using the embedder's method
        nabla_V, nabla_W = embedder.get_gradient(offsets=(offset_V, offset_W))

        # Embedder gradients match values calculated here based on offsets.
        self.assertTrue(torch.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(torch.allclose(original_W, embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided(self):

        # Set up conditions for the test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make an embedder, whose get_gradient method we are testing.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True, device=device)

        # Calculate the gradient manually here.
        original_V = embedder.V.clone()
        V = original_V
        M_hat = torch.mm(V.t(),V)
        #M = torch.tensor(M, dtype=torch.float32)
        delta = M - M_hat
        expected_nabla_V = torch.mm(V, delta)

        # Get the gradient using the embedders method (which we are testing).
        nabla_V = embedder.get_gradient()

        # Gradient from embedder should match that manually calculated.
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(torch.allclose(original_V.t(), embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided_with_offset(self):

        # Set up test conditions.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()
        offset_V = torch.rand(cooc_stats.Nxx.shape)

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make an embedder, whose get_gradient method we are testing.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True, device=device)

        # Manually calculate expected gradients
        original_V = embedder.V.clone()
        V =  original_V + offset_V
        M_hat = torch.mm(V.t(),V)
        #M = torch.tensor(M, dtype=torch.float32)
        delta = M - M_hat
        expected_nabla_V = torch.mm(V, delta)

        # Calculate gradients using embedder's method (which we are testing).
        nabla_V = embedder.get_gradient(offsets=offset_V)

        # Gradients from embedder should match those calculated manuall.
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset.
        self.assertTrue(torch.allclose(original_V.t(), embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))



    def test_integration_with_f_delta(self):

        # Set up conditions for test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        pass_args = {'a':True, 'b':False}

        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        # Make mock f_delta whose integration with an embedder is being tested.
        def get_mock_f_delta(cooc_stats, M_, implementation, device):
            def mock_f_delta(M_hat_, **kwargs):
                self.assertTrue(torch.allclose(
                    M_, torch.tensor(M, dtype=torch.float32)))
                self.assertEqual(kwargs, {'a':True, 'b':False})
                return M_ - M_hat_
            return mock_f_delta

        f_delta = get_mock_f_delta(
            cooc_stats, M, implementation='torch', device=device)

        # Make embedder whose integration with mock f_delta is being tested.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_delta, d, learning_rate, device=device)

        # Verify that all settings passed into the ebedder were registered,
        # and that the M matrix has been converted to a torch.Tensor.
        self.assertEqual(embedder.learning_rate, learning_rate)
        self.assertEqual(embedder.d, d)
        self.assertTrue(torch.allclose(
            embedder.M, torch.tensor(M, dtype=torch.float32)))
        self.assertEqual(embedder.f_delta, f_delta)

        # Clone current embeddings so we can manually calculate the expected
        # effect of one update cycle.
        old_W, old_V = embedder.W.clone(), embedder.V.clone()

        # Ask the embedder to progress through one update cycle.
        embedder.cycle(pass_args=pass_args, print_badness=False)

        # Calculate teh expected changes due to the update.
        #M = torch.tensor(M, dtype=torch.float32)
        M_hat = torch.mm(old_W, old_V)
        delta = M - M_hat
        new_V = old_V + learning_rate * torch.mm(old_W.t(), delta)
        new_W = old_W + learning_rate * torch.mm(delta, old_V.t())

        # New embeddings in embedder should match manually updated ones.
        self.assertTrue(torch.allclose(embedder.V, new_V))
        self.assertTrue(torch.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(abs(M - torch.mm(old_W, old_V))) / (d*d)
        self.assertEqual(expected_badness, embedder.badness)


    def test_arbitrary_f_delta(self):
        # Set up conditions for test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        delta_amount = 0.1

        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        # Test integration between an embedder and the following f_delta:
        delta_always = torch.zeros(M.shape) + delta_amount
        def get_f_delta(cooc_stats, M, implementation, device):
            def f_delta(M_hat):
                return delta_always
            return f_delta

        # Make the embedder whose integration with f_delta we are testing.
        f_delta = get_f_delta(
            cooc_stats, M, implementation='torch', device=device)
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_delta, d, learning_rate, device=device)

        # Clone current embeddings to manually calculate expected update.
        old_V = embedder.V.clone()
        old_W = embedder.W.clone()

        # Ask the embedder to advance through an update cycle.
        embedder.cycle(print_badness=False)

        # Check that the update was performed.
        new_V = old_V + learning_rate * torch.mm(old_W.t(), delta_always)
        new_W = old_W + learning_rate * torch.mm(delta_always, old_V.t())
        self.assertTrue(torch.allclose(embedder.V, new_V))
        self.assertTrue(torch.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(delta_always) / (d*d)
        self.assertEqual(expected_badness, embedder.badness)


    def test_update(self):

        # Set up conditions for test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats= h.corpus_stats.get_test_stats(2)

        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make the embedder whose update method we are testing.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Generate some random update to be applied
        old_W, old_V = embedder.W.clone(), embedder.V.clone()
        delta_V = torch.rand(M.shape)
        delta_W = torch.rand(M.shape)
        updates = delta_V, delta_W

        # Apply the updates.
        embedder.update(*updates)

        # Verify that the embeddings moved by the provided amounts.
        self.assertTrue(torch.allclose(old_W + delta_W, embedder.W))
        self.assertTrue(torch.allclose(old_V + delta_V, embedder.V))


    def test_update_with_constraints(self):

        # Set up test conditions.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        # Make the ebedder whose integration with constrainer we are testing.
        # Note that we have included a constrainer.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate,
            constrainer=h.constrainer.glove_constrainer,
            device='cpu'
        )

        # Clone the current embeddings, and apply a random update to them,
        # using the embedders update method.  Internally, the embedder should
        # apply the constraints after the update
        old_W, old_V = embedder.W.clone(), embedder.V.clone()
        delta_V = torch.rand(M.shape)
        delta_W = torch.rand(M.shape)
        updates = delta_V, delta_W
        embedder.update(*updates)

        # Calculate the expected updated embeddings, with application of
        # constraints.
        expected_updated_W = old_W + delta_W
        expected_updated_V = old_V + delta_V
        h.constrainer.glove_constrainer(expected_updated_W, expected_updated_V)

        # Verify that the resulting embeddings in the embedder match the ones
        # manually calculated here.
        self.assertTrue(torch.allclose(expected_updated_W, embedder.W))
        self.assertTrue(torch.allclose(expected_updated_V, embedder.V))

        # Verify that the contstraints really were applied.
        self.assertTrue(torch.allclose(embedder.W[:,1], torch.ones(d)))
        self.assertTrue(torch.allclose(embedder.V[0,:], torch.ones(d)))


    def test_update_one_sided_rejects_delta_W(self):

        # Set up conditions for test.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make a NON-one-sided embedder, whose `update` method we are testing.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Show that we can update covector embeddings for a non-one-sided model
        delta_W = torch.ones(M.shape)
        embedder.update(delta_W=delta_W)

        # Now make a ONE-SIDED embedder, which should reject covector updates.
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate, one_sided=True)
        delta_W = torch.ones(M.shape)
        with self.assertRaises(ValueError):
            embedder.update(delta_W=delta_W)


    def test_integration_with_constrainer(self):

        # Set up test conditions.
        d = 11
        device = 'cpu'
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make an embedder, to test its integration with constrainer.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate,
            constrainer=h.constrainer.glove_constrainer, device=device
        )

        # Copy the current embeddings so we can manually calculate the expected
        # updates.
        old_V = embedder.V.clone()
        old_W = embedder.W.clone()

        # Ask the embedder to advance through one update cycle.
        embedder.cycle(print_badness=False)

        # Calculate the expected update, with constraints applied.
        M_hat = torch.mm(old_W, old_V)
        delta = M - M_hat
        new_V = old_V + learning_rate * torch.mm(old_W.t(), delta)
        new_W = old_W + learning_rate * torch.mm(delta, old_V.t())

        # Apply the constraints.  Note that the constrainer operates in_place.

        # Verify that manually updated embeddings match those of the embedder.
        h.constrainer.glove_constrainer(new_W, new_V)
        self.assertTrue(torch.allclose(embedder.V, new_V))
        self.assertTrue(torch.allclose(embedder.W, new_W))

        # Verify that the contstraints really were applied.
        self.assertTrue(torch.allclose(embedder.W[:,1], torch.ones(d)))
        self.assertTrue(torch.allclose(embedder.V[0,:], torch.ones(d)))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(abs(delta)) / (d*d)
        self.assertEqual(expected_badness, embedder.badness)


    def test_mse_embedder(self):
        # Set up conditions for test.
        d = 11
        device = 'cpu'
        tolerance = 0.0001
        learning_rate = 0.01
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', neg_inf_val=0, device=device).load_all()

        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device=device)
        # Make the embedder, whose convergence we are testing.
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device=device)

        # Run the embdder for one hundred thousand update cycles.
        embedder.cycle(100000, print_badness=False)

        # Ensure that the embeddings have the right shape.
        self.assertEqual(embedder.V.shape, (M.shape[1],d))
        self.assertEqual(embedder.W.shape, (d,M.shape[0]))

        # Check that we have essentially reached convergence, based on the 
        # fact that the delta value for the embedder is near zero.
        M_hat = torch.mm(embedder.W, embedder.V)
        delta = f_MSE(M_hat)
        self.assertTrue(torch.sum(delta) < tolerance)
        





class MockObjective(object):

    def __init__(self, *param_shapes, implementation='torch', device='cuda'):
        self.param_shapes = param_shapes
        self.updates = []
        self.passed_args = []
        self.params = []
        self.implementation = implementation
        self.device = device
        h.utils.ensure_implementation_valid(implementation)
        self.initialize_params()


    def initialize_params(self):
        initial_params = []
        for shape in self.param_shapes:
            if self.implementation == 'torch':
                initial_params.append(
                    torch.tensor(np.random.random(shape), dtype=torch.float32)
                )
            else:
                initial_params.append(np.random.random(shape))
        self.params.append(initial_params)


    def get_gradient(self, offsets=None, pass_args=None):
        self.passed_args.append(pass_args)
        curr_gradient = []
        for i in range(len(self.param_shapes)):
            curr_gradient.append(
                self.params[-1][i] + 0.1 
                + (offsets[i] if offsets is not None else 0)
            )
        return curr_gradient


    def update(self, *updates):
        new_params = []
        for i in range(len(self.param_shapes)):
            new_params.append(self.params[-1][i] + updates[i])
        self.params.append(new_params)

        copied_updates = [
            a if np.isscalar(a) else a.copy() 
            if self.implementation == 'numpy' else a.clone()
            for a in updates
        ]
        self.updates.append(copied_updates)



class TestSolvers(TestCase):

    def test_momentum_solver(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mock_objective = MockObjective((1,), (3,3), implementation='numpy')

        solver = h.solver.MomentumSolver(
            mock_objective, learning_rate, momentum_decay, 
            implementation='numpy'
        )

        solver.cycle(times=times, pass_args={'a':1})

        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        expected_params = []
        np.random.seed(0)
        initial_params_0 = np.random.random((1,))
        initial_params_1 = np.random.random((3,3))
        expected_params.append((initial_params_0, initial_params_1))

        # Initialize the momentum at zero
        expected_momenta = [(np.zeros((1,)), np.zeros((3,3)))]

        # Compute successive updates
        for i in range(times):
            update_0 = (
                expected_momenta[-1][0] * momentum_decay
                + (expected_params[-1][0] + 0.1) * learning_rate
            )
            update_1 = (
                expected_momenta[-1][1] * momentum_decay
                + (expected_params[-1][1] + 0.1) * learning_rate
            )
            expected_momenta.append((update_0, update_1))

            expected_params.append((
                expected_params[-1][0] + expected_momenta[-1][0],
                expected_params[-1][1] + expected_momenta[-1][1]
            ))

        # Updates should be the successive momenta (excluding the first zero
        # value)
        for expected,found in zip(expected_momenta[1:], mock_objective.updates):
            for e, f in zip(expected, found):
                self.assertTrue(np.allclose(e, f))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mock_objective.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_momentum_solver_torch(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mock_objective = MockObjective(
            (1,), (3,3), implementation='torch', device='cpu')

        solver = h.solver.MomentumSolver(
            mock_objective, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )

        solver.cycle(times=times, pass_args={'a':1})

        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        expected_params = []


        np.random.seed(0)
        initial_params_0 = torch.tensor(
            np.random.random((1,)), dtype=torch.float32)

        initial_params_1 = torch.tensor(
            np.random.random((3,3)), dtype=torch.float32)


        expected_params.append((initial_params_0, initial_params_1))

        # Initialize the momentum at zero
        expected_momenta = [(torch.zeros((1,)), torch.zeros((3,3)))]

        # Compute successive updates
        for i in range(times):
            update_0 = (
                expected_momenta[-1][0] * momentum_decay
                + (expected_params[-1][0] + 0.1) * learning_rate
            )
            update_1 = (
                expected_momenta[-1][1] * momentum_decay
                + (expected_params[-1][1] + 0.1) * learning_rate
            )
            expected_momenta.append((update_0, update_1))

            expected_params.append((
                expected_params[-1][0] + expected_momenta[-1][0],
                expected_params[-1][1] + expected_momenta[-1][1]
            ))

        # Updates should be the successive momenta (excluding the first zero
        # value)
        iter_momenta_updates = zip(
            expected_momenta[1:], mock_objective.updates)
        for expected,found in iter_momenta_updates:
            for e, f in zip(expected, found):
                self.assertTrue(torch.allclose(e, f))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mock_objective.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_momentum_solver_torch_and_numpy_equivalent(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        # Do 3 iterations on a numpy-based objective and solver.
        np.random.seed(0)
        torch_mock_objective = MockObjective(
            (1,), (3,3), implementation='torch', device='cpu')
        torch_solver = h.solver.MomentumSolver(
            torch_mock_objective, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )
        torch_solver.cycle(times=times, pass_args={'a':1})


        # Do 3 iterations on a torch-based objective and solver.
        np.random.seed(0)
        numpy_mock_objective = MockObjective(
            (1,), (3,3), implementation='numpy')
        numpy_solver = h.solver.MomentumSolver(
            numpy_mock_objective, learning_rate, momentum_decay, 
            implementation='numpy'
        )
        numpy_solver.cycle(times=times, pass_args={'a':1})


        # They should be equal!
        iter_momenta_updates = zip(
            torch_mock_objective.updates, numpy_mock_objective.updates)
        for torch_update, numpy_update in iter_momenta_updates:
            for e, f in zip(torch_update, numpy_update):
                self.assertTrue(np.allclose(e, f))


    def test_nesterov_momentum_solver(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective((1,), (3,3), implementation='numpy')
        solver = h.solver.NesterovSolver(
            mo, learning_rate, momentum_decay, implementation='numpy')

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = self.calculate_expected_nesterov_params(
            times, learning_rate, momentum_decay)

        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(np.allclose(param, param_expected))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_nesterov_momentum_solver_torch_equivalent_to_numpy(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        torch_mo = MockObjective(
            (1,), (3,3), implementation='torch', device='cpu')
        torch_solver = h.solver.NesterovSolver(
            torch_mo, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )
        torch_solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        numpy_mo = MockObjective((1,), (3,3), implementation='numpy')
        numpy_solver = h.solver.NesterovSolver(
            numpy_mo, learning_rate, momentum_decay, implementation='numpy')
        numpy_solver.cycle(times=times, pass_args={'a':1})

        # Verify that the solver visited to the expected parameter values
        for i in range(len(torch_mo.params)):
            param_iterator = zip(torch_mo.params[i], numpy_mo.params[i])
            for torch_param, numpy_param in param_iterator:
                self.assertTrue(np.allclose(torch_param, numpy_param))


    def test_nesterov_momentum_solver_torch(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective((1,), (3,3), implementation='torch', device='cpu')
        solver = h.solver.NesterovSolver(
            mo, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )

        solver.cycle(times=times, pass_args={'a':1})


        np.random.seed(0)
        params_expected = self.calculate_expected_nesterov_params(
            times, learning_rate, momentum_decay
        )

        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(torch.allclose(
                    param,
                    torch.tensor(param_expected, dtype=torch.float32)
                ))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_nesterov_momentum_solver_optimized(self):

        learning_rate = 0.01
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective(
            (1,), (3,3), implementation='numpy', device='cpu')
        solver = h.solver.NesterovSolverOptimized(
            mo, learning_rate, momentum_decay, 
            implementation='numpy', device='cpu'
        )

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = self.calculate_expected_nesterov_optimized_params(
            times, learning_rate, momentum_decay
        )

        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(np.allclose(param, param_expected))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_nesterov_momentum_solver_optimized_torch(self):

        learning_rate = 0.01
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective(
            (1,), (3,3), implementation='torch', device='cpu')
        solver = h.solver.NesterovSolverOptimized(
            mo, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = self.calculate_expected_nesterov_optimized_params(
            times, learning_rate, momentum_decay
        )

        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(np.allclose(param, param_expected))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])



    def test_nesterov_momentum_solver_cautious_numpy(self):

        learning_rate = 0.01
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective(
            (1,), (3,3), implementation='numpy', device='cpu')
        solver = h.solver.NesterovSolverCautious(
            mo, learning_rate, momentum_decay, implementation='numpy')

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = (
            self.calculate_expected_nesterov_optimized_cautious_params(
                times, learning_rate, momentum_decay, implementation='numpy'
        ))


        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(np.allclose(param, param_expected))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_nesterov_momentum_solver_cautious_torch(self):

        learning_rate = 0.01
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective(
            (1,), (3,3), implementation='torch', device='cpu')
        solver = h.solver.NesterovSolverCautious(
            mo, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = (
            self.calculate_expected_nesterov_optimized_cautious_params(
                times, learning_rate, momentum_decay, implementation='torch'
        ))

        # Verify that the solver visited to the expected parameter values
        for i in range(len(params_expected)):
            for param, param_expected in zip(mo.params[i], params_expected[i]):
                self.assertTrue(np.allclose(param, param_expected))

        # Test that all the pass_args were received.  Note that the solver
        # will call get_gradient once at the start to determine the shape
        # of the parameters, and None will have been passed as the pass_arg.
        self.assertEqual(
            mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])



    def compare_nesterov_momentum_solver_to_optimized(self):

        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        mock_objective_1 = MockObjective((1,), (3,3))
        nesterov_solver = h.solver.NesterovSolver(
            mock_objective_1, learning_rate, momentum_decay)

        mock_objective_2 = MockObjective((1,), (3,3))
        nesterov_solver_optimized = h.solver.NesterovSolver(
            mock_objective_2, learning_rate, momentum_decay)

        for i in range(times):

            nesterov_solver.cycle()
            nesterov_solver_optimized.cycle()

            gradient_1 = nesterov_solver.gradient_steps
            gradient_2 = nesterov_solver_optimized.gradient_steps

            for param_1, param_2 in zip(gradient_1, gradient_2):
                self.assertTrue(np.allclose(param_1, param_2))



    def calculate_expected_nesterov_params(
        self, times, learning_rate, momentum_decay
    ):

        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        params_expected = [[]]
        params_expected[0].append(np.random.random((1,)))
        params_expected[0].append(np.random.random((3,3)))

        # Solver starts with zero momentum
        momentum_expected = [[np.zeros((1,)), np.zeros((3,3))]]

        # Compute successive updates
        gradient_steps = []
        for i in range(times):

            # In this test, the gradient is always equal to `params + 0.1`
            # Nesterov adds momentum to parameters before taking gradient.
            gradient_steps.append((
                learning_rate * (
                    params_expected[-1][0] + 0.1
                    + momentum_expected[-1][0] * momentum_decay),
                learning_rate * (
                    params_expected[-1][1] + 0.1
                    + momentum_expected[-1][1] * momentum_decay),
            ))

            momentum_expected.append((
                momentum_decay * momentum_expected[-1][0] 
                    + gradient_steps[-1][0],
                momentum_decay * momentum_expected[-1][1]
                    + gradient_steps[-1][1]
            ))

            # Do the accellerated update
            params_expected.append((
                params_expected[-1][0] + momentum_expected[-1][0],
                params_expected[-1][1] + momentum_expected[-1][1],
            ))

        return params_expected
            

    def calculate_expected_nesterov_optimized_params(
        self, times, learning_rate, momentum_decay
    ):

        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        params_expected = [[]]
        params_expected[0].append(np.random.random((1,)))
        params_expected[0].append(np.random.random((3,3)))

        # Solver starts with zero momentum
        momentum_expected = [[np.zeros((1,)), np.zeros((3,3))]]

        # Compute successive updates
        gradient_steps = []
        for i in range(times):

            # In this test, the gradient is always equal to `params + 0.1`
            gradient_steps.append((
                (params_expected[-1][0] + 0.1) * learning_rate,
                (params_expected[-1][1] + 0.1) * learning_rate
            ))

            momentum_expected.append((
                momentum_decay * momentum_expected[-1][0] 
                    + gradient_steps[-1][0],
                momentum_decay * momentum_expected[-1][1]
                    + gradient_steps[-1][1]
            ))

            # Do the accellerated update
            params_expected.append((
                params_expected[-1][0] + gradient_steps[-1][0] 
                    + momentum_decay * momentum_expected[-1][0],
                params_expected[-1][1] + gradient_steps[-1][1] 
                    + momentum_decay * momentum_expected[-1][1]
            ))

        return params_expected


    def calculate_expected_nesterov_optimized_cautious_params(
        self, times, learning_rate, momentum_decay, implementation, 
        device='cpu'
    ):

        h.utils.ensure_implementation_valid(implementation)
        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        params_expected = [[]]
        gradient_steps = []

        if implementation == 'numpy':
            params_expected[0].append(np.random.random((1,)))
            params_expected[0].append(np.random.random((3,3)))
            momentum_expected = [[np.zeros((1,)), np.zeros((3,3))]]
            last_gradient = [np.zeros((1,)), np.zeros((3,3))]

        else:
            params_expected[0].append(torch.tensor(
                np.random.random((1,)), dtype=torch.float32, device=device
            ))
            params_expected[0].append(torch.tensor(
                np.random.random((3,3)), dtype=torch.float32, device=device 
            ))
            momentum_expected = [[
                torch.tensor(
                    np.zeros((1,)), dtype=torch.float32, device=device), 
                torch.tensor(
                    np.zeros((3,3)), dtype=torch.float32, device=device)
            ]]
            last_gradient = [
                torch.tensor(
                    np.zeros((1,)), dtype=torch.float32, device=device),
                torch.tensor(
                    np.zeros((3,3)), dtype=torch.float32, device=device)
            ]


        for i in range(times):

            # In this test, the gradient is always equal to `params + 0.1`
            gradient = (
                (params_expected[-1][0] + 0.1),
                (params_expected[-1][1] + 0.1)
            )

            last_gradient_norm = np.sqrt(
                h.utils.norm(last_gradient[0])**2
                + h.utils.norm(last_gradient[1])**2
            )
            gradient_norm = np.sqrt(
                h.utils.norm(gradient[0])**2
                + h.utils.norm(gradient[1])**2
            )
            product = (
                h.utils.dot(last_gradient[0], gradient[0])
                + h.utils.dot(last_gradient[1], gradient[1])
            )

            norms = last_gradient_norm * gradient_norm

            if norms == 0:
                alignment = 1
            else:
                alignment = product / norms

            if implementation == 'torch':
                last_gradient = [gradient[0].clone(), gradient[1].clone()]
            else:
                last_gradient = [gradient[0].copy(), gradient[1].copy()]

            use_decay = max(0, alignment) * momentum_decay
            gradient_steps.append((
                (params_expected[-1][0] + 0.1) * learning_rate,
                (params_expected[-1][1] + 0.1) * learning_rate
            ))
            

            # Calculate the current momentum
            momentum_expected.append((
                use_decay * momentum_expected[-1][0] 
                    + gradient[0] * learning_rate,
                use_decay * momentum_expected[-1][1]
                    + gradient[1] * learning_rate
            ))

            # Do the accellerated update
            params_update = (
                gradient_steps[-1][0] 
                    + use_decay * momentum_expected[-1][0],
                gradient_steps[-1][1]
                    + use_decay * momentum_expected[-1][1]
            )

            new_params = (
                params_expected[-1][0] + params_update[0],
                params_expected[-1][1] + params_update[1]
            )
            params_expected.append(new_params)

        return params_expected
            


#TODO: add tests for torch embedder.
class TestEmbedderSolverIntegration(TestCase):

    def test_embedder_solver_integration(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi').load_all())

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(
            M, f_MSE, d, learning_rate)
        solver = h.solver.NesterovSolver(
            embedder, learning_rate, momentum_decay, implementation='numpy')
        solver.cycle(times=times)


    def test_embedder_momentum_solver_integration(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi').load_all())

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(M, f_MSE, d, learning_rate)
        solver = h.solver.MomentumSolver(
            embedder, learning_rate, momentum_decay, implementation='numpy')
        solver.cycle(times=times)


    def test_embedder_nesterov_solver_optimized_integration(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = np.array(h.M.M(cooc_stats, 'pmi').load_all())

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='numpy')
        embedder = h.embedder.HilbertEmbedder(M, f_MSE, d, learning_rate)
        solver = h.solver.NesterovSolverOptimized(
            embedder, learning_rate, momentum_decay, implementation='numpy')
        solver.cycle(times=times)


    def test_embedder_solver_integration_torch(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', device='cpu').load_all()

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(cooc_stats, M, implementation='torch')
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device='cpu')
        solver = h.solver.NesterovSolver(
            embedder, learning_rate, momentum_decay, 
            implementation='torch', device='cpu')
        solver.cycle(times=times)


    def test_embedder_momentum_solver_integration_torch(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', device='cpu').load_all()

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device='cpu')
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device='cpu')
        solver = h.solver.MomentumSolver(
            embedder, learning_rate, momentum_decay, 
            implementation='torch', device='cpu'
        )
        solver.cycle(times=times)


    def test_embedder_nesterov_solver_optimized_integration_torch(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        cooc_stats = h.corpus_stats.get_test_stats(2)
        M = h.M.M(cooc_stats, 'pmi', device='cpu').load_all()

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.get_f_MSE(
            cooc_stats, M, implementation='torch', device='cpu')
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_MSE, d, learning_rate, device='cpu')
        solver = h.solver.NesterovSolverOptimized(
            embedder, learning_rate, momentum_decay,
            implementation='torch', device='cpu'
        )
        solver.cycle(times=times)



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



class TestCoocStats(TestCase):


    def test_cooc_stats_unpacking(self):
        cooc_stats = h.corpus_stats.get_test_stats(2)
        Nxx, Nx, Nxt, N = cooc_stats
        self.assertTrue(np.allclose(Nxx, cooc_stats.denseNxx))
        self.assertTrue(np.allclose(Nx, cooc_stats.Nx))
        self.assertTrue(np.allclose(N, cooc_stats.N))



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

        # Create a cooccurrence instance using counts
        cooccurrence = h.cooc_stats.CoocStats(
            dictionary, Nxx=Nxx, verbose=False)

        # Currently the cooccurrence instance has no internal counter for
        # cooccurrences, because it is based on the cooccurrence_array
        self.assertTrue(cooccurrence._counts is None)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, Nxx))
        self.assertTrue(np.allclose(cooccurrence._Nx, Nx))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, Nx))

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
        self.assertEqual(cooccurrence._Nx, None)

        # Asking for Nxx forces it to sync itself.  
        # Ensure it it obtains the correct cooccurrence matrix
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        expected_Nx = np.sum(expected_Nxx, axis=1).reshape(-1,1)
        expected_N = np.sum(expected_Nx)
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, expected_Nx))
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
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), array))
        self.assertTrue(np.allclose(
            cooccurrence.Nx, np.sum(array, axis=1).reshape(-1,1)))
        self.assertEqual(cooccurrence.N, np.sum(array))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, array))

        # We can still add more counts.  This causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'rice')
        cooccurrence.add('rice', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._N, None)
        self.assertEqual(cooccurrence._denseNxx, None)

        # Asking for an array forces it to sync itself.  This time start with
        # denseNxx.
        expected_Nxx = np.append(array, [[1],[0],[0],[0]], axis=1)
        expected_Nxx = np.append(expected_Nxx, [[1,0,0,0,0]], axis=0)
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(
            cooccurrence.Nx, np.sum(expected_Nxx, axis=1).reshape(-1,1)))
        self.assertEqual(cooccurrence.N, np.sum(expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))

        # Adding more counts once again causes it to drop the stale Nxx.
        cooccurrence.add('banana', 'field')
        cooccurrence.add('field', 'banana')
        self.assertEqual(cooccurrence._Nxx, None)
        self.assertEqual(cooccurrence._Nx, None)
        self.assertEqual(cooccurrence._N, None)
        self.assertEqual(cooccurrence._denseNxx, None)

        # Asking for an array forces it to sync itself.  This time start with
        # Nx.
        expected_Nxx[0,3] += 1
        expected_Nxx[3,0] += 1
        self.assertTrue(np.allclose(cooccurrence.Nxx.toarray(), expected_Nxx))
        self.assertTrue(np.allclose(
            cooccurrence.Nx, np.sum(expected_Nxx, axis=1).reshape(-1,1)))
        self.assertEqual(cooccurrence.N, np.sum(expected_Nxx))
        self.assertTrue(np.allclose(cooccurrence.denseNxx, expected_Nxx))



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
        self.assertTrue(np.allclose(cooccurrence.denseNxx, sorted_array))
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

        # Save it, then load it
        cooccurrence.save(write_path)
        cooccurrence2 = h.cooc_stats.CoocStats.load(
            write_path, verbose=False)

        self.assertEqual(
            cooccurrence2.dictionary.tokens, 
            cooccurrence.dictionary.tokens
        )
        self.assertEqual(cooccurrence2.counts, cooccurrence.counts)
        self.assertTrue(np.allclose(
            cooccurrence2.denseNxx, cooccurrence.denseNxx))
        self.assertTrue(np.allclose(cooccurrence2.Nx, cooccurrence.Nx))

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
        trunc_N = np.sum(trunc_Nx)

        self.assertTrue(np.allclose(cooccurrence.denseNxx, trunc_Nxx))
        self.assertTrue(np.allclose(cooccurrence.Nx, trunc_Nx))
        self.assertTrue(np.allclose(cooccurrence.N, trunc_N))


    def test_dict_to_sparse(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        csr_matrix = h.cooc_stats.dict_to_sparse(counts)
        self.assertTrue(isinstance(csr_matrix, sparse.csr_matrix))
        self.assertTrue(np.allclose(csr_matrix.todense(), array))


    def test_deepcopy(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)

        cooccurrence2 = deepcopy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(
            cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.counts is not cooccurrence1.counts)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)

        self.assertTrue(np.allclose(
            cooccurrence2.denseNxx, cooccurrence1.denseNxx))
        self.assertTrue(np.allclose(cooccurrence2.Nx, cooccurrence1.Nx))
        self.assertEqual(cooccurrence2.N, cooccurrence1.N)
        self.assertEqual(cooccurrence2.counts, cooccurrence1.counts)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    def test_copy(self):
        dictionary, counts, dij, array = self.get_test_cooccurrence_stats()
        cooccurrence1 = h.cooc_stats.CoocStats(
            dictionary, counts, verbose=False)

        cooccurrence2 = copy(cooccurrence1)

        self.assertTrue(cooccurrence2 is not cooccurrence1)
        self.assertTrue(
            cooccurrence2.dictionary is not cooccurrence1.dictionary)
        self.assertTrue(cooccurrence2.counts is not cooccurrence1.counts)
        self.assertTrue(cooccurrence2.Nxx is not cooccurrence1.Nxx)
        self.assertTrue(cooccurrence2.Nx is not cooccurrence1.Nx)

        self.assertTrue(np.allclose(
            cooccurrence2.denseNxx, cooccurrence1.denseNxx))
        self.assertTrue(np.allclose(cooccurrence2.Nx, cooccurrence1.Nx))
        self.assertEqual(cooccurrence2.N, cooccurrence1.N)
        self.assertEqual(cooccurrence2.counts, cooccurrence1.counts)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)
        self.assertEqual(cooccurrence2.verbose, cooccurrence1.verbose)


    def test_add(self):
        """
        When CoocStats add, their counts add.
        """

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
        self.assertEqual(cooccurrence1.counts, counts)
        self.assertTrue(np.allclose(cooccurrence1.denseNxx, array))
        expected_Nx = np.sum(array, axis=1).reshape(-1,1)
        self.assertTrue(np.allclose(cooccurrence1.Nx, expected_Nx))
        self.assertEqual(cooccurrence1.N, np.sum(array))
        self.assertEqual(cooccurrence1.dictionary.tokens, dictionary.tokens)
        self.assertEqual(
            cooccurrence1.dictionary.token_ids, dictionary.token_ids)
        self.assertEqual(cooccurrence1.verbose, False)

        # Ensure that cooccurrence2 was not changed
        self.assertEqual(cooccurrence2.counts, counts2)

        self.assertTrue(np.allclose(cooccurrence2.denseNxx, array2))
        expected_Nx2 = np.sum(array2, axis=1).reshape(-1,1)
        self.assertTrue(np.allclose(cooccurrence2.Nx, expected_Nx2))
        self.assertEqual(cooccurrence2.N, np.sum(array2))
        self.assertEqual(cooccurrence2.dictionary.tokens, dictionary2.tokens)
        self.assertEqual(
            cooccurrence2.dictionary.token_ids, dictionary2.token_ids)
        self.assertEqual(cooccurrence2.verbose, False)
        

        # Ensure that cooccurrence_sum is as desired
        dictionary_sum = h.dictionary.Dictionary([
            'banana', 'socks', 'car', 'cave', 'field'])
        array_sum = np.array([
            [2, 4, 3, 0, 1],
            [4, 0, 1, 1, 0],
            [3, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])
        Nx_sum = np.sum(array_sum, axis=1).reshape(-1,1)
        counts_sum = Counter({
            (0, 0): 2, 
            (0, 1): 4, (1, 0): 4, (2, 0): 3, (0, 2): 3, (1, 2): 1, (3, 2): 1,
            (3, 1): 1, (2, 1): 1, (1, 3): 1, (2, 3): 1, (0, 4): 1, (4, 0): 1
        })
        #self.assertEqual(
        #    cooccurrence_sum.dictionary.tokens, dictionary_sum.tokens)
        #self.assertEqual(
        #    cooccurrence_sum.dictionary.token_ids, dictionary_sum.token_ids)
        self.assertTrue(np.allclose(cooccurrence_sum.denseNxx, array_sum))
        self.assertTrue(np.allclose(cooccurrence_sum.Nx, Nx_sum))
        self.assertTrue(cooccurrence_sum.N, cooccurrence1.N + cooccurrence2.N)
        self.assertEqual(cooccurrence_sum.counts, counts_sum)


def get_test_dictionary():
    return h.dictionary.Dictionary.load(
        os.path.join(h.CONSTANTS.TEST_DIR, 'dictionary'))


class TestEmbeddings(TestCase):

    def test_random(self):
        d = 300
        vocab = 5000
        shared = False
        dictionary = get_test_dictionary()

        # Can make random embeddings and provide a dictionary to use.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, implementation='torch', device='cpu')
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can have random embeddings with shared parameters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=True, implementation='torch', 
            device='cpu'
        )
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertTrue(embeddings.W is embeddings.V)
        self.assertTrue(embeddings.dictionary is dictionary)

        # Can omit the dictionary
        embeddings = h.embeddings.random(
            vocab, d, dictionary=None, shared=False, implementation='torch',
            device='cpu'
        )
        self.assertEqual(embeddings.V.shape, (vocab, d))
        self.assertEqual(embeddings.W.shape, (vocab, d))
        self.assertTrue(embeddings.dictionary is None)

        # Can use either numpy or torch
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=False, implementation='torch', 
            device='cpu'
        )
        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))

        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=False, implementation='numpy')
        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))


    def test_random_distribution(self):
        d = 300
        vocab = 5000
        shared = False

        dictionary = get_test_dictionary()

        # Can make numpy random embeddings with uniform distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='uniform', scale=0.2, seed=0,
            implementation='numpy', device='cpu',
        )

        np.random.seed(0)
        expected_uniform_V = np.random.uniform(-0.2, 0.2, (vocab, d))
        expected_uniform_W = np.random.uniform(-0.2, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_W))

        # Can make numpy random embeddings with normal distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='normal', scale=0.2, seed=0,
            implementation='numpy', device='cpu',
        )

        np.random.seed(0)
        expected_normal_V = np.random.normal(0, 0.2, (vocab, d))
        expected_normal_W = np.random.normal(0, 0.2, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='uniform', scale=1, seed=0,
            implementation='numpy', device='cpu',
        )

        np.random.seed(0)
        expected_uniform_scale_V = np.random.uniform(-1, 1, (vocab, d))
        expected_uniform_scale_W = np.random.uniform(-1, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_scale_W))

        # Scale matters.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='normal', scale=1, seed=0,
            implementation='numpy', device='cpu',
        )

        np.random.seed(0)
        expected_normal_scale_V = np.random.normal(0, 1, (vocab, d))
        expected_normal_scale_W = np.random.normal(0, 1, (vocab, d))

        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_scale_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_scale_W))

        # Can make torch random embeddings with uniform distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='uniform', scale=0.2, seed=0,
            implementation='torch', device='cpu',
        )

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_uniform_V))
        self.assertTrue(np.allclose(embeddings.W, expected_uniform_W))

        # Can make torch random embeddings with normal distribution
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            distribution='normal', scale=0.2, seed=0,
            implementation='torch', device='cpu',
        )

        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertTrue(np.allclose(embeddings.V, expected_normal_V))
        self.assertTrue(np.allclose(embeddings.W, expected_normal_W))



    def test_unk(self):

        d = 300
        vocab = 5000
        shared = False
        dictionary = get_test_dictionary()

        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            implementation='numpy', device='cpu'
        )
        self.assertTrue(np.allclose(embeddings.unk, embeddings.V.mean(0)))
        self.assertTrue(np.allclose(embeddings.unkV, embeddings.V.mean(0)))
        self.assertTrue(np.allclose(embeddings.unkW, embeddings.W.mean(0)))

        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared, 
            implementation='torch', device='cpu'
        )
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
        



    def test_embedding_implementation(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary, implementation="torch", device="cpu")
        self.assertTrue(isinstance(embeddings.V, torch.Tensor))
        self.assertTrue(isinstance(embeddings.W, torch.Tensor))
        self.assertEqual(embeddings.W.device.type, 'cpu')

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary, implementation="numpy")
        self.assertTrue(isinstance(embeddings.V, np.ndarray))
        self.assertTrue(isinstance(embeddings.W, np.ndarray))


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
        device='cpu'

        if os.path.exists(out_path):
            shutil.rmtree(out_path)


        # Create vectors using the numpy implementation, save them, then 
        # reload them alternately using either numpy or torch implementation.
        embeddings1 = h.embeddings.Embeddings(
            V, W, dictionary, implementation='torch',device='cpu')
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(
            out_path, implementation='torch', device='cpu')
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(
            out_path, implementation='numpy')
        self.assertTrue(isinstance(embeddings2.V, np.ndarray))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)

        # We can do the same save and load cycle, this time starting from 
        # torch embeddings.
        if os.path.exists(out_path):
            shutil.rmtree(out_path)

        embeddings1 = h.embeddings.Embeddings(
            V, W, dictionary, implementation='torch',device='cpu')
        embeddings1.save(out_path)

        embeddings2 = h.embeddings.Embeddings.load(
            out_path, implementation='torch', device='cpu')
        self.assertTrue(isinstance(embeddings2.V, torch.Tensor))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(torch.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(torch.allclose(embeddings1.W, embeddings2.W))

        embeddings2 = h.embeddings.Embeddings.load(
            out_path, implementation='numpy')
        self.assertTrue(isinstance(embeddings2.V, np.ndarray))

        self.assertTrue(embeddings1.V is not embeddings2.V)
        self.assertTrue(embeddings1.W is not embeddings2.W)
        self.assertTrue(np.allclose(embeddings1.V, embeddings2.V))
        self.assertTrue(np.allclose(embeddings1.W, embeddings2.W))

        shutil.rmtree(out_path)


    def test_embeddings_recognize_loading_normalized(self):

        in_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'normalized-test-embeddings')
        embeddings = h.embeddings.Embeddings.load(
            in_path, implementation='torch', device='cpu')
        self.assertTrue(embeddings.normed)
        self.assertTrue(embeddings.check_normalized())


    def test_normalize_embeddings(self):

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        device='cpu'

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary, implementation='torch',device='cpu')

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
        device='cpu'

        embeddings = h.embeddings.Embeddings(
            V, dictionary=dictionary, shared=True, 
            implementation='torch', device='cpu'
        )

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
        d = 300
        vocab = 5000
        V = torch.rand((vocab, d))
        W = torch.rand((vocab, d))

        with self.assertRaises(ValueError):
            embeddings = h.embeddings.Embeddings(V, W, shared=True)


    def test_greatest_product(self):
        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()

        # Some products are tied, and their sorting isn't stable.  But, when
        # we set fix the seed, the top and bottom ten are stably ranked.
        embeddings = h.embeddings.random(
            vocab, d, dictionary, shared=False, implementation='torch', 
            device='cpu', seed=0
        )

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
            vocab, d, dictionary, shared=False, implementation='torch', 
            device='cpu', seed=0
        )

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

        d = 300
        vocab = 5000
        dictionary = get_test_dictionary()
        V = torch.rand((vocab, d))
        W = torch.rand((vocab, d))
        device='cpu'

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary, implementation='torch',device='cpu'
        )

        self.assertTrue(torch.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(torch.allclose(
            embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(torch.allclose(
            embeddings.get_covec((slice(0,5000,1),slice(0,300,1))), 
            W
        ))


        V = np.random.random((vocab, d))
        W = np.random.random((vocab, d))
        device='cpu'

        embeddings = h.embeddings.Embeddings(
            V, W, dictionary, implementation='numpy')

        self.assertTrue(
            np.allclose(embeddings[0:5000:1,0:300:1], V))
        self.assertTrue(np.allclose(
                embeddings.get_vec((slice(0,5000,1),slice(0,300,1))), V))
        self.assertTrue(np.allclose(
                embeddings.get_covec((slice(0,5000,1),slice(0,300,1))), W))

        #  You can use negative step size for numpy arrays!
        self.assertTrue(np.allclose(
            embeddings[500:1000:-2,50:100:-10], V[500:1000:-2,50:100:-10])
        )
        self.assertTrue(np.allclose(
            embeddings.get_vec((slice(500,1000,-2),slice(50,100,-10))),
            V[500:1000:-2,50:100:-10])
        )
        self.assertTrue(np.allclose(
            embeddings.get_covec((slice(500,1000,-2),slice(50,100,-10))),
            W[500:1000:-2,50:100:-10])
        )









class TestUtils(TestCase):

    def test_normalize(self):

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

        V_torch = torch.rand((vocab, d))

        norm = torch.norm(V_torch, p=2, dim=1, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=1)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=1), torch.ones(vocab)))


        norm = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        expected = V_torch / norm
        found = h.utils.normalize(V_torch, ord=2, axis=0)
        self.assertTrue(torch.allclose(found, expected))
        self.assertTrue(V_torch.shape, found.shape)
        self.assertTrue(torch.allclose(
            torch.norm(found, p=2, dim=0), torch.ones(d)))



    def test_norm(self):

        d = 300
        vocab = 5000

        V_numpy = np.random.random((vocab, d))

        expected = np.linalg.norm(V_numpy, ord=2, axis=0, keepdims=True)
        found = h.utils.norm(V_numpy, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = np.linalg.norm(V_numpy, ord=3, axis=1, keepdims=False)
        found = h.utils.norm(V_numpy, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))

        V_torch = torch.rand((vocab, d))

        expected = torch.norm(V_torch, p=2, dim=0, keepdim=True)
        found = h.utils.norm(V_torch, ord=2, axis=0, keepdims=True)
        self.assertTrue(np.allclose(found, expected))

        expected = torch.norm(V_torch, p=3, dim=1, keepdim=False)
        found = h.utils.norm(V_torch, ord=3, axis=1, keepdims=False)
        self.assertTrue(np.allclose(found, expected))



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
    main()

