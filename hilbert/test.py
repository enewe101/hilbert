import sys
import os
import shutil
from unittest import main, TestCase
from copy import copy, deepcopy
from collections import Counter
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


class TestGetEmbedder(TestCase):

    def test_get_w2v_embedder(self):

        k = 15
        t = 0.1
        alpha = 0.75
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        bigram = h.corpus_stats.get_test_bigram(3)

        # Manually apply unigram_smoothing to the cooccurrence statistics.
        Nxx, Nx, Nxt, N = bigram
        bigram.unigram.apply_smoothing(alpha)
        uNx, uNxt, uN = bigram.unigram
        
        # Calculate the expected_M
        N_neg = k * (Nx-Nxx) * (uNxt / uN)
        expected_M = torch.log(Nxx) - torch.log(N_neg)

        # Calculate expected f_delta
        M_hat = expected_M + 1
        multiplier = Nxx + N_neg
        difference = 1/(1+np.e**(-expected_M)) - 1/(1+np.e**(-M_hat))
        expected_delta = multiplier * difference

        bigram = h.corpus_stats.get_test_bigram(3)

        # Note, in this case embedder is the solver, so the tuple return
        # is reduntant, but is done for consistency.
        found_embedder, found_solver = h.embedder.get_w2v_embedder(
            bigram, k=k, alpha=alpha, verbose=False
        )
        found_delta_calculator = found_embedder.delta
        found_delta = found_delta_calculator.calc_shard(M_hat)
        found_M = found_delta_calculator.M.load_all()

        self.assertTrue(torch.allclose(found_M, expected_M))
        self.assertTrue(torch.allclose(found_delta, expected_delta))


    def test_get_glove_embedder(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        d=300
        learning_rate = 1e-6
        one_sided = False
        constrainer = h.constrainer.glove_constrainer
        X_max = 100.0
        base = 'logNxx'
        neg_inf_val=0
        solver='sgd'

        np.random.seed(0)
        torch.manual_seed(0)
        M = h.M.M_logNxx(
            bigram=bigram, 
            neg_inf_val=neg_inf_val,
        ).load_all()
        delta_str = 'glove'
        delta = h.f_delta.DeltaGlove(
            bigram=bigram,
            M=M,
            X_max=X_max,
        )
        expected_embedder = h.embedder.HilbertEmbedder(
            delta=delta,
            d=d,
            learning_rate=learning_rate,
            one_sided=one_sided,
            constrainer=constrainer,
            verbose=False, 
        )

        np.random.seed(0)
        torch.manual_seed(0)
        found_embedder = h.embedder.get_embedder(
            bigram=bigram,
            delta=delta_str,
            base=base,
            solver=solver,
            X_max=X_max,

            # vvv Defaults
            k=None,
            smooth_unigram=None,
            shift_by=None,
            # ^^^ Defaults

            neg_inf_val=neg_inf_val,

            # vvv Defaults
            clip_thresh=None,
            diag=None,
            # ^^^ Defaults

            d=d,
            learning_rate=learning_rate,
            one_sided=one_sided,
            constrainer=constrainer,

            # vvv Defaults
            momentum_decay=0.9,
            # ^^^ Defaults

            verbose=False,
        )

        expected_embedder.cycle(times=10)
        found_embedder.cycle(times=10)

        self.assertTrue(torch.allclose(expected_embedder.V, found_embedder.V))
        self.assertTrue(torch.allclose(
            expected_embedder.badness, found_embedder.badness))



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
        bigram = h.corpus_stats.get_test_bigram(2)
        expected_PMI = np.load('test-data/expected_PMI.npz')['arr_0']
        found_PMI = h.corpus_stats.calc_PMI(bigram)
        self.assertTrue(np.allclose(found_PMI, expected_PMI))


    def test_sparse_PMI(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        expected_PMI = np.load('test-data/expected_PMI.npz')['arr_0']
        # PMI sparse treats all negative infinite values as zero
        expected_PMI[expected_PMI==-np.inf] = 0
        pmi_data, I, J = h.corpus_stats.calc_PMI_sparse(bigram)
        self.assertTrue(len(pmi_data) < np.product(bigram.Nxx.shape))
        found_PMI = sparse.coo_matrix((pmi_data,(I,J)),bigram.Nxx.shape)
        self.assertTrue(np.allclose(found_PMI.toarray(), expected_PMI))


    def test_PMI_star(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        expected_PMI_star_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'expected_PMI_star.npz')
        expected_PMI_star = np.load(expected_PMI_star_path)['arr_0']
        found_PMI_star = h.corpus_stats.calc_PMI_star(bigram)
        self.assertTrue(np.allclose(found_PMI_star, expected_PMI_star))


    def test_get_stats(self):
        # Next, test with a cooccurrence window of +/-2
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        bigram = h.corpus_stats.get_test_bigram(2)
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



class TestFDeltas(TestCase):


    def test_sigmoid(self):
        # This should work for np.array and torch.Tensor
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        bigram = h.corpus_stats.get_test_bigram(2)
        PMI = h.corpus_stats.calc_PMI(bigram)
        expected = np.array([
            [1/(1+np.e**(-pmi)) for pmi in row]
            for row in PMI
        ])
        result = h.f_delta.sigmoid(PMI)
        self.assertTrue(np.allclose(expected, result))

        PMI = torch.tensor(PMI, dtype=dtype, device=device)
        expected = torch.tensor([
            [1/(1+np.e**(-pmi)) for pmi in row]
            for row in PMI
        ], dtype=dtype, device=device)
        result = h.f_delta.sigmoid(PMI)
        self.assertTrue(torch.allclose(expected, result))


    def test_N_neg(self):
        k = 15.0
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram
        uNx, uNxt, uN = bigram.unigram
        expected = k * (Nx-Nxx) * (uNxt / uN)
        found = h.M.negative_sample(Nxx, Nx, uNxt, uN, k)
        self.assertTrue(np.allclose(expected, found))


    def test_f_w2v(self):
        k = 15
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram
        uNx, uNxt, uN = bigram.unigram
        N_neg = k * (Nx - Nxx) * (uNxt / uN ) 
        expected_M = torch.log(Nxx) - torch.log(N_neg)
        expected_M_hat = expected_M + 1
        expected_difference = (
            h.f_delta.sigmoid(expected_M) - h.f_delta.sigmoid(expected_M_hat))
        expected_multiplier = N_neg + Nxx
        expected = expected_multiplier * expected_difference

        M = h.M.M_w2v(bigram, k=k)
        M_ = M.load_all()
        M_hat = M_ + 1

        delta_w2v = h.f_delta.DeltaW2V(bigram, M, k)
        found = torch.zeros(bigram.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_w2v.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(expected, found))



    def test_f_glove(self):

        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        bigram = h.corpus_stats.get_test_bigram(2)
        bigram.truncate(10)

        Nxx, Nx, Nxt, N = bigram
        expected_M = torch.log(Nxx)
        # Zero out cells containing negative infinity, which are ignored
        # by glove.  We still need to zero them out to avoid nans.
        expected_M[expected_M==-np.inf] = 0
        expected_M_hat = expected_M + 1
        multiplier = torch.tensor([[
                2 * min(1, (bigram.Nxx[i,j] / 100.0)**0.75) 
                for j in range(bigram.Nxx.shape[1])
            ] for i in range(bigram.Nxx.shape[0])
        ], device=device, dtype=dtype)
        difference = torch.tensor([[
                expected_M[i,j] - expected_M_hat[i,j]
                if bigram.Nxx[i,j] > 0 else 0 
                for j in range(bigram.Nxx.shape[1])
            ] for i in range(bigram.Nxx.shape[0])
        ], device=device, dtype=dtype)
        expected = multiplier * difference

        M = h.M.M_logNxx(bigram, neg_inf_val=0)
        M_ = M.load_all()
        M_hat = M_ + 1
        delta_glove = h.f_delta.DeltaGlove(bigram, M)
        found = torch.zeros(bigram.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found[shard] = delta_glove.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(expected, found))

        # Try varying the X_max and alpha settings.
        alpha = 0.8
        X_max = 10
        delta_glove = h.f_delta.DeltaGlove(
            bigram, M, X_max=X_max, alpha=alpha)
        found2 = torch.zeros(bigram.Nxx.shape)
        shards = h.shards.Shards(2)
        for shard in shards:
            found2[shard] = delta_glove.calc_shard(M_hat[shard], shard)
        expected2 = torch.tensor([
            [
                2 * min(1, (bigram.Nxx[i,j] / X_max)**alpha) 
                    * (expected_M[i,j] - expected_M_hat[i,j])
                if bigram.Nxx[i,j] > 0 else 0 
                for j in range(bigram.Nxx.shape[1])
            ]
            for i in range(bigram.Nxx.shape[0])
        ], dtype=dtype, device=device)
        # The X_max setting has an effect, and matches a different expectation
        self.assertTrue(np.allclose(expected2, found2))
        self.assertFalse(np.allclose(expected2, expected))



    def test_f_MSE(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        bigram.truncate(10)  # Need a compound number for sharding
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()
        M_hat = M_ + 1
        expected = M_ - M_hat
        delta_mse = h.f_delta.DeltaMSE(bigram, M)
        found = delta_mse.calc_shard(M_hat)
        self.assertTrue(torch.allclose(expected, found))

        shards = h.shards.Shards(5)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()
        M_hat = M_ + 1
        expected = M_ - M_hat
        delta_mse = h.f_delta.DeltaMSE(bigram, M)
        found = torch.zeros(bigram.Nxx.shape, device=device, dtype=dtype)
        for shard in shards:
            found[shard] = delta_mse.calc_shard(M_hat[shard], shard)
        self.assertTrue(torch.allclose(expected, found))


    def test_f_swivel(self):
        bigram = h.corpus_stats.get_test_bigram(2)

        M = h.M.M_pmi_star(bigram)
        M_ = M.load_all()
        M_hat = M_ + 1

        expected = np.array([
            [
                np.sqrt(bigram.Nxx[i,j]) * (M_[i,j] - M_hat[i,j]) 
                if bigram.Nxx[i,j] > 0 else
                (np.e**(M_[i,j] - M_hat[i,j]) /
                    (1 + np.e**(M_[i,j] - M_hat[i,j])))
                for j in range(M_.shape[1])
            ]
            for i in range(M_.shape[0])
        ])

        delta_swivel = h.f_delta.DeltaSwivel(bigram, M)
        found = torch.zeros(bigram.Nxx.shape)
        shards = h.shards.Shards(5)
        for shard in shards:
            found[shard] = delta_swivel.calc_shard(M_hat[shard], shard)

        self.assertTrue(np.allclose(found, expected))


    def test_f_MLE(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        bigram.truncate(10)
        Nxx, Nx, Nxt, N = bigram

        expected_M = h.corpus_stats.calc_PMI(bigram)
        expected_M_hat = expected_M + 1
        N_indep_xx = bigram.Nx * bigram.Nx.T
        N_indep_max = np.max(N_indep_xx)
        multiplier = N_indep_xx / N_indep_max
        difference = np.e**expected_M - np.e**expected_M_hat
        expected = multiplier * difference

        M = h.M.M_pmi(bigram)
        M_ = M.load_all()
        M_hat = M_ + 1
        delta_mle = h.f_delta.DeltaMLE(bigram, M)
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



class TestConstrainer(TestCase):

    def test_glove_constrainer(self):
        W, V = np.zeros((3,3)), np.zeros((3,3))
        h.constrainer.glove_constrainer(W, V)
        self.assertTrue(np.allclose(W, np.array([[0,1,0]]*3)))
        self.assertTrue(np.allclose(V, np.array([[1,0,0]]*3)))



class TestHilbertEmbedder(TestCase):


    def test_cycle(self):
        torch.random.manual_seed(0)
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        shard_factor = 3
        device = h.CONSTANTS.MATRIX_DEVICE

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()
        f_MSE = h.f_delta.DeltaMSE(bigram, M)

        # Now make an embedder.
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate,
            verbose=False, 
            shard_factor=shard_factor
        )

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))

        # Calculate the expected update for one cycle.
        V = embedder.V.clone()
        W = embedder.W.clone()
        shards = h.shards.Shards(shard_factor)
        for shard_num, shard in enumerate(shards):
            M_hat = torch.mm(W[shard[0]], V[shard[1]].t())
            delta = M_[shard] - M_hat
            badness = torch.sum(abs(delta)) / (delta.shape[0] * delta.shape[1])
            nabla_V = torch.mm(delta.t(), W[shard[0]])
            nabla_W = torch.mm(delta, V[shard[1]])
            V[shard[1]] += learning_rate * nabla_V
            W[shard[0]] += learning_rate * nabla_W

        # Do one cycle on the actual embedder.
        embedder.cycle()

        # Check that the cycle produced the expected update.
        self.assertTrue(torch.allclose(embedder.V, V))
        self.assertTrue(torch.allclose(embedder.W, W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        self.assertTrue(torch.allclose(badness, embedder.badness))


    def test_one_sided(self):
        torch.random.manual_seed(0)
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        # First make a non-one-sided embedder.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=1,
            verbose=False
        )

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))
        self.assertTrue(isinstance(M_, torch.Tensor))

        # The covectors and vectors are not the same.
        self.assertFalse(torch.allclose(embedder.W, embedder.V))

        # Now make a one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, one_sided=True,
            verbose=False, 
            shard_factor=1
        )

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))

        # Now, the covectors and vectors are the same.
        self.assertTrue(torch.allclose(embedder.W, embedder.V))

        old_V = embedder.V.clone()
        embedder.cycle()

        self.assertTrue(isinstance(old_V, torch.Tensor))

        # Check that the update was performed.
        M_hat = torch.mm(old_V, old_V.t())
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        delta = f_MSE.calc_shard(M_hat) # No shard -> calculates full matrix
        nabla_V = torch.mm(delta.t(), old_V)
        nabla_W = torch.mm(delta, old_V)
        new_V = old_V + learning_rate * (nabla_V + nabla_W)
        self.assertTrue(torch.allclose(embedder.V, new_V))

        # Check that the vectors and covectors are still identical after the
        # update.
        self.assertTrue(torch.allclose(embedder.W, embedder.V))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        delta = abs(M_ - M_hat)
        badness = torch.sum(delta) / (vocab * vocab)
        self.assertTrue(torch.allclose(badness, embedder.badness))


    def test_one_sided_sharded(self):

        torch.random.manual_seed(0)
        d = 11
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()
        f_MSE = h.f_delta.DeltaMSE(bigram, M)

        # Make a one-sided embedder.
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, one_sided=True,
            verbose=False, shard_factor=3
        )

        # Ensure that the relevant variables are tensors
        self.assertTrue(isinstance(embedder.V, torch.Tensor))
        self.assertTrue(isinstance(embedder.W, torch.Tensor))

        # Now, the covectors and vectors are the same.
        self.assertTrue(torch.allclose(embedder.W, embedder.V))

        # Clone current embeddings to manually calculate expected update.
        old_V = embedder.V.clone()
        new_V = old_V.clone()

        # Ask the embedder to advance through an update cycle.
        embedder.cycle()

        # Check that the update was performed.
        shards = h.shards.Shards(3)
        for shard in shards:

            delta = M_[shard] - torch.mm(old_V[shard[0]], old_V[shard[1]].t())
            nabla_V = torch.mm(delta.t(), old_V[shard[0]])
            nabla_W = torch.mm(delta, old_V[shard[1]])
            new_V[shard[1]] += learning_rate * nabla_V
            new_V[shard[0]] += learning_rate * nabla_W

            old_V = new_V
            badness = torch.sum(abs(delta)) / (delta.shape[0] * delta.shape[1])

        self.assertTrue(torch.allclose(embedder.V, new_V, atol=0.0001))
        self.assertTrue(torch.allclose(embedder.W, new_V, atol=0.0001))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        self.assertTrue(torch.allclose(badness, embedder.badness))









        # Check that the vectors and covectors are still identical after the
        # update.
        self.assertTrue(torch.allclose(embedder.W, embedder.V))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        self.assertTrue(torch.allclose(badness, embedder.badness))



    def test_get_gradient(self):

        torch.random.manual_seed(0)
        # Set up conditions for the test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(3)
        #bigram.truncate(10)

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        # Make the embedder, whose method we are testing.
        delta_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            delta_MSE, d, learning_rate=learning_rate, verbose=False, 
            shard_factor=2
        )

        # Take the random starting embeddings.  We will compute the gradient
        # manually here to see if it matches what the embedder's method
        # returns.
        W, V = embedder.W.clone(), embedder.V.clone()

        # Since we are not doing one-sided, W and V should be unrelated.
        self.assertFalse(torch.allclose(W, V))

        # Calculate the expected gradient.
        M_hat = torch.mm(W,V.t())
        delta = M_ - M_hat

        expected_nabla_W = torch.mm(delta, V)
        expected_nabla_V = torch.mm(delta.t(), W)

        # Get the gradient according to the embedder.
        nabla_V, nabla_W = embedder.get_gradient()

        # Embedder's gradients should match manually calculated expectation.
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))
        self.assertTrue(torch.allclose(nabla_W, expected_nabla_W))


    def test_get_gradient_with_offsets(self):

        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        torch.random.manual_seed(0)
        # Set up conditions for the test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        offset_W = torch.rand(len(bigram.Nx),d, device=device, dtype=dtype)
        offset_V = torch.rand(len(bigram.Nx),d, device=device, dtype=dtype)

        # Create an embedder, whose get_gradient method we are testing.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=3,
            verbose=False
        )

        # Manually calculate the gradients we expect, applying offsets to the
        # current embeddings first.
        original_W, original_V = embedder.W.clone(), embedder.V.clone()
        W, V =  original_W + offset_W,  original_V + offset_V
        M_hat = torch.mm(W,V.t())
        delta = M_ - M_hat

        expected_nabla_W = torch.mm(delta, V)
        expected_nabla_V = torch.mm(delta.t(), W)

        # Get the gradient using the embedder's method
        nabla_V, nabla_W = embedder.get_gradient(offsets=(offset_V, offset_W))

        # Embedder gradients match values calculated here based on offsets.
        self.assertTrue(torch.allclose(nabla_W, expected_nabla_W))
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(torch.allclose(original_W, embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided(self):

        torch.random.manual_seed(0)
        # Set up conditions for the test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        # Make an embedder, whose get_gradient method we are testing.
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, one_sided=True,
            verbose=False
        )

        # Calculate the gradient manually here.
        original_V = embedder.V.clone()
        V = original_V
        M_hat = torch.mm(V,V.t())
        delta = M_ - M_hat
        expected_nabla_V = torch.mm(delta.t(), V)

        # Get the gradient using the embedders method (which we are testing).
        nabla_V, nabla_W = embedder.get_gradient()

        # Gradient from embedder should match that manually calculated.
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V, atol=1e-6))

        # Verify that the embeddings were not altered by the offset
        self.assertTrue(torch.allclose(original_V, embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))


    def test_get_gradient_one_sided_with_offset(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        torch.random.manual_seed(0)
        # Set up test conditions.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        offset_V = torch.rand(
            len(bigram.Nx), d, device=device, dtype=dtype)

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, one_sided=True,
            verbose=False
        )

        # Manually calculate expected gradients
        original_V = embedder.V.clone()
        V =  original_V + offset_V
        M_hat = torch.mm(V,V.t())
        delta = M_ - M_hat
        expected_nabla_V = torch.mm(delta.t(), V)

        # Calculate gradients using embedder's method (which we are testing).
        nabla_V, nabla_W = embedder.get_gradient(offsets=offset_V)

        # Gradients from embedder should match those calculated manuall.
        self.assertTrue(torch.allclose(nabla_V, expected_nabla_V))

        # Verify that the embeddings were not altered by the offset.
        self.assertTrue(torch.allclose(original_V, embedder.W))
        self.assertTrue(torch.allclose(original_V, embedder.V))



    def test_integration_with_f_delta(self):

        torch.random.manual_seed(0)
        # Set up conditions for test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        pass_args = {'a':True, 'b':False}

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        # Make mock f_delta whose integration with an embedder is being tested.
        class DeltaMock:

            def __init__(
                self,
                bigram,
                M,
                test_case,
                device=None,
            ):
                self.bigram = bigram
                self.M = M
                self.test_case = test_case
                self.device=device

            def calc_shard(self, M_hat, shard=None, **kwargs):
                self.test_case.assertTrue(self.M is M)
                self.test_case.assertEqual(kwargs, {'a':True, 'b':False})
                return self.M[shard] - M_hat
                

        f_delta = DeltaMock(bigram, M, self)

        # Make embedder whose integration with mock f_delta is being tested.
        embedder = h.embedder.HilbertEmbedder(
            f_delta, d, learning_rate=learning_rate, shard_factor=1,
            verbose=False
        )

        # Verify that all settings passed into the ebedder were registered,
        # and that the M matrix has been converted to a torch.Tensor.
        self.assertEqual(embedder.learning_rate, learning_rate)
        self.assertEqual(embedder.d, d)
        self.assertTrue(f_delta.M is M)
        self.assertEqual(embedder.delta, f_delta)

        # Clone current embeddings so we can manually calculate the expected
        # effect of one update cycle.
        old_W, old_V = embedder.W.clone(), embedder.V.clone()

        # Ask the embedder to progress through one update cycle.
        embedder.cycle(pass_args=pass_args)

        # Calculate teh expected changes due to the update.
        #M = torch.tensor(M, dtype=torch.float32)
        M_hat = torch.mm(old_W, old_V.t())
        delta = M_ - M_hat
        new_V = old_V + learning_rate * torch.mm(delta.t(), old_W)
        new_W = old_W + learning_rate * torch.mm(delta, old_V)

        # New embeddings in embedder should match manually updated ones.
        self.assertTrue(torch.allclose(embedder.V, new_V))
        self.assertTrue(torch.allclose(embedder.W, new_W))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(abs(M_ - M_hat)) / (vocab**2)
        self.assertTrue(torch.allclose(expected_badness, embedder.badness))


    def test_arbitrary_f_delta(self):
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        torch.random.manual_seed(0)
        # Set up conditions for test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        delta_amount = 0.1
        delta_always = torch.zeros(
            bigram.Nxx.shape, device=device, dtype=dtype) + delta_amount

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        # Test integration between an embedder and the following f_delta:
        class DeltaMock:
            def __init__(self, bigram, M):
                self.M = M
            def calc_shard(self, M_hat, shard=None, **kwargs):
                return delta_always[shard]
                

        # Make the embedder whose integration with f_delta we are testing.
        f_delta = DeltaMock(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            delta=f_delta, d=d, learning_rate=learning_rate, shard_factor=3,
            verbose=False
        )

        # Clone current embeddings to manually calculate expected update.
        old_V = embedder.V.clone()
        old_W = embedder.W.clone()

        # Ask the embedder to advance through an update cycle.
        embedder.cycle()

        # Check that the update was performed.
        new_V = old_V + learning_rate * torch.mm(delta_always.t(), old_W)
        new_W = old_W + learning_rate * torch.mm(delta_always, old_V)

        self.assertTrue(torch.allclose(embedder.V, new_V, atol=0.0001))
        self.assertTrue(torch.allclose(embedder.W, new_W, atol=0.0001))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(delta_always) / (vocab**2)
        self.assertTrue(torch.allclose(expected_badness, embedder.badness))


    def test_update(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        torch.random.manual_seed(0)
        # Set up conditions for test.
        d = 3
        learning_rate = 0.01
        bigram= h.corpus_stats.get_test_bigram(2)

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=2,
            verbose=False
        )

        # Generate some random update to be applied
        old_W, old_V = embedder.W.clone(), embedder.V.clone()
        delta_V = torch.rand(len(bigram.Nx), d, device=device, dtype=dtype)
        delta_W = torch.rand(len(bigram.Nx), d, device=device, dtype=dtype)
        updates = delta_V, delta_W

        # Apply the updates.
        embedder.update(*updates)

        # Verify that the embeddings moved by the provided amounts.
        self.assertTrue(torch.allclose(old_W + delta_W, embedder.W))
        self.assertTrue(torch.allclose(old_V + delta_V, embedder.V))


    def test_update_with_constraints(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        torch.random.manual_seed(0)
        # Set up test conditions.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        # Make the ebedder whose integration with constrainer we are testing.
        # Note that we have included a constrainer.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor = 3,
            constrainer=h.constrainer.glove_constrainer,
            verbose=False
        )

        # Clone the current embeddings, and apply a random update to them,
        # using the embedders update method.  Internally, the embedder should
        # apply the constraints after the update
        old_W, old_V = embedder.W.clone(), embedder.V.clone()
        delta_V = torch.rand(vocab, d, dtype=dtype, device=device)
        delta_W = torch.rand(vocab, d, dtype=dtype, device=device)
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
        self.assertTrue(torch.allclose(
            embedder.W[:,1], torch.ones(vocab, device=device, dtype=dtype)))
        self.assertTrue(torch.allclose(
            embedder.V[:,0], torch.ones(vocab, device=device, dtype=dtype)))


    def test_update_one_sided_rejects_delta_W(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        torch.random.manual_seed(0)
        # Set up conditions for test.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=5,
            verbose=False
        )

        # Show that we can update covector embeddings for a non-one-sided model
        delta_W = torch.ones(vocab, d, dtype=dtype, device=device)
        embedder.update(delta_W=delta_W)

        # Now make a ONE-SIDED embedder, which should reject covector updates.
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, one_sided=True,
            verbose=False, 
            shard_factor=5
        )
        delta_W = torch.ones(vocab, d, dtype=dtype, device=device)
        with self.assertRaises(ValueError):
            embedder.update(delta_W=delta_W)


    def test_integration_with_constrainer(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        torch.random.manual_seed(0)

        # Set up test conditions.
        d = 3
        learning_rate = 0.01
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=1,
            verbose=False, 
            constrainer=h.constrainer.glove_constrainer
        )

        # Copy the current embeddings so we can manually calculate the expected
        # updates.
        old_V = embedder.V.clone()
        old_W = embedder.W.clone()

        # Ask the embedder to advance through one update cycle.
        embedder.cycle()

        # Calculate the expected update, with constraints applied.
        M_hat = torch.mm(old_W, old_V.t())
        delta = M_ - M_hat

        new_V = old_V + learning_rate * torch.mm(delta.t(), old_W)
        new_W = old_W + learning_rate * torch.mm(delta, old_V)

        # Apply the constraints.  Note that the constrainer operates in_place.

        # Verify that manually updated embeddings match those of the embedder.
        h.constrainer.glove_constrainer(new_W, new_V)
        self.assertTrue(torch.allclose(embedder.V, new_V, atol=0.0001))
        self.assertTrue(torch.allclose(embedder.W, new_W, atol=0.0001))

        # Verify that the contstraints really were applied.
        self.assertTrue(torch.allclose(
            embedder.W[:,1], torch.ones(vocab, dtype=dtype, device=device)))
        self.assertTrue(torch.allclose(
            embedder.V[:,0], torch.ones(vocab, dtype=dtype, device=device)))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        expected_badness = torch.sum(abs(delta)) / (vocab*vocab)
        self.assertTrue(torch.allclose(expected_badness, embedder.badness))


    def test_mse_embedder(self):
        torch.random.manual_seed(0)
        # Set up conditions for test.
        d = 11
        num_cycles = 200
        tolerance = 0.002
        learning_rate = 0.1
        torch.random.manual_seed(0)
        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)
        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()

        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=1, 
            verbose=False
        )

        # Run the embdder for many update cycles.
        embedder.cycle(num_cycles)

        # Ensure that the embeddings have the right shape.
        self.assertEqual(embedder.V.shape, (vocab,d))
        self.assertEqual(embedder.W.shape, (vocab,d))

        # Check that we have essentially reached convergence, based on the 
        # fact that the delta value for the embedder is near zero.
        M_hat = torch.mm(embedder.W, embedder.V.t())
        delta = f_MSE.calc_shard(M_hat) # shard is None -> calculate full delta

        self.assertTrue(
            torch.sum(delta).item() < tolerance
        )
        

    # The premise of this test is not wrong.  Sharding will lead to a different
    # trajectory for parameters, because individual shards are carried through
    # full update cycles, rather than gathering gradient accross all shards
    # before updating.
    def test_sharding_equivalence(self):

        # Set up conditions for test.
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        device = h.CONSTANTS.MATRIX_DEVICE
        d = 11
        num_cycles = 20
        learning_rate = 0.01
        torch.random.manual_seed(0)

        bigram = h.corpus_stats.get_test_bigram(2)
        vocab = len(bigram.Nx)

        M = h.M.M_pmi(bigram, neg_inf_val=0)
        M_ = M.load_all()
        f_delta = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_delta, d, learning_rate=learning_rate, shard_factor=3, 
            verbose=False
        )

        # Clone current embeddings to manually calculate expected update.
        old_V = embedder.V.clone()
        old_W = embedder.W.clone()
        new_V = old_V.clone()
        new_W = old_W.clone()

        # Ask the embedder to advance through an update cycle.
        embedder.cycle()

        # Check that the update was performed.
        shards = h.shards.Shards(3)
        for shard in shards:

            delta = M_[shard] - torch.mm(old_W[shard[0]], old_V[shard[1]].t())
            new_V[shard[1]] = old_V[shard[1]] + learning_rate * torch.mm(
                delta.t(), old_W[shard[0]])
            new_W[shard[0]] = old_W[shard[0]] + learning_rate * torch.mm(
                delta, old_V[shard[1]])
            old_V = new_V
            old_W = new_W
            badness = torch.sum(abs(delta)) / (delta.shape[0] * delta.shape[1])

        self.assertTrue(torch.allclose(embedder.V, new_V, atol=0.0001))
        self.assertTrue(torch.allclose(embedder.W, new_W, atol=0.0001))

        # Check that the badness is correct 
        # (badness is based on the error before last update)
        self.assertTrue(torch.allclose(badness, embedder.badness))




class MockObjective(object):

    def __init__(self, *param_shapes, device=None):
        self.param_shapes = param_shapes
        self.updates = []
        self.passed_args = []
        self.params = []
        self.device = device
        self.initialize_params()


    def initialize_params(self):
        initial_params = []
        for shape in self.param_shapes:
            initial_params.append(torch.tensor(
                np.random.random(shape), 
                dtype=h.CONSTANTS.DEFAULT_DTYPE,
                device=h.CONSTANTS.MATRIX_DEVICE
            ))
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

        copied_updates = [a.clone() for a in updates]
        self.updates.append(copied_updates)



class TestSolvers(TestCase):

    def test_momentum_solver(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mock_objective = MockObjective((1,), (3,3))

        solver = h.solver.MomentumSolver(
            mock_objective, learning_rate, momentum_decay, verbose=False)

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
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mock_objective = MockObjective((1,), (3,3))

        solver = h.solver.MomentumSolver(
            mock_objective, learning_rate, momentum_decay, verbose=False)

        solver.cycle(times=times, pass_args={'a':1})

        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        expected_params = []


        np.random.seed(0)
        initial_params_0 = torch.tensor(
            np.random.random((1,)), dtype=dtype, device=device)

        initial_params_1 = torch.tensor(
            np.random.random((3,3)), dtype=dtype, device=device)


        expected_params.append((initial_params_0, initial_params_1))

        # Initialize the momentum at zero
        expected_momenta = [(
            torch.zeros((1,), device=device, dtype=dtype), 
            torch.zeros((3,3), device=device, dtype=dtype)
        )]

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


    #
    #   Obsolete, only torch implementation is supported.
    #
    #def test_momentum_solver_torch_and_numpy_equivalent(self):
    #    learning_rate = 0.1
    #    momentum_decay = 0.8
    #    times = 3

    #    # Do 3 iterations on a numpy-based objective and solver.
    #    np.random.seed(0)
    #    torch_mock_objective = MockObjective((1,), (3,3))
    #    torch_solver = h.solver.MomentumSolver(
    #        torch_mock_objective, learning_rate, momentum_decay)
    #    torch_solver.cycle(times=times, pass_args={'a':1})


    #    # Do 3 iterations on a torch-based objective and solver.
    #    np.random.seed(0)
    #    numpy_mock_objective = MockObjective(
    #        (1,), (3,3))
    #    numpy_solver = h.solver.MomentumSolver(
    #        numpy_mock_objective, learning_rate, momentum_decay, 
    #        implementation='numpy'
    #    )
    #    numpy_solver.cycle(times=times, pass_args={'a':1})


    #    # They should be equal!
    #    iter_momenta_updates = zip(
    #        torch_mock_objective.updates, numpy_mock_objective.updates)
    #    for torch_update, numpy_update in iter_momenta_updates:
    #        for e, f in zip(torch_update, numpy_update):
    #            self.assertTrue(np.allclose(e, f))


    def test_nesterov_momentum_solver(self):
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective((1,), (3,3))
        solver = h.solver.NesterovSolver(
            mo, learning_rate, momentum_decay, verbose=False)

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
        torch_mo = MockObjective((1,), (3,3))
        torch_solver = h.solver.NesterovSolver(
            torch_mo, learning_rate, momentum_decay, verbose=False)
        torch_solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        numpy_mo = MockObjective((1,), (3,3))
        numpy_solver = h.solver.NesterovSolver(
            numpy_mo, learning_rate, momentum_decay, verbose=False)
        numpy_solver.cycle(times=times, pass_args={'a':1})

        # Verify that the solver visited to the expected parameter values
        for i in range(len(torch_mo.params)):
            param_iterator = zip(torch_mo.params[i], numpy_mo.params[i])
            for torch_param, numpy_param in param_iterator:
                self.assertTrue(np.allclose(torch_param, numpy_param))


    def test_nesterov_momentum_solver_torch(self):
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective((1,), (3,3))
        solver = h.solver.NesterovSolver(
            mo, learning_rate, momentum_decay, verbose=False)

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
                    torch.tensor(param_expected, dtype=dtype, device=device)
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
        mo = MockObjective((1,), (3,3))
        solver = h.solver.NesterovSolverOptimized(
            mo, learning_rate, momentum_decay, verbose=False)

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
        mo = MockObjective((1,), (3,3))
        solver = h.solver.NesterovSolverOptimized(
            mo, learning_rate, momentum_decay, verbose=False)

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



    #
    #   This test is no longer needed because numpy isn't supported
    #
    #def test_nesterov_momentum_solver_cautious_numpy(self):

    #    learning_rate = 0.01
    #    momentum_decay = 0.8
    #    times = 3

    #    np.random.seed(0)
    #    mo = MockObjective(
    #        (1,), (3,3), implementation='numpy')
    #    solver = h.solver.NesterovSolverCautious(
    #        mo, learning_rate, momentum_decay, implementation='numpy')

    #    solver.cycle(times=times, pass_args={'a':1})

    #    np.random.seed(0)
    #    params_expected = (
    #        self.calculate_expected_nesterov_optimized_cautious_params(
    #            times, learning_rate, momentum_decay, implementation='numpy'
    #    ))


    #    # Verify that the solver visited to the expected parameter values
    #    for i in range(len(params_expected)):
    #        for param, param_expected in zip(mo.params[i], params_expected[i]):
    #            self.assertTrue(np.allclose(param, param_expected))

    #    # Test that all the pass_args were received.  Note that the solver
    #    # will call get_gradient once at the start to determine the shape
    #    # of the parameters, and None will have been passed as the pass_arg.
    #    self.assertEqual(
    #        mo.passed_args, [None, {'a':1}, {'a':1}, {'a':1}])


    def test_nesterov_momentum_solver_cautious_torch(self):

        learning_rate = 0.01
        momentum_decay = 0.8
        times = 3

        np.random.seed(0)
        mo = MockObjective((1,), (3,3))
        solver = h.solver.NesterovSolverCautious(
            mo, learning_rate, momentum_decay, verbose=False)

        solver.cycle(times=times, pass_args={'a':1})

        np.random.seed(0)
        params_expected = (
            self.calculate_expected_nesterov_optimized_cautious_params(
                times, learning_rate, momentum_decay)
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



    def compare_nesterov_momentum_solver_to_optimized(self):

        learning_rate = 0.1
        momentum_decay = 0.8
        times = 3

        mock_objective_1 = MockObjective((1,), (3,3))
        nesterov_solver = h.solver.NesterovSolver(
            mock_objective_1, learning_rate, momentum_decay, verbose=False)

        mock_objective_2 = MockObjective((1,), (3,3))
        nesterov_solver_optimized = h.solver.NesterovSolver(
            mock_objective_2, learning_rate, momentum_decay, verbose=False)

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
        self, times, learning_rate, momentum_decay
    ):

        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        # Initialize the parameters using the same random initialization as
        # used by the mock objective.
        params_expected = [[]]
        gradient_steps = []

        params_expected[0].append(torch.tensor(
            np.random.random((1,)), dtype=dtype, device=device))
        params_expected[0].append(torch.tensor(
            np.random.random((3,3)), dtype=dtype, device=device))
        momentum_expected = [[
            torch.tensor(
                np.zeros((1,)), dtype=dtype, device=device), 
            torch.tensor(
                np.zeros((3,3)), dtype=dtype, device=device)
        ]]
        last_gradient = [
            torch.tensor(
                np.zeros((1,)), dtype=dtype, device=device),
            torch.tensor(
                np.zeros((3,3)), dtype=dtype, device=device)
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

            norms = torch.tensor(
                last_gradient_norm * gradient_norm, dtype=dtype, device=device)

            if norms == 0:
                alignment = 1
            else:
                alignment = product / norms

            last_gradient = [gradient[0].clone(), gradient[1].clone()]

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
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=3,
            verbose=False
        )
        solver = h.solver.NesterovSolver(
            embedder, learning_rate, momentum_decay, verbose=False
        )
        solver.cycle(times=times)


    def test_embedder_momentum_solver_integration(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=3,
            verbose=False,
            )
        solver = h.solver.MomentumSolver(
            embedder, learning_rate, momentum_decay, verbose=False
        )
        solver.cycle(times=times)


    def test_embedder_nesterov_solver_optimized_integration(self):

        d = 5
        times = 3
        learning_rate = 0.01
        momentum_decay = 0.8
        bigram = h.corpus_stats.get_test_bigram(2)
        M = h.M.M_pmi(bigram, neg_inf_val=0)

        # This test just makes sure that the solver and embedder interface
        # properly.  All is good as long as this doesn't throw errors.
        f_MSE = h.f_delta.DeltaMSE(bigram, M)
        embedder = h.embedder.HilbertEmbedder(
            f_MSE, d, learning_rate=learning_rate, shard_factor=3,
            verbose=False,
        )
        solver = h.solver.NesterovSolverOptimized(
            embedder, learning_rate, momentum_decay, verbose=False
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



class TestUnigram(TestCase):

    def test_unigram_creation_from_corpus(self):
        # Make a unigram and fill it with tokens and counts.
        unigram = h.unigram.Unigram()
        for token in h.corpus_stats.load_test_tokens():
            unigram.add(token)

        # The correct number of counts are registered for each token
        counts = Counter(h.corpus_stats.load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])

        # Test sorting.
        unigram.sort()
        for i in range(len(unigram.Nx)-1):
            self.assertTrue(unigram.Nx[i] >= unigram.Nx[i+1])


    def test_unigram_creation_from_Nxx(self):
        tokens = h.corpus_stats.load_test_tokens()
        dictionary = h.dictionary.Dictionary(tokens)
        Nx = [0] * len(dictionary)
        for token in tokens:
            Nx[dictionary.get_id(token)] += 1

        # Must supply a dictionary to create a non-empty Unigram.
        with self.assertRaises(ValueError):
            unigram = h.unigram.Unigram(Nx=Nx)

        unigram = h.unigram.Unigram(dictionary=dictionary, Nx=Nx)

        # The correct number of counts are registered for each token
        counts = Counter(h.corpus_stats.load_test_tokens())
        for token in counts:
            token_id = unigram.dictionary.get_id(token)
            self.assertEqual(unigram.Nx[token_id], counts[token])


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

        # Sort to make the test easier.  Note that normally truncation does
        # not require sorting, and does not consider token frequency.
        unigram.sort()
        expected_tokens = unigram.dictionary.tokens[:5]
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


    def test_sort(self):
        unsorted_dictionary = h.dictionary.Dictionary([
            'field', 'car', 'socks', 'banana'
        ])
        unsorted_Nxx = np.array([
            [0,0,0,1],
            [0,0,1,1],
            [0,1,0,3],
            [1,1,3,0],
        ])
        sorted_dictionary = h.dictionary.Dictionary([
            'banana', 'socks', 'car', 'field'])
        sorted_Nxx = np.array([
            [0,3,1,1],
            [3,0,1,0],
            [1,1,0,0],
            [1,0,0,0]
        ])

        unsorted_unigram = h.unigram.Unigram(
            unsorted_dictionary, unsorted_Nxx.sum(axis=1))
        bigram = h.bigram.Bigram(unsorted_unigram, unsorted_Nxx, verbose=False)

        # Bigram is unsorted
        self.assertFalse(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
        self.assertTrue(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))

        # Sorting bigram works.
        bigram.sort()
        self.assertTrue(np.allclose(bigram.Nxx.toarray(), sorted_Nxx))
        self.assertFalse(np.allclose(bigram.Nxx.toarray(), unsorted_Nxx))

        # The unigram is also sorted
        self.assertTrue(np.allclose(bigram.unigram.Nx, sorted_Nxx.sum(axis=1)))
        self.assertFalse(
            np.allclose(bigram.unigram.Nx, unsorted_Nxx.sum(axis=1)))
        self.assertEqual(bigram.dictionary.tokens, sorted_dictionary.tokens)


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
        dictionary, array, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.Bigram(unigram, array, verbose=False)

        unigram_Nx = bigram.Nx.sum(axis=1)
        unigram_N = bigram.Nx.sum()

        trunc_Nxx = np.array([[0,3,1], [3,0,1], [1,1,0]])
        trunc_Nx = np.sum(trunc_Nxx, axis=1, keepdims=True)
        trunc_Nxt = np.sum(trunc_Nxx, axis=0, keepdims=True)
        trunc_N = np.sum(trunc_Nx)

        trunc_unigram_Nx = unigram_Nx[:3]
        trunc_unigram_N = unigram_Nx[:3].sum()

        bigram.truncate(3)
        Nxx, Nx, Nxt, N = bigram

        self.assertTrue(np.allclose(Nxx, trunc_Nxx))
        self.assertTrue(np.allclose(Nx, trunc_Nx))
        self.assertTrue(np.allclose(Nxt, trunc_Nxt))
        self.assertTrue(np.allclose(N, trunc_N))
        self.assertTrue(np.allclose(bigram.unigram.Nx, trunc_unigram_Nx))
        self.assertTrue(np.allclose(bigram.unigram.N, trunc_unigram_N))


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
        When Bigram add, their counts add.
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
        

        # Ensure that bigram_sum is as desired.  First, sort to make comparison
        # easier.
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


class TestCoocStatsAlterators(TestCase):


    def test_expectation_w2v_undersample(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        t = 0.1

        # Calc expected Nxx, Nx, Nxt, N
        orig_Nxx, orig_Nx, orig_Nxt, orig_N = bigram
        survival_probability = torch.clamp(
            torch.sqrt(t / (orig_Nx / orig_N)), 0, 1)
        pxx = survival_probability * survival_probability.t()
        expected_Nxx = orig_Nxx * pxx 
        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = orig_Nxt.clone()
        expected_N = orig_N.clone()

        # Found values from the function we are testing
        undersampled = h.cooc_stats.expectation_w2v_undersample(
            bigram, t, verbose=False)

        usamp_Nxx, usamp_Nx, usamp_Nxt, usamp_N = undersampled
        self.assertTrue(torch.allclose(usamp_Nxx, expected_Nxx))
        self.assertTrue(torch.allclose(usamp_Nx, expected_Nx))
        self.assertTrue(torch.allclose(usamp_Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(usamp_N, expected_N))

        # Check that the original bigram has not been altered
        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(torch.allclose(Nxx, orig_Nxx))
        self.assertTrue(torch.allclose(Nx, orig_Nx))
        self.assertTrue(torch.allclose(Nxt, orig_Nxt))
        self.assertTrue(torch.allclose(N, orig_N))


    def test_w2v_undersample(self):
        # For reproducibile test, seed randomness
        np.random.seed(0)

        device=h.CONSTANTS.MATRIX_DEVICE
        t = 0.1
        num_replicates = 100
        window = 2
        bigram = h.corpus_stats.get_test_bigram(window)

        # Calc expected Nxx, Nx, Nxt, N
        orig_Nxx, orig_Nx, orig_Nxt, orig_N = bigram
        survival_probability = torch.clamp(
            torch.sqrt(t / (orig_Nx / orig_N)), 0, 1)
        pxx = survival_probability * survival_probability.t()
        expected_Nxx = orig_Nxx * pxx 
        expected_Nx = torch.sum(expected_Nxx, dim=1, keepdim=True)
        expected_Nxt = orig_Nxt.clone()
        expected_N = orig_N.clone()

        # Found values from the function we are testing
        mean_Nxx = torch.zeros(expected_Nxx.shape, device=device)
        mean_Nx = torch.zeros(expected_Nx.shape, device=device)
        mean_Nxt = torch.zeros(expected_Nxt.shape, device=device)
        mean_N = torch.zeros(expected_N.shape, device=device)
        for i in range(num_replicates):
            bigram = h.corpus_stats.get_test_bigram(window)
            undersampled = h.cooc_stats.w2v_undersample(
                bigram, t, verbose=False)
            usamp_Nxx, usamp_Nx, usamp_Nxt, usamp_N = undersampled
            mean_Nxx += usamp_Nxx / num_replicates
            mean_Nx += usamp_Nx / num_replicates
            mean_Nxt += usamp_Nxt / num_replicates
            mean_N += usamp_N / num_replicates

        self.assertTrue(torch.allclose(mean_Nxx, expected_Nxx, atol=0.5))
        self.assertTrue(torch.allclose(mean_Nx, expected_Nx, atol=1))
        self.assertTrue(torch.allclose(mean_Nxt, expected_Nxt, atol=1))
        self.assertTrue(torch.allclose(mean_N, expected_N, atol=2))

        # Check that the original bigram has not been altered
        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(torch.allclose(Nxx, orig_Nxx))
        self.assertTrue(torch.allclose(Nx, orig_Nx))
        self.assertTrue(torch.allclose(Nxt, orig_Nxt))
        self.assertTrue(torch.allclose(N, orig_N))


    def test_smooth_unigram(self):
        t = 0.1
        num_replicates = 100
        window = 2
        alpha = 0.75
        bigram = h.corpus_stats.get_test_bigram(window)

        orig_Nxx, orig_Nx, orig_Nxt, orig_N = bigram
        # The Nxt and N values are altered to reflect a smoothed unigram dist.
        expected_Nxt = orig_Nxt ** 0.75
        expected_N = torch.sum(expected_Nxt)
        # ... however, we expect Nxx and Nx to be unchanged
        expected_Nxx = orig_Nxx
        expected_Nx = orig_Nx

        smoothed = h.cooc_stats.smooth_unigram(
            bigram, alpha, verbose=False)
        smooth_Nxx, smooth_Nx, smooth_Nxt, smooth_N = smoothed

        self.assertTrue(torch.allclose(smooth_Nxx, expected_Nxx))

        self.assertTrue(torch.allclose(smooth_Nx, expected_Nx))
        self.assertTrue(torch.allclose(smooth_Nxt, expected_Nxt))
        self.assertTrue(torch.allclose(smooth_N, expected_N))

        # Check that the original bigram has not been altered
        Nxx, Nx, Nxt, N = bigram
        self.assertTrue(torch.allclose(Nxx, orig_Nxx))
        self.assertTrue(torch.allclose(Nx, orig_Nx))
        self.assertTrue(torch.allclose(Nxt, orig_Nxt))
        self.assertTrue(torch.allclose(N, orig_N))



class TestEmbeddings(TestCase):

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

        # Sort the embeddings according to a new shuffled token order.
        embeddings_to_be_sorted.sort_like(embeddings_to_sort_by)

        # The pristine and sorted embeddings are no longer the same
        self.assertFalse(torch.allclose(
            embeddings_pristine.V, embeddings_to_be_sorted.V))
        self.assertFalse(torch.allclose(
            embeddings_pristine.W, embeddings_to_be_sorted.W))

        # The sorted embeddings' dictionary matches shuffled token order.
        # And is different from the pristine embeddings' dictionary order.
        self.assertEqual(
            embeddings_to_be_sorted.dictionary.tokens,
            sort_tokens
        )
        self.assertNotEqual(
            embeddings_pristine.dictionary.tokens,
            embeddings_to_be_sorted.dictionary.tokens
        )

        # The embeddings themselves are reordered too, but they are still bound
        # to the same tokens.
        for i, token in enumerate(sort_tokens):
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


class TestDeltaW2VSampleFullCorpus(TestCase):

    def test_delta_w2v_sample_full_corpus(self):

        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        d = 300

        Nxx = torch.tensor(np.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample-full-corpus', 'Nxx.npy'
        )), device=device, dtype=dtype)
        Nxx_neg = torch.tensor(np.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample-full-corpus', 
            'Nxx_neg.npy'
        )), device=device, dtype=dtype)

        # Make a random M_hat to use as a test input.
        vocab = Nxx.shape[0]
        V = torch.rand((vocab, d), device=device)
        W = torch.rand((vocab, d), device=device)
        M_hat = torch.mm(W,V.t())

        # Calculate expected delta.
        expected_M  = torch.log(Nxx) - torch.log(Nxx_neg)
        # Force nans to be zero. 
        expected_M[expected_M != expected_M] = 0
        expected_delta = (Nxx + Nxx_neg) * (
            h.f_delta.sigmoid(expected_M) - h.f_delta.sigmoid(M_hat)
        )

        # Now check if this is what we get from DeltaW2VSamplesFullCorpus
        f_delta = h.f_delta.DeltaW2VSamplesFullCorpus(Nxx, Nxx_neg)
        delta = f_delta.calc_shard(M_hat)

        self.assertTrue(torch.allclose(delta, expected_delta))




class TestDeltaW2VSample(TestCase):

    def test_sample_reader(self):

        sample_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'trace.txt')
        dictionary = h.dictionary.Dictionary.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'dictionary'))
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, verbose=False)

        with open(sample_path) as sample_file:
            lines = sample_file.readlines()

        num_lines = 0
        num_epochs = 0
        while True:

            try:
                next_sample = sample_reader.next_sample()
            except h.f_delta.NewEpoch:
                num_epochs += 1
                num_lines += 1
                continue
            except h.f_delta.NoMoreSamples:
                break

            for token_id1, token_id2, val in next_sample:
                token1 = dictionary.tokens[token_id1]
                token2 = dictionary.tokens[token_id2]
                self.assertEqual(
                    lines[num_lines],
                    '{}\t{}\t{}\n'.format(token1,token2,val)
                )
                num_lines += 1

        sample_reader.sample_file.close()


    def test_sample_reader_ignore_epoch(self):

        sample_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'trace.txt')
        dictionary = h.dictionary.Dictionary.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'dictionary'))
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, signal_epochs=False, verbose=False)

        with open(sample_path) as sample_file:
            lines = sample_file.readlines()

        line_num = 0
        while True:

            try:
                next_sample = sample_reader.next_sample()
            except h.f_delta.NewEpoch:
                self.assertTrue(False) # We should not see new epoch signals
            except h.f_delta.NoMoreSamples:
                break

            for token_id1, token_id2, val in next_sample:
                token1 = dictionary.tokens[token_id1]
                token2 = dictionary.tokens[token_id2]

                if lines[line_num].startswith('Epoch'):
                    line_num += 1

                self.assertEqual(
                    lines[line_num],
                    '{}\t{}\t{}\n'.format(token1,token2,val)
                )
                line_num += 1

        sample_reader.sample_file.close()
        

    def test_f_delta_w2v_sample(self):

        torch.random.manual_seed(0)
        dtype=h.CONSTANTS.DEFAULT_DTYPE
        device=h.CONSTANTS.MATRIX_DEVICE
        sample_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'trace.txt')
        dictionary_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'dictionary')
        d = 300
        dictionary = h.dictionary.Dictionary.load(dictionary_path)
        vocab = len(dictionary)

        # Create random vectors, and use them to make an M_hat
        V = h.utils.sample_sphere(vocab, d)
        W = h.utils.sample_sphere(vocab, d)
        M_hat = torch.mm(W,V.t())

        # Make a DeltaW2VSamples instance
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, signal_epochs=False, verbose=False)
        f_delta = h.f_delta.DeltaW2VSamples(sample_reader)

        # Make a sample reader, separate from the one used to make the 
        # DeltaW2V Samples instance, for generating the expected deltas.
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, signal_epochs=False, verbose=False)

        # Get some samples, and use them to make an expected delta
        for sample_num in range(6):
            sample = sample_reader.next_sample()
            Nxx = torch.zeros((vocab, vocab), device=device, dtype=dtype)
            Nxx_neg = torch.zeros((vocab, vocab), device=device, dtype=dtype)
            for token_id1, token_id2, val in sample:
                if val == 1:
                    Nxx[token_id1, token_id2] += 1
                elif val == 0:
                    Nxx_neg[token_id1, token_id2] += 1
                else:
                    assert False, 'Val must be 1 or 0'
            expected_delta = (
                (Nxx + Nxx_neg) * (
                    h.f_delta.sigmoid(torch.log(Nxx / Nxx_neg))
                    - h.f_delta.sigmoid(M_hat)
                )
            )
            # Set nans in expected_delta to zero.  They represent places where
            # both Nxx and Nxx_neg are zero, and so should be zero.  Use the 
            # trick that nan != nan.
            expected_delta[expected_delta != expected_delta] = 0
            found_delta = f_delta.calc_shard(M_hat)
            self.assertTrue(torch.allclose(found_delta, expected_delta))

        sample_reader.sample_file.close()
        f_delta.sample_reader.sample_file.close()


    #def test_w2v_sample_embedder(self):

    #    torch.random.manual_seed(0)

    #    dtype=h.CONSTANTS.DEFAULT_DTYPE
    #    device=h.CONSTANTS.MATRIX_DEVICE
    #    sample_path = os.path.join(
    #        h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'trace.txt')
    #    dictionary_path = os.path.join(
    #        h.CONSTANTS.TEST_DIR, 'delta-w2v-sample', 'dictionary')
    #    d = 300
    #    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    #    vocab = len(dictionary)

    #    # Make an embedder based on a DeltaW2VSamples instance.
    #    sample_reader = h.f_delta.SampleReader(
    #        sample_path, dictionary, verbose=False)
    #    f_delta = h.f_delta.DeltaW2VSamples(sample_reader)
    #    embedder = h.embedder.Embedder(
    #        f_delta,d=d, num_vecs=vocab, num_covecs=vocab, learning_rate=1e-3,
    #        shard_factor=1, verbose=False
    #    )

    #    # Create random vectors, and use them to make an M_hat
    #    V = embedder.V.clone()
    #    W = embedder.W.clone()

    #    M_hat = torch.mm(W,V.t())

    #    # Make a sample reader, separate from the one used to make the 
    #    # DeltaW2V Samples instance, for generating the expected deltas.
    #    sample_reader = h.f_delta.SampleReader(
    #        sample_path, dictionary, verbose=False)

    #    # Get some samples, and use them to make an expected delta
    #    for sample_num in range(6):
    #        sample = sample_reader.next_sample()
    #        Nxx = torch.zeros((vocab, vocab), device=device, dtype=dtype)
    #        Nxx_neg = torch.zeros((vocab, vocab), device=device, dtype=dtype)
    #        for token_id1, token_id2, val in sample:
    #            if val == 1:
    #                Nxx[token_id1, token_id2] += 1
    #            elif val == 0:
    #                Nxx_neg[token_id1, token_id2] += 1
    #            else:
    #                assert False, 'Val must be 1 or 0'
    #        expected_delta = (
    #            (Nxx + Nxx_neg) * (
    #                h.f_delta.sigmoid(torch.log(Nxx / Nxx_neg))
    #                - h.f_delta.sigmoid(M_hat)
    #            )
    #        )
    #        # Set nans in expected_delta to zero.  They represent places where
    #        # both Nxx and Nxx_neg are zero, and so should be zero.  Use the 
    #        # trick that nan != nan.
    #        expected_delta[expected_delta != expected_delta] = 0
    #        found_delta = f_delta.calc_shard(M_hat)
    #        self.assertTrue(torch.allclose(found_delta, expected_delta))

    #    sample_reader.sample_file.close()
    #    f_delta.sample_reader.sample_file.close()



class TestW2VReplica(TestCase):

    def test_w2v_updates(self):
        learning_rate = 0.025

        # Create a word2vec replica embedder
        sample_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-w2v', 'trace.txt')
        dictionary = h.dictionary.Dictionary.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-w2v', 'dictionary'))
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, verbose=False)

        w2v_replica = h.embedder.W2VReplica(
            sample_reader, learning_rate=learning_rate, delay_update=False)

        # Initialize it with vectors used to initialize a real w2v run
        embeddings_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-w2v', 'epoch-init')
        init_embeddings = h.embeddings.Embeddings.load(embeddings_path)
        w2v_replica.V = init_embeddings.V.clone()
        w2v_replica.W = init_embeddings.W.clone()

        # First read past the first epoch marker, which comes before any
        # samples.
        try:
            w2v_replica.cycle(None)
        except h.f_delta.NewEpoch:
            pass

        # Now track updates through 10 epochs, comparing the resulting
        # embeddings to those obtained by the original w2v algo.
        for i in range(5):

            try:
                w2v_replica.cycle(None)
            except h.f_delta.NewEpoch:
                pass

            # The w2v_replica's vectors should now match those obtained by w2v
            # after one epoch.
            embeddings_path = os.path.join(
                h.CONSTANTS.TEST_DIR, 'test-w2v', 'epoch{}'.format(i))
            expected_embeddings = h.embeddings.Embeddings.load(embeddings_path)
            self.assertTrue(torch.allclose(
                w2v_replica.V, expected_embeddings.V, atol=0.08))
            self.assertTrue(torch.allclose(
                w2v_replica.W, expected_embeddings.W, atol=0.08))


    def test_w2v_updates_no_delay(self):

        learning_rate = 0.025
        delay_update = False

        # Create a word2vec replica embedder
        sample_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-w2v', 'trace.txt')
        dictionary = h.dictionary.Dictionary.load(os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-w2v', 'dictionary'))
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, verbose=False)
        w2v_replica = h.embedder.W2VReplica(
            sample_reader,
            learning_rate=learning_rate,
            delay_update=delay_update
        )

        # Clone the initialized embeddings.  We'll use these as a starting
        # point for calculating the expected embeddings after an epoch.
        V = w2v_replica.V.clone()
        W = w2v_replica.W.clone()

        # To apply an epoch's worth of updates, it is necessary to pass
        # the first epoch marker, and proceed up until the second.
        for i in range(2):
            try:
                w2v_replica.cycle(None)
            except h.f_delta.NewEpoch:
                pass

        # Calculate the expected embeddings after one epoch
        sample_reader = h.f_delta.SampleReader(
            sample_path, dictionary, verbose=False)
        num_epochs = 0
        while num_epochs < 2:
            try:
                next_sample = sample_reader.next_sample()
            except h.f_delta.NewEpoch:
                num_epochs += 1
                continue

            for fields in next_sample:
                t1, t2, val = fields[:3]
                dot = torch.dot(V[t1], W[t2])
                g = ( val - 1/(1+np.e**(-dot)) ) * learning_rate
                W[t2] += g * V[t1]
                V[t1] += g * W[t2]

        self.assertTrue(torch.allclose(w2v_replica.V, V))

        self.assertTrue(torch.allclose(w2v_replica.W, W))






if __name__ == '__main__':

    if '--cpu' in sys.argv:
        print('\nTESTING DEVICE: CPU\n')
        sys.argv.remove('--cpu')
        h.CONSTANTS.MATRIX_DEVICE = 'cpu'
    else:
        print('\nTESTING DEVICE: CUDA.  Use --cpu to test on cpu.\n')

    main()

