from unittest import TestCase
import hilbert as h

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


