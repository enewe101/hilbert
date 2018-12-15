import sys
import hilbert as h
import torch
import numpy as np
from unittest import main, TestCase
from itertools import product

VERBOSE = False

def vprint(*args):
    if VERBOSE:
        print(*args)


class TestSharder(TestCase):

    def test_glove_sharder(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        sharder = h.msharder.GloveSharder(bigram)
        sharder._load_shard(None)
        Nxx, _,_,_ = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)

        xmax = 100.
        alpha = 0.75
        m = lambda t: (t / xmax).pow(alpha)
        em = m(Nxx)
        em[em > 1] = 1
        self.assertTrue(torch.allclose(em, sharder.multiplier))

        zidx = Nxx == 0
        self.assertTrue(all(sharder.M[zidx] == 0))
        self.assertTrue(all(sharder.multiplier[zidx] == 0))


    def test_ppmi_sharder(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        sharder = h.msharder.PPMISharder(bigram)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        M[M<0] = 0
        self.assertTrue(torch.allclose(M, sharder.M))


    def test_mse_minibatching_loss(self):
        bigram = h.corpus_stats.get_test_bigram(2)

        for keep in [0.1, 0.5, 1]:
            for scale, constructor in [(1, h.msharder.PPMISharder),
                                       (2, h.msharder.GloveSharder)]:

                sharder = constructor(bigram, update_density=keep)

                expected_scaler = float(np.prod(bigram.Nxx.shape) * keep)
                self.assertEqual(expected_scaler, sharder.criterion.rescale)

                sharder._load_shard(None)
                mhat = torch.ones(
                    sharder.M.shape, device=h.CONSTANTS.MATRIX_DEVICE)

                try:
                    weights = sharder.multiplier
                except AttributeError:
                    weights = 1

                torch.manual_seed(1)
                loss = sharder.calc_shard_loss(mhat, None)
                mse = scale * 0.5 * weights * ((mhat - sharder.M) ** 2)

                torch.manual_seed(1)
                exloss = torch.nn.functional.dropout(
                    mse, p=1-keep, training=True)
                exloss = keep * torch.sum(exloss) / expected_scaler

                self.assertEqual(loss, exloss)


    def test_max_likelihood_sharder(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        sharder = h.msharder.MaxLikelihoodSharder(bigram)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)

        self.assertTrue(torch.allclose(Pxx_data, sharder.Pxx_data))
        self.assertTrue(torch.allclose(Pxx_independent,sharder.Pxx_independent))


    def test_max_posterior_sharder(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        sharder = h.msharder.MaxPosteriorSharder(bigram)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)

        Pxx_independent = (Nx / N) * (Nxt / N)

        # These functions are assumed correct here, tested elsewhere
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            (Nxx, Nx, Nxt, N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)

        N_posterior = N + alpha + beta - 1
        Pxx_posterior = (Nxx + alpha) / N_posterior

        self.assertTrue(torch.allclose(Pxx_posterior, sharder.Pxx_posterior))
        self.assertTrue(torch.allclose(N_posterior, sharder.N_posterior))
        self.assertTrue(torch.allclose(Pxx_independent,sharder.Pxx_independent))


    def test_KL_sharder(self):

        bigram = h.corpus_stats.get_test_bigram(2)
        sharder = h.msharder.KLSharder(bigram)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            (Nxx, Nx, Nxt, N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1

        a = Nxx + alpha
        b = N - Nxx + beta
        digamma_a = torch.digamma(a) - torch.digamma(a+b)
        digamma_b = torch.digamma(b) - torch.digamma(a+b)

        self.assertTrue(torch.allclose(N_posterior, sharder.N_posterior))
        self.assertTrue(torch.allclose(Pxx_independent,sharder.Pxx_independent))
        self.assertTrue(torch.allclose(digamma_a, sharder.digamma_a))
        self.assertTrue(torch.allclose(digamma_b, sharder.digamma_b))



class TestLoss(TestCase):

    def test_max_likelihood_loss(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

        loss_class = h.hilbert_loss.MaxLikelihoodLoss(keep_prob, ncomponents)

        M_hat = torch.ones_like(Nxx)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        Pxx_model = Pxx_independent * torch.exp(M_hat)

        loss_term1 = Pxx_data * torch.log(Pxx_model)
        loss_term2 = (1-Pxx_data) * torch.log(1 - Pxx_model)
        expected_loss = -torch.sum(loss_term1 + loss_term2) / float(ncomponents)
        found_loss = loss_class(M_hat, Pxx_data, Pxx_independent)

        self.assertTrue(torch.allclose(found_loss, expected_loss))



    def test_max_posterior_loss(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

        loss_class = h.hilbert_loss.MaxPosteriorLoss(keep_prob, ncomponents)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            (Nxx, Nx, Nxt, N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1
        Pxx_posterior = (Nxx + alpha) / N_posterior
        M_hat = torch.ones_like(Nxx)
        Pxx_model = Pxx_independent * torch.exp(M_hat)

        loss_term1 = Pxx_posterior * torch.log(Pxx_model)
        loss_term2 = (1-Pxx_posterior) * torch.log(1 - Pxx_model)
        scaled_loss = (N_posterior / N) * (loss_term1 + loss_term2)
        expected_loss = - torch.sum(scaled_loss) / float(ncomponents)
        found_loss = loss_class(
            M_hat, N, N_posterior, Pxx_posterior, Pxx_independent 
        )

        self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_KL_loss(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

        loss_obj = h.hilbert_loss.KLLoss(keep_prob, ncomponents)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_independent = (Nx / N) * (Nxt / N)
        exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
            (Nxx, Nx, Nxt, N))
        alpha, beta = h.corpus_stats.calc_prior_beta_params(
            (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
        N_posterior = N + alpha + beta - 1

        a = Nxx + alpha
        b = N - Nxx + beta
        digamma_a = torch.digamma(a) - torch.digamma(a+b)
        digamma_b = torch.digamma(b) - torch.digamma(a+b)

        M_hat = torch.ones_like(Nxx)

        Pxx_model = Pxx_independent * torch.exp(M_hat)
        a_hat = N_posterior * Pxx_model
        b_hat = N_posterior * (1 - Pxx_model) + 1
        lbeta = torch.lgamma(a_hat) + torch.lgamma(b_hat) - torch.lgamma(
            a_hat + b_hat)
        KL = (lbeta - a_hat * digamma_a - b_hat * digamma_b) / N
        expected_loss = torch.sum(KL) / float(ncomponents)

        found_loss = loss_obj(
            M_hat, N, N_posterior, Pxx_independent, digamma_a, digamma_b)

        self.assertTrue(torch.allclose(found_loss, expected_loss))


class TestAutoEmbedder(TestCase):

    def test_embedder_functionality(self):
        terms, contexts = 100, 500
        d = 300
        shard = (None, None)

        V = torch.rand(terms, d)
        W = torch.rand(contexts, d)
        vbias = torch.rand(terms)
        wbias = torch.rand(contexts)

        # make the expected results
        exp_no_w_bias = V @ V.t()
        exp_no_bias = W @ V.t()
        exp_no_w = (V @ V.t()) + vbias.reshape(1, -1) + vbias.reshape(-1, 1)
        exp_all = (W @ V.t()) + vbias.reshape(1, -1) + wbias.reshape(-1, 1)

        options = [
            ({}, exp_no_w_bias) ,
            ({'W': W}, exp_no_bias),
            ({'v_bias': vbias}, exp_no_w),
            ({'W': W, 'v_bias': vbias, 'w_bias': wbias}, exp_all)
        ]
        for kwargs, expected_M in options:
            ae = h.autoembedder.AutoEmbedder(V, **kwargs)
            got_M = ae(shard)
            self.assertTrue(torch.allclose(got_M, expected_M))


    def test_emb_solver_functionality(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        glv_sharder = h.msharder.GloveSharder(bigram)
        ppmi_sharder = h.msharder.PPMISharder(bigram)
        w2v_sharder = h.msharder.Word2vecSharder(bigram, 15)
        opt = torch.optim.Adam
        shape = bigram.Nxx.shape

        # TODO: right now this doesn't work for sharding > 1.
        # Perhaps this is because the bigram matrix is 11x11, which
        # is too small?
        sharders = [glv_sharder, ppmi_sharder, w2v_sharder]
        shard_fs = [1, 2, 5]
        oss = [True, False]
        lbs = [True, False]
        options = product(sharders, shard_fs, oss, lbs)

        for sharder, shard_factor, one_sided, learn_bias in options:
            vprint('\n', sharder.__class__)
            vprint('one_sided =', one_sided, 'learn_bias =', learn_bias, 
                    'shard_factor =', shard_factor)

            solver = h.autoembedder.HilbertEmbedderSolver(
                sharder, opt, d=300,
                shape=shape if not one_sided else (shape[0],),
                learning_rate=0.003,
                shard_factor=shard_factor,
                one_sided=one_sided,
                learn_bias=learn_bias,
                device=h.CONSTANTS.MATRIX_DEVICE,
                verbose=False,
            )

            # check to make sure we get the same loss after resetting
            l1 = solver.cycle(epochs=10, shard_times=1, hold_loss=True)
            solver.restart()
            l2 = solver.cycle(epochs=10, shard_times=1, hold_loss=True)
            solver.restart()
            l3 = solver.cycle(epochs=5, shard_times=1, hold_loss=True)
            l3 += solver.cycle(epochs=5, shard_times=1, hold_loss=True)
            self.assertTrue(np.allclose(l1, l2))
            self.assertTrue(np.allclose(l1, l3))
            solver.restart()

            # here we're ensuring that the equality between the solver parameters
            # and the torch module parameters are always the same, before and
            # after learning
            for _ in range(3):
                V, W, vb, wb = solver.get_params()
                aV, aW, avb, awb = (
                    solver.learner.V, solver.learner.W,
                    solver.learner.v_bias, solver.learner.w_bias
                )
                for t1, t2 in [(V, aV), (W, aW), (vb, avb), (wb, awb)]:
                    if t1 is None and t2 is None:
                        continue
                    self.assertTrue(torch.allclose(t1, t2))

                solver.cycle(1)


    def test_solver_nan(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        ppmi_sharder = h.msharder.PPMISharder(bigram)
        opt = torch.optim.SGD
        shape = bigram.Nxx.shape

        solver = h.autoembedder.HilbertEmbedderSolver(
            ppmi_sharder, opt, d=300,
            shape=shape,
            learning_rate=1000,
            shard_factor=1,
            one_sided=False,
            learn_bias=False,
            device=h.CONSTANTS.MATRIX_DEVICE,
            verbose=VERBOSE,
        )

        # check to make sure we get the same loss after resetting
        with self.assertRaises(h.autoembedder.DivergenceError):
            solver.cycle(epochs=100, shard_times=1, hold_loss=True)


    def test_w2v_solver(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        shape = bigram.Nxx.shape
        w2v_sharder = h.msharder.Word2vecSharder(bigram, 15, update_density=1)
        opt = torch.optim.Adam

        solver = h.autoembedder.HilbertEmbedderSolver(
            w2v_sharder, opt, d=10,
            shape=shape,
            learning_rate=0.01,
            shard_factor=1,
            one_sided=False,
            learn_bias=False,
            device=h.CONSTANTS.MATRIX_DEVICE,
            verbose=VERBOSE,
        )
        w2v_sharder._load_shard(None)

        for _ in range(3):
            V, W, _, _ = solver.get_params()
            mhat = W @ V.t()

            loss_value = w2v_sharder._get_loss(mhat).item()

            # get expected
            smhat = mhat.sigmoid()
            total = -torch.mean(
                (w2v_sharder.Nxx * torch.log(smhat)) +
                (w2v_sharder.N_neg * torch.log(1 - smhat))
            )
            expected_loss = total.item()
            eps = 0.05
            self.assertTrue(
                expected_loss - eps < loss_value < expected_loss + eps)
            solver.cycle(5, True)



if __name__ == '__main__':

    if '--cpu' in sys.argv:
        print('\nTESTING DEVICE: CPU\n')
        sys.argv.remove('--cpu')
        h.CONSTANTS.MATRIX_DEVICE = 'cpu'
    else:
        print('\nTESTING DEVICE: CUDA.  Use --cpu to test on cpu.\n')

    if '-v' in sys.argv:
        VERBOSE = True

    main()
