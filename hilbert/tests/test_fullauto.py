import os
from itertools import product
from unittest import TestCase

import numpy as np

import hilbert as h

try:
    import torch
except ImportError:
    torch = None

VERBOSE = False

def vprint(*args):
    if VERBOSE:
        print(*args)



class TestLoss(TestCase):


    def test_w2v_loss(self):

        k = 15
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        uNx, uNxt, uN = bigram.unigram.load_shard(
            None, h.CONSTANTS.MATRIX_DEVICE) 
        ncomponents = np.prod(Nxx.shape)
        shard_id = None

        sigmoid = lambda a: 1/(1+torch.exp(-a))
        N_neg = h.msharder.Word2vecSharder.negative_sample(Nxx, Nx, uNxt, uN, k)
        M_hat = torch.ones_like(Nxx)
        loss_term_1 = Nxx * torch.log(sigmoid(M_hat))
        loss_term_2 = N_neg * torch.log(1-sigmoid(M_hat))
        loss_array = -(loss_term_1 + loss_term_2)

        for keep_prob in [1, 0.75, 0.1]:

            torch.manual_seed(0)
            rescale = float(keep_prob * ncomponents)
            loss_masked = h.hilbert_loss.keep(loss_array, keep_prob)
            expected_loss = torch.sum(loss_masked) / rescale

            torch.manual_seed(0)
            loss_obj = h.hilbert_loss.Word2vecLoss(keep_prob, ncomponents)
            found_loss = loss_obj(shard_id, M_hat, {'Nxx':Nxx, 'N_neg':N_neg})

            self.assertTrue(torch.allclose(found_loss, expected_loss))



    def test_max_likelihood_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

        M_hat = torch.ones_like(Nxx)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)
        Pxx_model = Pxx_independent * torch.exp(M_hat)

        loss_term1 = Pxx_data * M_hat
        loss_term2 = (1-Pxx_data) * torch.log(1 - Pxx_model)
        loss_array = loss_term1 + loss_term2

        shard_id = None
        for temperature in [1,10]:
            tempered_loss = loss_array * Pxx_independent**(1/temperature - 1)
            expected_loss = -torch.sum(tempered_loss) / float(ncomponents)
            loss_class = h.hilbert_loss.MaxLikelihoodLoss(
                keep_prob, ncomponents, temperature=temperature)
            found_loss = loss_class(shard_id, M_hat, {
                'Pxx_data': Pxx_data, 'Pxx_independent': Pxx_independent })
            self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_max_posterior_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

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

        loss_term1 = Pxx_posterior * M_hat
        loss_term2 = (1-Pxx_posterior) * torch.log(1 - Pxx_model)
        scaled_loss = (N_posterior / N) * (loss_term1 + loss_term2)

        shard_id = None
        for temperature in [1, 10]:
            tempered_loss = scaled_loss * Pxx_independent ** (1/temperature - 1)
            expected_loss = - torch.sum(tempered_loss) / float(ncomponents)
            loss_class = h.hilbert_loss.MaxPosteriorLoss(
                keep_prob, ncomponents, temperature=temperature)
            found_loss = loss_class(shard_id, M_hat, {
                'N': N, 'N_posterior': N_posterior, 
                'Pxx_posterior': Pxx_posterior,
                'Pxx_independent': Pxx_independent
            })
            self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_KL_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        ncomponents = np.prod(Nxx.shape)
        keep_prob = 1

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

        shard_id = None
        for temperature in [1, 10]:
            tempered_KL = KL * Pxx_independent ** (1/temperature - 1)
            expected_loss = torch.sum(tempered_KL) / float(ncomponents)
            loss_obj = h.hilbert_loss.KLLoss(
                keep_prob, ncomponents, temperature=temperature)
            found_loss = loss_obj(shard_id, M_hat, {
                'N': N, 'N_posterior': N_posterior, 
                'Pxx_independent': Pxx_independent, 
                'digamma_a': digamma_a, 'digamma_b': digamma_b, 
            })
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


#
#   This takes too long to run!  Need to make it quicker.  Hard-skipped right
#   Now to save my sanity :)
#
#    def test_emb_solver_functionality(self):
#
#        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
#        sector_factor = 3
#        num_loaders = 9
#        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
#        keep_prob = 1
#        opt = torch.optim.Adam
#        shape = bigram.Nxx.shape
#
#        # TODO: right now this doesn't work for sharding > 1.
#        # Perhaps this is because the bigram matrix is 11x11, which
#        # is too small?
#        loaders_losses = [
#            (h.bigram_loader.PPMILoader, h.hilbert_loss.MSELoss),
#            (h.bigram_loader.GloveLoader, h.hilbert_loss.MSELoss),
#            (h.bigram_loader.Word2vecLoader, h.hilbert_loss.Word2vecLoss),
#            (h.bigram_loader.MaxLikelihoodLoader, 
#                h.hilbert_loss.MaxLikelihoodLoss),
#            (h.bigram_loader.MaxPosteriorLoader, 
#                h.hilbert_loss.MaxPosteriorLoss),
#            (h.bigram_loader.KLLoader, h.hilbert_loss.KLLoss),
#        ]
#        shard_fs = [1, 3]
#        oss = [True, False]
#        lbs = [True, False]
#        options = product(loaders_losses, shard_fs, oss, lbs)
#
#        for (loader_class, loss_class), sf, one_sided, learn_bias in options:
#
#            vprint('\n', loader_class)
#            vprint('one_sided =', one_sided, 'learn_bias =', learn_bias, 
#                    'shard_factor =', sf)
#
#            loader = loader_class(
#                bigram_path, sector_factor, sf, num_loaders, verbose=False)
#            loss = loss_class(keep_prob, bigram.vocab**2)
#            solver = h.autoembedder.HilbertEmbedderSolver(
#                loader, loss, opt, d=300, learning_rate=0.003,
#                shape=shape if not one_sided else (shape[0],),
#                one_sided=one_sided, learn_bias=learn_bias, verbose=False,
#            )
#
#            # check to make sure we get the same loss after resetting
#            l1 = solver.cycle(epochs=10, shard_times=1, hold_loss=True)
#            solver.restart()
#            l2 = solver.cycle(epochs=10, shard_times=1, hold_loss=True)
#            solver.restart()
#            l3 = solver.cycle(epochs=5, shard_times=1, hold_loss=True)
#            l3 += solver.cycle(epochs=5, shard_times=1, hold_loss=True)
#            self.assertTrue(np.allclose(l1, l2))
#            self.assertTrue(np.allclose(l1, l3))
#            solver.restart()
#
#            # here we're ensuring that the equality between the solver
#            # parameters and the torch module parameters are always the same,
#            # before and after learning
#            for _ in range(3):
#                V, W, vb, wb = solver.get_params()
#                aV, aW, avb, awb = (
#                    solver.learner.V, solver.learner.W,
#                    solver.learner.v_bias, solver.learner.w_bias
#                )
#                for t1, t2 in [(V, aV), (W, aW), (vb, avb), (wb, awb)]:
#                    if t1 is None and t2 is None:
#                        continue
#                    self.assertTrue(torch.allclose(t1, t2))
#
#                solver.cycle(1)


    def test_solver_nan(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        keep = 1
        sector_factor = 3
        shard_factor = 4
        num_loaders = 9
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        loader = hilbert.bigram.bigram_preloader.PPMILoader(
            bigram_path, sector_factor, shard_factor, num_loaders, verbose=False)
        loss = h.hilbert_loss.MSELoss(keep, bigram.vocab**2) 
        opt = torch.optim.SGD
        shape = bigram.Nxx.shape
        solver = h.autoembedder.HilbertEmbedderSolver(
            loader=loader, 
            loss=loss, 
            optimizer_constructor=opt, 
            d=300,
            shape=shape,
            learning_rate=1e5,
            one_sided=False,
            learn_bias=False,
            device=h.CONSTANTS.MATRIX_DEVICE,
            verbose=VERBOSE,
        )

        # The high learning rate causes `nan`s, which should raise an error.
        with self.assertRaises(h.autoembedder.DivergenceError):
            solver.cycle(iters=1, shard_times=100, hold_loss=True)


    def test_w2v_solver(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        shape = bigram.Nxx.shape
        scale = bigram.vocab**2
        learning_rate = 0.000001
        keep = 1
        sector_factor = 1
        shard_factor = 1
        num_loaders = 1
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        loader = hilbert.bigram.bigram_preloader.Word2vecLoader(
            bigram_path, sector_factor, shard_factor, num_loaders, verbose=False)
        outer_loader = hilbert.bigram.bigram_preloader.Word2vecLoader(
            bigram_path, sector_factor, shard_factor, num_loaders, verbose=False)
        loss = h.hilbert_loss.Word2vecLoss(keep, bigram.vocab**2) 

        solver = h.autoembedder.HilbertEmbedderSolver(
            loader=loader, 
            loss=loss,
            optimizer_constructor=torch.optim.SGD, 
            d=300,
            shape=shape,
            learning_rate=learning_rate,
            one_sided=False,
            learn_bias=False,
            device=h.CONSTANTS.MATRIX_DEVICE,
            verbose=VERBOSE,
        )

        V, W, _, _ = solver.get_params()
        expected_V = V.clone()
        expected_W = W.clone()

        # Loader is superfluous because there's just one shard and just one 
        # sector.  So just give us the data!
        shard_id, shard_data = next(iter(outer_loader))

        for iteration in range(5):

            # Manually calculate the gradient and expected update
            mhat = expected_W @ expected_V.t()
            N_sum = (shard_data['Nxx'] + shard_data['N_neg'])
            delta = shard_data['Nxx'] - mhat.sigmoid() * N_sum
            neg_grad_V = torch.mm(delta.t(), expected_W) / scale
            neg_grad_W = torch.mm(delta, expected_V) / scale
            expected_V += learning_rate * neg_grad_V
            expected_W += learning_rate * neg_grad_W

            # Let the solver make one update
            solver.cycle()
            found_V, found_W, _, _ = solver.get_params()

            # Check that the solvers update matches expectation.
            self.assertTrue(torch.allclose(found_V, expected_V))
            self.assertTrue(torch.allclose(found_W, expected_W))


