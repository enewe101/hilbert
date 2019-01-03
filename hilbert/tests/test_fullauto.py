import hilbert as h
import numpy as np
from unittest import TestCase
from itertools import product

try:
    import torch
except ImportError:
    torch = None

VERBOSE = False

def vprint(*args):
    if VERBOSE:
        print(*args)


class MockMSELossMaskedDiag(h.hilbert_loss.HilbertLoss):
    """
    Provides an implementation of MSELoss that always masks the diagonal using
    multiplication with a mask, to help test the correctness of the
    mask_diagonal flag supported by other HilbertLoss instances.
    """
    def _forward(self, M_hat, shard, M, weights=None):
        weights = 1 if weights is None else weights
        loss_array = 0.5 * weights * ((M_hat - M) ** 2)
        device = h.CONSTANTS.MATRIX_DEVICE

        # Mask the main diagonal
        if h.shards.on_diag(shard):
            mask = torch.ones_like(loss_array, device=device) 
            for i in range(min(mask.shape)):
                mask[i,i] = 0
            loss_array = loss_array * mask

        return loss_array


class MockPPMISharder(h.msharder.PPMISharder):
    criterion_class = MockMSELossMaskedDiag


class TestSharder(TestCase):

    def test_glove_sharder(self):
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
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
        bigram, unigram, Nxx = h.corpus_stats.get_test_bigram_base()
        sharder = h.msharder.PPMISharder(bigram)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
        M[M<0] = 0
        self.assertTrue(torch.allclose(M, sharder.M))


    def test_loss_with_masked_diagonal(self):
        """
        Tests that masking the diagonal behaves in the expected way, for the
        PPMI sharder, by comparing it to a mocked version that is hard-coded
        to mask the diagonal of the loss function in a way that will certainly
        produce the desired effect.  Masking diagonal is tested with other
        sharding classes (but without mocks) in another test.
        """


        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        keep = 1

        # Test for different shard factors to make sure the diagonal elements
        # are always correctly found
        for shard_factor in [1,2,3]:

            shards = h.shards.Shards(shard_factor)

            # Run an ordinary sharder without any diagonal masking.
            sharder = h.msharder.PPMISharder(
                bigram, update_density=keep, mask_diagonal=False)
            mhat = torch.nn.Parameter(torch.ones(
                (bigram.vocab, bigram.vocab), device=h.CONSTANTS.MATRIX_DEVICE
            ))
            optimizer = torch.optim.SGD((mhat,), lr=1.)
            for shard in shards:
                optimizer.zero_grad()
                loss = sharder.calc_shard_loss(mhat[shard], shard)
                loss.backward()
                optimizer.step()

            # Run the a masked sharder for one update.
            masked_diag_sharder = h.msharder.PPMISharder(
                bigram, update_density=keep, mask_diagonal=True)
            masked_mhat = torch.nn.Parameter(torch.ones(
                (bigram.vocab, bigram.vocab), device=h.CONSTANTS.MATRIX_DEVICE
            ))
            optimizer = torch.optim.SGD((masked_mhat,), lr=1.)
            for shard in shards:
                optimizer.zero_grad()
                loss = masked_diag_sharder.calc_shard_loss(
                    masked_mhat[shard], shard)
                loss.backward()
                optimizer.step()

            # Run the mock masked sharder for one update.
            mock_diag_sharder = MockPPMISharder(
                bigram, update_density=keep, mask_diagonal=False)
            mock_mhat = torch.nn.Parameter(torch.ones(
                (bigram.vocab, bigram.vocab), device=h.CONSTANTS.MATRIX_DEVICE
            ))
            optimizer = torch.optim.SGD((mock_mhat,), lr=1.)
            for shard in shards:
                optimizer.zero_grad()
                loss = mock_diag_sharder.calc_shard_loss(mock_mhat[shard],shard)
                loss.backward()
                optimizer.step()

            # Every model should have undergone an update.
            ones = torch.ones_like(mhat)
            self.assertFalse(torch.allclose(mhat, ones))
            self.assertFalse(torch.allclose(masked_mhat, ones))
            self.assertFalse(torch.allclose(mock_mhat, ones))

            # Masked sharder should match manually masked mock sharder.
            self.assertTrue(torch.allclose(masked_mhat, mock_mhat))

            # Masked and non-masked sharders should differ along the diagonal,
            # but be the same elsewhere.  First construct indices needed to
            # address these regions.
            device = h.CONSTANTS.MATRIX_DEVICE
            dtype = torch.uint8
            diagonal = torch.eye(mhat.shape[0], dtype=dtype, device=device)
            off_diagonal = torch.ones(
                mhat.shape,dtype=dtype,device=device) - diagonal

            # Now show they are different on diagonal, but same off diagonal. 
            self.assertFalse(torch.allclose(
                masked_mhat[diagonal], mhat[diagonal]))
            self.assertTrue(torch.allclose(
                masked_mhat[off_diagonal], mhat[off_diagonal]))


    def test_all_sharders_can_mask_diagonal(self):
        """
        Tests that every combination of sharder class and masking or not 
        masking the diagonal of the loss function works as expected.
        """

        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        keep = 1

        # We will need to address diagonal and non-diagonal entries in 
        # M-matrices below.  Prepare the indices now.
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = torch.uint8
        diagonal = torch.eye(bigram.shape[0], dtype=dtype, device=device)
        off_diagonal = torch.ones(
            bigram.shape, dtype=dtype,device=device) - diagonal

        # Test for different shard factors to make sure the diagonal elements
        # are always correctly found
        sharder_classes_and_learning_rates = [
            (h.msharder.PPMISharder, 1.0), 
            (h.msharder.Word2vecSharder, 0.1),
            (h.msharder.GloveSharder, 0.1),
            (h.msharder.MaxLikelihoodSharder, 1000),
            (h.msharder.MaxPosteriorSharder, 1000),
            (h.msharder.KLSharder, 1000)
        ]
        runs = product([1,3], sharder_classes_and_learning_rates)

        for shard_factor, (sharder_class, learning_rate) in runs:

            shards = h.shards.Shards(shard_factor)

            # Run an ordinary sharder without any diagonal masking.
            sharder = sharder_class(
                bigram=bigram, update_density=keep, mask_diagonal=False)
            mhat = torch.nn.Parameter(torch.ones(
                (bigram.vocab, bigram.vocab), device=h.CONSTANTS.MATRIX_DEVICE))
            optimizer = torch.optim.SGD((mhat,), lr=learning_rate)
            for shard in shards:
                optimizer.zero_grad()
                loss = sharder.calc_shard_loss(mhat[shard], shard)
                loss.backward()
                optimizer.step()

            # Run the a masked sharder for one update.
            masked_diag_sharder = sharder_class(
                bigram, update_density=keep, mask_diagonal=True)
            masked_mhat = torch.nn.Parameter(torch.ones(
                (bigram.vocab, bigram.vocab), device=h.CONSTANTS.MATRIX_DEVICE))
            optimizer = torch.optim.SGD((masked_mhat,), lr=learning_rate)
            for shard in shards:
                optimizer.zero_grad()
                loss = masked_diag_sharder.calc_shard_loss(
                    masked_mhat[shard], shard)
                loss.backward()
                optimizer.step()

            # Every model should have undergone an update.
            ones = torch.ones_like(mhat)
            self.assertFalse(torch.allclose(mhat, ones))
            self.assertFalse(torch.allclose(masked_mhat, ones))

            # Masked and non-masked sharders should differ along the diagonal,
            # but be the same elsewhere.
            self.assertFalse(torch.allclose(
                masked_mhat[diagonal], mhat[diagonal]))
            self.assertTrue(torch.allclose(
                masked_mhat[off_diagonal], mhat[off_diagonal]))



    def test_mse_minibatching_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()

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

                self.assertTrue(torch.allclose(loss, exloss))


    def test_max_likelihood_sharder(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        temperature = 10
        sharder = h.msharder.MaxLikelihoodSharder(bigram, temperature)
        sharder._load_shard(None)

        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        Pxx_data = Nxx / N
        Pxx_independent = (Nx / N) * (Nxt / N)

        self.assertTrue(torch.allclose(Pxx_data, sharder.Pxx_data))
        self.assertTrue(torch.allclose(Pxx_independent,sharder.Pxx_independent))
        self.assertEqual(temperature, sharder.temperature)


    def test_max_posterior_sharder(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        temperature = 10
        sharder = h.msharder.MaxPosteriorSharder(bigram, temperature)
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
        self.assertEqual(temperature, sharder.temperature)


    def test_KL_sharder(self):

        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        temperature = 10
        sharder = h.msharder.KLSharder(bigram, temperature)
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
        self.assertEqual(temperature, sharder.temperature)



class TestLoss(TestCase):

    def test_mask_diagonal(self):
        """
        Tests the mask_diagonal function, independent from its use by sharders.
        """

        # Make two identical square tensors
        tensor = torch.arange(64).reshape(8,8)
        clone = torch.clone(tensor)

        # Mask the diagonal of one of them
        h.hilbert_loss.mask_diagonal(tensor)

        # Make indices for the diagonal and off-diagonal entries
        device = h.CONSTANTS.MATRIX_DEVICE
        dtype = torch.uint8
        diagonal = torch.eye(8, dtype=dtype, device=device)
        off_diagonal = torch.ones((8,8), dtype=dtype, device=device) - diagonal

        # tensor and clone should differ only along the diagonal, and tensor's
        # diagonal should be zero.
        self.assertFalse(all(torch.eq(
            tensor[diagonal], clone[diagonal]
        )))
        self.assertTrue(all(torch.eq(
            tensor[off_diagonal], clone[off_diagonal]
        )))
        self.assertTrue(all(torch.eq(
            tensor[diagonal], torch.zeros(8, dtype=torch.int64)
        )))


    def test_w2v_loss(self):

        k = 15
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        Nxx, Nx, Nxt, N = bigram.load_shard(None, h.CONSTANTS.MATRIX_DEVICE)
        uNx, uNxt, uN = bigram.unigram.load_shard(
            None, h.CONSTANTS.MATRIX_DEVICE) 
        ncomponents = np.prod(Nxx.shape)
        shard = None

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
            loss_class = h.hilbert_loss.W2VLoss(keep_prob, ncomponents)
            found_loss = loss_class(M_hat, shard, Nxx, N_neg)

            self.assertTrue(torch.allclose(found_loss, expected_loss))



    def test_max_likelihood_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
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
        loss_array = loss_term1 + loss_term2

        shard = None
        for temperature in [1,10]:
            tempered_loss = loss_array * Pxx_independent**(1/temperature - 1)
            expected_loss = -torch.sum(tempered_loss) / float(ncomponents)
            found_loss = loss_class(
                M_hat, shard, Pxx_data, Pxx_independent, temperature)
            self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_max_posterior_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
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

        shard = None
        for temperature in [1, 10]:
            tempered_loss = scaled_loss * Pxx_independent ** (1/temperature - 1)
            expected_loss = - torch.sum(tempered_loss) / float(ncomponents)
            found_loss = loss_class(
                M_hat, shard, N, N_posterior, Pxx_posterior, Pxx_independent, 
                temperature
            )
            self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_KL_loss(self):
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
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

        shard = None
        for temperature in [1, 10]:
            tempered_KL = KL * Pxx_independent ** (1/temperature - 1)
            expected_loss = torch.sum(tempered_KL) / float(ncomponents)
            found_loss = loss_obj(
                M_hat, shard, N, N_posterior, Pxx_independent, digamma_a,
                digamma_b, temperature
            )
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
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
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
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        ppmi_sharder = h.msharder.PPMISharder(bigram)
        opt = torch.optim.SGD
        shape = bigram.Nxx.shape

        solver = h.autoembedder.HilbertEmbedderSolver(
            ppmi_sharder, opt, d=300,
            shape=shape,
            learning_rate=10000,
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
        bigram, _, _ = h.corpus_stats.get_test_bigram_base()
        shape = bigram.Nxx.shape
        w2v_sharder = h.msharder.Word2vecSharder(bigram, 15, update_density=1)
        opt = torch.optim.Adam
        shard = None

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

            loss_value = w2v_sharder._get_loss(mhat, shard).item()

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


