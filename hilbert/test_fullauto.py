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


    def test_minibatching_loss(self):
        bigram = h.corpus_stats.get_test_bigram(2)

        for keep in [0.1, 1]:
            for scale, constructor in [(1, h.msharder.PPMISharder),
                                       (2, h.msharder.GloveSharder)]:
                sharder = constructor(bigram, update_density=keep)
                sharder._load_shard(None)
                mhat = torch.ones(sharder.M.shape)
                try:
                    weights = sharder.multiplier
                except AttributeError:
                    weights = 1

                torch.manual_seed(1)
                loss = sharder.calc_shard_loss(mhat, None)
                torch.manual_seed(1)
                mse = weights * ((mhat - sharder.M) ** 2)
                exloss = torch.nn.functional.dropout(
                    mse, p=1-keep, training=True)
                exloss = 0.5 * torch.sum(exloss) * scale

                if keep != 1:
                    self.assertNotEqual(loss, exloss)
                else:
                    self.assertEqual(loss, exloss)

                exloss *= keep
                self.assertEqual(loss, exloss)



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
        opt = torch.optim.Adam
        shape = bigram.Nxx.shape

        # TODO: right now this doesn't work for sharding > 1.
        # Perhaps this is because the bigram matrix is 11x11, which
        # is too small?
        sharders = [glv_sharder, ppmi_sharder]
        shard_fs = [1]
        oss = [True, False]
        lbs = [True, False]
        options = product(sharders, shard_fs, oss, lbs)

        for sharder, shard_factor, one_sided, learn_bias in options:
            vprint('\n', sharder.__class__)
            vprint('one_sided =', one_sided, 'learn_bias =', learn_bias)

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


    def test_solver_nan(self):
        bigram = h.corpus_stats.get_test_bigram(2)
        ppmi_sharder = h.msharder.PPMISharder(bigram)
        opt = torch.optim.SGD
        shape = bigram.Nxx.shape

        solver = h.autoembedder.HilbertEmbedderSolver(
            ppmi_sharder, opt, d=300,
            shape=shape,
            learning_rate=10,
            shard_factor=1,
            one_sided=False,
            learn_bias=False,
            device=h.CONSTANTS.MATRIX_DEVICE,
            verbose=VERBOSE,
        )

        # check to make sure we get the same loss after resetting
        with self.assertRaises(h.autoembedder.DivergenceError):
            solver.cycle(epochs=1000, shard_times=1, hold_loss=True)




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
