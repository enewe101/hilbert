import os
import hilbert as h
import torch
from unittest import TestCase, main
from scipy import sparse



def get_test_cooccurrence(verbose=True):
    """
    For testing purposes, builds a cooccurrence from constituents (not using
    it's own load function) and returns the cooccurrence along with the
    constituents used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
    unigram = h.unigram.Unigram.load(path, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    cooccurrence = h.cooccurrence.Cooccurrence(
        unigram, Nxx, verbose=verbose)

    return cooccurrence, unigram, Nxx


# TODO: add tests for mle, mle_sample, glove
class TestModels(TestCase):
    """
    Push each model through one update cycle, and ensure that the parameters
    are updated according to expectations.
    """
    def test_w2v(self):
        cooccurrence, _, _ = get_test_cooccurrence()
        shape = cooccurrence.Nxx.shape
        scale = cooccurrence.vocab**2
        learning_rate = 0.001
        keep = 1
        k = 15
        d = 300
        sector_factor = 1
        shard_factor = 1
        opt_str = 'sgd'
        device = h.CONSTANTS.MATRIX_DEVICE
        bias=False

        # Changed from being cooccurrence-sector.
        cooccurrence_path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
        dictionary = h.dictionary.Dictionary.load(
            os.path.join(cooccurrence_path, 'dictionary'))

        loss = h.loss.SGNSLoss(len(dictionary)**2, k=k, device=device)

        loader = h.loader.DenseLoader(
            cooccurrence_path,
            shard_factor,
            include_unigrams=loss.REQUIRES_UNIGRAMS,
            undersampling=None,
            smoothing=0.75,
            device=device,
            verbose=False,
        )

        learner = h.learner.DenseLearner(
            vocab=len(dictionary),
            covocab=len(dictionary),
            d=d,
            bias=bias,
            init=None,
            device=device
        )

        optimizer = h.factories.get_optimizer(
            opt_str, learner.parameters(), learning_rate)

        solver = h.solver.Solver(
            loader=loader,
            loss=loss,
            learner=learner,
            optimizer=optimizer,
            schedulers=None,
            dictionary=dictionary,
            verbose=False,
        )

        V, W, _, _ = solver.get_params()
        expected_V = V.clone()
        expected_W = W.clone()

        # Loader is superfluous because there's just one shard and just one 
        # sector.  So just give us the data!
        shard_id, shard_data = next(iter(loader))

        for iteration in range(5):

            # Manually calculate the gradient and expected update
            mhat = expected_W @ expected_V.t()

            cooccurrence_data, unigram_data = shard_data
            Nxx, Nx, Nxt, N = cooccurrence_data
            uNx, uNxt, uN = unigram_data
            N_neg  = k * (Nx - Nxx) * (uNxt / uN)

            N_sum = Nxx + N_neg
            delta = Nxx - mhat.sigmoid() * N_sum
            neg_grad_V = torch.mm(delta.t(), expected_W) / scale
            neg_grad_W = torch.mm(delta, expected_V) / scale
            expected_V += learning_rate * neg_grad_V
            expected_W += learning_rate * neg_grad_W

            # Let the solver make one update
            solver.cycle()
            found_V, found_W, _, _ = solver.get_params()

            # Check that the solvers update matches expectation.
            self.assertTrue(torch.allclose(found_V, expected_V, atol=1e-3))
            self.assertTrue(torch.allclose(found_W, expected_W, atol=1e-3))


