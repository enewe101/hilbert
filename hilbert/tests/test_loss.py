import os
import numpy as np
import hilbert as h
import torch
from unittest import TestCase, main
from scipy import sparse

def get_test_cooccurrence(device=None, verbose=True):
    """
    For testing purposes, builds a cooccurrence from constituents (not using
    it's own load function) and returns the cooccurrence along with the
    constituents used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
    unigram = h.unigram.Unigram.load(path, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    cooccurrence = h.cooccurrence.Cooccurrence(unigram, Nxx, verbose=verbose)

    return cooccurrence, unigram, Nxx


class TestLoss(TestCase):


    def test_w2v_loss(self):

        # Setup the scenario.
        k = 15
        cooccurrence, _, _ = get_test_cooccurrence()
        cooccurrence_data = cooccurrence.load_shard(
            None, h.CONSTANTS.MATRIX_DEVICE)
        Nxx, Nx, Nxt, N = cooccurrence_data
        unigram_data = cooccurrence.unigram.load_shard(
            None, h.CONSTANTS.MATRIX_DEVICE) 
        uNx, uNxt, uN = unigram_data
        ncomponents = np.prod(Nxx.shape)

        # Calculate loss
        M_hat = torch.ones_like(Nxx)
        loss_obj = h.loss.SGNSLoss(ncomponents)
        found_loss = loss_obj(M_hat, (cooccurrence_data, unigram_data))

        # Calculate expected loss
        N_neg = h.loss.SGNSLoss.negative_sample(Nxx, Nx, uNxt, uN, k)
        sigmoid = lambda a: 1/(1+torch.exp(-a))
        loss_term_1 = Nxx * torch.log(sigmoid(M_hat))
        loss_term_2 = N_neg * torch.log(1-sigmoid(M_hat))
        loss_array = -(loss_term_1 + loss_term_2)
        expected_loss = torch.sum(loss_array) #/ float(ncomponents)

        # They should be the same!
        self.assertTrue(torch.allclose(found_loss, expected_loss))



    def test_mle_loss(self):

        cooccurrence, _, _ = get_test_cooccurrence()
        cooccurrence_data = cooccurrence.load_shard(
            None, h.CONSTANTS.MATRIX_DEVICE)
        Nxx, Nx, Nxt, N = cooccurrence_data
        ncomponents = np.prod(Nxx.shape)
        M_hat = torch.ones_like(Nxx)


        for temperature in [1,10]:

            # Calculate expected loss for this test.
            Pxx_data = Nxx / N
            Pxx_independent = (Nx / N) * (Nxt / N)
            Pxx_model = Pxx_independent * torch.exp(M_hat)
            loss_array = -(Pxx_data * M_hat - Pxx_model)
            tempered_loss = loss_array * Pxx_independent**(1/temperature - 1)
            expected_loss = torch.sum(tempered_loss) #/ float(ncomponents)

            # Calculate loss calculated by the code being tested.
            loss_class = h.loss.MLELoss(ncomponents, temperature=temperature)
            found_loss = loss_class(M_hat, (cooccurrence_data, None))

            self.assertTrue(torch.allclose(found_loss, expected_loss))


    def test_sample_mle_loss(self):
        batch_size = 100
        M_hat_pos = torch.rand(batch_size)
        M_hat_neg = torch.rand(batch_size)
        numerator = -(M_hat_pos.sum() - torch.exp(M_hat_neg).sum())
        expected_loss = numerator / batch_size
        loss_obj = h.loss.SampleMLELoss()
        found_loss = loss_obj(torch.cat((M_hat_pos, M_hat_neg)), None)
        self.assertTrue(torch.allclose(found_loss, expected_loss))


