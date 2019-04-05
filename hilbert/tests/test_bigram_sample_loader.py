import hilbert as h
import torch
from unittest import TestCase
import scipy.sparse as sparse
import numpy as np

import os

class TestBigramSampleLoader(TestCase):

    def test_bigram_sample_loader_probabilities(self):
        for temperature in [1,2,5,10]:
            self.do_bigram_sample_loader_probabilities_test(temperature)


    def do_bigram_sample_loader_probabilities_test(self, temperature):
        """
        Draw a large number of samples, and calculate the empirical probability
        for each outcome.  It should be close to the probability vector with
        which Categorical was created.
        """
        torch.manual_seed(3141592)
        batch_size = 10000000
        sector_factor = 3

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sampler = h.bigram.BigramSampleLoader(
            bigram_path, sector_factor, temperature=temperature,
            batch_size=batch_size, verbose=False)
        Nxx_data, I, J, Nx, Nxt = h.generic_datastructs.get_Nxx_coo(
            bigram_path, sector_factor, verbose=False)

        positive_counts = torch.zeros(
            (Nx.shape[0], Nxt.shape[1]), dtype=torch.int32)
        negative_counts = torch.zeros(
            (Nx.shape[0], Nxt.shape[1]), dtype=torch.int32)

        IJ_sample = sampler.sample(batch_size)
        self.assertEqual(IJ_sample.shape, (batch_size*2, 2))

        # Acumulate the counts within the samples, and then derive the empirical
        # probabilities based on the counts.
        positive_counts = self.as_counts(IJ_sample[:batch_size])
        found_pij = positive_counts / positive_counts.sum()
        negative_counts = self.as_counts(IJ_sample[batch_size:])
        found_pi = negative_counts.sum(axis=1) / negative_counts.sum()
        found_pj = negative_counts.sum(axis=0) / negative_counts.sum()

        # Calculate the expected probabilities.  Take temperature into account.
        expected_pi_untempered = Nx.reshape((-1,)) / Nx.sum()
        expected_pi_raised = expected_pi_untempered ** (1/temperature-1)
        expected_pi_tempered = expected_pi_raised * expected_pi_untempered
        expected_pi_tempered = expected_pi_tempered / expected_pi_tempered.sum()

        expected_pj_untempered = Nxt.reshape((-1,)) / Nxt.sum()
        expected_pj_raised = expected_pj_untempered ** (1/temperature-1)
        expected_pj_tempered = expected_pj_raised * expected_pj_untempered
        expected_pj_tempered = expected_pj_tempered / expected_pj_tempered.sum()

        expected_pij_untempered = sparse.coo_matrix(
            (Nxx_data.numpy(), (I.numpy(), J.numpy()))).toarray()
        temper_adjuster = (
            expected_pi_raised.view((-1,1)) * expected_pj_raised.view((1,-1)))
        expected_pij_tempered = expected_pij_untempered * temper_adjuster
        expected_pij_tempered /= expected_pij_tempered.sum()

        # Check that empirical probabilities of samples match the probabilities
        # prescribed by the bigram data read by the sampler.
        self.assertTrue(np.allclose(
            found_pij, expected_pij_tempered, atol=1e-3))
        self.assertTrue(np.allclose(found_pi, expected_pi_tempered, atol=1e-3))
        self.assertTrue(np.allclose(found_pj, expected_pj_tempered, atol=1e-3))


    def as_counts(self, IJ):
        """
        Take advantage of scipy's coo_matrix constructor as a way to accumulate
        counts for i,j-samples.
        """
        return sparse.coo_matrix((
            np.ones((IJ.shape[0],)), 
            (self.cpu_1d_np(IJ[:,0]), self.cpu_1d_np(IJ[:,1]))
        )).toarray()


    def cpu_1d_np(self, tensor):
        """Move tensor to cpu; cast and reshape to a 1-d numpy array"""
        return tensor.cpu().view((-1,)).numpy()


    def test_bigram_sample_loader_interface(self):
        """
        Draw a large number of samples, and calculate the empirical probability
        for each outcome.  It should be close to the probability vector with
        which Categorical was created.
        """
        torch.manual_seed(3141592)
        batch_size = 3
        sector_factor = 3

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sampler = h.bigram.BigramSampleLoader(
            bigram_path, sector_factor, batch_size=batch_size, verbose=False)

        # Figure out the number of batches we expect, given the total number
        # of cooccurrence counts in the data, and the chosen batch_size.
        Nxx_data, I, J, Nx, Nxt = h.generic_datastructs.get_Nxx_coo(
            bigram_path, sector_factor, verbose=False)
        expected_num_batches = int(np.ceil(Nx.sum() / batch_size))

        # Confirm the shape, number, and dtype of batches.
        num_batches = 0
        for batch_id, batch_data in sampler:
            num_batches += 1
            self.assertEqual(batch_id.shape, (batch_size*2, 2))
            self.assertEqual(batch_data, None)
            self.assertEqual(batch_id.dtype, torch.LongTensor.dtype)

        self.assertEqual(num_batches, expected_num_batches)

