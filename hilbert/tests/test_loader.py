import os
import sys
from unittest import TestCase, main
import hilbert as h
import torch
import itertools
import scipy.sparse as sparse
import numpy as np


class TestLoader(TestCase):

    # Assume that the bigram sector is correct, only test loader itself
    def test_dense_loader(self):

        cooccurrence_path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
        shard_factor = 1, 2
        include_unigrams = False, True
        undersampling = None, torch.tensor(1e-5)
        smoothing = None, 3/4
        verbose = False

        sector_factor = h.cooccurrence.CooccurrenceSector.get_sector_factor(
            cooccurrence_path)
        sectors = h.shards.Shards(sector_factor)
        options = itertools.product(
            shard_factor, include_unigrams, undersampling, smoothing)

        for sh_factor, uni, usamp, smooth in options:

            loader = h.loader.DenseLoader(
                cooccurrence_path=cooccurrence_path,
                shard_factor=sh_factor,
                include_unigrams=uni,
                undersampling=usamp,
                smoothing=smooth,
                verbose=verbose,
            )
            shards = h.shards.Shards(sh_factor)

            for i, (shard_id, batch_data) in enumerate(loader):
                cooccurrence, unigram = batch_data
                sector = sectors[i//sh_factor**2]
                shard = shards[i%sh_factor**2]
                cooc_sector = h.cooccurrence.CooccurrenceSector.load(
                    cooccurrence_path, sector, verbose=False)
                if usamp is not None:
                    cooc_sector.apply_w2v_undersampling(usamp)
                if smooth is not None:
                    cooc_sector.apply_unigram_smoothing(smooth)
                expected_cooccurrence = cooc_sector.load_relative_shard(shard)
                for found, expected in zip(cooccurrence, expected_cooccurrence):
                    self.assertTrue(torch.allclose(expected, found))
                if uni:
                    expected_unigram = cooc_sector.load_relative_unigram_shard(
                        shard)
                    for found, expected in zip(unigram, expected_unigram):
                        self.assertTrue(torch.allclose(expected, found))
                else:
                    self.assertTrue(unigram is None)


class TestCPUSampleLoader(TestCase):

    def test_cpu_sample_loader_probabilities(self):

        num_samples = 50
        batch_size = 10000
        num_draws = num_samples * batch_size
        torch.random.manual_seed(1)

        # Construct a sample-based loader.  Sample from it, and check tha tthe 
        # statistics are as desired.  But first, construct it.
        cooc_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'cooccurrence-10')
        loader = h.loader.CPUSampleLoader(cooc_path, device='cpu')

        # We'll need to know the vocabulary.
        vocab = h.dictionary.Dictionary.check_vocab(
            os.path.join(cooc_path, 'dictionary'))

        # Make some embeddings that will be used to calculate this sampler's 
        # ability to generate the desired expectations using importance 
        # sampling.
        embeddings = h.embeddings.random(vocab, 50)

        # Calculate the sampled expectation and probability.
        Nxx_sample = torch.zeros((vocab, vocab), dtype=torch.int32)
        Vxx_sample = torch.zeros((vocab, vocab), dtype=torch.float32)
        sum_exp_pmi = 0
        importance_sum = 0
        for sample_num in range(num_samples):
            sys.stdout.write('_'); sys.stdout.flush()
            I,J,exp_pmis = loader.sample(batch_size)
            for i, j, exp_pmi in zip(I,J,exp_pmis):
                Nxx_sample[i,j] += 1
                Vxx_sample[i,j] += exp_pmi
                sum_exp_pmi += exp_pmi
        Qxx_sample = Nxx_sample.float() / num_draws
        Pxx_sample = Vxx_sample / num_draws
        avg_exp_pmi = sum_exp_pmi / num_draws

        # Now calculate samples cooccurrence.
        cooc = h.cooccurrence.Cooccurrence.load(cooc_path)
        Nxx = torch.tensor(cooc.Nxx.toarray(), dtype=torch.float32)
        Pxx_expected = Nxx / cooc.N
        Qxx_expected = (cooc.Nx / cooc.N) * (cooc.Nxt / cooc.N)

        # Did we reproduce the target distribution (Pxx)?
        self.assertTrue(torch.allclose(Pxx_sample, Pxx_expected, atol=5e-4))

        # Was the proposal distribution as expected?
        self.assertTrue(torch.allclose(Qxx_sample, Qxx_expected, atol=5e-4))
        


        




class TestGPUSampleLoader(TestCase):


    def test_cooccurrence_sample_loader_probabilities(self):
        for temperature in [1,2,5,10]:
            self.do_cooccurrence_sample_loader_probabilities_test(temperature)


    def do_cooccurrence_sample_loader_probabilities_test(self, temperature):
        """
        Draw a large number of samples, and calculate the empirical probability
        for each outcome.  It should be close to the probability vector with
        which Categorical was created.
        """
        torch.manual_seed(3141592)
        batch_size = 10000000
        sector_factor = 3

        cooccurrence_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sampler = h.loader.GPUSampleLoader(
            cooccurrence_path, temperature=temperature,
            batch_size=batch_size, verbose=False
        )
        Nxx_data, I, J, Nx, Nxt = h.cooccurrence.CooccurrenceSector.load_coo(
            cooccurrence_path, verbose=False)

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
        # prescribed by the cooccurrence data read by the sampler.
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


    def test_cooccurrence_sample_loader_interface(self):
        """
        Even though test_cooccurrence_sampler does not yield multiple batches,
        It is implemented as an iterator, for consistency with DenseLoader.
        Test that the iterator interface is satisfied.
        """
        torch.manual_seed(3141592)
        batch_size = 3
        sector_factor = 3
        num_batches = 10

        cooccurrence_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sampler = h.loader.GPUSampleLoader(
            cooccurrence_path, batch_size=batch_size, verbose=False)

        # Figure out the number of batches we expect, given the total number
        # of cooccurrence counts in the data, and the chosen batch_size.
        Nxx_data, I, J, Nx, Nxt = h.cooccurrence.CooccurrenceSector.load_coo(
            cooccurrence_path, sector_factor, verbose=False)

        # Confirm the shape, number, and dtype of batches.
        num_batches_seen = 0
        for batch_num in range(num_batches):
            for batch_id, batch_data in sampler:
                num_batches_seen += 1
                self.assertEqual(batch_id.shape, (batch_size*2, 2))
                self.assertEqual(batch_data, None)
                self.assertEqual(batch_id.dtype, torch.LongTensor.dtype)
        self.assertEqual(num_batches_seen, num_batches)





if __name__ == '__main__':
    main()
