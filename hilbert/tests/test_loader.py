import os
import sys
from unittest import TestCase, main
import hilbert as h
import torch
from pytorch_categorical import Categorical
from collections import Counter
import itertools
import scipy.sparse as sparse
import numpy as np
import seaborn as sns
from hilbert import factories as f
import matplotlib.pyplot as plt



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
            IJ, batch_data = loader.sample(batch_size)
            for (i, j), exp_pmi in zip(IJ, batch_data['exp_pmi']):
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

        expected_pij_untempered = torch.tensor(
            sparse.coo_matrix((Nxx_data.numpy(), (I.numpy(), J.numpy())))
            .toarray()
        )
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


class TestGibbsSampleLoader(TestCase):

    def sample_distribution_from_counter(self, counter):
        distr = dict(counter)
        total_cnt = sum(counter.values())
        # self.assertEqual(total_cnt, 100000)
        ordered_distr = torch.zeros(6)
        for k in distr:
            distr[k] = distr[k]/total_cnt
            ordered_distr[k] = distr[k]
        return ordered_distr


    def initialization(self, get_distr=False):
        torch.manual_seed(616)
        batch_size = 10
        cooccurrence_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'gibbs_sampling_data')
        dictionary = h.dictionary.Dictionary.load(
            os.path.join(cooccurrence_path, 'dictionary'))
        learner = h.learner.SampleLearner(
            vocab=len(dictionary),
            covocab=len(dictionary),
            d=50,
            device='cpu'
        )

        sampler = h.loader.GibbsSampleLoader(
            cooccurrence_path, learner, temperature=1,
            batch_size=batch_size, verbose=False, gibbs_iteration=1, get_distr=get_distr
        )
        return sampler

    def test_toy_model_distribution(self):
        """
        Only test 1 cycle.

        For each Gibbs sampling step, we are sampling 1 unit from the categorical distribution conditioning on the given
        positive sample.
        To examine such unit is drawn from the expected model distribution, we draw a large number of sample from each
        conditional distribution instead, and calculate the empirical probability of the sample. The probability vector
        should be close to probability vector with which Categorical was created.
        """

        # Initialization..
        gibbs_sampler = self.initialization()
        IJ_sample = gibbs_sampler.sample(gibbs_sampler.batch_size)
        # make sure number of sample we need
        self.assertEqual(IJ_sample.shape, (gibbs_sampler.batch_size * 2, 2))

        positive_samples_IJ = IJ_sample[:gibbs_sampler.batch_size]
        # expected counts
        for ind, (i, j) in enumerate(positive_samples_IJ):
            # model_distribution = torch.nn.functional.softmax(sampler.conditional_dist_J[ind])
            model_distribution = gibbs_sampler.conditional_dist_J[ind]/gibbs_sampler.conditional_dist_J[ind].sum()
            # p(j'|i)
            self.assertEqual(model_distribution.shape[0], len(gibbs_sampler.learner.vocab))
            condition_sampler = Categorical(model_distribution, normalized=True)
            j_prime_samples_ind = condition_sampler.sample((1000000,)).cpu().numpy()
            j_prime_cntr = Counter(j_prime_samples_ind)
            sample_dist = self.sample_distribution_from_counter(j_prime_cntr)

            expected_dist_prenorm = gibbs_sampler.Pj * torch.exp(gibbs_sampler.learner.V[i] @ gibbs_sampler.learner.W.t())
            self.assertEqual(expected_dist_prenorm.shape[0], len(gibbs_sampler.learner.vocab))
            # expected_dist = torch.nn.functional.softmax(expected_dist_prenorm)
            expected_dist = expected_dist_prenorm/expected_dist_prenorm.sum()

            # print(sample_dist,expected_dist)
            # print("")
            # normalized sample distribution and true distribution
            self.assertTrue(torch.allclose(sample_dist, expected_dist.cpu(), atol=1e-3))
            self.assertTrue(torch.allclose(expected_dist, model_distribution))

            # self.assertEqual(expected_dist.shape[0], len(sample_dist))

    def test_iterative_gibbs(self):
        """
        Test iterative gibbs sampling by making sure the first iteration by iterative method should be the same as
        hard coded toy example, which has been tested.

        """
        toy_gibbs_sampler = self.initialization()
        IJ_sample_toy = toy_gibbs_sampler.sample(toy_gibbs_sampler.batch_size)
        iter_gibbs_sampler = self.initialization()
        IJ_sample_iter = iter_gibbs_sampler.sample(iter_gibbs_sampler.batch_size)
        self.assertTrue(torch.equal(IJ_sample_toy, IJ_sample_iter))


    def test_gibbs_batch_prob(self):
        """
        Test if P(J'|I) given I and P(I'|J) given J are correct using synthetic data
        :return:
        """

        sampler = self.initialization()
        positive_samples = sampler.sample(10)[:10]
        positive_I = positive_samples[:,0]
        positive_J = positive_samples[:,1]
        empirical_J_conditional_on_I_dist = sampler.Pj * torch.exp(sampler.learner.V[positive_I] @ sampler.learner.W.t())

        empirical_I_conditional_on_J_dist = sampler.Pi * torch.exp(sampler.learner.W[positive_J] @ sampler.learner.V.t())

        model_I_conditional_on_J_dist, _ = sampler.iterative_gibbs_sampling(positive_J,
                                                                            input_I_flag=False,
                                                                            steps=2,
                                                                            get_distr=True)

        _, model_J_conditional_on_I_dist = sampler.iterative_gibbs_sampling(positive_I,
                                                                            input_I_flag=True,
                                                                            steps=2,
                                                                            get_distr=True)

        self.assertTrue(torch.equal(empirical_J_conditional_on_I_dist, model_J_conditional_on_I_dist))
        self.assertTrue(torch.equal(empirical_I_conditional_on_J_dist, model_I_conditional_on_J_dist))


    def draw_sample_heatmap(self, samples, size):
        matrix_for_heatmap = np.zeros((size, size), dtype=int)
        for i, j in samples:
            matrix_for_heatmap[i][j] += 1
        print(matrix_for_heatmap)


    def test_model_distr_sampling(self):
        sampler = self.initialization()
        # first iteration model distribution
        model_pmi = torch.exp(sampler.learner.V @ sampler.learner.W.t())
        model_dist = (sampler.Pi.view((-1,1)) * sampler.Pj.view((1,-1))) * model_pmi
        model_dist /= model_dist.sum()

        model_I_dist = sampler.Pi * model_pmi
        # print(sampler.Pi)
        # print(model_pmi)
        # print(model_I_dist)
        # self.draw_sample_heatmap(model_samples, model_pmi.shape[0])

        size = sampler.learner.V.shape[0]
        self.assertEqual(model_dist.shape, torch.randn((size,size)).shape)

        for gibbs_iter in [1, 5, 10]:
            sampler.gibbs_iteration = gibbs_iter
            samples = sampler.sample(1000)
            found_ij_model = samples[1000:]
            self.draw_sample_heatmap(found_ij_model, model_pmi.shape[0])


        expected_pij_untempered = sparse.coo_matrix(
            (sampler.Nxx_data.cpu().numpy(), (sampler.I.cpu().numpy(), sampler.J.cpu().numpy()))).toarray()
        print(expected_pij_untempered)

class GibbsSamplingIntegrationTest(TestCase):


    def initialization(self, learner, cooc, get_distr=False):
        torch.manual_seed(616)
        np.random.seed(616)
        batch_size = 10000

        sampler = h.loader.GibbsSampleLoader(
            cooc, learner, temperature=2,
            batch_size=batch_size, verbose=False, gibbs_iteration=1, get_distr=get_distr, device='cpu'
        )
        return sampler

    def build_solver(self):
        cooccurrence_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'Kylie_sampler_test')

        dictionary = h.dictionary.Dictionary.load(
            os.path.join(cooccurrence_path, 'dictionary'))

        loss = h.loss.GibbsSampleMLELoss()

        learner = h.learner.SampleLearner(
            vocab=len(dictionary),
            covocab=len(dictionary),
            d=50,
            bias=False,
            init=None,
            device='cpu'
        )
        optimizer = f.get_optimizer('adam', learner, learning_rate=0.01)

        loader = self.initialization(learner, cooccurrence_path)

        solver = h.solver.Solver(
            loader=loader,
            loss=loss,
            learner=learner,
            dictionary=dictionary,
            optimizer=optimizer,
            verbose=True,
        )
        return solver

    def draw_heat_map(self, solver, ax):

        sampler = solver.loader
        samples = sampler.sample(10000)
        samples_from_model = samples[10000:]

        indep_samples = samples[:10000]
        matrix_for_heatmap = np.zeros((len(solver.dictionary),len(solver.dictionary)), dtype=int)
        for i,j in samples_from_model:
            matrix_for_heatmap[i][j] += 1
        matrix_for_heatmap = matrix_for_heatmap / 100000

        sns.heatmap(matrix_for_heatmap,ax=ax)
        return ax

    def test_training(self):
        solver = self.build_solver()
        np.set_printoptions(precision=3,linewidth=150)
        print(solver.loader.Pi)
        fig,axs = plt.subplots(2,5,figsize=(30,6),dpi=80)
        try:
            for i in range(100):
                solver.cycle(1)
                if i%10 == 0:
                    # draw heat map for every 10 steps
                    self.draw_heat_map(solver, axs.flat[int(i/10)])
            Nxx = sparse.coo_matrix(
                (solver.loader.Nxx_data.cpu().numpy(),
                 (solver.loader.I.cpu().numpy(), solver.loader.J.cpu().numpy()))).toarray()
            print(Nxx)

        except ValueError:
            Nxx = sparse.coo_matrix(
                (solver.loader.Nxx_data.cpu().numpy(), (solver.loader.I.cpu().numpy(), solver.loader.J.cpu().numpy()))).toarray()
            print(Nxx)
        plt.savefig("test_heatmap.jpg")

        ax = sns.heatmap(Nxx)

        plt.savefig("empirical_heatmap.jpg")

        # plt.savefig("empirical_heatmap.jpg")


# TODO: test mask: should be uint, should mask the right things.
# TODO: should we mask the ROOT?
class TestDependencyLoader(TestCase):

    def test_dependency_loader(self):
        batch_size = 3
        loader = h.loader.DependencyLoader(
            h.tests.load_test_data.dependency_corpus_path(),
            batch_size=batch_size
        )

        dependency_corpus = h.tests.load_test_data.load_dependency_corpus()

        for batch_num, (positives, mask) in loader:

            found_batch_size, _, padded_length = positives.shape

            start = batch_num * batch_size
            stop = start + batch_size

            # Assemble the expected batch
            expected_idxs = dependency_corpus.sort_idxs[start:stop]
            expected_sentences = [
                dependency_corpus.sentences[idx.item()]
                for idx in expected_idxs
            ]
            expected_lengths = [
                dependency_corpus.sentence_lengths[idx.item()]
                for idx in expected_idxs
            ]
            expected_max_length = max(expected_lengths)

            expected_mask = torch.zeros(
                (len(expected_lengths), expected_max_length),
                dtype=torch.uint8
            )
            for i, length in enumerate(expected_lengths):
                expected_mask[i][length:] = 1

            self.assertTrue(torch.equal(mask, expected_mask))

            # Did we get the batch size we expected?
            expected_batch_size = len(expected_sentences)
            self.assertEqual(found_batch_size, expected_batch_size)

            zipped_sentences = enumerate(zip(positives, expected_sentences))
            for i, (found_sentence, expected_sentence) in zipped_sentences:
                expected_length = expected_lengths[i]

                _, found_length = found_sentence.shape
                expected_padding_length = expected_max_length - expected_length

                # Words are as expected
                self.assertTrue(torch.equal(
                    found_sentence[0][:expected_length],
                    torch.tensor(expected_sentence[0])
                ))

                # Heads are as expected
                self.assertTrue(torch.equal(
                    found_sentence[1][:expected_length],
                    torch.tensor(expected_sentence[1])
                ))

                # Arc types are as expected.
                self.assertTrue(torch.equal(
                    found_sentence[2][:expected_length],
                    torch.tensor(expected_sentence[2])
                ))

                # The first token shoudl be root
                self.assertEqual(found_sentence[0][0], 1)

                # Sentence should be padded.
                expected_padded = expected_sentence[0] + [h.CONSTANTS.PAD] * (
                        expected_max_length - expected_length
                ).item()
                self.assertTrue(torch.equal(
                    found_sentence[0],
                    torch.tensor(expected_padded)
                ))

                # The root has no head (indicated by padding)
                self.assertEqual(found_sentence[1][0].item(), h.dependency.PAD)

                # The list of heads for the sentence is padded.
                expected_head_padded = expected_sentence[1] + [
                    h.CONSTANTS.PAD] * (
                                               expected_max_length - expected_length
                                       ).item()

                self.assertTrue(torch.equal(
                    found_sentence[1],
                    torch.tensor(expected_head_padded)
                ))

                # The root has no incoming arc_type (has padding)
                self.assertEqual(found_sentence[2][0].item(), h.CONSTANTS.PAD)

                # The list of arc-types should be padded.
                expected_arc_types_padded = expected_sentence[2] + (
                        [h.CONSTANTS.PAD] * (
                            expected_max_length - expected_length
                            ).item())

                self.assertTrue(torch.equal(
                    found_sentence[2],
                    torch.tensor(expected_arc_types_padded)
                ))



if __name__ == '__main__':
    main()
