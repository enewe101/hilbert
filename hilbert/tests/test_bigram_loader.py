import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h
from hilbert.loader import Loader, MultiLoader, BufferedLoader


try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class TestBigramLoader(TestCase):

    def test_bigram_loader(self):

        # Make a bigram loader
        t = 1e-5
        alpha = 0.6
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        sector_factor = 3
        shard_factor = 4
        num_loaders = sector_factor**2
        loader = h.bigram_loader.BigramLoader(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t, 
            alpha_unigram_smoothing=alpha
        )

        expected_bigram = h.bigram.BigramSector.load(
            bigram_path, h.shards.whole)
        expected_bigram.apply_w2v_undersampling(t)
        expected_bigram.apply_unigram_smoothing(alpha)

        expected_shards = list(h.shards.Shards(shard_factor * sector_factor))
        num_shards_iterated = 0
        for found_shard_id, bigram_data, unigram_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_id in expected_shards)
            expected_bigram_data = expected_bigram.load_shard(found_shard_id)
            expected_unigram_data = expected_bigram.load_unigram_shard(
                found_shard_id)

            comparisons = [
                (bigram_data, expected_bigram_data),
                (unigram_data, expected_unigram_data)]

            for found_data, expected_data in comparisons:
                for f_tensor, ex_tensor in zip(found_data, expected_data):
                    self.assertTrue(torch.allclose(f_tensor, ex_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))


class TestBigramMultiLoader(TestCase):

    def test_bigram_multi_loader(self):

        # Make a bigram loader
        t = 1e-5
        alpha = 0.6
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        sector_factor = 3
        shard_factor = 4
        num_loaders = sector_factor**2

        loader = h.bigram_loader.BigramMultiLoader(
            bigram_path, sector_factor, shard_factor, num_loaders,
            t_clean_undersample=t, 
            alpha_unigram_smoothing=alpha
        )

        expected_bigram = h.bigram.BigramSector.load(
            bigram_path, h.shards.whole)
        expected_bigram.apply_w2v_undersampling(t)
        expected_bigram.apply_unigram_smoothing(alpha)

        expected_shards = list(h.shards.Shards(sector_factor * shard_factor))
        num_shards_iterated = 0
        for found_shard_id, bigram_data, unigram_data in loader:
            num_shards_iterated += 1
            self.assertTrue(found_shard_id in expected_shards)
            expected_bigram_data = expected_bigram.load_shard(found_shard_id)
            expected_unigram_data = expected_bigram.load_unigram_shard(
                found_shard_id)

            comparisons = [
                (bigram_data, expected_bigram_data),
                (unigram_data, expected_unigram_data)]

            for found_data, expected_data in comparisons:
                for f_tensor, ex_tensor in zip(found_data, expected_data):
                    self.assertTrue(torch.allclose(f_tensor, ex_tensor))

        self.assertEqual(num_shards_iterated, len(expected_shards))



class TestConcreteLoaders(TestCase):

    def test_glove_loader(self):

        sector_factor = 3
        shard_factor = 4
        num_loaders = 9
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')

        for base_loader in [Loader, MultiLoader, BufferedLoader]:

            loader = h.bigram_loader.get_loader(
                h.bigram_loader.GloveLoader, base_loader,
                bigram_path, sector_factor, shard_factor, num_loaders, 
                verbose=False
            )

            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_base()

            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id)
                expected_M = torch.log(Nxx)
                expected_M[Nxx==0] = 0
                self.assertTrue(torch.allclose(shard_data['M'], expected_M))
                xmax = 100.
                alpha = 0.75
                m = lambda t: (t / xmax).pow(alpha)
                em = m(Nxx)
                em[em>1] = 1
                em *= 2
                self.assertTrue(torch.allclose(em, shard_data['weights']))
                zidx = Nxx == 0
                self.assertTrue(all(shard_data['M'][zidx] == 0))
                self.assertTrue(all(shard_data['weights'][zidx] == 0))


    def test_word2vec_loader(self):
        sector_factor = 1
        shard_factor = 1
        num_loaders = 9
        k = 10
        bigram_path = "/home/rldata/hilbert-embeddings/cooccurrence/1.2048-5w-dynamic-10k"


        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.Word2vecLoader, base_loader, bigram_path,
                sector_factor, shard_factor, num_loaders, k=10, verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_small()
            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id)
                uNx, uNxt, uN = expected_bigram.load_unigram_shard(shard_id)
                N_neg = k * (Nx - Nxx) * (uNxt / uN)

                self.assertTrue(torch.allclose(shard_data['Nxx'], Nxx))
                self.assertTrue(torch.allclose(shard_data['N_neg'], N_neg))


    def test_ppmi_loader(self):
        sector_factor = 3
        shard_factor = 4
        num_loaders = 9
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')

        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.PPMILoader, base_loader, bigram_path,
                sector_factor, shard_factor, num_loaders, verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_base()
            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id )
                expected_M = h.corpus_stats.calc_PMI((Nxx, Nx, Nxt, N))
                expected_M[expected_M<0] = 0
                self.assertTrue(torch.allclose(shard_data['M'], expected_M))



    def test_diff_loader(self):
        sector_factor = 1
        shard_factor = 1
        num_loaders = 1
        w = 5
        bigram_path = "/home/rldata/hilbert-embeddings/cooccurrence/1.2048-5w-dynamic-10k"

        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.DiffLoader, base_loader, bigram_path,
                sector_factor, shard_factor, num_loaders, w=5,
                verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_small()
            for shard_id, shard_data in loader:
                print("loaded shard from loader")
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id)
                print("loaded shard")

                P_j_given_i = (Nxx / N) * (N / Nx)
                self.assertTrue(torch.allclose(shard_data['Nxx'], Nxx))
                self.assertTrue(torch.allclose(shard_data['trans_M'], P_j_given_i))
                self.assertTrue(torch.allclose(shard_data['N'], N))

                newPji = w * P_j_given_i 
                for i in range(w - 1):
                    k = i + 2
                    newPji = newPji + ((w - k + 1) * torch.matrix_power(P_j_given_i, k))

                normalization = 0
                for i in range(w):
                    normalization += i + 1

                newPji = newPji / normalization

                stationary = torch.matrix_power(P_j_given_i, 1000)[0]
                stationary = stationary.view(stationary.size()[0], 1)

                Pxx_data = torch.mm(newPji, stationary)
                Pxx_independent = torch.t(stationary) * stationary
                self.assertTrue(torch.allclose(stationary, shard_data["pi"]))
                self.assertTrue(torch.allclose(newPji, shard_data["altered"]))
                print("got pxx's")
                self.assertTrue(torch.allclose(
                    Pxx_data, shard_data['Pxx_data']))
                self.assertTrue(torch.allclose(
                    Pxx_independent, shard_data['Pxx_independent']))
            print("done asserting") 

    def test_max_likelihood_loader(self):
        sector_factor = 1
        shard_factor = 1
        num_loaders = 9
        bigram_path = "/home/rldata/hilbert-embeddings/cooccurrence/1.2048-5w-dynamic-10k"

        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.MaxLikelihoodLoader, base_loader,
                bigram_path, sector_factor, shard_factor, num_loaders, 
                verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_small()
            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id )
                print("loaded shard")
                Pxx_data = Nxx / N
                Pxx_independent = (Nx / N) * (Nxt / N)
                print("got pxx's")
                self.assertTrue(torch.allclose(
                    Pxx_data, shard_data['Pxx_data']))
                self.assertTrue(torch.allclose(
                    Pxx_independent, shard_data['Pxx_independent']))


    def test_max_posterior_loader(self):
        sector_factor = 3
        shard_factor = 4
        num_loaders = 9
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')

        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.MaxPosteriorLoader, base_loader,
                bigram_path, sector_factor, shard_factor, num_loaders, 
                verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_base()
            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id )
                Pxx_independent = (Nx / N) * (Nxt / N)
                exp_mean, exp_std =  h.corpus_stats.calc_exp_pmi_stats(
                    (Nxx, Nx, Nxt, N))
                alpha, beta = h.corpus_stats.calc_prior_beta_params(
                    (Nxx, Nx, Nxt, N), exp_mean, exp_std, Pxx_independent)
                N_posterior = N + alpha + beta - 1
                Pxx_posterior = (Nxx + alpha) / N_posterior
                self.assertTrue(torch.allclose(
                    Pxx_posterior, shard_data['Pxx_posterior']))
                self.assertTrue(torch.allclose(
                    N_posterior, shard_data['N_posterior']))
                self.assertTrue(torch.allclose(
                    Pxx_independent, shard_data['Pxx_independent']))


    def test_KL_loader(self):
        sector_factor = 3
        shard_factor = 4
        num_loaders = 9
        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')

        for base_loader in [Loader, MultiLoader, BufferedLoader]:
            loader = h.bigram_loader.get_loader(
                h.bigram_loader.KLLoader, base_loader, bigram_path,
                sector_factor, shard_factor, num_loaders, verbose=False
            )
            expected_bigram, _, _ = h.corpus_stats.get_test_bigram_base()
            for shard_id, shard_data in loader:
                Nxx, Nx, Nxt, N = expected_bigram.load_shard(shard_id )
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
                self.assertTrue(torch.allclose(N, shard_data['N']))
                self.assertTrue(torch.allclose(
                    N_posterior, shard_data['N_posterior']))
                self.assertTrue(torch.allclose(
                    Pxx_independent,shard_data['Pxx_independent']))
                self.assertTrue(torch.allclose(
                    digamma_a, shard_data['digamma_a']))
                self.assertTrue(torch.allclose(
                    digamma_b, shard_data['digamma_b']))


