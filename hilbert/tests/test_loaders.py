from unittest import TestCase, main
from itertools import product
import hilbert as h
import torch
import scipy
import numpy as np
import os

BIGRAM_PATH = 'test-data/bigram-sectors/'
SPARSE_BIGRAM_PATH = 'test-data/bigram'
SECTOR_FACTOR = 3


def key_requirements_satisfied(mdict, mname):
    if mname == 'glv':
        return 'M' in mdict and 'weights' in mdict
    elif mname == 'w2v':
        return 'Nxx' in mdict and 'N_neg' in mdict
    elif mname == 'mle':
        return 'Pxx_data' in mdict and 'Pxx_independent' in mdict
    elif mname == 'map':
        return 'Pxx_independent' in mdict and 'Pxx_posterior' in mdict \
            and 'N_posterior' in mdict and 'N' in mdict
    elif mname == 'kl':
        return 'digamma_a' in mdict and 'digamma_b' in mdict \
            and 'N' in mdict and 'N_posterior' in mdict \
            and 'Pxx_independent' in mdict
    else:
        raise ValueError('No model name `{mname}`!'.format(mname))


class TestLoader(TestCase):


    def get_test_cooccurrence_stats(self):
        dictionary = h.dictionary.Dictionary([
            'banana','socks','car','field','radio','hamburger'
        ])
        Nxx = np.array([
            [0,3,1,1,0,0],
            [3,0,1,0,1,0],
            [1,1,0,0,0,1],
            [1,0,0,1,0,0],
            [0,1,0,0,0,1],
            [0,0,1,0,1,0]
        ])
        unigram = h.unigram.Unigram(dictionary, Nxx.sum(axis=1))
        return dictionary, Nxx, unigram


    def test_sample_max_likelihood_preloader(self):

        # Settings for test
        sector_factor = 3

        # Create sharded bigram data to test the loader
        dictionary, Nxx, unigram = self.get_test_cooccurrence_stats()
        bigram = h.bigram.BigramMutable(unigram, Nxx)

        # Save the bigram data on disk in shards to test the loader
        save_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-sample-loader')
        sectors = h.shards.Shards(sector_factor)
        bigram.save_sectors(save_path, sectors)

        # Get the loader to re-produce the original matrix
        loader = h.bigram.SampleMaxLikelihoodLoader(save_path, sector_factor)
        loader.accumulate_statistics()
        sparse = scipy.sparse.coo_matrix((
            np.array(loader.data), 
            (np.array(loader.I), np.array(loader.J))
        ))
        dense = sparse.toarray()

        # Test equality
        self.assertTrue(np.allclose(dense, Nxx))




    def test_sparse_preload_functionality(self):
        device = torch.device('cpu')
        filter_repeats = False
        zk = 100

        preloader = SparsePreloader(
            SPARSE_BIGRAM_PATH,
            zk=zk,
            filter_repeats=filter_repeats,
            device=device,
        )

        all_batches = []
        for i in preloader.preload_iter():
            torch.manual_seed(i)
            batch_id, bigram_data, unigram_data = preloader.prepare(i)
            all_batches.append(batch_id[0])

            self.assertEqual(len(bigram_data), 4)
            nijs, ni, njs, n = bigram_data

            # check that it's going correctly
            self.assertEqual(len(nijs), len(njs))
            self.assertEqual(len(ni.shape), 0) # constant
            self.assertEqual(len(n.shape), 0) # constant

            if not filter_repeats:
                i, all_js = batch_id
                a_nijs = preloader.sparse_nxx[i][1] # values

                # these are z-samples we will draw, given that we hardcode the seed
                torch.manual_seed(i)
                expected_zs = torch.randint(preloader.n_batches,
                                            device=device,
                                            size=(min(len(a_nijs), zk,),)).sort()[0]

                got_z_nijs = nijs[-len(expected_zs):]
                self.assertEqual(len(expected_zs), len(got_z_nijs))

                n_zeds = len(expected_zs)
                self.assertEqual(len(a_nijs) + n_zeds, len(nijs))
                self.assertEqual(sum(nijs[-n_zeds:]), 0)

        self.assertEqual(len(all_batches), preloader.n_batches)


    def test_model_with_sparse_preloader(self):
        model_constructors = [
            (h.model_loaders.GloveLoader, 'glv'),
            (h.model_loaders.Word2vecLoader, 'w2v'),
            (h.model_loaders.MaxLikelihoodLoader, 'mle'),
            (h.model_loaders.MaxPosteriorLoader, 'map'),
            (h.model_loaders.KLLoader, 'kl')
        ]

        for constructor, mname in model_constructors:
            model_loader = constructor(
                SparsePreloader(SPARSE_BIGRAM_PATH,
                                zk=1000,
                                filter_repeats=True,
                                device='cpu',
                                include_unigram_data=mname=='w2v'),
                verbose=False,
                device='cpu'
            )
            n_expected_iters = model_loader.preloader.n_batches

            # double checking that the construction fills it up!
            self.assertEqual(len(model_loader.preloaded_batches), n_expected_iters)

            # check that resetting works as intended
            model_loader.preload_all_batches()
            self.assertEqual(len(model_loader.preloaded_batches), n_expected_iters)

            # testing model shard iteration
            all_batch_ids = []
            for batch_id, data_dict in model_loader:
                all_batch_ids.append(batch_id[0])
                self.assertTrue(key_requirements_satisfied(data_dict, mname))

            # ensuring batching is properly done
            self.assertEqual(len(all_batch_ids), n_expected_iters)
            self.assertEqual(len(all_batch_ids), len(set(all_batch_ids)))


    def test_dense_preload_functionality(self):
        # things we will be testing over
        test_combos = list(product(
            [1, 3], # sector factors
            [1, 2, 3, 4], # shard factors
            [None, 1e-6, 1e-4], # t_clean_undersample
            [None, 0.25, 0.75], # alpha_unigram_smoothing
        ))

        # iterate over each combo
        for sef, shf, t, al in test_combos:
            preloader = h.bigram.DenseShardPreloader(
                BIGRAM_PATH, sef, shf,
                t_clean_undersample=t,
                alpha_unigram_smoothing=al,
            )

            n_expected_iters = (sef ** 2) * (shf ** 2)
            all_shard_ids = []
            for shard, bigram, unigram in preloader.preload_iter():
                all_shard_ids.append((shard.i, shard.j))

            self.assertEqual(len(all_shard_ids), n_expected_iters)
            self.assertEqual(len(all_shard_ids), len(set(all_shard_ids)))


    def test_model_with_dense_preloader(self):
        shard_factor = 1
        n_expected_iters = (SECTOR_FACTOR ** 2) * (shard_factor ** 2)

        model_constructors = [
            (ml.GloveLoader, 'glv'),
            (ml.Word2vecLoader, 'w2v'),
            (ml.MaxLikelihoodLoader, 'mle'),
            (ml.MaxPosteriorLoader, 'map'),
            (ml.KLLoader, 'kl')
        ]

        for constructor, mname in model_constructors:
            model_loader = constructor(
                h.bigram.DenseShardPreloader(BIGRAM_PATH, SECTOR_FACTOR, shard_factor,
                                    None, None),
                verbose=False,
                device='cpu'
            )

            # double checking that the construction fills it up!
            self.assertEqual(len(model_loader.preloaded_batches), n_expected_iters)

            # check that resetting works as intended
            model_loader.preload_all_batches()
            self.assertEqual(len(model_loader.preloaded_batches), n_expected_iters)

            # testing model shard iteration
            all_shard_ids = []
            for shard, data_dict in model_loader:
                all_shard_ids.append((shard.i, shard.j))
                self.assertTrue(key_requirements_satisfied(data_dict, mname))

            # ensuring sharding is properly done
            self.assertEqual(len(all_shard_ids), n_expected_iters)
            self.assertEqual(len(all_shard_ids), len(set(all_shard_ids)))



if __name__ == '__main__':
    main()
