from unittest import TestCase, main
from itertools import product
import hilbert as h
import torch
import scipy
import numpy as np
import os


SECTOR_FACTOR = 3
COOCCURRENCE_PATH = os.path.join(
    h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors/')
SPARSE_COOCCURRENCE_PATH = os.path.join(
    h.CONSTANTS.TEST_DIR, 'cooccurrence')


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




# Need to totally rewrite this test.

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
            preloader = h.cooccurrence.DenseShardPreloader(
                COOCCURRENCE_PATH, sef, shf,
                t_clean_undersample=t,
                alpha_unigram_smoothing=al,
                verbose=False
            )

            n_expected_iters = (sef ** 2) * (shf ** 2)
            all_shard_ids = []
            for shard, cooccurrence, unigram in preloader.preload_iter():
                all_shard_ids.append((shard.i, shard.j))

            self.assertEqual(len(all_shard_ids), n_expected_iters)
            self.assertEqual(len(all_shard_ids), len(set(all_shard_ids)))


    def test_model_with_dense_preloader(self):
        shard_factor = 1
        n_expected_iters = (SECTOR_FACTOR ** 2) * (shard_factor ** 2)

        model_constructors = [
            (h.loaders.GloveLoader, 'glv'),
            (h.loaders.Word2vecLoader, 'w2v'),
            (h.loaders.MaxLikelihoodLoader, 'mle'),
            (h.loaders.MaxPosteriorLoader, 'map'),
            (h.loaders.KLLoader, 'kl')
        ]

        for constructor, mname in model_constructors:
            model_loader = constructor(
                h.cooccurrence.DenseShardPreloader(
                    COOCCURRENCE_PATH, SECTOR_FACTOR, shard_factor, None, None,
                    verbose=False
                ),
                verbose=False,
                device='cpu'
            )

            # double checking that the construction fills it up!
            self.assertEqual(
                len(model_loader.preloaded_batches), n_expected_iters)

            # check that resetting works as intended
            model_loader.preload_all_batches()
            self.assertEqual(
                len(model_loader.preloaded_batches), n_expected_iters)

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
