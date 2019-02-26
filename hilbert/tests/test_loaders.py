from hilbert.bigram import BigramPreloader
from unittest import TestCase, main
from itertools import product
import hilbert.model_loaders as ml

BIGRAM_PATH = 'test-data/bigram-sectors/'
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
        raise ValueError(f'No model name `{mname}`!')


class LoaderTest(TestCase):

    def test_preload_functionality(self):

        # things we will be testing over
        test_combos = list(product(
            [1, 2, 3, 4], # shard factors
            [None, 1e-6, 1e-4], # t_clean_undersample
            [None, 0.25, 0.75], # alpha_unigram_smoothing
            [None, 'cpu'] # device
        ))

        # iterate over each combo
        for sf, t, al, dev in test_combos:
            preloader = BigramPreloader(
                BIGRAM_PATH, SECTOR_FACTOR, sf,
                t_clean_undersample=t,
                alpha_unigram_smoothing=al,
                device=dev,
            )

            n_expected_iters = (SECTOR_FACTOR ** 2) * (sf ** 2)
            all_shard_ids = []
            for shard, bigram, unigram in preloader.preload_iter():
                all_shard_ids.append((shard.i, shard.j))

            self.assertEqual(len(all_shard_ids), n_expected_iters)
            self.assertEqual(len(all_shard_ids), len(set(all_shard_ids)))


    def test_model_with_preloader(self):
        shard_factor = 1
        n_expected_iters = (SECTOR_FACTOR ** 2) * (shard_factor ** 2)

        model_constructors = [
            (ml.GloveLoaderModel, 'glv'),
            (ml.Word2VecLoaderModel, 'w2v'),
            (ml.MaxLikelihoodLoaderModel, 'mle'),
            (ml.MaxPosteriorLoaderModel, 'map'),
            (ml.KLLoaderModel, 'kl')
        ]

        for constructor, mname in model_constructors:
            model_loader = constructor(
                BigramPreloader(BIGRAM_PATH, SECTOR_FACTOR, shard_factor,
                                None, None, 'cpu'),
                verbose=False
            )

            # double checking that the construction fills it up!
            self.assertEqual(len(model_loader.preloaded_shards), n_expected_iters)

            # check that resetting works as intended
            model_loader.preload_all_shards()
            self.assertEqual(len(model_loader.preloaded_shards), n_expected_iters)

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
