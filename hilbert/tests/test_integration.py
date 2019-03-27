import hilbert as h
import sys
import os
from unittest import TestCase, main


BIGRAM_PATH = 'test-data/bigram-sectors/'
SAVE_EMBEDDINGS = 'test-data/test-embeddings/'
SECTOR_FACTOR = 3


class IntegrationTests(TestCase):

    def test_runners(self):
        for sparse in [True, False]:

            common_kwargs = {
                'bigram_path': BIGRAM_PATH,
                'save_embeddings_dir': SAVE_EMBEDDINGS,
                'epochs': 2,
                'iters_per_epoch': 5,
                'init_embeddings_path': None,
                'd': 50,
                'update_density': 1.,
                'learning_rate': 0.0001,
                'opt_str': 'sgd',
                'sector_factor': SECTOR_FACTOR,
                'shard_factor': 1,
                'shard_times': 1,
                'seed': 1,
                'sparse': sparse,
                'device': h.CONSTANTS.MATRIX_DEVICE,
            }

            # combine common with the special kwargs
            w2v_kwargs = {**common_kwargs, **{'k': 15,
                                              't_clean_undersample': 2.45e-5,
                                              'alpha_smoothing': 0.75}}
            glv_kwargs = {**common_kwargs, **{'X_max': 100.,
                                              'alpha': 0.75}}
            temp_kwargs = {**common_kwargs, **{'temperature': 2.0}}

            # make a nice stew with 3 carrots and a tomato!
            runners_args = [
                (h.runners.run_w2v, w2v_kwargs),
                (h.runners.run_glv, glv_kwargs),
                (h.runners.run_map, temp_kwargs),
                (h.runners.run_mle, temp_kwargs),
                (h.runners.run_kl, temp_kwargs),
            ]

            # supress printing for the runners
            with open(os.devnull, "w") as supressed:
                sys.stdout = supressed
                for run, kwargs in runners_args:
                    run(**kwargs)
            sys.stdout = sys.__stdout__

            # pro unit testing
            self.assertTrue(True)


if __name__ == '__main__':
    main()
