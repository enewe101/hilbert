import hilbert as h
import sys
import os
from unittest import TestCase, main


SECTOR_FACTOR = 3
COOCCURRENCE_PATH = os.path.join(
    h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors/')
SAVE_EMBEDDINGS = os.path.join(
    h.CONSTANTS.TEST_DIR,'test-data/test-embeddings/')


class IntegrationTests(TestCase):

    def test_runners(self):
        for dm in ['dense', 'tupsparse', 'lilsparse']:

            common_kwargs = {
                'cooccurrence_path': COOCCURRENCE_PATH,
                'save_embeddings_dir': SAVE_EMBEDDINGS,
                'epochs': 2,
                'iters_per_epoch': 3,
                'init_embeddings_path': None,
                'd': 50,
                'update_density': 1.,
                'learning_rate': 0.0001,
                'opt_str': 'sgd',
                'sector_factor': SECTOR_FACTOR,
                'shard_factor': 1,
                'shard_times': 1,
                'seed': 1,
                'datamode': dm,
                'tup_n_batches': 13,
                'zk': 100,
                'verbose': 2,
                'device': h.CONSTANTS.MATRIX_DEVICE,
            }

            # combine common with the special kwargs
            w2v_kwargs = {**common_kwargs, **{'k': 15,
                                              't_clean_undersample': 2.45e-5,
                                              'alpha_smoothing': 0.75}}
            glv_kwargs = {**common_kwargs, **{'X_max': 100.,
                                              'alpha': 0.75}}
            temp_kwargs = {**common_kwargs, **{'temperature': 2.0}}

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
