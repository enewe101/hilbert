import hilbert as h
import shutil
import sys
import os
from unittest import TestCase, main


SECTOR_FACTOR = 3
COOCCURRENCE_PATH = os.path.join(
    h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors/')
SAVE_EMBEDDINGS_DIR = os.path.join(
    h.CONSTANTS.TEST_DIR,'test-data/test-integration/')


class TestIntegration(TestCase):


    #
    # TODO: Right now this just runs the runners, and the test passes as long
    # as no exceptions are raised.  Would be good to add a few conditions to 
    # assert as well: 
    #   - all arguments are reflected built solver.
    #   - embeddings get written to disk.
    #

    def test_runners(self):

        # We're going to test the factories.  First, these arguments are common
        # to all factories.
        base_kwargs = {
            'save_embeddings_dir': SAVE_EMBEDDINGS_DIR,
            'num_writes': 10,
            'num_updates': 100,
            'cooccurrence_path': COOCCURRENCE_PATH,
            'bias': False,
            'init_embeddings_path': None,
            'dimensions': 300,
            'learning_rate': 0.01,
            'opt_str': 'adam',
            'seed': 1917,
            'device': None,
            'verbose': False,
        }

        # Each factory also has its own particular arguments
        mle_kwargs = {
            **base_kwargs,
            'shard_factor':1, 'simple_loss':False, 'temperature':2}
        glove_kwargs = {
            **base_kwargs, 'shard_factor':1, 'X_max':100., 'alpha':0.75}
        sgns_kwargs = {
            **base_kwargs, 'shard_factor':1, 'k':15,
            'undersampling':2.45e-5, 'smoothing':0.75}
        mle_sample_kwargs = {
            **base_kwargs, 'batch_size':1000, 'temperature':2.0}

        # Pair up the factories with the arguments.
        runners_args = [
            (h.runners.run_sgns.run, sgns_kwargs),
            (h.runners.run_glove.run, glove_kwargs),
            (h.runners.run_mle.run, mle_kwargs),
            (h.runners.run_mle_sample.run, mle_sample_kwargs),
        ]

        # Run all of the runners.
        for run, kwargs in runners_args:
            run(**kwargs)

        # Cleanup after the test.
        shutil.rmtree(SAVE_EMBEDDINGS_DIR)



if __name__ == '__main__':
    main()
