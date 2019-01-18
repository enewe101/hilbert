import os
import shutil
from unittest import TestCase
from copy import copy, deepcopy
import hilbert as h


try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


class TestIntegration(TestCase):


    def get_opt(self, string):
        s = string.lower()
        d = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
        }
        return d[s]


    def test_runners(self):
        num_epochs = 3
        iters_per_epoch = 10
        shard_times = 2

        k = 10
        alpha = 0.7
        xmax = 80
        temperature = 5
        alpha_smoothing = 0.8
        t_clean_undersample = 1e-4

        test_runners_dir = os.path.join(h.CONSTANTS.TEST_DIR, 'test-runners')

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        init_embeddings_path = os.path.join(h.CONSTANTS.TEST_DIR, 'init-500')
        d = 350
        update_density = 0.01
        mask_diagonal = False
        learning_rate = 0.001
        opt_str = 'sgd'
        loader_policy = 'buffered'

        sector_factor = 3
        shard_factor = 1
        num_loaders = 9
        queue_size = 32
        seed = 1998
        device = 'cuda:1'
        verbose = False

        runners = [
            h.run_mle.run_mle, 
            h.run_map.run_map,
            h.run_kl.run_kl, 
            h.run_hbt_glv.run_glv,
            h.run_hbt_w2v.run_w2v
        ]
        extra_kwargs = [
            {'temperature':temperature},
            {'temperature':temperature},
            {'temperature':temperature},
            {'xmax':xmax, 'alpha':alpha},
            {
                'k':k, 't_clean_undersample':t_clean_undersample,
                'alpha_smoothing':alpha_smoothing}
        ]

        for runner, kwargs in zip(runners, extra_kwargs):

            # Clear the directory in which embeddings will be saved
            if os.path.exists(test_runners_dir):
                shutil.rmtree(test_runners_dir)
            os.makedirs(test_runners_dir)
            save_embeddings_dir = os.path.join(test_runners_dir, 'embeddings')

            runner(
                bigram_path,
                save_embeddings_dir=save_embeddings_dir,
                epochs=num_epochs,
                iters_per_epoch=iters_per_epoch,
                init_embeddings_path=init_embeddings_path,
                d=d,
                update_density=update_density,
                mask_diagonal=mask_diagonal,
                learning_rate=learning_rate,
                opt_str=opt_str,
                sector_factor=sector_factor,
                shard_factor=shard_factor,
                shard_times=shard_times,
                num_loaders=num_loaders,
                queue_size=queue_size,
                loader_policy=loader_policy,
                seed=seed,
                device=device,
                **kwargs
            )

            for epoch in range(1, num_epochs+1):
                count = iters_per_epoch * epoch
                save_embeddings_path = os.path.join(
                    save_embeddings_dir, 'iter-{}'.format(count))
                h.embeddings.Embeddings.load(save_embeddings_path)



