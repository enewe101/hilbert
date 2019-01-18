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



class TestFactory(TestCase):


    def get_opt(self, string):
        s = string.lower()
        d = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adagrad': torch.optim.Adagrad,
        }
        return d[s]


    def test_factory(self):

        k = 10
        alpha = 0.7
        xmax = 80
        temperature = 5

        bigram_path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram-sectors')
        expected_bigrams = h.bigram.BigramBase.load(bigram_path)
        expected_dictionary_tokens = expected_bigrams.dictionary.tokens
        init_embeddings_path = os.path.join(h.CONSTANTS.TEST_DIR, 'init-500')
        d = 350
        t_clean_undersample = 1e-4
        alpha_unigram_smoothing = 0.6
        update_density = 0.01
        mask_diagonal = True
        learning_rate = 0.1
        opt_str = 'sgd'
        loader_policy = 'buffered'
        sector_factor = 3
        shard_factor = 4
        num_loaders = 4
        queue_size = 32
        seed = 1998
        device = 'cuda:1'
        verbose = False

        constructors = [
            h.factories.construct_w2v_solver,
            h.factories.construct_glv_solver,
            h.factories.construct_max_likelihood_solver,
            h.factories.construct_max_posterior_solver,
            h.factories.construct_KL_solver
        ]

        extra_kwargs = [
            {'k': k}, 
            {'alpha': alpha, 'xmax': xmax},
            {'temperature': temperature},
            {'temperature': temperature},
            {'temperature': temperature},
        ]

        extra_equalities = [
            [(lambda solver: solver.loader.k, k)], 
            [
                (lambda solver: solver.loader.alpha, alpha),
                (lambda solver: solver.loader.X_max, xmax),
            ], 
            [(lambda solver: solver.loss.temperature, temperature)],
            [(lambda solver: solver.loss.temperature, temperature)],
            [(lambda solver: solver.loss.temperature, temperature)]
        ]

        tests = zip(constructors, extra_kwargs, extra_equalities)

        for constructor, kwargs, equalities in tests:

            solver = constructor(
                bigram_path=bigram_path,
                init_embeddings_path=init_embeddings_path,
                d=d,
                t_clean_undersample=t_clean_undersample,
                alpha_unigram_smoothing=alpha_unigram_smoothing,
                update_density=update_density,
                mask_diagonal=mask_diagonal,
                learning_rate=learning_rate,
                opt_str=opt_str,
                sector_factor=sector_factor,
                shard_factor=shard_factor,
                num_loaders=num_loaders,
                queue_size=queue_size,
                loader_policy=loader_policy,
                seed=seed,
                device=device,
                verbose=verbose,
                **kwargs
            )

            loss = solver.loss
            loader = solver.loader
            init_embeddings = h.embeddings.Embeddings.load(init_embeddings_path)

            # Gather the things that should be equal
            common_equalities = [
                (loader.bigram_path, bigram_path),
                (solver.V.shape[1], d),
                (loader.t_clean_undersample, t_clean_undersample),
                (loader.alpha_unigram_smoothing, alpha_unigram_smoothing),
                (loss.keep_prob, update_density),
                (loss.mask_diagonal, mask_diagonal),
                (solver.learning_rate, learning_rate),
                (solver.optimizer.__class__, self.get_opt(opt_str)),
                (loader.sector_factor, sector_factor),
                (loader.shard_factor, shard_factor),
                (loader.num_loaders, num_loaders),
                (loader.queue_size, queue_size),
                (loader.device, 'cuda:1'),
                (solver.device, 'cuda:1'),
                (loader.verbose, verbose),
                (solver.verbose, verbose),
                (solver.get_dictionary().tokens, expected_dictionary_tokens)
            ]
            for item1, item2 in common_equalities:
                self.assertEqual(item1, item2)
            for item1_getter, item2 in equalities:
                self.assertEqual(item1_getter(solver), item2)

        self.assertTrue(isinstance(loader, h.loader.BufferedLoader), True)
        self.assertTrue(torch.allclose(solver.V, init_embeddings.V))
        self.assertTrue(torch.allclose(solver.W, init_embeddings.W))

        # finally, run the solver for a couple steps
        solver.cycle(epochs=2,shard_times=2)



