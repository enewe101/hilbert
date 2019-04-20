import os
import hilbert as h
import torch
from unittest import TestCase, main

class TestSolver(TestCase):

    #TODO: test get_params and get_embeddings

    def test_solver_nan(self):
        """
        If nan's appear in the loss function, the solver should raise
        DivergenceError.  Usually this is caused by a learning rate being
        too high, which is how we trigger it here.
        """

        cooccurrence_path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
        learning_rate = 1000
        solver = h.factories.build_mle_solver(
            cooccurrence_path=cooccurrence_path,
            shard_factor=1,
            dimensions=300,
            learning_rate=learning_rate,
            opt_str='adam',
            seed=1917,
            verbose=False,
        )

        # The high learning rate causes `nan`s, which should raise an error.
        with self.assertRaises(h.exceptions.DivergenceError):
            solver.cycle(updates_per_cycle=1000)


if __name__ == '__main__':
    main()
