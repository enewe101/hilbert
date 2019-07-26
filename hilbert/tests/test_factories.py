import os
import shutil
import hilbert as h
import random
import numpy as np
import torch
from unittest import TestCase


class TestBuildDependencySolver(TestCase):

    def test_build_dependency_solver(self):

        dependency_path= h.tests.load_test_data.dependency_corpus_path()
        batch_size = 10000
        init_embeddings_path = None
        dimensions = 300
        learning_rate = 0.01
        opt_str = 'adam'
        num_updates = 1
        num_negative_samples = 1
        seed = 1917

        h.tracer.tracer.verbose = False

        solver = h.factories.build_dependency_solver(
            dependency_path,
            batch_size=batch_size,
            init_embeddings_path=init_embeddings_path,
            dimensions=dimensions,
            learning_rate=learning_rate,
            opt_str=opt_str,
            num_updates=num_updates,
            num_negative_samples=num_negative_samples,
            seed=seed
        )

        solver.cycle(10)
