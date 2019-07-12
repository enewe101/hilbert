import os
from unittest import TestCase
import hilbert as h
try:
    import torch
except ImportError:
    torch = None


class TestDependencyLearner(TestCase):

    def test_dependency_learner(self):
        vocab = int(1e3)
        covocab = int(1e3)
        d = 50
        learner = h.learner.DependencyLearner(vocab=vocab, covocab=covocab, d=d)
        dep_corpus = h.load_test_data.load_dependency_corpus()
        


