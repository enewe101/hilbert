import os
import numpy as np
import hilbert as h
import torch
from unittest import TestCase, main
from scipy import sparse


VERBOSE = False


def vprint(*args):
    if VERBOSE:
        print(*args)


def get_test_cooccurrence(device=None, verbose=True):
    """
    For testing purposes, builds a cooccurrence from constituents (not using
    it's own load function) and returns the cooccurrence along with the
    constituents used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence')
    unigram = h.unigram.Unigram.load(path, device=device, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    cooccurrence = h.cooccurrence.Cooccurrence(
        unigram, Nxx, device=device, verbose=verbose)

    return cooccurrence, unigram, Nxx


def get_test_cooccurrence_sector(device=None, verbose=True):
    """
    For testing purposes, builds a cooccurrence from constituents (not using
    it's own load function) and returns the cooccurrence along with the
    constituents used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'cooccurrence-sectors')
    unigram = h.unigram.Unigram.load(path, device=device, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    cooccurrence = h.cooccurrence.Cooccurrence(
        unigram, Nxx, device=device, verbose=verbose)

    return cooccurrence, unigram, Nxx



class TestLearner(TestCase):

    def test_learner_forward(self):
        # In these, use the factories, but test that we get the correct values.
        terms, contexts = 100, 500
        d = 300
        shard = (None, None)
        for use_bias in [True, False]:

            # Make the learner, and get M according to it.
            learner = h.learner.DenseLearner(
                vocab=terms,
                covocab=contexts,
                d=d,
                bias=False
            )
            got_M = learner(shard)

            # Now figure out what M is supposed to be, given the learners inits.
            W = learner.W.clone()
            V = learner.V.clone()
            vbias = torch.tensor([0.0], device=h.utils.get_device())
            wbias = torch.tensor([0.0], device=h.utils.get_device())
            if learner.vb is not None:
                vbias, wbias = learner.vb.clone(), learner.wb.clone()
            expected = (W @ V.t()) + vbias.reshape(1, -1) + wbias.reshape(-1, 1)
            self.assertTrue(torch.allclose(got_M, expected))


