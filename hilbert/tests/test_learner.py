import os
import itertools
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



class TestDenseLearner(TestCase):

    def test_dense_learner_forward(self):
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

class TestSampleLearner(TestCase):
    
    def test_one_sided_learner(self):
        vocab = 10
        covocab = 10
        d = 5
        bias = True

        for one_sided in ['yes', 'R', 'no']:

            learner = h.learner.SampleLearner(vocab=vocab, covocab=covocab, d=d,
                bias=bias, one_sided=one_sided)

            I = [0, 1, 3, 4]
            J = [1, 2, 5, 7]
            IJ = torch.tensor(list(zip(I,J)))

            found = learner(IJ)

            V = learner.V.clone()
            W = learner.W.clone() if one_sided == 'no' else None
            R = learner.R.clone() if one_sided == 'R' else None
            vb = learner.vb.clone() if bias else None
            wb = learner.wb.clone() if bias and one_sided == 'no' else None
            
            expected = torch.zeros((len(IJ),), device=learner.device)

            for q, (i,j) in enumerate(zip(I,J)):
                bias_term = torch.zeros((1,))
                if one_sided == 'yes':
                    dot_terms = V[i] * V[j]
                    if bias:
                        bias_term = vb[i] + vb[j]

                elif one_sided == 'R':
                    transformed = torch.mv(R.transpose(0,1),V[j])
                    dot_terms = transformed * V[i]
                    if bias:
                        bias_term = vb[i] + vb[j]
                
                else:
                    dot_terms = V[i] * W[j]
                    if bias:
                        bias_term = vb[i] + wb[j]

                dot_product = torch.sum(dot_terms)
                biased_product = dot_product + bias_term
                expected[q] = biased_product


            self.assertTrue(torch.allclose(found, expected))


class TestMultisenseLearner(TestCase):

    def test_multisense_learner(self):
        vocab = 10
        covocab = 20
        d = 8
        num_senses = 3
        bias = True

        for bias in [True, False]:

            learner = h.learner.MultisenseLearner(
                vocab=vocab, covocab=covocab, d=d, num_senses=num_senses, bias=bias)

            # Select an arbitrary sample.
            I = [0, 1, 4, 9]
            J = [0, 2, 19, 12]
            IJ = torch.tensor(list(zip(I,J)))

            found = learner(IJ)

            V = learner.V.clone()
            W = learner.W.clone()
            vb = learner.vb.clone() if bias else None
            wb = learner.wb.clone() if bias else None

            expected = torch.zeros((len(IJ),))
            for q, (i,j) in enumerate(zip(I,J)):

                # Calculate the product of all sense-combinations, 
                # plus associated biases.
                matrix_product = torch.zeros((num_senses,num_senses))
                bias_matrix = torch.zeros((num_senses,num_senses))
                iter_senses = itertools.product(
                    range(num_senses), range(num_senses))
                for r,s in iter_senses:
                    matrix_product[r,s] = torch.dot(W[j,:,r], V[i,:,s])
                    if bias:
                        bias_matrix[r,s] = wb[j,r] + vb[i,s]

                biased_matrix_product = matrix_product + bias_matrix
                expected[q] = torch.log(torch.exp(biased_matrix_product).sum())

            self.assertTrue(torch.allclose(found, expected))
            









        
