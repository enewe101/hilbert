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



class TestMultisenseLearner(TestCase):

    def test_multisense_learner(self):
        vocab = 10
        covocab = 20
        d = 8
        num_senses = 3
        bias = True

        for bias in [True, False]:

            learner = h.learner.MultisenseLearner(
                vocab=vocab, covocab=covocab, d=d,
                num_senses=num_senses, bias=bias
            )

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




class TestDependencyLearner(TestCase):

    def test_dependency_learner(self):
        vocab = int(1e3)
        covocab = int(1e3)
        d = 50
        learner = h.learner.DependencyLearner(vocab=vocab, covocab=covocab, d=d)
        dependency_corpus = h.tests.load_test_data.load_dependency_corpus()
        
        batch_size = 3
        loader = h.loader.DependencyLoader(
            h.tests.load_test_data.dependency_corpus_path(),
            batch_size=batch_size
        )

        for batch_num, (positives, mask) in loader:

            learner.forward(positives, mask)

            found_batch_size, _, padded_length = positives.shape

            start = batch_num * batch_size
            stop = start + batch_size

            # Assemble the expected batch
            expected_idxs = dependency_corpus.sort_idxs[start:stop]
            expected_sentences = [
                dependency_corpus.sentences[idx.item()]
                for idx in expected_idxs
            ]
            expected_lengths = [
                dependency_corpus.sentence_lengths[idx.item()]
                for idx in expected_idxs
            ]
            expected_max_length = max(expected_lengths)

            expected_mask = torch.zeros((
                len(expected_lengths), expected_max_length))
            for i, length in enumerate(expected_lengths):
                expected_mask[i][:length] = 1

            self.assertTrue(torch.equal(mask, expected_mask))

            # Did we get the batch size we expected?
            expected_batch_size = len(expected_sentences)
            self.assertEqual(found_batch_size, expected_batch_size)

            zipped_sentences = enumerate(zip(positives, expected_sentences))
            for i, (found_sentence, expected_sentence) in zipped_sentences:

                expected_length = expected_lengths[i]

                _, found_length = found_sentence.shape
                expected_padding_length = expected_max_length - expected_length

                # Words are as expected
                self.assertTrue(torch.equal(
                    found_sentence[0][:expected_length], 
                    torch.tensor(expected_sentence[0])
                ))

                # Heads are as expected
                self.assertTrue(torch.equal(
                    found_sentence[1][:expected_length],
                    torch.tensor(expected_sentence[1])
                ))

                # Arc types are as expected.
                self.assertTrue(torch.equal(
                    found_sentence[2][:expected_length],
                    torch.tensor(expected_sentence[2])
                ))

                # The first token shoudl be root
                self.assertEqual(found_sentence[0][0], 1)

                # Sentence should be padded.
                expected_padded = expected_sentence[0] + [h.CONSTANTS.PAD] * (
                    expected_max_length - expected_length
                ).item()
                self.assertTrue(torch.equal(
                    found_sentence[0],
                    torch.tensor(expected_padded)
                ))


                # The root has no head (indicated by padding)
                self.assertEqual(found_sentence[1][0].item(), h.dependency.PAD)

                # The list of heads for the sentence is padded.
                expected_head_padded = expected_sentence[1]+[h.CONSTANTS.PAD]*(
                    expected_max_length - expected_length
                ).item()

                self.assertTrue(torch.equal(
                    found_sentence[1],
                    torch.tensor(expected_head_padded)
                ))
                
                # The root has no incoming arc_type (has padding)
                self.assertEqual(found_sentence[2][0].item(), h.CONSTANTS.PAD)

                # The list of arc-types should be padded.
                expected_arc_types_padded = expected_sentence[2] + (
                    [h.CONSTANTS.PAD] * (expected_max_length - expected_length
                ).item())

                self.assertTrue(torch.equal(
                    found_sentence[2],
                    torch.tensor(expected_arc_types_padded)
                ))





if __name__ == '__main__':
    main()

