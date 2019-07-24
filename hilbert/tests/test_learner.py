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

    def test_calculate_score(self):

        torch.random.manual_seed(0)
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

        V = learner.V.clone()
        W = learner.W.clone()
        vb = learner.vb.clone()
        wb = learner.wb.clone()

        for batch_num, batch_data in loader:
            positives, mask = batch_data
            words = positives[:,0,:]
            head_ids = positives[:,1,:]
            batch_size, max_sentence_length = words.shape

            expected_score = torch.zeros((batch_size,))
            for row in range(batch_size):
                for word_idx in range(max_sentence_length):

                    # Skip ROOT, it does not choose a head.
                    if word_idx == 0:
                        continue

                    # Skip masked positions
                    if mask[row][word_idx] == 1:
                        continue

                    # Get ahold of the word and head tokens.
                    word = words[row][word_idx]
                    head_idx = head_ids[row][word_idx]
                    head = words[row][head_idx]

                    # Calculate the score for this specific attachment.
                    score = (W[word]*V[head]).sum() + wb[word] + vb[head]
                    expected_score[row] += score

            # Calculate the score using the method under test.
            found_score = learner.calculate_score(words, head_ids, mask) 
            self.assertTrue(torch.allclose(found_score, expected_score))


    def test_parse_energy(self):
        torch.random.manual_seed(0)
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

        V = learner.V.clone()
        W = learner.W.clone()
        vb = learner.vb.clone()
        wb = learner.wb.clone()

        for batch_num, batch_data in loader:
            positives, mask = batch_data
            words = positives[:,0,:]
            head_ids = positives[:,1,:]
            batch_size, max_sentence_length = words.shape

            expected_energy = torch.zeros((
                batch_size,max_sentence_length, max_sentence_length))
            for row in range(batch_size):

                sent_words = words[row].clone()
                sent_words[mask[row]] = 0
                
                # Get modifier and head embeddings.
                modifier_embeddings = W[sent_words]
                head_embeddings = V[sent_words]
                modifier_biases = wb[sent_words].unsqueeze(1)
                head_biases = vb[sent_words].unsqueeze(0)

                # Calculate the energy (illegal selections need to be masked)
                expected_energy[row, :, :] = torch.mm(
                    modifier_embeddings, head_embeddings.t()
                ) + head_biases + modifier_biases

                # Apply masking...
                # ... Padding does not choose a head
                expected_energy[row, mask[row], :] = float('-inf')
                # ... Padding cannot be chosen as a head
                expected_energy[row, :, mask[row]] = float('-inf')
                # ... A token cannot chose itself as a head
                I = (range(max_sentence_length), range(max_sentence_length))
                expected_energy[row][I] = float('-inf')
                # ... ROOT should not choose a head
                expected_energy[row][0,:] = float('-inf')

            # Calculate the score using the method under test.
            found_energy = learner.parse_energy(words, mask) 

            # Did we get what we expected?
            self.assertTrue(torch.allclose(found_energy, expected_energy))


    def test_probs(self):
        pass


    def test_inference(self):
        torch.random.manual_seed(0)
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
        num_inferred_samples = 10000

        V = learner.V.clone()
        W = learner.W.clone()
        vb = learner.vb.clone()
        wb = learner.wb.clone()

        for batch_num, batch_data in loader:
            positives, mask = batch_data
            words = positives[:,0,:]
            head_ids = positives[:,1,:]
            batch_size, sentence_length = words.shape

            counts = torch.zeros((batch_size, sentence_length, sentence_length))
            for i in range(num_inferred_samples):
                # Just work on the first sentence for simplicity
                try:
                    sample = learner.do_inference(words, mask) 
                except IndexError:
                    import pdb; pdb.set_trace()
                sample[mask] = 0
                sample[:,0] = 0
                idx1 = [
                    i for i in range(batch_size) 
                    for j in range(sentence_length)
                ]
                idx2 = [
                    j for i in range(batch_size)
                    for j in range(sentence_length)
                ]
                counts[idx1, idx2, sample.view(-1)] += 1

            found_probs = counts / num_inferred_samples

            # The root is forced to always choose itself as its own head
            # (but these samples are always ignored).
            head_choices = found_probs[:,0,:].clone()
            num_head_choices, sentence_length = head_choices.shape
            chose_head = torch.tensor(
                [1] + [0]*(sentence_length-1), dtype=torch.float32)
            self.assertTrue(torch.equal(
                head_choices, chose_head.expand(num_head_choices, -1)
            ))

            # Padding is forced to always choose [ROOT] as its head
            # (but these samples are always ignored).
            padding_choices = found_probs[mask].clone()
            num_padding_choices, sentence_length = padding_choices.shape
            self.assertTrue(torch.equal(
                padding_choices, chose_head.expand(num_padding_choices, -1)
            ))

            # For all other probabilities, compare them to expected probs
            expected_probs = learner.parse_probs(words, mask) 

            # We already checked probabilities for root and padding.
            # Overwrite these values because they don't need to be tested.
            # The values returned by expected probs handles the head and 
            # padding differently, so this overwriting step is the easiest
            # way to dodge root and padding during the next test.
            pad_prob = 1/sentence_length
            found_probs[:,0,:] = pad_prob
            found_probs[mask] = pad_prob

            # For all non-root non-padding entries, found probabilities should
            # match the expected ones!
            self.assertTrue(torch.allclose(
                found_probs, expected_probs, atol=0.01
            ))


if __name__ == '__main__':
    main()

