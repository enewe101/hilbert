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

        for one_sided in ['yes','R','no','arc_labels']:

            learner = h.learner.SampleLearner(vocab=vocab, covocab=covocab, d=d,
                bias=bias, one_sided=one_sided)

            I = [0, 1, 3, 4]
            J = [1, 2, 5, 7]
            K = [0, 1, 2, 0]

            IJ = torch.tensor(list(zip(I,J,K)))

            found = learner(IJ)

            V = learner.V.clone()
            W = learner.W.clone() if one_sided == 'no' or one_sided == 'arc_labels' else None
            R = learner.R.clone() if one_sided == 'R' or one_sided == 'arc_labels' else None
            vb = learner.vb.clone() if bias else None
            wb = learner.wb.clone() if bias and (one_sided == 'no' or one_sided == 'arc_labels') else None
            kb = learner.kb.clone() if bias and one_sided == 'arc_labels' else None


            expected = torch.zeros((len(IJ),), device=learner.device)

            for q, (i,j,k) in enumerate(zip(I,J,K)):
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
                
                elif one_sided == 'arc_labels':
                    transformed =  torch.mv(R[k].transpose(0,1),W[j])
                    dot_terms = transformed * V[i]
                    if bias:
                        bias_term = vb[k,i] + wb[k,j] + kb[k]

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




class TestDependencySampler(TestCase):
    torch.random.manual_seed(0)


    def test_dependency_sampler(self):
        V = torch.tensor([
            [0,0,0,0],
            [1,1,1,1],
            [-1,1,-1,1],
            [2,0,2,0],
            [-1,-1,1,1],
            [-1,-1,-1,-1],
        ], dtype=torch.float32)
        W = torch.tensor([
            [-1,1,-1,1],
            [0.5,0.5,0.5,0.5],
            [-1,-1,1,1],
            [1,1,1,1],
            [-1,-1,-1,-1],
            [2,0,2,0],
        ], dtype=torch.float32)
        sampler = h.loader.DependencySampler(V=V, W=W)
        positives = torch.tensor([
            [[0,1,2,3,0], [3,2,1,0,0]],
            [[1,2,3,0,0], [3,2,0,1,0]],
            [[2,3,4,5,0], [5,4,2,3,0]],
            [[1,2,0,0,0], [2,1,0,0,0]]
        ], dtype=torch.int64)
        mask = torch.tensor([
            [1,1,1,1,0],
            [1,1,1,0,0],
            [1,1,1,1,0],
            [1,1,0,0,0],
        ], dtype=torch.uint8)
      
        sampling_rows = [1,2,3,6,7,11,12,13,16]
        padding_rows = [0,4,5,8,9,10,14,15,17,18,19]

        words = positives[:,0,:]
        covectors = W[words]
        self.assertEqual(covectors.size(), (4,5,4))
        self.assertTrue(torch.equal(covectors[0,0,:], torch.tensor([-1,1,-1,1], dtype=torch.float32)))
        
        vectors = V[words]
        product = torch.bmm(covectors, vectors.transpose(1,2))
        #print(product)
        
        reflected_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).byte()
        self.assertTrue(torch.equal(reflected_mask[0,:,:], torch.tensor([[1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [1,1,1,1,0],
                                                                         [0,0,0,0,0]], dtype=torch.uint8)))
        identities_idx = (slice(None), range(5), range(5))
        product[identities_idx] = torch.tensor(-float('inf'))
        product[1-reflected_mask] = torch.tensor(-float('inf'))

        self.assertEqual(product.size(), (4,5,5))
        self.assertTrue(torch.equal(product[0,:,:], torch.tensor([[-float('inf'),0,4,-4,-float('inf')],
                                                                  [0,-float('inf'),0,2,-float('inf')],
                                                                  [0,0,-float('inf'),0,-float('inf')],
                                                                  [0,4,0,-float('inf'),-float('inf')],
                                                                  [-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf')]])))

        unnormalized_probs = torch.exp(product)
        unnormalized_probs_2d = unnormalized_probs.view(-1, 5)
        unnormalized_probs_2d[1-mask.reshape(-1),:] = 1
        totals = unnormalized_probs_2d.sum(dim=1, keepdim=True)
        probs_full = unnormalized_probs_2d / totals
 
        #We only care about the probs from non-padding rows
        probs = torch.zeros(len(sampling_rows),5)

        k = 0
        for i in range(20):
            if i in sampling_rows:
                probs[k,:] = probs_full[i,:]
                k += 1
        
        print("\n")
        print(probs)

        counter = torch.zeros(len(sampling_rows),5)
        iterations = 50000

        # draw negative samples
        # validate shape and range of negative samples
        # check that negative samples are distributed as expected

        for i in range(iterations):
            negatives = sampler.sample(positives, mask)
            for j in range(len(padding_rows)):
                row_idx = padding_rows[j]
                selection = negatives[row_idx//5,1,row_idx%5]
                self.assertTrue(torch.equal(selection,h.dependency.PAD))
            for j in range(len(sampling_rows)):
                row_idx = sampling_rows[j]
                selection = int(negatives[row_idx//5,1,row_idx%5])
                selected_row = row_idx - row_idx%5 + selection
                self.assertTrue(selected_row in sampling_rows or selected_row % 5 == 0)
                counter[j,selection] += 1
        
        #print("\n")
        #print(negatives)
        self.assertEqual(negatives.size(),positives.size())

        found_probs = counter / iterations

        print("\n")
        print(found_probs)

        self.assertTrue(torch.allclose(probs,found_probs,atol=1e-02))


if __name__ == '__main__':
    main()

