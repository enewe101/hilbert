import os
import shutil
import random
from collections import Counter
import copy

import numpy as np
from scipy import sparse
from unittest import main, TestCase

import shared
import hilbert as h
import data_preparation as dp

import warnings
import logging
logging.captureWarnings(True)





class MockUnigram(h.unigram.Unigram):
    def freq(self, token):
        return {
            "the": 0.1,
            "apple": 0.2,
            "doesn't": 0.3,
            "fall": 0.2,
            "far": 0.1,
            "from": 0.1,
            "tree": 0,
            ".": 0
        }[token]


class TestSampling(TestCase):

    def test_get_count_prob(self):

        token_list_len = 5
        drop_probs = np.array([0.2] * token_list_len)

        # These settings should have no effect on the test.
        window = 2
        thresh = 1

        # Make a sampler (we need empty unigram and bigram instances to do this
        # but they have no direct involvement in the test).
        unigram = h.unigram.Unigram()
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)

        found_count_prob = sampler.get_count_prob_right(drop_probs)
        expected_count_prob = np.load(os.path.join(
            dp.CONSTANTS.TEST_DATA_DIR, 'count_probabilities.npy'))
        self.assertTrue(np.allclose(found_count_prob, expected_count_prob))


    def test_sample_w2v(self):
        np.random.seed(0)

        tokens = "the apple doesn't fall far from tree .".split()
        unigram = MockUnigram()
        for token in tokens:
            unigram.add(token)

        # First look for a match when threshold is high enough that no
        # tokens will be dropped.  Expected cooccurrences are just given 
        # by the "dynamic sampling" kernel sliding according to token position.
        window = 2
        thresh = 1
        expected_cooccurrences = np.array([
            [0, 1, 1/2, 0, 0, 0, 0, 0],
            [1, 0, 1, 1/2, 0, 0, 0, 0],
            [1/2, 1, 0, 1, 1/2, 0, 0, 0],
            [0, 1/2, 1, 0, 1, 1/2, 0, 0],
            [0, 0, 1/2, 1, 0, 1, 1/2, 0],
            [0, 0, 0, 1/2, 1, 0, 1, 1/2],
            [0, 0, 0, 0, 1/2, 1, 0, 1],
            [0, 0, 0, 0, 0, 1/2, 1, 0],
        ])
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)
        sampler.sample(tokens)
        self.assertTrue(
            np.allclose(bigram.Nxx.toarray(), expected_cooccurrences))

        # Now set the threshold so that some tokens get dropped.  Simulate
        # many replicates of the w2v dynamic sampling, and check that the
        # the average number of counts mathes the direct calculation of
        # counts expectation.
        window = 3
        thresh = 0.05
        mean_weight = np.zeros((8,8), dtype=np.float32)

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)

        # Repeatedly bigram counting with common-word undersampling, to
        # estimate the expectation bigram counts.
        drop_probs = sampler.drop_prob(tokens)
        num_replicates = 1000
        for rep in range(num_replicates):
            kept_tokens = [
                token for token, prob in zip(tokens, drop_probs)
                if prob < random.random()
            ]
            for i in range(len(kept_tokens)):
                use_window = random.randint(1,window)
                for j in range(i-use_window, i+use_window+1):
                    if j == i or j < 0 or j >= len(kept_tokens):
                        continue
                    idx1 = unigram.dictionary.get_id(kept_tokens[i])
                    idx2 = unigram.dictionary.get_id(kept_tokens[j])
                    mean_weight[idx1, idx2] += 1

        # Calculate expectation bigram counts. 
        mean_weight = mean_weight / num_replicates

        # Use the sampler to get bigram counts.
        #sampler.sample(tokens)
        sampler.sample(tokens)

        self.assertTrue(np.allclose(
            bigram.Nxx.toarray(), mean_weight, atol=0.05))
    

class TestUnigramExtraction(TestCase):

    def test_extract_unigram(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        with open(corpus_path) as test_corpus:
            tokens = test_corpus.read().strip().split()

        expected_counts = Counter(tokens)

        unigram = h.unigram.Unigram()
        dp.bigram_extraction.extract_unigram(
            corpus_path, unigram, verbose=False)
        self.assertEqual(len(unigram), len(expected_counts))
        for token in expected_counts:
            self.assertEqual(unigram.count(token), expected_counts[token])


    def test_extract_unigram_parallel(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        with open(corpus_path) as test_corpus:
            tokens = test_corpus.read().strip().split()

        expected_counts = Counter(tokens)

        # Try extraction with different numbers of workers
        for num_workers in range(1,5):
            unigram = dp.bigram_extraction.extract_unigram_parallel(
                corpus_path, 1, verbose=False)
            self.assertEqual(len(unigram), len(expected_counts))
            for token in expected_counts:
                self.assertEqual(unigram.count(token), expected_counts[token])

    




class TestBigramExtraction(TestCase):

    def test_extract_bigram(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        window = 3
        with open(corpus_path) as test_file:
            documents = [doc.split() for doc in test_file.read().split('\n')]
        unigram = h.unigram.Unigram()
        dp.bigram_extraction.extract_unigram(
            corpus_path, unigram, verbose=False)

        # Test extracting using flat weighting
        expected_counts = Counter()
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    expected_counts[doc[i],doc[j]] += 1

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerFlat(bigram, window)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )

        # Test extracting using harmonic weighting
        expected_counts = Counter()
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    expected_counts[doc[i],doc[j]] += 1.0/abs(i-j)

        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerHarmonic(bigram, window)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )

        # Test dynamic weight
        expected_counts = Counter()
        random.seed(0)
        for doc in documents:
            for i in range(len(doc)):
                for j in range(i-window, i+window+1):
                    if j==i or j<0 or j>=len(doc):
                        continue
                    d = abs(i-j)
                    weight = (window - d + 1) / window
                    expected_counts[doc[i],doc[j]] += weight

        thresh = 1 # Disables common-word undersampling
        bigram = h.bigram.Bigram(unigram)
        sampler = dp.bigram_sampler.SamplerW2V(bigram, window, thresh)
        dp.bigram_extraction.extract_bigram(
            corpus_path, sampler, verbose=False)

        for token1 in unigram.dictionary.tokens:
            for token2 in unigram.dictionary.tokens:
                self.assertEqual(
                    bigram.count(token1, token2),
                    expected_counts[token1, token2]
                )



    def test_extract_bigram_parallel(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        window = 3
        min_count = None
        thresh = 1
        with open(corpus_path) as test_file:
            documents = [doc.split() for doc in test_file.read().split('\n')]
        unigram = h.unigram.Unigram()
        dp.bigram_extraction.extract_unigram(
            corpus_path, unigram, verbose=False)
        pristine_unigram = copy.deepcopy(unigram)

        # Test for different numbers of worker processes
        for num_workers in range(1, 5):

            # Test extracting using flat weighting
            expected_counts = Counter()
            for doc in documents:
                for i in range(len(doc)):
                    for j in range(i-window, i+window+1):
                        if j==i or j<0 or j>=len(doc):
                            continue
                        expected_counts[doc[i],doc[j]] += 1

            bigram = dp.bigram_extraction.extract_bigram_parallel(
                corpus_path, num_workers, unigram, 'flat', window, min_count,
                thresh, verbose=False
            )

            for token1 in unigram.dictionary.tokens:
                for token2 in unigram.dictionary.tokens:
                    self.assertEqual(
                        bigram.count(token1, token2),
                        expected_counts[token1, token2]
                    )

            # Verify that the unigram is unaffected
            self.assertTrue(np.array_equal(bigram.unigram.Nx, unigram.Nx))


            # Test extracting using harmonic weighting
            expected_counts = Counter()
            for doc in documents:
                for i in range(len(doc)):
                    for j in range(i-window, i+window+1):
                        if j==i or j<0 or j>=len(doc):
                            continue
                        expected_counts[doc[i],doc[j]] += 1.0/abs(i-j)

            bigram = dp.bigram_extraction.extract_bigram_parallel(
                corpus_path, num_workers, unigram, 'harmonic', window,
                min_count, thresh, verbose=False
            )

            for token1 in unigram.dictionary.tokens:
                for token2 in unigram.dictionary.tokens:
                    self.assertEqual(
                        bigram.count(token1, token2),
                        expected_counts[token1, token2]
                    )

            # Verify that the unigram is unaffected
            self.assertTrue(np.array_equal(bigram.unigram.Nx, unigram.Nx))

            # Test w2v sampler
            expected_counts = Counter()
            random.seed(0)
            for doc in documents:
                for i in range(len(doc)):
                    for j in range(i-window, i+window+1):
                        if j==i or j<0 or j>=len(doc):
                            continue
                        d = abs(i-j)
                        weight = (window - d + 1) / window
                        expected_counts[doc[i],doc[j]] += weight

            bigram = dp.bigram_extraction.extract_bigram_parallel(
                corpus_path, num_workers, unigram, 'w2v', window, min_count, 
                thresh, verbose=False
            )

            for token1 in unigram.dictionary.tokens:
                for token2 in unigram.dictionary.tokens:
                    self.assertEqual(
                        bigram.count(token1, token2),
                        expected_counts[token1, token2]
                    )

            # Verify that the unigram is unaffected
            self.assertTrue(np.array_equal(bigram.unigram.Nx, unigram.Nx))

            # Test dynamic weight
            expected_counts = Counter()
            random.seed(0)
            for doc in documents:
                for i in range(len(doc)):
                    for j in range(i-window, i+window+1):
                        if j==i or j<0 or j>=len(doc):
                            continue
                        d = abs(i-j)
                        weight = (window - d + 1) / window
                        expected_counts[doc[i],doc[j]] += weight

            bigram = dp.bigram_extraction.extract_bigram_parallel(
                corpus_path, num_workers, unigram, 'dynamic', window, min_count,
                thresh, verbose=False
            )

            for token1 in unigram.dictionary.tokens:
                for token2 in unigram.dictionary.tokens:
                    self.assertEqual(
                        bigram.count(token1, token2),
                        expected_counts[token1, token2]
                    )

            # Verify that the unigram is unaffected
            self.assertTrue(np.array_equal(bigram.unigram.Nx, unigram.Nx))



class TestFileAccess(TestCase):

    def test_file_access(self):
        fname = 'tokenized-cat-test-long.txt'
        test_path = os.path.join(dp.CONSTANTS.TEST_DATA_DIR, fname)

        with open(test_path) as test_file: 
            expected_lines = test_file.readlines()

        # Run the test for many different chunk sizes, to ensure that even
        # when edgecases occur, we always read every line once, and only once.
        # Important edgecases are when chunk boundaries occur right at, right
        # before, or right after a newline, and when a chunk contains no lines
        # at all.
        for num_chunks in range(1,12):
            found_lines = []
            for chunk in range(num_chunks):
                add_lines = list(
                    dp.file_access.open_chunk(test_path, chunk, num_chunks)
                )
                found_lines.extend(add_lines)

            self.assertEqual(expected_lines, found_lines)

        # Trying to read a chunk greater than or equal to num_chunks is an 
        # error.
        with self.assertRaises(ValueError):
            dp.file_access.open_chunk(test_path, 0, 0)
        with self.assertRaises(ValueError):
            dp.file_access.open_chunk(test_path, 2, 2)
        with self.assertRaises(ValueError):
            dp.file_access.open_chunk(test_path, 2, 1)



if __name__ == '__main__':
    main()


