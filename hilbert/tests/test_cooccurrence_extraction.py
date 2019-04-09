import os
import shutil
import random
from collections import Counter
import copy
import itertools

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




# Expected results using a flat cooccurrence extractor
def get_expected_flat_counts(documents, window):
    # Test extracting using flat weighting
    expected_counts = Counter()
    for doc in documents:
        for i in range(len(doc)):
            for j in range(i-window, i+window+1):
                if j==i or j<0 or j>=len(doc):
                    continue
                expected_counts[doc[i],doc[j]] += 1
    return expected_counts


# Expected results using a harmonic cooccurrence extractor
def get_expected_harmonic_counts(documents, window):
    # Test extracting using harmonic weighting
    expected_counts = Counter()
    for doc in documents:
        for i in range(len(doc)):
            for j in range(i-window, i+window+1):
                if j==i or j<0 or j>=len(doc):
                    continue
                expected_counts[doc[i],doc[j]] += 1.0/abs(i-j)
    return expected_counts


# Expected results using a dynamic cooccurrence extractor
def get_expected_dynamic_counts(documents, window):
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
    return expected_counts


# Expected results using a custom cooccurrence extractor
def get_expected_custom_counts(documents, weights):
    right_weights, left_weights = weights
    expected_counts = Counter()
    random.seed(0)
    for doc in documents:
        for i in range(len(doc)):
            for offset in range(1, len(right_weights)+1):
                j = i + offset
                if j==i or j<0 or j>=len(doc):
                    continue
                weight = right_weights[offset-1]
                expected_counts[doc[i],doc[j]] += weight
            for offset in range(1, len(left_weights)+1):
                j = i - offset
                if j==i or j<0 or j>=len(doc):
                    continue
                weight = left_weights[offset-1]
                expected_counts[doc[i],doc[j]] += weight
    return expected_counts


def get_expected_counts(extractor_str, documents, weights, window):
    if extractor_str == 'flat':
        return get_expected_flat_counts(documents, window)
    elif extractor_str == 'harmonic':
        return get_expected_harmonic_counts(documents, window)
    elif extractor_str == 'dynamic':
        return get_expected_dynamic_counts(documents, window)
    elif extractor_str == 'custom':
        return get_expected_custom_counts(documents, weights)
    else:
        raise ValueError("Unexpected extractor_str: {}.".format(extractor_str))


class TestUnigramExtraction(TestCase):

    def test_extract_unigram(self):

        corpus_path = dp.CONSTANTS.TOKENIZED_CAT_TEST_PATH
        with open(corpus_path) as test_corpus:
            tokens = test_corpus.read().strip().split()

        expected_counts = Counter(tokens)

        unigram = h.unigram.Unigram()
        h.cooccurrence.extraction.extract_unigram(
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
            unigram = h.cooccurrence.extraction.extract_unigram_parallel(
                corpus_path, 1, verbose=False)
            self.assertEqual(len(unigram), len(expected_counts))
            for token in expected_counts:
                self.assertEqual(unigram.count(token), expected_counts[token])

    

def counts_are_equal(unigram, cooccurrence, expected_counts):
    atol = 1e-4
    for token1 in unigram.dictionary.tokens:
        for token2 in unigram.dictionary.tokens:
            found = cooccurrence.count(token1, token2)
            expected = expected_counts[token1, token2]
            if abs(found-expected) > atol:
                print(token1, token2, found, expected)
                return False
    return True



def read_test_corpus(corpus_path):
    with open(corpus_path) as test_file:
        documents = [doc.split() for doc in test_file.read().split('\n')]
    unigram = h.unigram.Unigram()
    h.cooccurrence.extraction.extract_unigram(
        corpus_path, unigram, verbose=False)
    return documents, unigram


class TestCooccurrenceExtraction(TestCase):


    def setup(self):
        corpus_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-cooccurrence.txt')
        with open(corpus_path) as test_file:
            documents = [doc.split() for doc in test_file.read().split('\n')]
        unigram = h.unigram.Unigram()
        h.cooccurrence.extraction.extract_unigram(
            corpus_path, unigram, verbose=False)
        return corpus_path, unigram, documents


    def test_read_weights_file(self):
        path = os.path.join(h.CONSTANTS.TEST_DIR, 'example-weights.txt')
        weights = h.cooccurrence.extractor.read_weights_file(path)
        expected_weights = (
            [1.0, 0.5, 0.3333333333333333, 0.25, 0.2],
            [1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.1]
        )
        self.assertEqual(weights, expected_weights)

        path = os.path.join(h.CONSTANTS.TEST_DIR, 'example-weights-bad.txt')
        with self.assertRaises(ValueError):
            weights = h.cooccurrence.extractor.read_weights_file(path)



    def test_extract_cooccurrence(self):
        corpus_path, unigram, documents = self.setup()
        for args in get_extractor_options(with_workers=False):

            # Build the cooccurrence, the extractor, and extract the statistics
            del args['vocab']   # We don't want this in this test.
            cooccurrence = h.cooccurrence.CooccurrenceMutable(unigram)
            extractor = h.cooccurrence.extractor.get_extractor(
                cooccurrence=cooccurrence, **args)
            h.cooccurrence.extraction.extract_cooccurrence(
                corpus_path, extractor, verbose=False)

            # Calculate the counts we expect for this test
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=args['extractor_str'],
                weights=args['weights'],
                window=args['window'],
            )

            # Check that the counts are as expected
            counts_are_equal(unigram, cooccurrence, expected_counts)


    def test_extract_cooccurrence_parallel(self):
        corpus_path, unigram, documents = self.setup()
        for args in get_extractor_options():

            # Run extraction code to get counts
            del args['vocab']   # We don't want this in this test.
            cooccurrence = (
                h.cooccurrence.extraction.extract_cooccurrence_parallel(
                    corpus_path=corpus_path, unigram=unigram, **args,
                    verbose=False
                )
            )

            # Calculate the counts we expect for this test
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=args['extractor_str'],
                weights=args['weights'],
                window=args['window'],
            )

            # Check that counts are as expected
            counts_are_equal(unigram, cooccurrence, expected_counts)

            # Verify that the unigram was not altered
            self.assertTrue(np.array_equal(cooccurrence.unigram.Nx, unigram.Nx))


def get_extractor_options(with_workers=True):
    preset_extractors = ['flat', 'harmonic', 'dynamic']
    windows = [
        {'window': 2, 'weights':None},
        {'window': 5, 'weights':None},
    ]
    preset_extractor_combos = [
        {'extractor_str':extractor, **window} for 
        extractor, window in itertools.product(preset_extractors, windows)
    ]
    custom_extractor_combos = [
        {
            'extractor_str': 'custom', 'window': None,
            'weights':([0.0, 0.0, 1.25], [1.0, 0.5, 0.25])
        },
        {
            'extractor_str': 'custom', 'window': None,
            'weights':([1.0, 0.5, 0.25], [0.0, 0.0, 0.1])
        }
    ]
    extractor_combos = preset_extractor_combos + custom_extractor_combos
    count_vocabs = [
        {'min_count':None, 'vocab':None},
    ]
    all_option_combos = [
        {**extractor_combo, **count_vocab} for extractor_combo, count_vocab
        in itertools.product(extractor_combos, count_vocabs)
    ]
    if not with_workers:
        return all_option_combos
    all_option_combos = [
        {'num_workers':num_workers, **combo} for combo, num_workers
        in itertools.product(all_option_combos, [1,2,5])
    ]
    return all_option_combos



class TestExtractUnigramAndCooccurrence(TestCase):


    def test_extract_unigram_and_cooccurrence(self):
        corpus_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-cooccurrence.txt')
        save_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-unigram-and-cooccurrence.txt')
        documents, unigram_pristine = read_test_corpus(corpus_path)
        num_workers = 2
        for extractor_options in get_extractor_options(with_workers=False):

            # Make sure any output from previous iterations is removed.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Run extraction (writes to disk), then load the result.
            h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                corpus_path=corpus_path, save_path=save_path, 
                num_workers=num_workers, **extractor_options, verbose=False
            )
            cooccurrence = h.cooccurrence.Cooccurrence.load(save_path)

            # Calculate the expected counts
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=extractor_options['extractor_str'],
                weights=extractor_options['weights'],
                window=extractor_options['window'],
            )

            # Verify we got what we expected
            #unigram = copy.copy(unigram_pristine)
            #if extractor_options['vocab'] is not None:
            #    unigram.truncate(extractor_options['vocab'])
            #if extractor_options['min_count'] is not None:
            #    unigram.prune(extractor_options['min_count'])
            if not counts_are_equal(
                unigram_pristine,cooccurrence,expected_counts
            ):
                import pdb; pdb.set_trace()

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)


    def test_extract_unigram_and_cooccurrence_enforces_vocab(self):
        corpus_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-cooccurrence.txt')
        save_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-unigram-and-cooccurrence.txt')
        documents, unigram_pristine = read_test_corpus(corpus_path)
        weights = ([1.0, 0.5, 0.25], [0.0, 0.0, 0.1])
        window = 3
        num_workers = 2
        extractor_args = [
            {'weights': None, 'window':window, 'extractor_str':'flat'},
            {'weights': None, 'window':window, 'extractor_str':'harmonic'},
            {'weights': None, 'window':window, 'extractor_str':'dynamic'},
            {'weights': weights, 'window':None, 'extractor_str':'custom'},
        ]
        vocab_args = [
            {'vocab': None, 'min_count':None},
            {'vocab': None, 'min_count':0},
            {'vocab': None, 'min_count':1},
            {'vocab': None, 'min_count':2},
            {'vocab': 5, 'min_count':None},
            {'vocab': 10, 'min_count':None},
        ]
        all_args = itertools.product(extractor_args, vocab_args)
        for extractor_arg, vocab_arg in all_args:

            # Make sure any output from previous iterations is removed.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Run extraction (writes to disk), then load the result.
            h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                corpus_path=corpus_path, save_path=save_path, 
                num_workers=num_workers, **extractor_arg, **vocab_arg,
                verbose=False
            )
            cooccurrence = h.cooccurrence.Cooccurrence.load(save_path)
            vocab = vocab_arg['vocab']
            if vocab is not None:
                self.assertEqual(cooccurrence.shape, (vocab, vocab))
                self.assertEqual(cooccurrence.unigram.shape[0], vocab)
            min_count = vocab_arg['min_count']
            if min_count is not None:
                self.assertTrue(all([
                    n >= min_count for n in cooccurrence.unigram.Nx]))

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)



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


