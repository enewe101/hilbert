import os
import math
import copy
import shutil
import random
import itertools
from collections import Counter
from unittest import main, TestCase

import numpy as np
from scipy import sparse

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
        for processes in range(1,5):
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



    # We're keeping around a function that extracts cooccurrences serially
    # as opposed to in parallel.
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
            self.assertTrue(
                counts_are_equal(unigram, cooccurrence, expected_counts))




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
        {'processes':processes, **combo} for combo, processes
        in itertools.product(all_option_combos, [1,2,5])
    ]
    return all_option_combos



class TestExtractUnigramAndCooccurrence(TestCase):

    def test_sector_or_monolithic(self):
        """
        Writes Nxx.npz only if `save_monolithic`.
        Writes sectors (e.g. 0-0-4.Nxx.npz) if `save_sectorized`.
        Raises ValueError if `not save_monolithic and not save_sectorized`. 
        """
        corpus_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-doc-long.txt')
        save_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-unigram-and-cooccurrence.txt')
        documents, unigram = read_test_corpus(corpus_path)
        processes = 1
        save_monolithic = True
        min_count = None
        vocab = None
        window = 5
        extractor_str = 'flat'
        weights = None
        h.CONSTANTS.RC['max_sector_size'] = 700

        options = [(False, False), (False, True), (True, False), (True, True)]
        for save_monolithic, save_sectorized in options:

            # Cleanup anything left over from previous run or loop cycle.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Calculate the expected counts.
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=extractor_str,
                weights=weights,
                window=window
            )

            # Make sure any output from previous iterations is removed.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Run extraction provided at least one of the write modes is 
            # active.  If neither is active, it should raise an error.
            if not save_monolithic and not save_sectorized:
                with self.assertRaises(ValueError):
                    h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                        corpus_path=corpus_path, save_path=save_path,
                        processes=processes, window=window,
                        extractor_str=extractor_str, weights=weights,
                        min_count=min_count, vocab=vocab,
                        save_sectorized=save_sectorized,
                        save_monolithic=save_monolithic, verbose=False
                    )
                continue
            else:
                h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                    corpus_path=corpus_path, save_path=save_path,
                    processes=processes, window=window,
                    extractor_str=extractor_str, weights=weights,
                    min_count=min_count, vocab=vocab,
                    save_sectorized=save_sectorized,
                    save_monolithic=save_monolithic, verbose=False
                )

            # Read sectorized counts.  They should only exist if we asked for 
            # them by setting `save_sectorized=True`.
            # But first check that the number of sectors written is what we
            # expect.
            if save_sectorized:
                found_sector_factor = (
                    h.cooccurrence.CooccurrenceSector.get_sector_factor(
                        save_path))
                expected_sector_factor = (
                    math.ceil(len(unigram) / h.CONSTANTS.RC['max_sector_size']))
                self.assertEqual(found_sector_factor, expected_sector_factor)

                Nxx = np.zeros((len(unigram), len(unigram)))
                for sector in h.shards.Shards(expected_sector_factor):
                    cooccurrence = h.cooccurrence.CooccurrenceSector.load(
                        save_path, sector)
                    Nxx[sector] += cooccurrence.Nxx.toarray()
                found_unigram = cooccurrence.unigram
                sect_cooccurrence = h.cooccurrence.Cooccurrence(
                    found_unigram, Nxx)
                self.assertTrue(counts_are_equal(
                    unigram, sect_cooccurrence, expected_counts))
            else:
                with self.assertRaises(FileNotFoundError):
                    Nxx = np.zeros((len(unigram), len(unigram)))
                    for sector in h.shards.Shards(expected_sector_factor):
                        cooccurrence = h.cooccurrence.CooccurrenceSector.load(
                            save_path, sector)
                        Nxx[sector] += cooccurrence.Nxx.toarray()

            if save_monolithic:
                monolithic_cooccurrence = h.cooccurrence.Cooccurrence.load(
                    save_path)
                self.assertTrue(counts_are_equal(
                    unigram, monolithic_cooccurrence, expected_counts))
            else:
                with self.assertRaises(FileNotFoundError):
                    monolithic_cooccurrence = h.cooccurrence.Cooccurrence.load(
                        save_path)

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def test_correct_sector_written(self):
        """
        If the vocabulary is larger than max_sector_size, it should choose a
        sector factor that makes it so that the size of a sector is as big
        as possible, but not bigger than max_sector_size.
        """

        corpus_path = os.path.join(h.CONSTANTS.TEST_DIR, 'test-doc-long.txt')
        save_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-unigram-and-cooccurrence.txt')
        documents, unigram = read_test_corpus(corpus_path)
        processes = 1
        save_monolithic = True
        min_count = None
        vocab = None
        window = 5
        extractor_str = 'flat'
        weights = None

        for sector_size in [500, 700]:
            h.CONSTANTS.RC['max_sector_size'] = sector_size

            # Calculate the expected counts.
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=extractor_str,
                weights=weights,
                window=window
            )

            # Make sure any output from previous iterations is removed.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Run extraction (writes to disk), then load the result.
            h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                corpus_path=corpus_path, save_path=save_path,
                processes=processes, window=window,
                extractor_str=extractor_str, weights=weights,
                min_count=min_count, vocab=vocab,
                save_monolithic=save_monolithic, verbose=False
            )

            # Check that the number of sectors written is what we expect.
            found_sector_factor = (
                h.cooccurrence.CooccurrenceSector.get_sector_factor(save_path))
            expected_sector_factor = (
                math.ceil(len(unigram) / h.CONSTANTS.RC['max_sector_size']))
            self.assertEqual(found_sector_factor, expected_sector_factor)

            # Read counts from disk.  Accumulate over all the sectors.
            Nxx = np.zeros((len(unigram), len(unigram)))
            for sector in h.shards.Shards(expected_sector_factor):
                cooccurrence = h.cooccurrence.CooccurrenceSector.load(
                    save_path, sector)
                Nxx[sector] += cooccurrence.Nxx.toarray()
            found_unigram = cooccurrence.unigram
            sect_cooccurrence = h.cooccurrence.Cooccurrence(found_unigram, Nxx)

            # The counts stored monolithically, and the counts stored in
            # sectors, should be equal.
            monolithic_cooccurrence = h.cooccurrence.Cooccurrence.load(
                save_path)
            self.assertTrue(np.allclose(
                sect_cooccurrence.Nxx.toarray(),
                monolithic_cooccurrence.Nxx.toarray()
            ))

            # Compare counts from accumulating over all sectors to expected
            # counts.
            self.assertTrue(counts_are_equal(
                unigram, sect_cooccurrence, expected_counts))

        # Cleanup
        if os.path.exists(save_path):
            shutil.rmtree(save_path)



    def test_extract_unigram_and_cooccurrence(self):
        corpus_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-cooccurrence.txt')
        save_path = os.path.join(
            h.CONSTANTS.TEST_DIR, 'test-extract-unigram-and-cooccurrence.txt')
        documents, unigram_pristine = read_test_corpus(corpus_path)
        processes = 2
        for extractor_options in get_extractor_options(with_workers=False):

            # Make sure any output from previous iterations is removed.
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            # Run extraction (writes to disk), then load the result.
            h.cooccurrence.extraction.extract_unigram_and_cooccurrence(
                corpus_path=corpus_path, save_path=save_path, 
                processes=processes, **extractor_options, verbose=False
            )
            sector = h.shards.Shards(1)[0]
            cooccurrence = h.cooccurrence.CooccurrenceSector.load(
                save_path, sector)

            # Calculate the expected counts
            expected_counts = get_expected_counts(
                documents=documents,
                extractor_str=extractor_options['extractor_str'],
                weights=extractor_options['weights'],
                window=extractor_options['window'],
            )

            self.assertTrue(
                counts_are_equal(unigram_pristine,cooccurrence,expected_counts))

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
        processes = 2
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
                processes=processes, **extractor_arg, **vocab_arg,
                verbose=False, save_monolithic=True, save_sectorized=False
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


