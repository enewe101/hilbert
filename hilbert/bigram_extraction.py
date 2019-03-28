import os
import sys
import random
import codecs
import argparse
import itertools
from collections import Counter
import time

import numpy as np

import hilbert as h
import shared
import data_preparation as dp
from multiprocessing import Pool


def extract_unigram(corpus_path, unigram, verbose=True):
    """
    Slowly accumulate the vocabulary and unigram statistics for the corpus
    path at ``corpus_path``, and using the provided ``unigram`` instance.

    To do this more quickly using parallelization, use 
    ``extract_unigram_parallel`` instead.
    """
    with open(corpus_path) as corpus_f:
        for line_num, line in enumerate(corpus_f):
            if verbose and line_num % 1000 == 0:
                print(line_num)
            tokens = line.strip().split()
            for token in tokens:
                unigram.add(token)
    unigram.sort()


def extract_unigram_parallel(
    corpus_path, num_workers, save_path=None, verbose=True
):
    """
    Accumulate the vocabulary and unigram statistics for the corpus
    path at ``corpus_path``, by parallelizing across ``num_workers`` processes.
    Optionally save the unigram statistics to the directory ``save_path``
    (making it if it doesn't exist).
    """
    pool = Pool(num_workers)
    args = (
        (corpus_path, worker_id, num_workers, verbose) 
        for worker_id in range(num_workers)
    )
    unigrams = pool.map(extract_unigram_parallel_worker, args)
    unigram = sum(unigrams, h.unigram.Unigram())
    if save_path is not None:
        unigram.save(save_path)
    return unigram


def extract_unigram_parallel_worker(args):
    corpus_path, worker_id, num_workers, verbose = args
    unigram = h.unigram.Unigram()
    file_chunk = dp.file_access.open_chunk(corpus_path, worker_id, num_workers)
    for line_num, line in enumerate(file_chunk):
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            print(line_num)
        tokens = line.strip().split()
        for token in tokens:
            unigram.add(token)
    return unigram




def extract_bigram(corpus_path, sampler, verbose=True):
    """
    Slowly extracts cooccurrence statistics from ``corpus_path`` using
    the ``Sampler`` instance ``sampler``.

    The input file should be space-tokenized, and normally should have one
    document per line.  Cooccurrences are only considered within the same line
    (i.e. words on different lines aren't considered as cooccurring no matter
    how close together they are).

    To do this more quickly using parallelization, use
    ``extract_bigram_parallel`` instead.
    """
    with codecs.open(corpus_path, 'r', 'utf8') as in_file:
        for line_num, line in enumerate(in_file):
            if verbose and line_num % 1000 == 0:
                print(line_num)
            sampler.sample(line.split())



def extract_bigram_parallel(
    corpus_path, num_workers, unigram, sampler_type, window, min_count, thresh,
    save_path=None, verbose=True
):
    pool = Pool(num_workers)
    args = (
        (
            corpus_path, worker_id, num_workers, unigram,
            sampler_type, window, min_count, thresh, verbose
        ) 
        for worker_id in range(num_workers)
    )
    bigrams = pool.map(extract_bigram_parallel_worker, args)

    merged_bigram = bigrams[0]
    for bigram in bigrams[1:]:
        merged_bigram.merge(bigram)

    if save_path is not None:
        merged_bigram.save(save_path)
    return merged_bigram



def extract_bigram_parallel_worker(args):
    (
        corpus_path, worker_id, num_workers, unigram, sampler_type, 
        window, min_count, thresh, verbose
    ) = args
    bigram = h.bigram.BigramMutable(unigram)
    sampler = h.bigram.sampler.get_sampler(
        sampler_type, bigram, window, min_count, thresh)
    file_chunk = dp.file_access.open_chunk(corpus_path, worker_id, num_workers)
    start = time.time()
    for line_num, line in enumerate(file_chunk):
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            print('elapsed', time.time() - start)
            start = time.time()
            print(line_num)
        sampler.sample(line.split())

    return bigram



def extract_unigram_and_bigram(
    corpus_path,
    out_dir,
    sampler_type,
    window,
    processes=1,
    min_count=None,
    thresh=None,
    vocab=None
):

    print('Corpus path:\t{}\n'.format(corpus_path))
    print('Output path:\t{}\n'.format(out_dir))
    print('Sampler type:\t{}\n'.format(sampler_type))
    print('Window size:\t{}\n'.format(window))

    # Attempt to read unigram, if none exists, then train it and save to disc.
    try:

        print('Attempting to read unigram data...')
        unigram = h.unigram.Unigram.load(out_dir)
        if vocab is not None and len(unigram) > vocab:
            raise ValueError(
                'An existing unigram object was found on disk, having a '
                'vocabulary size of {}, but a vocabulary size of {} was '
                'requested.  Either truncate it manually, or run extraction '
                'for existing vocabulary size.'.format(len(unigram), vocab)
            )
        elif min_count is not None and min(unigram.Nx) < min_count:
            raise ValueError(
                'An existing unigram object was found on disk, containing '
                'tokens occuring only {} times (less than the requested '
                'min_count of {}).  Either prune it manually, or run '
                'extraction with `min_count` reduced.'.format(
                    min(unigram.Nx), min_count))

    except IOError:
        print('None found.  Training unigram data...')
        unigram = extract_unigram_parallel(corpus_path, processes)
        if vocab is not None:
            unigram.truncate(vocab)
        if min_count is not None:
            unigram.prune(min_count)

        print('Saving unigram data...')
        unigram.save(out_dir)

    # Train the bigram, and save it to disc.
    print('Training bigram data...')
    bigram = extract_bigram_parallel(
        corpus_path, processes, unigram, sampler_type, window, min_count, 
        thresh
    )
    print('Saving bigram data...')
    bigram.save(out_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=(
        "Extracts unigram and bigram statistics from a corpus, "
        "and stores to disk"
    ))
    parser.add_argument(
        '--corpus', '-c', required=True, dest='corpus_filename',
        help="File name for input corpus"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, 
        help="Name of directory in which to store cooccurrence data"
    )
    parser.add_argument(
        '--thresh', '-t', type=float, help=(
            "Threshold for common-word undersampling, "
            "for use with w2v sampler only"
        )
    )
    parser.add_argument(
        '--sampler', '-s', help="Type of sampler to use",
        choices=('w2v', 'flat', 'harmonic', 'dynamic'), required=True,
        dest="sampler_type"
    )
    parser.add_argument(
        '--window', '-w', help="Cooccurrence window size",
        required=True, type=int
    )
    parser.add_argument(
        '--vocab', '-v', type=int, default=None,
        help="Prune vocabulary to the most common `vocab` number of words"
    )
    parser.add_argument(
        '--processes', '-p', help="Number of processes to spawn",
        default=1, type=int
    )
    parser.add_argument(
        '--min-count', '-m', default=None, type=int,
        help="Minimum number of occurrences below which token is ignored",
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Corpus path and output directory are relative to standard locations.
    args['corpus_path'] = os.path.join(
        shared.CONSTANTS.TOKENIZED_CAT_DIR, args['corpus_filename']
    )
    del args['corpus_filename']
    args['out_dir'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['out_dir']
    )

    if args['min_count'] is not None and args['vocab'] is not None:
        raise ValueError('Use either --vocab or --min-count, not both.')

    # thresh should only be specified if the sampler is w2v
    if args['thresh'] is not None and args['sampler_type'] != 'w2v':
        raise ValueError('thresh argument is only valid for w2v sampler.')

    extract_unigram_and_bigram(**args)

