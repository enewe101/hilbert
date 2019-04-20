import os
import sys
import math
import random
import codecs
import argparse
import itertools
from collections import Counter
import time
import numpy as np
import hilbert as h
import shared
from multiprocessing import Pool



# TODO: test min_count and vocab


def extract_unigram_parallel(
    corpus_path, processes, save_path=None, verbose=True
):
    """
    Accumulate the vocabulary and unigram statistics for the corpus
    path at ``corpus_path``, by parallelizing across ``processes`` processes.
    Optionally save the unigram statistics to the directory ``save_path``
    (making it if it doesn't exist).
    """
    pool = Pool(processes)
    args = (
        (corpus_path, worker_id, processes, verbose) 
        for worker_id in range(processes)
    )
    unigrams = pool.map(extract_unigram_parallel_worker, args)
    unigram = sum(unigrams, h.unigram.Unigram())
    if save_path is not None:
        unigram.save(save_path)
    return unigram


def extract_unigram_parallel_worker(args):
    corpus_path, worker_id, processes, verbose = args
    unigram = h.unigram.Unigram()
    file_chunk = h.file_access.open_chunk(corpus_path, worker_id, processes)
    start = time.time()
    for line_num, line in enumerate(file_chunk):
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            sys.stdout.write(
                '\rTime elapsed: %0.f sec.;  '
                'Lines read (in one process): %d'
                % (time.time() - start, line_num)
            )
        tokens = line.strip().split()
        for token in tokens:
            unigram.add(token)
    if worker_id == 0 and verbose:
        print()
    return unigram



def extract_and_write_cooccurrence_parallel(
    corpus_path, processes, unigram, extractor_str, window=None,
    min_count=None, weights=None, save_path=None, save_sectorized=True,
    save_monolithic=False, verbose=True
):
    if not save_monolithic and not save_sectorized:
        raise ValueError(
            "You need to choose at least one of `save_monolithic` "
            "or `save_sectorized` to be `True`.  They were both `False`!"
        )

    pool = Pool(processes)
    extractor_constructor_args = {
        'extractor_str': extractor_str, 
        'window': window,
        'min_count': min_count,
        'weights': weights,
    }
    args = (
        (
            corpus_path, worker_id, processes, unigram,
            extractor_constructor_args, verbose
        ) 
        for worker_id in range(processes)
    )
    cooccurrence = pool.map(extract_cooccurrence_parallel_worker, args)

    merged_cooccurrence = cooccurrence[0]
    for _cooccurrence in cooccurrence[1:]:
        merged_cooccurrence.merge(_cooccurrence)

    max_sector_size = h.CONSTANTS.RC['max_sector_size']
    sector_factor = int(math.ceil(len(unigram) / max_sector_size))
    sectors = h.shards.Shards(sector_factor)
    if save_path is not None:
        if save_sectorized:
            merged_cooccurrence.save_sectors(save_path, sectors)
        if save_monolithic:
            merged_cooccurrence.save_cooccurrences(save_path)

        #merged_cooccurrence.save(save_path)
    return merged_cooccurrence



def extract_cooccurrence_parallel_worker(args):
    (
        corpus_path, worker_id, processes, unigram, 
        extractor_constructor_args, verbose
    ) = args
    cooccurrence = h.cooccurrence.CooccurrenceMutable(unigram)
    extractor = h.cooccurrence.extractor.get_extractor(
        cooccurrence=cooccurrence, **extractor_constructor_args)
    file_chunk = h.file_access.open_chunk(corpus_path, worker_id, processes)
    start = time.time()
    for line_num, line in enumerate(file_chunk):
        print(worker_id, line)
        if worker_id == 0 and verbose and line_num % 1000 == 0:
            sys.stdout.write(
                '\rTime elapsed: %0.f sec.;  '
                'Lines read (in one process): %d'
                % (time.time() - start, line_num)
            )
        extractor.extract(line.split())
    if worker_id == 0 and verbose:
        print()
    return cooccurrence


def l(s):
    """lengthen the string `s`."""
    return s.ljust(12, ' ')


def extract_unigram_and_cooccurrence(
    corpus_path,
    save_path,
    extractor_str,
    window=None,
    weights=None,
    processes=1,
    min_count=None,
    vocab=None,
    save_sectorized=True,
    save_monolithic=False,
    verbose=True
):

    if verbose:
        print()
        print(l('Processes:'), processes)
        print(l('Corpus:'), corpus_path)
        print(l('Output:'), save_path)
        print(l('Extractor:'),extractor_str)
        print(l('Weights:'), weights)
        print(l('Window:'), window)

    # Attempt to read unigram, if none exists, then train it and save to disc.
    try:

        if verbose:
            sys.stdout.write('\nAttempting to read unigram data...')

        unigram = h.unigram.Unigram.load(save_path)

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

        elif verbose:
            print('Found.')

    except IOError:
        if verbose:
            print('None found.  Collecting unigram data...')
        unigram = extract_unigram_parallel(
            corpus_path, processes, verbose=verbose)
        if vocab is not None:
            unigram.truncate(vocab)
        if min_count is not None:
            unigram.prune(min_count)
        if verbose:
            print('Saving unigram data...')
        unigram.save(save_path)

    # Extract the cooccurrence, and save it to disc.
    if verbose:
        print('\nCollecting cooccurrence data...')
    cooccurrence = extract_and_write_cooccurrence_parallel(
        corpus_path=corpus_path, processes=processes, unigram=unigram,
        extractor_str=extractor_str, window=window,
        min_count=min_count, weights=weights, save_path=save_path,
        save_sectorized=save_sectorized, save_monolithic=save_monolithic,
        verbose=verbose
    )   # This call both extracts and writes to disk.
    if verbose:
        print('\nSaving cooccurrence data...')



###
### FOSSILS: 
###
### Single-process implementations of the parallel extractors.
### It represents ground truth if the parallel extractors are in doubt.
###


def extract_cooccurrence(corpus_path, extractor, verbose=True):
    """
    Slowly extracts cooccurrence statistics from ``corpus_path`` using
    the ``Extractor`` instance ``extractor``.

    The input file should be space-tokenized, and normally should have one
    document per line.  Cooccurrences are only considered within the same line
    (i.e. words on different lines aren't considered as cooccurring no matter
    how close together they are).

    To do this more quickly using parallelization, use
    ``extract_and_write_cooccurrence_parallel`` instead.
    """
    with codecs.open(corpus_path, 'r', 'utf8') as in_file:
        start = time.time()
        for line_num, line in enumerate(in_file):
            if verbose and line_num % 1000 == 0:
                sys.stdout.write(
                    '\rTime elapsed: %0.f sec.;  '
                    'Lines read (in one process): %d'
                    % (time.time() - start, line_num)
                )
            extractor.extract(line.split())
    if verbose:
        print()



def extract_unigram(corpus_path, unigram, verbose=True):
    """
    Slowly accumulate the vocabulary and unigram statistics for the corpus
    path at ``corpus_path``, and using the provided ``unigram`` instance.

    To do this more quickly using parallelization, use 
    ``extract_unigram_parallel`` instead.
    """
    with open(corpus_path) as corpus_f:
        start = time.time()
        for line_num, line in enumerate(corpus_f):
            if verbose and line_num % 1000 == 0:
                sys.stdout.write(
                    '\rTime elapsed: %0.f sec.;  '
                    'Lines read (in one process): %d'
                    % (time.time() - start, line_num)
                )
            tokens = line.strip().split()
            for token in tokens:
                unigram.add(token)
    if verbose:
        print()
    unigram.sort()


