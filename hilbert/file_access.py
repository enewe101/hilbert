"""
This module makes it easier to access specific chunks in a file, without
having to read sequentially to that chunk.  This makes it easier to parallelize
the extraction of occurrence statistics from a large file.
"""

import os
import math
import time
from multiprocessing import Pool

def f(a):
    time.sleep(1)
    return a*a


def readlines_parallel(path, num_readers):
    p = Pool(5)
    for result in p.imap_unordered(f, range(15)):
        print(result)


    



def open_chunk(path, chunk, num_chunks):
    """
    Iterate contiguous lines corresponding to a fraction of the file, generally
    starting mid-file, but always ensuring it yields full lines of the 
    original file.  To ensure that whole lines are yielded, the precice 
    starting and ending points will correspond to the byte following the first
    newline after byte `chunk/num_chunks` and the byte corresponding to the 
    first newline after `(chunk+1)/num_chunks`.
    """
    # Do a few validations.  These can't be done inside _open_chunk, because
    # it is a generator, and no lines within it will be run until the first
    # element is requested in the calling frame.  We would rather fail
    # immediately if there is a basic problem.
    _fail_fast(path, chunk, num_chunks)
    return _open_chunk(path, chunk, num_chunks)



def _fail_fast(path, chunk, num_chunks):
    # Test that the file is readable.
    with open(path) as file:
        pass
    # Ensure valid values for chunk and num_chunks.
    if chunk >= num_chunks:
        raise ValueError('`chunk` must be less than `num_chunks`.')



def _open_chunk(path, chunk, num_chunks):
    """
    Generator supporting the functionality of open_chunk.  Do not call this
    directly, call `open_chunk` instead.
    """
    total_bytes = os.path.getsize(path)
    start_point = math.ceil(total_bytes / num_chunks * chunk)
    end_point = math.ceil(total_bytes / num_chunks * (chunk + 1))
    with open(path) as f:
        f.seek(start_point)

        # We are generally landing midline, and we don't know where the 
        # last newline was, so we discard this partial line (it is included 
        # in the previous chunk, which must always read through until the 
        # newline occurring on or after its own endpoint, putting cursor
        # strictly after its endpoint).  But, in the case of the fist chunk
        # we must include the first line.
        if chunk > 0:
            f.readline()
        cursor = f.tell()

        while cursor <= end_point and cursor < total_bytes:
            yield f.readline()
            cursor = f.tell()

        raise StopIteration
        


def open_chunk_slow(path, chunk, num_chunks):
    """
    Equivalent to `open_chunk` in terms of the file data yielded, but much
    slower in practice, because it reads sequentially until startpoint, rather
    than seeking to it directly.
    """
    total_bytes = os.path.getsize(path)
    start_point = math.ceil(total_bytes / num_chunks * chunk)
    end_point = math.ceil(total_bytes / num_chunks * (chunk + 1))
    with open(path) as f:
        cursor = f.tell()
        while cursor < start_point:
            f.readline()

        while cursor <= end_point:
            yield f.readline()
            cursor = f.tell()

        raise StopIteration
        
