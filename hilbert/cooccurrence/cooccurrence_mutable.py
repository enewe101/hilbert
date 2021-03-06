import os
import time
from copy import deepcopy
from collections import Counter

try:
    import numpy as np
    from scipy import sparse, stats
except ImportError:
    np = None
    sparse = None
    stats = None

import hilbert as h
from .cooccurrence import Cooccurrence


def read_stats(path):
    return CooccurrenceMutable.load(path)


def sectorize(path, sector_factor, out_path=None, verbose=True):
    out_path = out_path if out_path is not None else path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    start = time.time()
    cooccurrence = CooccurrenceMutable.load(path)
    if verbose:
        print("Loaded ({} seconds)".format(time.time() - start))
    start = time.time()
    save_extras = True
    for i, sector in enumerate(h.shards.Shards(sector_factor)):
        cooccurrence.save_sector(out_path, sector, save_extras, save_extras)
        save_extras = False
        if verbose:
            print("Saved sector {} ({} seconds)".format(i, time.time() - start))
        start = time.time()


def write_marginals(path):
    """
    An old version of CooccurrenceMutable did not save marginals.  This reads the old
    version, and writes out the marginal, so that they can be loaded according
    to the new version.
    """
    Nxx_path = os.path.join(path, 'Nxx.npz')
    Nx_path = os.path.join(path, 'Nx.npy')
    Nxt_path = os.path.join(path, 'Nxt.npy')
    Nxx = sparse.load_npz(Nxx_path).tolil()
    np.save(Nx_path, np.asarray(np.sum(Nxx, axis=1)))
    np.save(Nxt_path, np.asarray(np.sum(Nxx, axis=0)))


def truncate(in_path, out_path, k):
    b = CooccurrenceMutable.load(in_path)
    b.truncate(k)
    b.save(out_path)


class CooccurrenceMutable(Cooccurrence):
    """
    Similar to Cooccurrence, but supporting various mutation and 
    writing operations.  useful for accumulating cooccurrence 
    while reading through a corpus.
    """

    def __init__(
        self,
        unigram,
        Nxx=None,
        marginalize=True,
        verbose=True
    ):
        if Nxx is None:
            Nxx = np.zeros((len(unigram), len(unigram)))
            Nxx = sparse.lil_matrix(Nxx)
        super(CooccurrenceMutable, self).__init__(
            unigram, Nxx=Nxx, marginalize=True, verbose=verbose)


    def __copy__(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        unigram_copy = deepcopy(self.unigram, memo)
        result = CooccurrenceMutable(
            unigram=unigram_copy, Nxx=self.Nxx, verbose=self.verbose)
        memo[id(self)] = result
        return result


    def add(self, token1, token2, count=1, skip_unk=False):

        # Get token idxs.
        id1 = self.dictionary.get_id_safe(token1)
        id2 = self.dictionary.get_id_safe(token2)

        # Handle tokens that aren't in the dictionary
        if id1 == None or id2 == None: 
            missing_token = token1 if id1 is None else token2
            if skip_unk: 
                return
            raise ValueError(
                'Cannot add cooccurrence, no entry for "{}" in dictionary'
                .format(missing_token)
            )

        # Add counts.
        self.add_id([id1], [id2], count)


    def add_id(self, focal_ids, context_ids, count=1):
        # I wrote this originally hoping that I could "vectorize" the addition
        # of counts, and hence avoid the for-loop.  Doing so is faster, but it
        # repeated indices in a vectorized addition are not accumulated.
        for focal_id, context_id in zip(focal_ids, context_ids):
            self.Nxx[focal_id, context_id] += np.array([count])
            self.Nx[focal_id] += count
            self.Nxt[0][context_id] += count
            self.N += count


    #def sort(self, force=False):
    #    """Adopt decreasing order of unigram frequency."""
    #    raise NotImplementedError("`CooccurrenceMutable`s are always sorted.")
    #    top_indices = self.unigram.sort()
    #    self.Nxx = self.Nxx.tocsr()[top_indices][:,top_indices].tocsr()
    #    self.Nx = self.Nx[top_indices]
    #    self.Nxt = self.Nxt[:,top_indices]


    def truncate(self, k):
        """Drop all but the `k` most common words."""

        #if not self.sorted:
        #    self.sort()

        self.Nxx = self.Nxx[:k][:,:k]
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)
        self.unigram.truncate(k)


    def save(self, path, save_unigram=True, save_marginals=True):
        """
        Save the cooccurrence data to disk.  A new directory will be created
        at `path`, and two files will be created within it to store the 
        token-ID mapping and the cooccurrence data.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        self.save_cooccurrences(path)
        if save_unigram:
            self.unigram.save(path)
        if save_marginals:
            self.save_marginals(path)


    def save_cooccurrences(self, path):
        sparse.save_npz(os.path.join(path, 'Nxx.npz'), self.Nxx.tocsr())


    def save_marginals(self, path):
        Nx_path = os.path.join(path, 'Nx.npy')
        np.save(Nx_path, self.Nx)
        Nxt_path = os.path.join(path, 'Nxt.npy')
        np.save(Nxt_path, self.Nxt)


    def save_sectors(self, path, sectors):
        first = True
        for sector in sectors:
            self.save_sector(
                path, sector, save_marginal=first, save_unigram=first)
            first = False


    def save_sector(
        self, path, sector, 
        save_marginal=True, save_unigram=True
    ):
        """
        Saves the indicated sector, does not save marginalized or unigram data.
        Equivalent to calling 
        `save(path, sector, seave_marginalized=False, save_unigram=False)`.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        Nxx_fname = 'Nxx-{}-{}-{}.npz'.format(*h.shards.serialize(sector))
        sparse.save_npz(os.path.join(path, Nxx_fname), self.Nxx[sector].tocsr())

        # Save the marginalized cooccurrence statistics.  This differs from
        # unigram statistics by approximately a factor of 2 * window_size - 1.
        if save_marginal:
            self.save_marginals(path)

        if save_unigram:
            self.unigram.save(path)


    @staticmethod
    def load(path, marginalize=True, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        unigram = h.unigram.Unigram.load(path, verbose=verbose)
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
        return CooccurrenceMutable(
            unigram, Nxx, marginalize=True, verbose=verbose)


    @staticmethod
    def load_unigram(path, verbose=True):
        unigram = h.unigram.Unigram.load(path)
        return CooccurrenceMutable(unigram, verbose=verbose)


