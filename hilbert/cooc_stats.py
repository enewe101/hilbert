import os
from collections import Counter

import numpy as np
from scipy import sparse

import data_preparation as dp
import hilbert as h


def get_stats(name):
    path = dp.path_iteration.get_cooccurrence_path(name)
    return CoocStats.load(path)



class CoocStats(object):
    """Represents cooccurrence statistics."""

    def __init__(
        self,
        dictionary=None,
        counts=None,
        Nxx=None,
        verbose=True,
        copy=True
    ):
        '''
        Keeps track of token cooccurrences.  No arguments are needed to create
        an empty instance.  To create an instance that already contains counts
        supply (1) a dictionary and (2) either a 2-D numpy array of 
        CoocStats or a collections.Counter instance.

        dictionary (data_preparation.dictionary.Dictionary):
            A two-way mapping between tokens and ids.  Can be None if starting
            an empty CoocStats instance, otherwise required.

        counts (collections.Counter):
            Used to accumulates counts as a corpus is read.  Leave blank if
            starting an empty CoocStats.  Otherwise, it should have
            pairs (tuples) of token_ids as keys, and number of coocccurrences 
            as values.

        Nxx (numpy.array or scipy.sparse.csr matrix):
            Represents counts in a sparse format that is efficient for
            calculations, but not so convinient for accumulating counts as a
            corpus is read.
        '''

        self.validate_args(dictionary, counts, Nxx)
        self._dictionary = dictionary or h.dictionary.Dictionary()

        if counts is not None and copy:
            self._counts = Counter(counts)
        else:
            self._counts = counts

        if Nxx is not None and copy:
            self._Nxx = sparse.csr_matrix(Nxx)
        else:
            self._Nxx = Nxx
        if Nxx is not None:
            self._Nx = np.array(np.sum(self._Nxx, axis=1)).reshape(-1)
        else:
            self._Nx = None

        # If no prior cooccurrence stats are given, start as empty.
        if counts is None and Nxx is None:
            self._counts = Counter()

        self.verbose = verbose
        self._denseNxx = None


    def validate_args(self, dictionary, counts, Nxx):

        if counts is not None and Nxx is not None:
            raise ValueError(
                'Non-empty CoocStats objects should be '
                'instantiated by providing either a cooccurrence matrix (Nxx) '
                'or a cooccurrence Counter (counts)---not both.'
            )

        if counts is not None or Nxx is not None:
            if dictionary is None:
                raise ValueError(
                    'A dictionary must be provided to create a non-empty '
                    'CoocStats object.'
                )

    @property
    def counts(self):
        if self._counts is None:
            self.decompile()
        return self._counts


    @property
    def dictionary(self):
        # Use the existence of _Nxx as an indicator of whether we need to
        # compile before returning the dictionary.
        if self._Nxx is None:
            self.compile()
        return self._dictionary


    @property
    def Nxx(self):
        if self._Nxx is None:
            self.compile()
        return self._Nxx


    @property
    def Nx(self):
        if self._Nx is None:
            self.compile()
        return self._Nx

    @property
    def N(self):
        if self._N is None:
            self.compile()
        return self._N


    @property
    def denseNxx(self):
        if self._denseNxx is None:
            self._denseNxx = self.Nxx.toarray()
        return self._denseNxx


    def add(self, token1, token2):
        id1 = self._dictionary.add_token(token1)
        id2 = self._dictionary.add_token(token2)
        self.counts[id1, id2] += 1

        # We are no longer synced with Nxx, Nx, and denseNxx.
        self._Nxx = None
        self._Nx = None
        self._denseNxx = None


    def decompile(self, force=False):
        if self._counts is not None:
            raise ValueError(
                'Cannot decompile CooccurrenceStats: already decompiled')
        if self.verbose:
            print('Decompiling cooccurrence stats...')

        Nxx_coo = self.Nxx.tocoo()
        self._counts = Counter()
        for i,j,v in zip(Nxx_coo.row, Nxx_coo.col, Nxx_coo.data):
            self._counts[i,j] = v


    def compile(self):
        if self._Nxx is not None:
            raise ValueError(
                'Cannot compile CoocStats: already compiled.')
        if self.verbose:
            print('Compiling cooccurrence stats...')

        vocab_size = len(self._dictionary)
        self._Nxx = dict_to_sparse(self.counts, (vocab_size,vocab_size))
        self._Nx = np.array(np.sum(self._Nxx, axis=1)).reshape(-1)
        self._N = np.sum(self._Nx)
        self.sort(True)

        self.synced = True
        self._denseNxx = None


    def sort(self, force=False):
        top_indices = np.argsort(-self.Nx.reshape((-1,)))
        self._Nxx = self.Nxx[top_indices][:,top_indices]
        self._Nx = self.Nx[top_indices]
        self._dictionary = h.dictionary.Dictionary([
            self._dictionary.tokens[i] for i in top_indices])
        index_map = {
            old_idx: new_idx 
            for new_idx, old_idx in enumerate(top_indices)
        }
        new_counts = Counter()
        for (i,j), count in self.counts.items():
            new_counts[index_map[i], index_map[j]] = count
        self._counts = new_counts


    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        sparse.save_npz(os.path.join(path, 'Nxx.npz'), self.Nxx)
        #np.savez(os.path.join(path, 'Nx.npz'), Nx)
        self.dictionary.save(os.path.join(path, 'dictionary'))


    def density(self, threshold_count=0):
        num_cells = np.prod(self.Nxx.shape)
        num_filled = (
            self.Nxx.getnnz() if threshold_count == 0 
            else np.sum(self.Nxx>threshold_count)
        )
        return float(num_filled) / num_cells


    def truncate(self, k):
        self._Nxx = self.Nxx[:k][:,:k]
        self._Nx = self.Nx[:k]
        dictionary = h.dictionary.Dictionary(self.dictionary.tokens[:k])


    @staticmethod
    def load(path, verbose=True):
        return CoocStats(
            dictionary=h.dictionary.Dictionary.load(
                os.path.join(path, 'dictionary')),
            Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')),
            verbose=verbose
        )




def dict_to_sparse(d, shape=None):
    I, J, V = [], [], []
    for (idx1, idx2), value in d.items():
        I.append(idx1)
        J.append(idx2)
        V.append(value)

    return sparse.coo_matrix((V,(I,J)), shape).tocsr()


