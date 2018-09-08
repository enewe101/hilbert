import os
from collections import Counter

import numpy as np
from scipy import sparse

import data_preparation as dp
import hilbert as h


def load_stats(path):
    return CoocStats.load(path)


class CoocStats(object):
    """Represents cooccurrence statistics."""

    def __init__(
        self,
        dictionary=None,
        counts=None,
        Nxx=None,
        verbose=True
    ):
        '''
        `dictionary` -- A hilbert.dictionary.Dictionary instance mapping tokens
            from/to integer IDs; 
        `counts` -- A collections.Counter instance with token-IDs-2-tuple as
            keys and number of cooccurrences as values; 
        `Nxx` -- A 2D array-like instance (e.g. numpy.ndarray, scipy.sparse.csr
            matrix, or a list of lists), in which the (i,j)th element contains
            the number of cooccurrences for words having IDs i and j.

        Provide `counts` or `Nxx` or neither. Not both!

        CoocStats Keeps track of token cooccurrences, and saves/loads from
        disk.  Provide no arguments to create an empty instance, e.g. to
        accumulate cooccurrence while reading through a corpus.

        Cooccurence statistics are represented in two ways:

            (1) as a collections.Counter, whose keys are pairs of token
                indices; and

            (2) as a 2D numpy.ndarray, whose (i,j)th element contains the 
                number of cooccurrences of the tokens with indices i and j.

        This dual representation supports extending the vocabulary while 
        accumulating statisitcs (using self.add method), and makes cooccurrence
        matrix available as a numpy array for fast calculations.

        Synchronization between these two representations is done
        lazily when you access or call methods that rely on one representation.
        '''

        self.validate_args(dictionary, counts, Nxx)
        self._dictionary = dictionary or h.dictionary.Dictionary()

        self._counts = counts
        if counts is not None:
            self._counts = Counter(counts)

        self._Nxx = Nxx
        self._Nx = None
        self._N = None
        if Nxx is not None:
            if sparse.issparse(Nxx):
                self._Nxx = Nxx.toarray()
            else:
                self._Nxx = np.array(Nxx)

            self._Nx = np.sum(self._Nxx, axis=1).reshape(-1,1)
            self._N = np.sum(self._Nx)

        # If no prior cooccurrence stats are given, start as empty.
        if counts is None and Nxx is None:
            self._counts = Counter()

        self.verbose = verbose


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


    def add(self, token1, token2):
        id1 = self._dictionary.add_token(token1)
        id2 = self._dictionary.add_token(token2)
        self.counts[id1, id2] += 1

        # Nxx, Nx, and N are all stale, so set them to None.
        self._Nxx = None
        self._Nx = None
        self._N = None


    def decompile(self, force=False):
        """
        Convert the cooccurrence data stored in `Nxx` into a counter.
        """
        if self._counts is not None:
            raise ValueError(
                'Cannot decompile CooccurrenceStats: already decompiled')
        if self.verbose:
            print('Decompiling cooccurrence stats...')

        Nxx_coo = sparse.coo_matrix(self.Nxx)
        self._counts = Counter()
        for i,j,v in zip(Nxx_coo.row, Nxx_coo.col, Nxx_coo.data):
            self._counts[i,j] = v


    def compile(self):
        """
        Convert the cooccurrence data stored in `counts` into a numpy array.
        """
        if self._Nxx is not None:
            raise ValueError(
                'Cannot compile CoocStats: already compiled.')
        if self.verbose:
            print('Compiling cooccurrence stats...')

        vocab_size = len(self._dictionary)
        self._Nxx = dict_to_sparse(
            self.counts, (vocab_size,vocab_size)).toarray()
        self._Nx = np.array(np.sum(self._Nxx, axis=1)).reshape(-1,1)
        self._N = np.sum(self._Nx)
        self.sort(True)


    def sort(self, force=False):
        """
        Re-assign token indices providing lower indices to more common words.
        This affects the dictionary mapping, the IDs used in `counts`, and 
        The indexing of Nxx and Nx.
        """
        top_indices = np.argsort(-self.Nx.reshape(-1))
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


    def save(self, path, save_as_sparse=True):
        """
        Save the cooccurrence data to disk.  A new directory will be created
        at `path`, and two files will be created within it to store the 
        token-ID mapping and the cooccurrence data.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if save_as_sparse:
            sparse.save_npz(
                os.path.join(path, 'Nxx.npz'), sparse.coo_matrix(self.Nxx))
        else:
            np.savez(os.path.join(path, 'Nxx.npz'), self.Nxx)
        #np.savez(os.path.join(path, 'Nx.npz'), Nx)
        self.dictionary.save(os.path.join(path, 'dictionary'))


    def density(self, threshold_count=0):
        """
        Return the number of cells whose value is greater than
        `threshold_count`.
        """
        num_cells = np.prod(self.Nxx.shape)
        num_filled = np.sum(self.Nxx>threshold_count)
        return float(num_filled) / num_cells


    def truncate(self, k):
        """Drop all but the `k` most common words."""
        self._Nxx = self.Nxx[:k][:,:k]
        self._Nx = self.Nx[:k]
        dictionary = h.dictionary.Dictionary(self.dictionary.tokens[:k])


    @staticmethod
    def load(path, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        dictionary = h.dictionary.Dictionary.load(
            os.path.join(path, 'dictionary'))
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz'))
        return CoocStats(dictionary=dictionary, Nxx=Nxx, verbose=verbose)




def dict_to_sparse(counts, shape=None):
    """
    Given a dict-like `counts` whose keys are 2-tuples of token indices,
    return a scipy.sparse.coo.coo_matrix containing the same values.
    """
    I, J, V = [], [], []
    for (idx1, idx2), value in counts.items():
        I.append(idx1)
        J.append(idx2)
        V.append(value)

    return sparse.coo_matrix((V,(I,J)), shape)


