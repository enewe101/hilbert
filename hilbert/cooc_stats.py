import os
from copy import deepcopy
from collections import Counter

try:
    import numpy as np
    from scipy import sparse, stats
    import torch
except ImportError:
    np = None
    sparse = None
    stats = None
    torch = None

import hilbert as h


def read_stats(path):
    return CoocStats.load(path)


class CoocStats(object):
    """Represents cooccurrence statistics."""

    def __init__(
        self,
        dictionary=None,
        counts=None,
        Nxx=None,
        device=None,
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
        self._Nxt = None
        self._N = None
        if Nxx is not None:
            self._Nxx = sparse.csr_matrix(Nxx)
            self._Nx = np.asarray(np.sum(self._Nxx, axis=1))
            self._Nxt = np.asarray(np.sum(self._Nxx, axis=0))
            self._N = np.sum(self._Nx)

        loaded_Nxx = None
        loaded_Nx = None
        loaded_Nxt = None
        loaded_N = None

        # If no prior cooccurrence stats are given, start as empty.
        if counts is None and Nxx is None:
            self._counts = Counter()

        self.device = device
        self.verbose = verbose


    def __getitem__(self, shard):
        return self.load_shard(shard)

        
    def load_shard(self, shard=None, device=None):

        if shard is None:
            shard = h.shards.whole

        device = device or self.device or h.CONSTANTS.MATRIX_DEVICE

        loaded_Nxx = h.utils.load_shard(self.Nxx, shard, device=device)
        loaded_Nx = h.utils.load_shard(self.Nx, shard[0], device=device)
        loaded_Nxt = h.utils.load_shard(
            self.Nxt, (slice(None), shard[1]), device=device)
        loaded_N = h.utils.load_shard(self.N, device=device)

        return loaded_Nxx, loaded_Nx, loaded_Nxt, loaded_N


    def __copy__(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        result = CoocStats(
            dictionary=deepcopy(self.dictionary),
            counts=Counter(self.counts), 
            verbose=self.verbose
        )
        memo[id(self)] = result
        return result


    def __iter__(self):
        """
        Returns Nxx, Nx, Nxt, N, which means that the CoocStats instance can
        easily unpack into cooccurrence counts, unigram counts, and the total
        number of tokens.  Useful for functions expecting such a stats triplet,
        and for getting raw access to the data.
        """
        return iter(self[h.shards.whole])

    
    #def __radd__(self, other):
    #    """
    #    Create a new CoocStats that has counts from both operands.
    #    """
    #    # Just delegate to add.
    #    return self.__add__(other)


    def __add__(self, other):
        """
        Create a new CoocStats that has counts from both operands.
        """
        if not isinstance(other, CoocStats):
            return NotImplemented

        result = deepcopy(self)
        result.__iadd__(other)
        return result


    def __iadd__(self, other):
        """
        Add counts from `other` to `self`, in place.
        """

        # For better performance, this is implemented so as to avoid 
        # decompiling the CoocStats instances.  Instead, we get ijv triples
        # for non-zero elements in the other CoocStats, and then tack them
        # on to ijv triples for self.  Conversion between the CSR and COO 
        # sparse matrix formats is much faster than conversion between dict
        # and sparse matrix format.

        if not isinstance(other, CoocStats):
            return NotImplemented

        # Avoid unnecessarily decompiling other's Nxx into counts: if it does
        # not have its counts already in memory, use the coo-format of it's 
        # Nxx array to more quickly provide the same info.
        if other._counts is None:
            other_Nxx_coo = other.Nxx.tocoo()
            I, J, V = other_Nxx_coo.row, other_Nxx_coo.col, other_Nxx_coo.data
        else:
            I, J, V = dict_to_IJV(other.counts)

        self_Nxx_coo = self.Nxx.tocoo()

        self_row = list(self_Nxx_coo.row) 
        self_col = list(self_Nxx_coo.col) 
        self_data = list(self_Nxx_coo.data)

        add_to_row = [
            self.dictionary.add_token(other.dictionary.get_token(i))
            for i in I
        ]
        self_row += add_to_row

        add_to_col = [
            self.dictionary.add_token(other.dictionary.get_token(j))
            for j in J
        ]
        self_col += add_to_col

        self_data += list(V) 

        vocab_size = len(self._dictionary)
        self._Nxx = sparse.coo_matrix(
            (self_data, (self_row, self_col)),
            (vocab_size,vocab_size)
        ).tocsr()
        self._Nx = np.array(np.sum(self._Nxx, axis=1))
        self._Nxt = np.array(np.sum(self._Nxx, axis=0))
        self._N = np.sum(self._Nx)
        self.sort(True)

        return self


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
        # So that it always returns a consistent result, we want the 
        # dictionary to undergo sorting, which happens during compilation.
        # Testing whether self._Nxx is None here is a way to test if the
        # correctness of the dictionary's ordering has gone stale.
        if self._Nxx is None:
            self.compile()
        return self._dictionary


    @property
    def Nxx(self):
        if self._Nxx is None:
            self.compile()
        return self._Nxx


    @property
    #TODO: deprecate this
    def denseNxx(self):
        print('WARNING: denseNxx is deprecated.  Use load_shard or indexing.')
        if self._denseNxx is None:
            self._denseNxx = self.Nxx.toarray()
        return self._denseNxx


    @property
    def Nx(self):
        if self._Nx is None:
            self.compile()
        return self._Nx


    @property
    def Nxt(self):
        if self._Nxt is None:
            self.compile()
        return self._Nxt


    @property
    def N(self):
        if self._N is None:
            self.compile()
        return self._N


    def add(self, token1, token2, count=1):
        id1 = self._dictionary.add_token(token1)
        id2 = self._dictionary.add_token(token2)
        self.counts[id1, id2] += count

        # Nxx, Nx, Nxt, and N are all stale, so set them to None.
        self._Nxx = None
        self._Nx = None
        self._Nxt = None
        self._N = None
        self._denseNxx = None


    def decompile(self, force=False):
        """
        Convert the cooccurrence data stored in `Nxx` into a counter.
        """
        if self._counts is not None:
            raise ValueError(
                'Cannot decompile CooccurrenceStats: already decompiled')
        if self.verbose:
            print('Decompiling cooccurrence stats...')

        self._counts = Counter()
        Nxx_coo = self._Nxx.tocoo()
        for i,j,v in zip(Nxx_coo.row, Nxx_coo.col, Nxx_coo.data):
            self._counts[i,j] = v
        self._Nxx, self._Nx, self._Nxt, self_N = None,None,None,None


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
            self.counts, (vocab_size,vocab_size))
        self._Nx = np.array(np.sum(self._Nxx, axis=1))
        self._Nxt = np.array(np.sum(self._Nxx, axis=0))
        self._N = np.sum(self._Nx)
        self.sort(True)


    def sort(self, force=False):
        """
        Re-assign token indices providing lower indices to more common words.
        This affects the dictionary mapping, the IDs used in `counts`, and 
        The indexing of Nxx and Nx.  `counts` will simply be dropped, since it
        can be calculated lazily later if needed.
        """
        top_indices = np.argsort(-self.Nx.reshape(-1))
        self._Nxx = self.Nxx.tocsr()[top_indices][:,top_indices].tocsr()
        self._Nx = self.Nx[top_indices]
        self._Nxt = self.Nxt[:,top_indices]
        self._dictionary = h.dictionary.Dictionary([
            self._dictionary.tokens[i] for i in top_indices])
        index_map = {
            old_idx: new_idx 
            for new_idx, old_idx in enumerate(top_indices)
        }
        self._counts = None


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
                os.path.join(path, 'Nxx.npz'), self.Nxx)
        else:
            np.savez(os.path.join(path, 'Nxx.npz'), self.Nxx.todense())
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
        self._Nx = np.array(np.sum(self._Nxx, axis=1))
        self._Nxt = np.array(np.sum(self._Nxx, axis=0))
        self._N = np.sum(self._Nx)
        dictionary = h.dictionary.Dictionary(self.dictionary.tokens[:k])


    @staticmethod
    def load(path, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        dictionary = h.dictionary.Dictionary.load(
            os.path.join(path, 'dictionary'))
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tocsr()
        return CoocStats(dictionary=dictionary, Nxx=Nxx, verbose=verbose)



def dict_to_IJV(counts):
    """
    Given a dict-like `counts` whose keys are 2-tuples of token indices,
    return a the parallel arrays I, J, and V, where I contains all first-token
    indices, J contains all second-token indices, and V contains all values.
    """
    I, J, V = [], [], []
    for (idx1, idx2), value in counts.items():
        I.append(idx1)
        J.append(idx2)
        V.append(value)
    return I, J, V


def dict_to_sparse(counts, shape=None):
    """
    Given a dict-like `counts` whose keys are 2-tuples of token indices,
    return a scipy.sparse.coo.coo_matrix containing the same values.
    """
    I, J, V = dict_to_IJV(counts)

    return sparse.coo_matrix((V,(I,J)), shape).tocsr()


### COOC_STATS ALTERATTIONS ###

def w2v_undersample(cooc_stats, t, verbose=True):
    """
    Given a h.cooc_stats.CoocStats instance returns an altered version (leaves
    original unchanged) to reflect undersampling of common words as done in
    word2vec.

    The Nxx matrix is altered by multiplying by p_xx, which is equal to
    the product of the probability of having *not* rejected the focal word and 
    not rejected the context word.

    The Nx matrix contains new sums, based on the undersampling expectation
    values.

    The Nxt matrix, on the other hand, is left unchanged, as is N, which
    preserves a representation of the unigram distribution.  In the 
    word2vec's normal choices for M and Delta, this naturally provides
    for the fact that negative samples draw from the unigram distribution,
    according to the natural appearance of Nxt and N.
    """
    if t is None:
        return cooc_stats

    if verbose:
        print('undersample_method\tsample')
        print('t_undersample\t{}'.format(t))

    new_cooc_stats = h.cooc_stats.CoocStats(dictionary=cooc_stats.dictionary)

    p_xx = calc_w2v_undersample_survival_probability(cooc_stats, t)

    # Take the original number of observations as a number of trials, 
    # and the survaval probability in a binomial distribution for undersampling.
    I, J = cooc_stats.Nxx.nonzero()
    keep_Nxx_data = stats.binom.rvs(
        cooc_stats.Nxx[I,J].astype(np.int32), p_xx[I,J])


    # Modify Nxx, and Nx to reflect undersampling; leave Nxt, and N unchanged.
    # unigram distribution unchanged).
    new_cooc_stats._Nxx = sparse.coo_matrix(
        (keep_Nxx_data, (I,J)), cooc_stats.Nxx.shape).tocsr()
    new_cooc_stats._Nx = np.asarray(np.sum(new_cooc_stats._Nxx, axis=1))
    new_cooc_stats._Nxt = cooc_stats.Nxt.copy()
    new_cooc_stats._N = float(cooc_stats.N)

    return new_cooc_stats



def expectation_w2v_undersample(cooc_stats, t, verbose=True):
    """
    Given a h.cooc_stats.CoocStats instance, returns an altered version (leaves
    original unchanged) by reducing counts for common words, simulating the
    rejection of common words in word2vec.  The counts are changed into the
    expectation of counts under the undersampled distribution.

    The Nxx matrix is altered by multiplying by p_xx, which is equal to
    the product of the probability of having *not* rejected the focal word and 
    not rejected the context word.

    The Nx matrix contains new sums, based on the undersampling expectation
    values.

    The Nxt matrix, on the other hand, is left unchanged, as is N, which
    preserves a representation of the unigram distribution.  In the 
    word2vec's normal choices for M and Delta, this naturally provides
    for the fact that negative samples draw from the unigram distribution,
    according to the natural appearance of Nxt and N.

    (A separate but related modification that word2vec can make is to 
    distort the unigram distribution... see h.cooc_stats.distort_unigram.)
    """

    if t is None:
        return cooc_stats

    if verbose:
        print('undersample_method\texpectation')
        print('t_undersample\t{}'.format(t))

    # We will copy count statistics to a new cooc_stats object, but with
    # alterations.
    new_cooc_stats = h.cooc_stats.CoocStats(dictionary=cooc_stats.dictionary)

    p_xx = calc_w2v_undersample_survival_probability(cooc_stats, t)
    new_cooc_stats._Nxx = cooc_stats.Nxx.multiply(p_xx)
    new_cooc_stats._Nx = np.asarray(np.sum(new_cooc_stats.Nxx, axis=1))
    new_cooc_stats._Nxt = cooc_stats.Nxt.copy()
    new_cooc_stats._N = float(cooc_stats.N)

    return new_cooc_stats



def calc_w2v_undersample_survival_probability(cooc_stats, t):
    # Probability that token of a given type is kept.
    p_x = np.sqrt(t * cooc_stats.N / cooc_stats.Nx)
    p_x[p_x > 1] = 1

    # Calculate the elements of p_x * p_x.T that correspond to nonzero elements
    # of Nxx.  That way we keep it sparse.
    p_x = sparse.csr_matrix(p_x)
    I, J = cooc_stats.Nxx.nonzero()
    nonzero_mask = sparse.coo_matrix(
        (np.ones(I.shape),(I,J)),cooc_stats.Nxx.shape).tocsr()
    p_xx = nonzero_mask.multiply(p_x).multiply(p_x.T)
    return p_xx



def smooth_unigram(cooc_stats, alpha=None, verbose=True):

    if verbose:
        print('smooth_unigram_alpha\t{}'.format(alpha))

    if alpha is None:
        return cooc_stats

    new_cooc_stats = h.cooc_stats.CoocStats(dictionary=cooc_stats.dictionary)

    # We consider Nxt and N to be the holders of unigram frequency info,
    new_cooc_stats._Nxt = cooc_stats.Nxt**alpha
    new_cooc_stats._N = np.sum(new_cooc_stats._Nxt)
    # Nxx, and Nx remain unchanged
    new_cooc_stats._Nxx = cooc_stats.Nxx.copy()
    new_cooc_stats._Nx = cooc_stats.Nx.copy()

    return new_cooc_stats




