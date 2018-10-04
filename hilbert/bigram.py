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
    return Bigram.load(path)


class Bigram(object):
    """Represents cooccurrence statistics."""

    def __init__(
        self,
        unigram,
        Nxx=None,
        device=None,
        verbose=True
    ):
        '''
        `unigram` -- A hilbert.unigram.Unigram instance containing unigram
            frequences and a dictionary mapping from/to integer IDs; 
        `Nxx` -- A 2D array-like instance (e.g. numpy.ndarray, scipy.sparse.csr
            matrix, or a list of lists), in which the (i,j)th element contains
            the number of cooccurrences for words having IDs i and j.

        Bigram Keeps track of token cooccurrences, and saves/loads from
        disk.  Provide no Nxx to create an empty instance, e.g. to
        accumulate cooccurrence while reading through a corpus.

        Cooccurence statistics are represented as a scipy.sparse.lil_matrix.
        '''

        self.validate_args(unigram, Nxx)
        self.unigram = unigram

        if Nxx is not None:
            self._Nxx = sparse.lil_matrix(Nxx)
            self._Nx = np.asarray(np.sum(self._Nxx, axis=1))
            self._Nxt = np.asarray(np.sum(self._Nxx, axis=0))
            self._N = np.sum(self._Nx)
        else:
            self._Nxx = sparse.lil_matrix(self.vocab, self.vocab))
            self._Nx = np.zeros((self.vocab, 1))
            self._Nxt = np.zeros((1, self.vocab))
            self._N = 0

        self.device = device
        self.verbose = verbose


    def validate_args(self, unigram, Nxx):

        if Nxx is not None:
            if Nxx.shape[0] != len(unigram) or Nxx.shape[1] != len(unigram):
                raise ValueError(
                    'Nxx length and width equal should unigram length. '
                    'Got %d x %d (unigram length was %d).' 
                    % (Nxx.shape[0], Nxx.shape[1], len(unigram))
                )


    def __getitem__(self, shard):
        return self.load_shard(shard)

        
    def load_shard(self, shard=None, device=None):

        if shard is None:
            shard = h.shards.whole

        device = device or self.device or h.CONSTANTS.MATRIX_DEVICE

        loaded_Nxx = h.utils.load_shard(
            self.Nxx, shard, from_sparse=True, device=device)
        loaded_Nx = h.utils.load_shard(
            self.Nx, shard[0], device=device)
        loaded_Nxt = h.utils.load_shard(
            self.Nxt, (slice(None), shard[1]), device=device)
        loaded_N = h.utils.load_shard(self.N, device=device)

        return loaded_Nxx, loaded_Nx, loaded_Nxt, loaded_N


    def __copy__(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        result = Bigram(
            dictionary=deepcopy(self.dictionary),
            counts=Counter(self.counts), 
            verbose=self.verbose
        )
        memo[id(self)] = result
        return result


    def __iter__(self):
        """
        Returns Nxx, Nx, Nxt, N, which means that the Bigram instance can
        easily unpack into cooccurrence counts, unigram counts, and the total
        number of tokens.  Useful for functions expecting such a stats triplet,
        and for getting raw access to the data.
        """
        return iter(self[h.shards.whole])

    
    def __add__(self, other):
        """
        Create a new Bigram that has counts from both operands.
        """
        if not isinstance(other, Bigram):
            return NotImplemented

        result = deepcopy(self)
        result.__iadd__(other)
        return result


    def __iadd__(self, other):
        """
        Add counts from `other` to `self`, in place.
        """

        if not isinstance(other, Bigram):
            return NotImplemented

        # Find an shared ordering that matches other's ordering, with any words
        # unique to self's vocab placed at the end.
        token_order = other.unigram.dictionary.tokens
        other_vocab = len(token_order)
        remaining_tokens = set(self.unigram.dictionary.tokens)-set(token_order)
        token_order += remaining_tokens
        idx_order = [self.dictionary.add_token(token) for token in token_order]

        # Copy self's counts into a large enough matrix
        new_Nxx = sparse.lil_matrix((len(idx_order), len(idx_order)))
        new_Nxx += self._Nxx
        self._Nxx = new_Nxx

        # Reorder self, ensure that unigram adopts the same ordering.
        self._Nxx = self._Nxx[idx_order][:,idx_order]
        self.unigram.sort_by_tokens(token_order)

        # Add other's counts to self
        self._Nxx[:other_vocab,:other_vocab] += other.Nxx
        self.unigram += other.unigram

        # Compile tallies.
        self._Nx = np.array(np.sum(self._Nxx, axis=1))
        self._Nxt = np.array(np.sum(self._Nxx, axis=0))
        self._N = np.sum(self._Nx)

        return self


    @property
    def vocab(self):
        return len(self.unigram)


    @property
    def dictionary(self):
        return self.unigram.dictionary


    def add(self, token1, token2, count=1):

        # Get token idx's.
        id1 = self._dictionary.add_token(token1)
        id2 = self._dictionary.add_token(token2)

        # Add counts.
        self._Nxx[id1, id2] += count
        self._Nx[id1] += count
        self._Nxt[id2] += count
        self._N += count


    def sort(self, force=False):
        """
        Re-assign token indices providing lower indices to more common words.
        The unigram and its dictionary are forced to adopt the same ordering.
        """
        top_indices = np.argsort(-self.Nx.reshape(-1))
        self._Nxx = self.Nxx.tocsr()[top_indices][:,top_indices].tocsr()
        self._Nx = self.Nx[top_indices]
        self._Nxt = self.Nxt[:,top_indices]
        self.unigram.sort_by_idxs(top_indices)


    def save(self, path, save_unigram=True):
        """
        Save the cooccurrence data to disk.  A new directory will be created
        at `path`, and two files will be created within it to store the 
        token-ID mapping and the cooccurrence data.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        sparse.save_npz(
            os.path.join(path, 'Nxx.npz'), self.Nxx)
        if save_unigram:
            self.unigram.save(path)


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
        self.unigram.truncate(k)


    @staticmethod
    def load(path, verbose=None):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        verbose = verbose if verbose is not None else self.verbose
        unigram = h.unigram.Unigram.load(path)
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tocsr()
        return Bigram(unigram, Nxx=Nxx, verbose=verbose)



### COOC_STATS ALTERATTIONS ###
def w2v_undersample(cooc_stats, t, verbose=True):
    """
    Given a h.cooc_stats.Bigram instance returns an altered version (leaves
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

    new_cooc_stats = h.cooc_stats.Bigram(dictionary=cooc_stats.dictionary)

    p_xx = calc_w2v_undersample_survival_probability(cooc_stats, t)

    # Take the original number of observations as a number of trials, 
    # and the survaval probability in a binomial distribution for undersampling.
    I, J = cooc_stats.Nxx.nonzero()
    keep_Nxx_data = stats.binom.rvs(cooc_stats.Nxx[I,J], p_xx[I,J])


    # Modify Nxx, and Nx to reflect undersampling; leave Nxt, and N unchanged.
    # unigram distribution unchanged).
    new_cooc_stats._Nxx = sparse.coo_matrix(
        (keep_Nxx_data, (I,J)), cooc_stats.Nxx.shape).tocsr()
    new_cooc_stats._Nx = np.asarray(np.sum(new_cooc_stats._Nxx, axis=1))
    new_cooc_stats._Nxt = cooc_stats.Nxt.copy()
    new_cooc_stats._N = cooc_stats.N.copy()

    return new_cooc_stats



def expectation_w2v_undersample(cooc_stats, t, verbose=True):
    """
    Given a h.cooc_stats.Bigram instance, returns an altered version (leaves
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
    new_cooc_stats = h.cooc_stats.Bigram(dictionary=cooc_stats.dictionary)

    p_xx = calc_w2v_undersample_survival_probability(cooc_stats, t)
    new_cooc_stats._Nxx = cooc_stats.Nxx.multiply(p_xx)
    new_cooc_stats._Nx = np.asarray(np.sum(new_cooc_stats.Nxx, axis=1))
    new_cooc_stats._Nxt = cooc_stats.Nxt.copy()
    new_cooc_stats._N = cooc_stats.N.copy()

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

    new_cooc_stats = h.cooc_stats.Bigram(dictionary=cooc_stats.dictionary)

    # We consider Nxt and N to be the holders of unigram frequency info,
    new_cooc_stats._Nxt = cooc_stats.Nxt**alpha
    new_cooc_stats._N = np.sum(new_cooc_stats._Nxt)
    # Nxx, and Nx remain unchanged
    new_cooc_stats._Nxx = cooc_stats.Nxx.copy()
    new_cooc_stats._Nx = cooc_stats.Nx.copy()

    return new_cooc_stats





