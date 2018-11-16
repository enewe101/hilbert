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


#TODO: ensure the dtype is float32 not float64
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
            self.Nxx = sparse.lil_matrix(Nxx)
            self.Nx = np.asarray(np.sum(self.Nxx, axis=1))
            self.Nxt = np.asarray(np.sum(self.Nxx, axis=0))
            self.N = np.sum(self.Nx)
        else:
            self.Nxx = sparse.lil_matrix((self.vocab, self.vocab))
            self.Nx = np.zeros((self.vocab, 1))
            self.Nxt = np.zeros((1, self.vocab))
            self.N = 0

        self.device = device
        self.verbose = verbose


    # TODO: test
    def apply_w2v_undersampling(self, t):
        """
        Simulate undersampling of common words, like how is done in word2vec.
        However, when applied here (as opposed to within the corpus sampler,
        we are taking expectation values cooccurrence statistics under 
        undersampling, and undersampling is applied in the "clean" way which
        does not alter the effective size of the sample window.
        """

        # For each pair of words, calculate the probability that a given 
        # cooccurrence would still be observed given undersampling.
        freqs = np.array(self.unigram.Nx) / self.unigram.N
        drop_probs = np.clip((freqs - t)/freqs - np.sqrt(t/freqs), 0, 1)
        keep_probs = 1 - drop_probs
        p_i = sparse.lil_matrix(keep_probs.reshape((-1,1)))
        p_it = sparse.lil_matrix(keep_probs.reshape((1, -1)))

        # Calculate the expectation cooccurrence counts given undersampling.
        self.Nxx = self.Nxx.multiply(p_i).multiply(p_it).tolil()
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)


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
            self.Nxx, shard, device=device)
        loaded_Nx = h.utils.load_shard(
            self.Nx, shard[0], device=device)
        loaded_Nxt = h.utils.load_shard(
            self.Nxt, (slice(None), shard[1]), device=device)
        loaded_N = h.utils.load_shard(self.N, device=device)

        return loaded_Nxx, loaded_Nx, loaded_Nxt, loaded_N


    def __copy__(self):
        return deepcopy(self)


    def __len__(self):
        return self.vocab


    def __deepcopy__(self, memo):
        result = Bigram(
            unigram=deepcopy(self.unigram), Nxx=self.Nxx, verbose=self.verbose)
        memo[id(self)] = result
        return result


    def __iter__(self):
        """
        Returns Nxx, Nx, Nxt, N, which means that the Bigram instance can
        easily unpack into cooccurrence counts, unigram counts, and the total
        number of tokens.  Useful for functions expecting such a stats tuple,
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
        token_order = list(other.unigram.dictionary.tokens)
        other_vocab = len(token_order)
        self_vocab = self.vocab
        remaining_tokens = set(self.unigram.dictionary.tokens)-set(token_order)
        token_order += remaining_tokens

        # Use a temporary dictionary based on current vocabulary to provide
        # the ordering to be used
        ordering_dict = deepcopy(self.dictionary)
        idx_order = [ordering_dict.add_token(token) for token in token_order]

        # Copy self's counts into a large enough matrix
        new_Nxx = sparse.lil_matrix((len(idx_order), len(idx_order)))
        new_Nxx[:self_vocab,:self_vocab] += self.Nxx
        self.Nxx = new_Nxx

        # Adopt the shared ordering and add others counts
        self.Nxx = self.Nxx[idx_order][:,idx_order]
        self.Nxx[:other_vocab,:other_vocab] += other.Nxx

        # Add unigrams, and adopt the shared ordering there too.
        self.unigram += other.unigram
        self.unigram.sort_by_tokens(token_order)

        # Compile tallies.
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)

        return self


    @property
    def sorted(self):
        # Sorted order is determined by the unigram frequencies.
        return self.unigram.sorted


    @property
    def vocab(self):
        return len(self.unigram)


    @property
    def dictionary(self):
        return self.unigram.dictionary


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
        self.Nxx[id1, id2] += count
        self.Nx[id1][0] += count
        self.Nxt[0][id2] += count
        self.N += count


    def count(self, token1, token2):
        id1 = self.dictionary.get_id(token1)
        id2 = self.dictionary.get_id(token2)
        return self.Nxx[id1, id2]


    def sort(self, force=False):
        """Adopt decreasing order of unigram frequency."""
        top_indices = self.unigram.sort()
        self.Nxx = self.Nxx.tocsr()[top_indices][:,top_indices].tocsr()
        self.Nx = self.Nx[top_indices]
        self.Nxt = self.Nxt[:,top_indices]


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
        if not self.sorted:
            self.sort()

        self.Nxx = self.Nxx[:k][:,:k]
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)
        self.unigram.truncate(k)


    def save(self, path, save_unigram=True):
        """
        Save the cooccurrence data to disk.  A new directory will be created
        at `path`, and two files will be created within it to store the 
        token-ID mapping and the cooccurrence data.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        sparse.save_npz(
            os.path.join(path, 'Nxx.npz'), self.Nxx.tocsr())
        if save_unigram:
            self.unigram.save(path)


    @staticmethod
    def load(path, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        unigram = h.unigram.Unigram.load(path)
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
        return Bigram(unigram, Nxx=Nxx, verbose=verbose)


    @staticmethod
    def load_unigram(path, verbose=True):
        unigram = h.unigram.Unigram.load(path)
        return Bigram(unigram, verbose=verbose)

