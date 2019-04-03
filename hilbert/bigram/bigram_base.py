import hilbert as h
import os
import warnings
import numpy as np
import torch
from scipy import sparse

import time

class BigramBase(object):

    def __init__(
        self,
        unigram,
        Nxx,
        device=None,
        marginalize=True,
        verbose=True
    ):
        """
        Keeps track of token cooccurrences, and saves/loads from disk.  
        Providing only unigram data will create an empty instance.

        `unigram` 
            A hilbert.unigram.Unigram instance containing unigram
            frequences and a dictionary mapping from/to integer IDs; 
        `Nxx`
            A 2D array-like instance (e.g. numpy.ndarray, scipy.sparse.csr
            matrix, or a list of lists), in which the (i,j)th element contains
            the number of cooccurrences for words having IDs i and j.

        Cooccurence statistics are represented as a scipy.sparse.lil_matrix.
        """

        if not unigram.sorted:
            raise ValueError(
                'Bigram instances must be built from a sorted Unigram instance')

        # Own some things
        self.device = device
        self.verbose = verbose

        dtype = h.CONSTANTS.DEFAULT_DTYPE
        mem_device = h.CONSTANTS.MEMORY_DEVICE

        # Own unigram statistics.
        self.unigram = unigram
        self.uNx = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=mem_device).view(-1, 1)
        self.uNxt = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=mem_device).view(1, -1)
        self.uN = torch.tensor(
            self.unigram.N, dtype=dtype, device=mem_device)

        # Own cooccurrence statistics and marginalized totals.
        self.Nxx = sparse.lil_matrix(Nxx)

        # Marginalizing counts actually takes long.  This seems to be faster
        # than just calling self.Nxx.sum().
        if marginalize:
            self.Nx = np.zeros((self.Nxx.shape[0],1))
            self.Nxt = np.zeros((1,self.Nxx.shape[1]))
            for i in range(len(self.Nxx.data)):
                # put in the marginal sums!
                self.Nx[i] = sum(self.Nxx.data[i])
                self.Nxt[:,self.Nxx.rows[i]] += self.Nxx.data[i]
            self.N = self.Nx.sum()
        self.validate_shape()
        self.undersampled = False


    def validate_shape(self):
        uNx_match = self.Nxx.shape[0] == self.uNx.shape[0] 
        uNxt_match = self.Nxx.shape[1] == self.uNxt.shape[1]

        if not all([uNx_match, uNxt_match]):
            raise ValueError(
                'Nxx length and width equal should match unigram and '
                'dictionary. Got: \nNxx:\t{}x{}\nuNx:\t{}\nuNxt:\t{}'.format(
                    self.Nxx.shape[0], self.Nxx.shape[1], 
                    self.uNx.shape[0], self.uNxt.shape[1],
                )
            )

        if not self.Nxx.shape[0] == len(self.dictionary):
            raise ValueError(
                "Nxx length should match dictionary. Got: "
                "\nNxx:        {}x{}"
                "\ndictionary  {}.".format(
                    self.Nxx.shape[0], self.Nxx.shape[1], len(self.dictionary)
                )
            )


    @staticmethod
    def load(path, device=None, verbose=True, marginalize=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        unigram = h.unigram.Unigram.load(path, device=device, verbose=verbose)
        Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
        return BigramBase(unigram, Nxx, device=device, verbose=verbose, marginalize=marginalize)


    @property
    def sorted(self):
        # Sorted order is determined by the unigram frequencies.
        warnings.warn(
            '`Bigram.sorted` is deprecated.  Sorted state should be '
            'guaranteed.', DeprecationWarning
        )
        return self.unigram.sorted


    @property
    def dictionary(self):
        return self.unigram.dictionary


    @property
    def shape(self):
        return self.Nxx.shape


    @property
    def vocab(self):
        warnings.warn(
            '`Bigram.vocab` is deprecated.  Use `Bigram.shape`.',
            DeprecationWarning
        )
        return self.Nxx.shape[0]


    def __len__(self):
        return self.Nxx.shape[0]


    def __getitem__(self, shard):
        return self.load_shard(shard)


    def __iter__(self):
        """
        Returns Nxx, Nx, Nxt, N, which means that the Bigram instance can
        easily unpack into cooccurrence counts, unigram counts, and the total
        number of tokens.  Useful for functions expecting such a stats tuple,
        and for getting raw access to the data.
        """
        warnings.warn(
            "Implicit unpacking is deprecated.  Use `Bigram.load_shard()` "
            "instead.",
            DeprecationWarning
        )
        return iter(self[h.shards.whole])


    def density(self, threshold_count=0):
        """
        Return the number of cells whose value is greater than
        `threshold_count`.
        """
        num_cells = np.prod(self.Nxx.shape)
        num_filled = np.sum(self.Nxx>threshold_count)
        return float(num_filled) / num_cells


    def count(self, token1, token2):
        id1 = self.dictionary.get_id(token1)
        id2 = self.dictionary.get_id(token2)
        return self.Nxx[id1, id2]


    def load_shard(self, shard=None, device=None):
        """
        Provides tensors corresponding to the cooccurrence data, marginalized
        data, and total count, specific to the requested shard.

        INPUTS
        `shard`
            A pair of slice objects specifying a particular subset of cells, 
            or None (in which case the entire dataset is loaded).
        """

        if shard is None:
            shard = h.shards.whole

        device = device or self.device

        loaded_Nxx = h.utils.load_shard(
            self.Nxx, shard, device=device)
        loaded_Nx = h.utils.load_shard(
            self.Nx, shard[0], device=device)
        loaded_Nxt = h.utils.load_shard(
            self.Nxt, (slice(None), shard[1]), device=device)
        loaded_N = h.utils.load_shard(self.N, device=device)

        return loaded_Nxx, loaded_Nx, loaded_Nxt, loaded_N


    def load_unigram_shard(self, shard=None, device=None):

        if shard is None:
            shard = h.shards.whole

        device = device or self.device

        loaded_uNx = h.utils.load_shard(self.uNx, shard[0], device=device)
        loaded_uNxt = h.utils.load_shard(
            self.uNxt, (slice(None), shard[1]), device=device)
        loaded_uN = h.utils.load_shard(self.uN, device=device)

        return loaded_uNx, loaded_uNxt, loaded_uN


    def merge(self, other):
        """
        Add counts from `other` to `self`, in place, without altering
        the underlying dictionary or unigram.  `self` and `other` must
        use completely identical dictionaries (same vocabulary and ordering).
        """

        if not isinstance(other, BigramBase):
            return NotImplemented

        self.Nxx += other.Nxx
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)

        return self


    def validate_undersampling(self):
        # Performing undersampling multiple times is a mistake.
        if self.undersampled:
            raise ValueError(
                "Undersampling was already done.  Cannot perform "
                "undersampling multiple times."
            )
        # Undersampling using a smoothed unigram would give wrong results.
        if self.unigram.smoothed:
            raise ValueError(
                "Cannot perform undersampling based on smoothed unigram "
                "frequencies.  Perform undersampling before smoothing"
            )


    def apply_w2v_undersampling(self, t):
        """
        Simulate undersampling of common words, like how is done in word2vec.
        However, when applied here (as opposed to within the corpus sampler,
        we are taking expectation values cooccurrence statistics under 
        undersampling, and undersampling is applied in the "clean" way which
        does not alter the effective size of the sample window.
        """

        if t == 1 or t is None:
            return 
        self.validate_undersampling()
        self.undersampled = True

        # First calculate probability of dropping row-word and col-words
        p_i = h.corpus_stats.w2v_prob_keep(self.uNx, self.uN, t)
        p_i = sparse.lil_matrix(p_i)
        p_j = h.corpus_stats.w2v_prob_keep(self.uNxt, self.uN, t)
        p_j = sparse.lil_matrix(p_j)

        # Calculate the expectation cooccurrence after undersampling
        self.Nxx = self.Nxx.multiply(p_i).multiply(p_j).tolil()

        # Recalculate marginals.
        self.Nx = np.array(np.sum(self.Nxx, axis=1))
        self.Nxt = np.array(np.sum(self.Nxx, axis=0))
        self.N = np.sum(self.Nx)


    def apply_unigram_smoothing(self, alpha):
        """
        Smooth the unigram distribution by raising all frequencies to the 
        exponent `alpha`, followed by re-normalization.  This irreversibly
        mutates the underlying unigram object.
        """

        if alpha == 1 or alpha is None:
            return

        self.unigram.apply_smoothing(alpha)

        # We keep unigram data locally as a tensor, so we need to recopy it all
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        mem_device = h.CONSTANTS.MEMORY_DEVICE
        self.uNx = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=mem_device).view(-1, 1)
        self.uNxt = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=mem_device).view(1, -1)
        self.uN = torch.tensor(self.unigram.N, dtype=dtype, device=mem_device)


    def get_sector(self, sector, device=None, verbose=None):
        device = device if device is not None else self.device
        verbose = verbose if verbose is not None else self.verbose
        return h.bigram.BigramSector(
            self.unigram, self.Nxx[sector], self.Nx, self.Nxt, sector,
            device=device, verbose=verbose
        )

