import os
import warnings
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
from .bigram_base import BigramBase

class BigramSector(BigramBase):
    """Represents cooccurrence statistics."""

    def __init__(
        self, unigram, Nxx, Nx, Nxt, sector,
        device=None,
        verbose=True
    ):
        '''
        BigramSector represents a subset of the bigram data coming from a 
        corpus.
        '''

        if not unigram.sorted:
            raise ValueError(
                'Bigram instances must be built from a sorted Unigram instance')

        # Own some things
        self.device = device
        self.verbose = verbose

        dtype = h.CONSTANTS.DEFAULT_DTYPE

        # Store unigram data.
        self.unigram = unigram
        self._uNx = torch.tensor(self.unigram.Nx, dtype=dtype).view(-1,1)
        self._uNxt = torch.tensor(self.unigram.Nx, dtype=dtype).view(1,-1)
        self.uN = torch.tensor(self.unigram.N, dtype=dtype)

        # Own cooccurrence statistics and marginalized totals.
        self.Nxx = sparse.lil_matrix(Nxx)
        self._Nx = torch.tensor(Nx, dtype=h.CONSTANTS.DEFAULT_DTYPE)
        self._Nxt = torch.tensor(Nxt, dtype=h.CONSTANTS.DEFAULT_DTYPE)
        self.N = torch.sum(self._Nx)

        self.sector = sector

        # Distinct subsamples of the dictionary to index rows and columns.
        self.row_dictionary = h.dictionary.Dictionary(
            self.unigram.dictionary.tokens[sector[0]])
        self.column_dictionary = h.dictionary.Dictionary(
            self.unigram.dictionary.tokens[sector[1]])

        # Check all is good
        self.validate_shape()


    @property
    def Nx(self):
        return self._Nx[self.sector[0]]

    @property
    def Nxt(self):
        return self._Nxt[:,self.sector[1]]

    @property
    def uNx(self):
        return self._uNx[self.sector[0]]

    @property
    def uNxt(self):
        return self._uNxt[:,self.sector[1]]

    @property
    def dictionary(self):
        warnings.warn(
            '`The conversion from tokens to indices requires the use of '
            '`BigramSector.row_dictionary` and '
            '`BigramSector.column_dictionary`.', DeprecationWarning
        )
        return self.unigram.dictionary


    def validate_shape(self):

        # Ensure that Nx, Nxt, uNx, uNxt match prior to viewing through 
        # self.sector
        Nx_match = self._uNx.shape == self._Nx.shape
        Nxt_match = self._uNxt.shape == self._Nxt.shape
        if not all([Nx_match, Nxt_match]):
            raise ValueError(
                'Shape of Bigram marginal stats should be identical to shape '
                'of unigram stats. Got:' 
                '\nuNx:\t{}\nuNxt:\t{}\nNx:\t{}\nNxt:\t{}'.format(
                    self.uNx.shape, self.uNxt.shape,
                    self.Nx.shape, self.Nxt.shape,
                )
            )

        # Ensure that Nxx matches shape of Nx, Nxt, uNx, uNxt viewed through
        # self.sector
        uNx_match = self.Nxx.shape[0] == self.uNx.shape[0] 
        uNxt_match = self.Nxx.shape[1] == self.uNxt.shape[1]
        Nx_match = self.Nxx.shape[0] == self.Nx.shape[0]
        Nxt_match = self.Nxx.shape[1] == self.Nxt.shape[1]
        if not all([uNx_match, uNxt_match, Nx_match, Nxt_match]):
            raise ValueError(
                'Nxx length and width should unigram. '
                'Got: \nNxx:\t{}x{}\nuNx:\t{}\nuNxt:\t{}'
                '\nNx:\t{}\nNxt:\t{}'
                .format(
                    self.Nxx.shape[0], self.Nxx.shape[1], 
                    self.uNx.shape[0], self.uNxt.shape[1],
                    self.Nx.shape[0], self.Nxt.shape[1],
                )
            )

        # Ensure that row and column dictionary lengths match Nxx's shape
        rd_match = self.Nxx.shape[0] == len(self.row_dictionary) 
        cd_match = self.Nxx.shape[1] == len(self.column_dictionary)
        if not all([rd_match, cd_match]):
            raise ValueError(
                'Nxx length and width should match '
                'dictionary. Got: \nNxx:\t{}x{}\n'
                'r_dict\t{}\nc_dict\t{}\n.'.format(
                    self.Nxx.shape[0], self.Nxx.shape[1], 
                    len(self.row_dictionary), len(self.column_dictionary)
                )
            )


    def count(self, token1, token2):
        id1 = self.row_dictionary.get_id(token1)
        id2 = self.column_dictionary.get_id(token2)
        return self.Nxx[id1, id2]


    @staticmethod
    def load(path, sector, device=None, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        # Read Unigram
        unigram = h.unigram.Unigram.load(path, device=device, verbose=verbose)

        # Read Nxx, Nx, and Nxt.
        if sector == h.shards.whole:
            Nxx_fname = 'Nxx.npz'
        else:
            Nxx_fname = 'Nxx-{}-{}-{}.npz'.format(*h.shards.serialize(sector))
        Nxx = sparse.load_npz(os.path.join(path, Nxx_fname)).tolil()
        Nx = np.load(os.path.join(path, 'Nx.npy'))
        Nxt = np.load(os.path.join(path, 'Nxt.npy'))

        return BigramSector(
            unigram, Nxx=Nxx, Nx=Nx, Nxt=Nxt, sector=sector,
            device=device, verbose=verbose
        )


    def load_shard(self, shard=None, device=None):
        """
        Provides tensors corresponding to the cooccurrence data, marginalized
        data, and total count, specific to the requested shard.

        INPUTS
        `shard`
            A pair of slice objects specifying a particular subset of cells, 
            or None (in which case the entire sector is loaded). 
        """

        # If no shard is provided, by default load the entire sector
        if shard is None:
            shard = self.sector

        relative_shard = h.shards.relativize(shard, self.sector)
        return self.load_relative_shard(relative_shard, device)


    def load_relative_shard(self, shard=None, device=None):
        """
        Provides tensors corresponding to the cooccurrence data, marginalized
        data, and total count, specific to the requested shard, when indexing
        relative to the current sector.

        I.e. the data yielded will correspond to `shard * sector`.

        INPUTS
        `shard`
            A pair of slice objects specifying a particular subset of cells, 
            or None (in which case the entire sector is loaded). 
        """

        # If no shard is provided, by default load the entire sector
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


    def load_unigram_shard(self, shard=None, device=None):
        """
        Provides tensors corresponding to the word occurrence (unigram) data,
        specific to the requested shard.

        I.e. the data yielded will correspond to `shard * sector`.

        INPUTS
        `shard`
            A pair (tuple) of slice objects specifying a particular subset of
            cells, or None (in which case the entire sector is loaded). 
        """
        # If no shard is provided, by default load the entire sector
        if shard is None:
            shard = self.sector

        relative_shard = h.shards.relativize(shard, self.sector)
        return self.load_relative_unigram_shard(relative_shard, device)


    def load_relative_unigram_shard(self, shard=None, device=None):
        """
        Provides tensors corresponding to the word occurrence (unigram) data,
        specific to the requested shard, when indexing relative to the current
        sector.

        I.e. the data yielded will correspond to `shard * sector`.

        INPUTS
        `shard`
            A pair (tuple) of slice objects specifying a particular subset of
            cells, or None (in which case the entire sector is loaded). 
        """
        # If no shard is provided, by default load the entire sector
        if shard is None:
            shard = h.shards.whole

        device = device or self.device or h.CONSTANTS.MATRIX_DEVICE

        loaded_uNx = h.utils.load_shard(
            self.uNx, shard[0], device=device)
        loaded_uNxt = h.utils.load_shard(
            self.uNxt, (slice(None), shard[1]), device=device)
        loaded_uN = h.utils.load_shard(self.uN, device=device)

        return loaded_uNx, loaded_uNxt, loaded_uN


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

        # First calculate probability of dropping row-word and col-words
        p_i = h.corpus_stats.w2v_prob_keep(self._uNx, self.uN, t)
        p_j = h.corpus_stats.w2v_prob_keep(self._uNxt, self.uN, t)

        # This approximates the effect of undersampling on marginal counts
        # without actually having to re-sum marginals (since a sector does not
        # possess the necessary cooccurrence data to do that)
        self._Nx = self._Nx * p_i * torch.sum(self._uNxt / self.uN * p_j)
        self._Nxt = self._Nxt * p_j * torch.sum(self._uNx / self.uN * p_i)
        self.N = torch.sum(self._Nx)

        # Calculate the expectation cooccurrence counts given undersampling.
        # This needs to be estimated in a different way.
        p_i = sparse.lil_matrix(p_i[self.sector[0]].numpy())
        p_j = sparse.lil_matrix(p_j[:,self.sector[1]].numpy())
        self.Nxx = self.Nxx.multiply(p_i).multiply(p_j).tolil()



    def merge(self, other):
        """
        Add counts from `other` to `self`, in place, without altering
        the underlying dictionary or unigram.  `self` and `other` must
        use completely identical dictionaries (same vocabulary and ordering).
        And represent the same sector.
        """

        if not isinstance(other, BigramSector):
            return NotImplemented

        if self.sector != other.sector:
            raise ValueError(
                "`BigramSector`s must represent the same sector to be "
                "mergeable."
            )

        self.Nxx += other.Nxx
        self._Nx += other._Nx
        self._Nxt += other._Nxt
        self.N += other.N

        return self


    def get_sector(self, *args):
        raise NotImplementedError("`BigramSector`s cannot `get_sector`.")


