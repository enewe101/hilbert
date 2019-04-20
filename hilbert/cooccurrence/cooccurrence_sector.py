import re
import os
import warnings
import numpy as np
import torch
import hilbert as h
from scipy import sparse




MEM_DEVICE = h.CONSTANTS.MEMORY_DEVICE

class CooccurrenceSector(object):

    def __init__(
            self, unigram, Nxx, Nx, Nxt, sector, verbose=True
        ):
        """
        CooccurrenceSector represents a subset of the cooccurrence data coming
        from a corpus. That is, after the corpus Nxx data is divided by
        CooccurrenceMutable.sectorize(), this deals with those sectors in a
        useful way.
        """

        if not unigram.sorted:
            raise ValueError(
                'Cooccurrence instances must be built from a sorted Unigram '
                'instance')

        # Own some things
        self.dtype = h.utils.get_dtype()
        self.verbose = verbose

        # Store unigram data.
        self.unigram = unigram
        self._uNx = torch.tensor(
            self.unigram.Nx, dtype=self.dtype, device=MEM_DEVICE).view(-1,1)
        self._uNxt = torch.tensor(
            self.unigram.Nx, dtype=self.dtype, device=MEM_DEVICE).view(1,-1)
        self.uN = torch.tensor(
            self.unigram.N, dtype=self.dtype, device=MEM_DEVICE)

        # Own cooccurrence statistics and marginalized totals.
        self.Nxx = sparse.lil_matrix(Nxx)
        self._Nx = torch.tensor(Nx, dtype=self.dtype, device=MEM_DEVICE)
        self._Nxt = torch.tensor(Nxt, dtype=self.dtype, device=MEM_DEVICE)
        self.N = torch.sum(self._Nx)

        self.sector = sector

        # Distinct subsamples of the dictionary to index rows and columns.
        self.row_dictionary = h.dictionary.Dictionary(
            self.unigram.dictionary.tokens[sector[0]])
        self.column_dictionary = h.dictionary.Dictionary(
            self.unigram.dictionary.tokens[sector[1]])

        # Check all is good
        self.validate_shape()
        self.undersampled = False


    @staticmethod
    def load(path, sector, verbose=True):
        """
        Load the token-ID mapping and cooccurrence data previously saved in
        the directory at `path`.
        """
        # Read Unigram
        unigram = h.unigram.Unigram.load(path, verbose=verbose)

        # Read Nxx, Nx, and Nxt.
        if sector is None:
            Nxx_fname = 'Nxx.npz'
        else:
            Nxx_fname = 'Nxx-{}-{}-{}.npz'.format(*h.shards.serialize(sector))
        Nxx = sparse.load_npz(os.path.join(path, Nxx_fname)).tolil()
        Nx = np.load(os.path.join(path, 'Nx.npy'))
        Nxt = np.load(os.path.join(path, 'Nxt.npy'))

        return CooccurrenceSector(
            unigram, Nxx=Nxx, Nx=Nx, Nxt=Nxt, sector=sector, verbose=verbose)


    @staticmethod
    def load_coo(cooccurrence_path, include_marginals=True, verbose=True):
        """ 
        Reads in sectorized cooccurrence data from disk, and converts it
        into a sparse tensor representation using COO format.  If desired,
        marginal sums are included.
        """

        # Go though each sector and accumulate all of the non-zero data
        # into a single sparse tensor representation.
        data = torch.tensor([], dtype=h.utils.get_dtype(), device=MEM_DEVICE)
        I = torch.tensor([], dtype=torch.int32, device=MEM_DEVICE)
        J = torch.tensor([], dtype=torch.int32, device=MEM_DEVICE)

        sector_factor = h.cooccurrence.CooccurrenceSector.get_sector_factor(
            cooccurrence_path)

        for sector_id in h.shards.Shards(sector_factor):

            if verbose:
                print('loading sector {}'.format(sector_id.serialize()))

            # Read the sector, and get the statistics in sparse COO-format
            sector = h.cooccurrence.CooccurrenceSector.load(
                cooccurrence_path, sector_id)
            sector_coo = sector.Nxx.tocoo()

            # Tensorfy the data, and the row and column indices
            add_Nxx = torch.tensor(sector_coo.data, dtype=h.utils.get_dtype())
            add_i_idxs = torch.tensor(sector_coo.row, dtype=torch.int)
            add_j_idxs = torch.tensor(sector_coo.col, dtype=torch.int)

            # Adjust the row and column indices to account for sharding
            add_i_idxs = add_i_idxs * sector_id.step + sector_id.i
            add_j_idxs = add_j_idxs * sector_id.step + sector_id.j

            # Concatenate
            data = torch.cat((data, add_Nxx))
            I = torch.cat((I, add_i_idxs))
            J = torch.cat((J, add_j_idxs))

        if include_marginals:
            # Every sector has global marginals, so get marginals from last
            # sector.
            Nx = torch.tensor(sector._Nx, dtype=h.utils.get_dtype())
            Nxt = torch.tensor(sector._Nxt, dtype=h.utils.get_dtype())
            return data, I, J, Nx, Nxt
        else:
            return data, I, J


    @property
    def shape(self):
        return self.Nxx.shape

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
            '`CooccurrenceSector.row_dictionary` and '
            '`CooccurrenceSector.column_dictionary`.', DeprecationWarning
        )
        return self.unigram.dictionary


    def validate_shape(self):

        # Ensure that Nx, Nxt, uNx, uNxt match prior to viewing through 
        # self.sector
        Nx_match = self._uNx.shape == self._Nx.shape
        Nxt_match = self._uNxt.shape == self._Nxt.shape
        if not all([Nx_match, Nxt_match]):
            raise ValueError(
                'Shape of Cooccurrence marginal stats should be identical to '
                'shape of unigram stats. Got:' 
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

        device = h.utils.get_device(device)

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

        device = h.utils.get_device(device)

        loaded_uNx = h.utils.load_shard(
            self.uNx, shard[0], device=device)
        loaded_uNxt = h.utils.load_shard(
            self.uNxt, (slice(None), shard[1]), device=device)
        loaded_uN = h.utils.load_shard(self.uN, device=device)

        return loaded_uNx, loaded_uNxt, loaded_uN


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
        self._uNx = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=MEM_DEVICE).view(-1,1)
        self._uNxt = torch.tensor(
            self.unigram.Nx, dtype=dtype, device=MEM_DEVICE).view(1,-1)
        self.uN = torch.tensor(self.unigram.N, dtype=dtype, device=MEM_DEVICE)


    def apply_w2v_undersampling(self, t):
        """
        Simulate undersampling of common words, like how is done in word2vec.
        However, when applied here (as opposed to within the corpus sampler),
        we are taking expectation values given undersampling, rather than
        actually undersampling.  This pseudo-ndersampling is applied in the
        "clean" way which does not alter the effective size of the sample
        window.
        """

        if t == 1 or t is None:
            return 
        self.validate_undersampling()
        self.undersampled = True

        # For each pair of words, calculate the probability that a given 
        # cooccurrence would still be observed given undersampling.

        # First calculate probability of dropping row-word and col-words
        p_i = h.cooccurrence.cooccurrence.w2v_prob_keep(self._uNx, self.uN, t)
        p_j = h.cooccurrence.cooccurrence.w2v_prob_keep(self._uNxt, self.uN, t)

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

        if not isinstance(other, CooccurrenceSector):
            return NotImplemented

        if self.sector != other.sector:
            raise ValueError(
                "`CooccurrenceSector`s must represent the same sector to be "
                "mergeable."
            )

        self.Nxx += other.Nxx
        self._Nx += other._Nx
        self._Nxt += other._Nxt
        self.N += other.N

        return self

    @staticmethod
    def get_sector_factor(path):
        # Check for presence of auxiliary files
        found_files = set(os.listdir(path))
        if 'Nx.npy' not in found_files:
            raise ValueError()
        elif 'Nxt.npy' not in found_files:
            raise ValueError()

        sector_matcher = re.compile('Nxx-\d+-\d+-(\d+).npz')
        sector_factors = [
            int(sector_matcher.match(p).groups()[0]) for p in os.listdir(path) 
            if sector_matcher.match(p)
        ]
        if len(sector_factors) == 0:
            if 'Nxx.npz' not in found_files:
                raise ValueError(
                    "No cooccurrence sectors found on disk at {}.".format(path)
                )
            return None
        sector_factor = sector_factors[0]
        if not all([s == sector_factor for s in sector_factors]):
            raise ValueError(
                "Sector factor is ambiguous.  Did you write other files to "
                "disk at {} that would match '{}'?".format(
                    path, sector_matcher.pattern
                )
            )
        if not len(sector_factors) == sector_factor**2:
            raise ValueError(
                "Some sectors appear to be missing.  Detected a sector "
                "factor of {}, but only found {} sectors (expected {}).".format(
                    sector_factor, len(sector_factors), sector_factor**2
                )
            )
        return sector_factor




    def get_sector(self, *args):
        raise NotImplementedError(
            "`CooccurrenceSector`s cannot `get_sector`.")


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


