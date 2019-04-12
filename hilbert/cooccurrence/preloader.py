import abc
import hilbert as h
import torch
import torch.distributions as dist
from math import ceil
from hilbert.generic_datastructs import Describable, \
    build_sparse_lil_nxx, build_sparse_tup_nxx


class BatchPreloader(Describable):
    """
    Abstract class for batch preloading.
    """

    @abc.abstractmethod
    def __init__(
            self, cooccurrence_path, *args,
            undersampling=None,
            smoothing=None, **kwargs
        ):

        self.cooccurrence_path = cooccurrence_path
        self.undersampling = undersampling
        self.smoothing = smoothing


    @abc.abstractmethod
    def preload_iter(self, *args, **kwargs):
        return


    def prepare(self, preloaded):
        return preloaded


    @abc.abstractmethod
    def describe(self):
        s = '\tcooccurrence_path = {}\n'.format(self.cooccurrence_path)
        s += '\tt_clean_undersample = {}\n'.format(self.undersampling)
        s += '\talpha_unigram_smoothing = {}\n'.format(
            self.smoothing)
        return s




class DenseShardPreloader(BatchPreloader):
    """
    Class for dense matrix factorization data loading.
    """

    def __init__(
        self,
        cooccurrence_path,
        shard_factor,
        undersampling=None,
        smoothing=None,
        verbose=True
    ):
        """
        Base class for more specific loaders `CooccurrenceLoader` yields tensors 
        representing shards of text cooccurrence data.  Each shard has unigram
        and cooccurrence data, for words and word-pairs, along with totals.

        cooccurrence data:
            `Nxx`   number of times ith word seen with jth word.
            `Nx`    marginalized (summed) counts: num pairs containing ith word
            `Nxt`   marginalized (summed) counts: num pairs containing jth word
            `N`     total number of pairs.

            Note: marginalized counts aren't equal to frequency of the word,
            one word occurrence means participating in ~2 x window-size number
            of pairs.

        unigram data `(uNx, uNxt, uN)`
            `uNx`   Number of times word i occurs.
            `uNxt`  Number of times word j occurs.
            `uN`    total number of words

            Note: Due to unigram-smoothing (e.g. in w2v), uNxt may not equal
            uNx.  In w2v, one gets smoothed, the other is left unchanged (both
            are needed).
        """
        super(DenseShardPreloader, self).__init__(
            cooccurrence_path, undersampling=undersampling,
            smoothing=smoothing
        )
        self.shard_factor = shard_factor
        self.cooccurrence_sector = None
        self.verbose = verbose


    def preload_iter(self, *args, **kwargs):
        super(DenseShardPreloader, self).preload_iter(*args, **kwargs)

        sector_factor = h.cooccurrence.CooccurrenceSector.get_sector_factor(
            self.cooccurrence_path)

        for i, sector_id in enumerate(h.shards.Shards(sector_factor)):
            if self.verbose:
                print('loading sector {}'.format(i))

            # Read the sector of cooccurrence data into memory, and transform
            # distributions as desired.
            self.cooccurrence_sector = h.cooccurrence.CooccurrenceSector.load(
                self.cooccurrence_path, sector_id)

            self.cooccurrence_sector.apply_w2v_undersampling(
                self.undersampling)

            self.cooccurrence_sector.apply_unigram_smoothing(
                self.smoothing)

            # Start yielding cRAM-preloaded shards
            for shard_id in h.shards.Shards(self.shard_factor):

                cooccurrence_data = (
                    self.cooccurrence_sector.load_relative_shard(
                        shard=shard_id, device='cpu'
                    )
                )

                unigram_data = (
                    self.cooccurrence_sector.load_relative_unigram_shard(
                        shard=shard_id, device='cpu'
                    )
                )

                yield shard_id * sector_id, cooccurrence_data, unigram_data
        return


    def describe(self):
        s = super(DenseShardPreloader, self).describe()
        s += 'Dense Preloader\n'
        s += '\tshard_factor = {}\n'.format(self.shard_factor)
        return s







