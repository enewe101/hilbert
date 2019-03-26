import abc
import hilbert as h
import torch
from hilbert.generic_interfaces import Describable


class BatchPreloader(Describable):
    """
    Abstract class for batch preloading.
    """

    @abc.abstractmethod
    def __init__(self, bigram_path, *args,
                 t_clean_undersample=None,
                 alpha_unigram_smoothing=None, **kwargs):

        self.bigram_path = bigram_path
        self.t_clean_undersample = t_clean_undersample
        self.alpha_unigram_smoothing = alpha_unigram_smoothing

    @abc.abstractmethod
    def preload_iter(self, *args, **kwargs):
        return

    def prepare(self, preloaded):
        return preloaded

    @abc.abstractmethod
    def describe(self):
        s = '\tbigram_path = {}\n'.format(self.bigram_path)
        s += '\tt_clean_undersample = {}\n'.format(self.t_clean_undersample)
        s += '\talpha_unigram_smoothing = {}\n'.format(
            self.alpha_unigram_smoothing)
        return s



"""
Class for hardcore matrix factorization data loading.
"""
class DenseShardPreloader(BatchPreloader):

    def __init__(
        self, bigram_path, sector_factor, shard_factor,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
    ):
        """
        Base class for more specific loaders `BigramLoader` yields tensors 
        representing shards of text cooccurrence data.  Each shard has unigram
        and bigram data, for words and word-pairs, along with totals.

        bigram data:
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
        super(DenseShardPreloader, self).__init__(bigram_path,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing
        )
        self.sector_factor = sector_factor
        self.shard_factor = shard_factor
        self.bigram_sector = None


    def preload_iter(self, *args, **kwargs):
        super(DenseShardPreloader, self).preload_iter(*args, **kwargs)

        for i, sector_id in enumerate(h.shards.Shards(self.sector_factor)):

            # Read the sector of bigram data into memory, and transform
            # distributions as desired.
            self.bigram_sector = h.bigram.BigramSector.load(
                self.bigram_path, sector_id)

            self.bigram_sector.apply_w2v_undersampling(
                self.t_clean_undersample)

            self.bigram_sector.apply_unigram_smoothing(
                self.alpha_unigram_smoothing)

            # Start yielding cRAM-preloaded shards
            for shard_id in h.shards.Shards(self.shard_factor):

                bigram_data = self.bigram_sector.load_relative_shard(
                    shard=shard_id, device='cpu')

                unigram_data = self.bigram_sector.load_relative_unigram_shard(
                    shard=shard_id, device='cpu')

                yield shard_id * sector_id, bigram_data, unigram_data
        return


    def describe(self):
        s = super(DenseShardPreloader, self).describe()
        s += '\tsector_factor = {}\n'.format(self.sector_factor)
        s += '\tshard_factor = {}\n'.format(self.shard_factor)
        return s




"""
Class for smart compressed representation loading.
"""
class SparsePreloader(BatchPreloader):

    def __init__(self, bigram_path,
                 t_clean_undersample=None,
                 alpha_unigram_smoothing=None,
                 include_unigram_data=False,
                 device=None):

        super(SparsePreloader, self).__init__(bigram_path,
          t_clean_undersample=t_clean_undersample,
          alpha_unigram_smoothing=alpha_unigram_smoothing,
        )
        self.device = device
        self.n_nonzeros = 0
        self.n_batches = 0
        self.include_unigram_data = include_unigram_data


    def preload_iter(self, *args, **kwargs):
        super(SparsePreloader, self).preload_iter(*args, **kwargs)

        bigram = h.bigram.BigramBase.load(self.bigram_path)

        # number of nonzero elements
        self.n_nonzeros = bigram.Nxx.nnz
        self.n_batches = len(bigram.Nxx.data)

        # iterate over each row index in the sparse matrix
        self.sparse_nxx = []

        # iterate over each row in the sparse matrix
        for i in range(len(bigram.Nxx.data)):
            js_tensor = torch.LongTensor(bigram.Nxx.rows[i])
            nijs_tensor = torch.FloatTensor(bigram.Nxx.data[i])

            # store the implicit sparse matrix as a series
            # of tuples, J-indexes, then Nij values.
            self.sparse_nxx.append(
                ( js_tensor.to(self.device),
                  nijs_tensor.to(self.device), )
            )

        # now we need to store the other statistics
        self.Nx = bigram.Nx.flatten().to(self.device)
        self.Nxt = bigram.Nxt.flatten().to(self.device)
        self.N = bigram.N.to(self.device)

        if self.include_unigram_data:
            self.uNx = bigram.uNx.flatten().to(self.device)
            self.uNxt = bigram.uNxt.flatten().to(self.device)
            self.uN = bigram.uN.to(self.device)

        # implicitly store the preloaded batches
        return range(self.n_batches)


    def prepare(self, preloaded):
        i, js = preloaded, self.sparse_nxx[preloaded][0]
        bigram_data = (self.sparse_nxx[preloaded][1],
                       self.Nx[i],
                       self.Nxt[js],
                       self.N)

        # fill up unigram data only if necessary
        unigram_data = None
        if self.include_unigram_data:
            unigram_data = (self.uNx[i], self.uNxt[js], self.uN)

        batch_id = (i, js)
        return batch_id, bigram_data, unigram_data


    def describe(self):
        s = super(SparsePreloader, self).describe()
        return 'Sparse preloader\n' + s


