import hilbert as h
from abc import ABC, abstractmethod
from hilbert.loader import Loader, MultiLoader
try:
    import torch
    from torch.multiprocessing import JoinableQueue, Process
except ImportError:
    torch = None
    JoinableQueue, Process = None, None


class BigramLoaderBase():

    def __init__(
        self, bigram_path, sector_factor, shard_factor, num_loaders, 
        queue_size=1, device=None
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

        Subclasses can override `_load`, to more specifically choose what
        bigram / unigram data to load, and what other preparations to do to
        make the shard ready to be fed to the model.
        """
        if num_loaders != sector_factor**2:
            raise ValueError(
                "`num_loaders` must equal `sector_factor**2`, so that each "
                "sector can be assigned to one loader."
            )
        self.bigram_path = bigram_path
        self.sector_factor = sector_factor
        self.shard_factor = shard_factor

        super(BigramLoaderBase, self).__init__(
            num_loaders=num_loaders, queue_size=queue_size)


    def _preload_iter(self, loader_id):
        sector_id = h.shards.Shards(self.sector_factor)[loader_id]
        bigram_sector = h.bigram.BigramSector.load(
            self.bigram_path, sector_id)
        for shard_id in h.shards.Shards(self.shard_factor):
            bigram_data = bigram_sector.load_relative_shard(
                shard=shard_id, device='cpu')
            unigram_data = bigram_sector.load_relative_unigram_shard(
                shard=shard_id, device='cpu')
            yield shard_id * sector_id, bigram_data, unigram_data


    def _load(self, preloaded):
        device = h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(device) for tensor in bigram_data)
        unigram_data = tuple(tensor.to(device) for tensor in unigram_data)
        return shard_id, bigram_data, unigram_data


class BigramLoader(BigramLoaderBase, Loader):
    pass

class BigramMultiLoader(BigramLoaderBase, MultiLoader):
    pass

