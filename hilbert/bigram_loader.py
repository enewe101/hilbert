import warnings
import hilbert as h
import torch
from hilbert.loader import BufferedLoader



class BigramLoaderBase(object):

    def __init__(
        self, bigram_path, sector_factor, shard_factor,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
        device=None,
        verbose=True
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

        self.bigram_path = bigram_path
        self.sector_factor = sector_factor
        self.shard_factor = shard_factor
        self.t_clean_undersample = t_clean_undersample
        self.alpha_unigram_smoothing = alpha_unigram_smoothing
        self.device = device
        self.loaded_sector = None
        self.bigram_sector = None

        super(BigramLoaderBase, self).__init__(verbose=verbose)


    def _preload_iter(self, loader_id):

        for i, sector_id in enumerate(h.shards.Shards(self.sector_factor)):

            # Each worker should handle a subset of the sectors
            if i % self.num_loaders != loader_id:
                continue

            # If we're doing the same sector as last time, no need to reload it
            # This is the advantage of having 
            # num_workers = num_sectors = sector_factor**2, since a given worker
            # will have a dedicated sector.
            if self.loaded_sector != sector_id:
                self.loaded_sector = sector_id

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


    def _load(self, preloaded):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        shard_id, bigram_data, unigram_data = preloaded
        bigram_data = tuple(tensor.to(device) for tensor in bigram_data)
        unigram_data = tuple(tensor.to(device) for tensor in unigram_data)
        return shard_id, bigram_data, unigram_data


    def describe(self):
        s = '\tbigram_path = {}\n'.format(self.bigram_path)
        s += '\tsector_factor = {}\n'.format(self.sector_factor)
        s += '\tshard_factor = {}\n'.format(self.shard_factor)
        s += '\tt_clean_undersample = {}\n'.format(self.t_clean_undersample)
        s += '\talpha_unigram_smoothing = {}\n'.format(
            self.alpha_unigram_smoothing)
        s += '\tdevice = {}\n'.format(self.device)
        s += '\tverbose = {}\n'.format(self.verbose)
        return s



class BigramBufferedLoader(BigramLoaderBase, BufferedLoader):
    pass


