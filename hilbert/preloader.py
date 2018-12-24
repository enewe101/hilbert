import hilbert as h

class Preloader(object):

    def setup(self):
        """
        This function will be called only once, when the prelaoder process
        first starts up.  Use it for any one-time preparation.  For example,
        read a portion of a dataset into memory or connect to a database.
        database.

        Anything preparations that produce state which is difficult to pickle
        and transmit through a pipe can be done here, since this is done after
        the preloader process starts (after the transmission has occured).

        By default this is a no-op, so it is not necessary to override.
        """
        pass

    def __getitem__(self, shard_id):
        """
        This method should yield a single `(shard_id, cpu_tensors)` tuple,
        where: 

            `shard_id` 
                should be a shard_spec, i.e. a 2-tuple of SliceObjecgts, and

            `cpu_tensors`
                is a tensor, tuple of tensors, or any kind of object that
                collects together a bunch of tensors that represent the data
                for one shard, loaded onto the CPU.
        """
        raise NotImplementedError(
            'Subcpasses should override Preloader.__getitem__')



class BigramPreloader(Preloader):

    def __init__(self, bigram_path):
        """
        This preloader wraps a Bigram object so that it supports the preloader
        interface.  This helps with preloading tensors correspondong to bigram
        data onto the CPU in background processes. 

        `bigram_path` is the path to directory storing data from which a 
        `hilbert.bigram.Bigram` can be loaded.
        """
        self.bigram_path = bigram_path

    def setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def __getitem__(self, shard_id):
        return self.bigram.load_shard(shard=shard_id, device='cpu')


