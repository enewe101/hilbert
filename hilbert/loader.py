import hilbert as h
import torch
from abc import ABC, abstractmethod


class Loader(ABC):

    def __init__(self, verbose=True):
        """
        Iterable that provides GPU-loaded minibatches.

        This loader is a drop-in replacement for `MultiLoader`.  They are
        identical in effect, except that (1) no preloading happens in the
        background, and (2) the ordering of minibatches is deterministic.

        While the whole purpose of a having a loader is to preload data in the
        background, `Loader` as a drop-in replacement when debugging classes
        that are intended to subclass MultiLoader.  Simply temporarily subclass
        `Loader` instead of `MultiLoader` to keep execution in a single process
        while debugging.

        INPUTS
        `num_loaders`, 
            Number of times self._preload_iter() will be called, each time with
            a different `loader_id`.  In `MultiLoader`, these calls are run
            concurrently in separate processes.  Here they run sequentially
            in the main process.
        
        `queue_size`, `verbose`
            Both have no effect.  Included for compatibility with `MultiLoader`.
        """
        self.verbose = verbose


    def __iter__(self):
        """
        Yields GPU-loaded shards.  This is the only element of the public
        interface.  The training loop should treat loader as an iterable, and
        calculate forward, backward passes using the shards yielded.
        """
        for preloaded in self._preload_iter(loader_id):
            yield self._load(preloaded)


    @abstractmethod
    def _load(self, preloaded):
        """
        Returns a gpu-loaded shard, usually in the form of a tuple
        `(shard_id, gpu_data)`, where `shard_id` is an index representing
        the rows and columns in the shard and `gpu_data` is some collection of
        gpu-tensors representing the data in that shard (e.g. tuple of tensors,
        dict of tensors, single tensor, etc.).'
        """
        pass


    @abstractmethod
    def _preload_iter(self, loader_id):
        """
        Generator that yields CPU-loaded shards usually in the form of a tuple
        `(shard_id, cpu_data)`.  It will be iterated over inside a loader
        process.  Based on the `loader_id` and `self.num_loaders`, it should
        yield only a portion of the data, so that together all calls to 
        `_preload_iter` yield the full dataset without overlap.
        """
        pass



class BufferedLoader(Loader):

    def __init__(self, *args, **kwargs):
        super(BufferedLoader, self).__init__(*args, **kwargs)
        self.cpu_preloads = []

    def __iter__(self):
        """
        Yields GPU-loaded shards.

        During first iteration, all cpu shards will be buffered into memory and
        held there.  Subsequent iterations will simply load onto the GPU from
        the in-memory cpu shards.
        
        This is the only element of the public
        interface.  The training loop should treat loader as an iterable, and
        calculate forward, backward passes using the shards yielded.
        """
        if self.cpu_preloads is None:
            self.cpu_preloads = [
                preload for loader_id in range(self.num_loaders)
                for preload in self._preload_iter(loader_id)
            ]
        for preload in self.cpu_preloads:
            yield self._load(preload)



