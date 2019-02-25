import hilbert as h
from abc import ABC, abstractmethod
try:
    import torch
    from torch.multiprocessing import JoinableQueue, Process
except ImportError:
    torch = None
    JoinableQueue, Process = None, None


class Loader(ABC):

    def __init__(self, num_loaders=1, queue_size=1, verbose=True):
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
        self.num_loaders = num_loaders
        self.queue_size = queue_size
        self.verbose = verbose


    def __iter__(self):
        """
        Yields GPU-loaded shards.  This is the only element of the public
        interface.  The training loop should treat loader as an iterable, and
        calculate forward, backward passes using the shards yielded.
        """
        for loader_id in range(self.num_loaders):
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


class MultiLoader(ABC):

    def __init__(self, num_loaders, queue_size=1, verbose=True):
        """
        Iterable that yields GPU loaded minibatches.  Override
        `_preload_iter()` with a generator that yields minibatches preloaded
        (into cRAM), and override `_load()` with a function that moves those
        preloaded minibatches onto the GPU.  Several loader processes will
        execute `_preload_iter()` in the background, placing the yielded
        results onto a queue, to await their turn on the GPU, the transfer of
        data onto the GPU faster.

        When writing a subclass of `MultiLoader`, temporarily inherit from
        `Loader`, which supports the same interface, but keeps execution in the
        main process for easier debugging, then change the inheritance once
        things are working.

        INPUTS
        ``````
        `num_loaders`
            Number of worker processes to spawn.  The `_setup()` and
            `_preload()` methods (which need to be overriden by subclasses) are
            run in separate loader processes, so that CPUs can be busy
            preparing shards in the background while the GPU works.

        `queue_size`
            How large the queue of preloaded shards can be.  In general, if
            there are enough workers to keep up with the demand for batches
            by the GPU, then there will be `num_workers + queue_size` number
            of shards preloaded into cRAM awaiting their turn on the GPU.
            Normally setting it to 1 is fine, because it means that each worker
            is allowed to assemble it's own shard, but will only start on a
            new shard once the one it made is accepted onto the queue.
        """
        self.queue_size = queue_size
        self.num_loaders = num_loaders
        self.verbose = verbose
        self.already_iterated = False
        self._start_preloading()


    def _start_preloading(self):
        """
        Starts child loader processes, which will load cpu_shards in the
        background.
        """

        # Loaders will place finished shards (shards loaded into cRAM) onto the
        # result_queue
        self.result_queue = JoinableQueue(maxsize=self.queue_size)
        self.epoch_queue = JoinableQueue()

        # Start the loader processes.
        self.loading_processes = []
        for loader_id in range(self.num_loaders):
            p = Process(target=self._manage_preloading, args=(loader_id,))
            p.daemon = True
            p.start()
            self.loading_processes.append(p)


    def __iter__(self):
        """
        Yields GPU-loaded shards.  This is the only element of the public
        interface.  The training loop should treat loader as an iterable, and
        calculate forward, backward passes using the shards yielded.
        """
        #if self.already_iterated:
        #    self._start_preloading()
        #self.already_iterated = True

        # Dispatch all the workers to start one epoch
        for worker_id in range(self.num_loaders):
            self.epoch_queue.put(True)

        # Iterates through preloaded shards, as they become available
        preload_iterator = h.utils.iterate_queue(
            self.result_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=self.num_loaders, verbose=self.verbose
        )
        for preloaded in preload_iterator:
            yield self._load(preloaded)
            #self.result_queue.task_done()


    def _manage_preloading(self, loader_id):
        """
        Preloads data shards into CPU-ram, so that they can be more quickly
        loaded onto the GPU when their turn comes.  Runs in a background loader
        process.  Once all shards have been preloaded, it will signal to the
        main process that it has no more work to send using a StopIteration as
        a sentinal.  It will wait until the main process has consumed
        everything off the queue (to avoid prematurely closing the queue and
        generating exception in the main process).
        """
        epoch_iterator = h.utils.iterate_queue(
            self.epoch_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=1
        )
        for epoch in epoch_iterator:
            for preloaded in self._preload_iter(loader_id):
                self.result_queue.put(preloaded)
            self.result_queue.put(StopIteration())

        self.result_queue.join()


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
        Generator that yields CPU-loaded shards, usually in the form of a tuple
        `(shard_id, cpu_data)`.  It will be iterated over inside a loader
        process.  Based on the `loader_id` and `self.num_loaders`, it should
        yield only a portion of the data, so that together all calls to 
        `_preload_iter` yield the full dataset without overlap.
        """
        pass


class BufferedLoaderGPU(Loader):
    def __init__(self, *args, **kwargs):
        super(BufferedLoaderGPU, self).__init__(*args, **kwargs)
        self.gpu_preloaded= None

    def __iter__(self):
        if self.gpu_preloaded = None:
            self.gpu_preloaded = [
                self._load(preload) for loader_id in range(self.num_loaders)
                for cpu_preloads in self._preload_iter(loader_id)
                for preload in self.cpu_preloads
            ]

        for preload in self.gpu_preloaded:
            yield preload

class BufferedLoader(Loader):

    def __init__(self, *args, **kwargs):
        super(BufferedLoader, self).__init__(*args, **kwargs)
        self.cpu_preloads = None

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



class BufferedMultiLoader(MultiLoader):

    def __init__(self, *args, **kwargs):
        super(BufferedMultiLoader, self).__init__(*args, **kwargs)
        self.cpu_preloads = None

    def __iter__(self):
        """
        Yields GPU-loaded shards.  This is the only element of the public
        interface.  The training loop should treat loader as an iterable, and
        calculate forward, backward passes using the shards yielded.
        """

        if self.cpu_preloads is None:
            # Dispatch all the workers to start one epoch
            for worker_id in range(self.num_loaders):
                self.epoch_queue.put(True)

            # Iterates through preloaded shards, as they become available
            preload_iterator = h.utils.iterate_queue(
                self.result_queue, stop_when_empty=False,
                sentinal=StopIteration, num_sentinals=self.num_loaders,
                verbose=self.verbose
            )
            self.cpu_preloads = [preload for preload in preload_iterator]

        for preloaded in self.cpu_preloads:
            yield self._load(preloaded)
            #self.result_queue.task_done()

