import hilbert as h
from abc import ABC, abstractmethod
try:
    import torch
    from torch.multiprocessing import JoinableQueue, Process
except ImportError:
    torch = None
    JoinableQueue, Process = None, None



class Loader(ABC):

    def __init__(self, num_loaders=1, queue_size=1):
        """
        Iterable that provides GPU-loaded data shards (similar to minibatches).

        INPUTS
        `num_loaders`, 
            Number of times self._preload_iter() will be called, each time with
            a different `loader_id`.  In `MultiLoader`, these calls are run
            concurrently in separate processes.  Here they run sequentially
            in the main process.
        
        `queue_size`
            Has no effect.  Included for compatibility with `MultiLoader`.

        Shards are like minibatches, but for matrix data.  A "shard_spec" is 
        a subset of rows and columns representing the (usually non-contiguous)
        subset of matrix data selected for a given minibatch.

        Shards are first constructed by the CPU then loaded onto the GPU 
        where they may undergo further initialization, but ultimately are
        yielded to the training loop where they are used to run forward /
        backward training updates.

        Compared to `MultiLoader`, which uses multiprocessing to prepare
        shards in the background for faster loading, 
        this loader runs in a single process. Any CPU manipulations involved
        in loading are performed synchronously, when the next shard is requested
        in the training loop.  This is generally slower, but provides a simpler
        approach that is appropriate if the CPU-based operations to prepare a 
        shard are minimal, or when troubleshooting specific loading logic.

        Compatibility Note
        This class is constructed so that one can easily write concrete
        implementations that are simultaneously valid subclasses for Loader and
        MultiLoader, making them easy to toggle between by changing the
        inheritence in the subclass's definition.  The `__init__()` the inputs
        `num_loaders`, `queue_size` have no effect.  The `worker_id` will
        be passed into `_setup(worker_id)`, as in `MultiLoader`, and `_setup()`
        will be run at the end of the `__init__()` function.
        """
        self.num_loaders = num_loaders


    def __iter__(self):
        for loader_id in range(self.num_loaders):
            for preloaded in self._preload_iter(loader_id):
                yield self._load(preloaded)


    @abstractmethod
    def _load(self, preloaded):
        """
        Returns a gpu-loaded shard, in the form of a tuple
        `(shard_spec, gpu_data)`, where `shard_spec` is an index representing
        the rows and columns in the shard and `gpu_data` is some collection of
        gpu-tensors representing the data in that shard (e.g. tuple of tensors,
        dict of tensors, single tensor, etc.).'
        """
        pass


    @abstractmethod
    def _preload_iter(self, loader_id):
        """
        Returns a cpu-loaded shard, in the form of a tuple
        `(shard_spec, cpu_data)`, where `shard_spec` is an index representing
        the rows and columns in the shard and `cpu_data` is some collection of
        cpu-tensors representing the data in that shard (e.g. tuple of tensors,
        dict of tensors, single tensor, etc.).'
        """
        pass



class MultiLoader(ABC):

    def __init__(self, num_loaders, queue_size=1):
        """
        Like `Loader`, `MultiLoader` is an iterable that yields `(shard_spec,
        gpu_tensors)` pairs.  However, unlike Loader, it runs preloading code
        in separate loader processes, so that it can keep up with GPU demand
        and provide fast loading times.

        This is an abstract base_class.  Concrete subclasses should override
        `_setup`, `_preload`, and `_load`.

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
            new shard once the one it made is taken, so it should naturally
            keep memory usage low, while allowing you to have enough workers
            to keep up with the GPU.

        `device`
            The target device for final loading.  Shards are preloaded onto the
            CPU, but are finally loaded onto this device when it is time to 
            learn from the given shard.  Normally this can be left to it's 
            default, which is to target the globally set
            h.CONSTANTS.MATRIX_DEVICE.
        """
        self.queue_size = queue_size
        self.num_loaders = num_loaders
        self._start_preloading()


    def _start_preloading(self):
        """
        Start a number child loader processes.  The number of processes started
        is controlled by `self.num_loaders`.  In the child process, shards will
        be preloaded by running the `_preload()` function on shard_specs,
        as yielded by `self.shard_schedule`.  Preloading of shards is
        asynchronous, so ordering in which shards will finally be loaded onto
        the GPU is not guaranteed, but is roughly the same as that provided by
        `self.shard_scheudle`.
        """

        # Place the desired shards onto the request queue, which serves as a 
        # listing of work from which loaders draw.  Loaders will place 
        # finished shards (shards loaded into cRAM) onto the result_queue
        self.result_queue = JoinableQueue(maxsize=self.queue_size)

        # Start the loader processes.
        self.loading_processes = []
        for loader_id in range(self.num_loaders):
            p = Process(target=self._manage_preloading, args=(loader_id,))
            p.start()
            self.loading_processes.append(p)


    def __iter__(self):
        # Loads and yields shards onto (on gRAM).  This is the public interface
        # of the class which is used by the training loop to request the next
        # shard to be loaded onto the GPU to run forward / backward learning
        # iterations.

        # Iterates through preloaded shards, as they become available
        preload_iterator = h.utils.iterate_queue(
            self.result_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=self.num_loaders
        )
        for preloaded in preload_iterator:
            yield self._load(preloaded)


    def _manage_preloading(self, loader_id):
        """
        Target function that runs inside child loader processes.  Using 
        Queues, it simply takes shard specs, generates CPU-loaded shards
        by delegation to `self._preload()`, and places the result onto a queue
        for consumption in the main process.  Usually, don't override this,
        but rather, override the `_preload()` to which it delegates.  Notice
        that before 
        """
        for preloaded in self._preload_iter(loader_id):
            self.result_queue.put(preloaded)
        self.result_queue.put(StopIteration())


    @abstractmethod
    def _load(self, preloaded):
        """
        Returns a gpu-loaded shard, in the form of a tuple
        `(shard_spec, gpu_data)`, where `shard_spec` is an index representing
        the rows and columns in the shard and `gpu_data` is some collection of
        gpu-tensors representing the data in that shard (e.g. tuple of tensors,
        dict of tensors, single tensor, etc.).'
        """
        pass


    @abstractmethod
    def _preload_iter(self, loader_id):
        """
        This should generate `shard_id`, `cpu_data` tuples.  It will be 
        iterated over inside a loader process.
        """
        pass



