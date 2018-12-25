import hilbert as h
try:
    import torch
    from torch.multiprocessing import Queue, Process
except ImportError:
    torch = None
    Queue, Process = None, None


def construct_bigram_loader(shard_schedule, bigram_path, num_workers):
    preloaders = [
        h.preloader.BigramPreloader(bigram_path) for w in range(num_workers)]
    return MultiLoader(shard_schedule, preloaders, bigram_path)



class Loader(object):

    def __init__(
        self, shard_schedule, 
        num_loaders=1, queue_size=1, worker_id=0, device=None
    ):
        """
        The `Loader`'s main responsibility is to provide GPU-loaded data
        shards (similar to minibatches by supporting the iterable interface.

        INPUTS
        `shard_schedule` 
            An iterator that yields shard_specs, which are pairs of 
            slice objects designating the rows and columns defining a shard.

        `num_loaders`, `queue_size`, `worker_id`
            For compatibility with `MultiLoader`.  See "Compatibility Note".

        `device`
            Target device onto which shards are ultimately loaded for training.

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
        self.shard_schedule = shard_schedule
        self.device = device
        self._setup(worker_id)


    def _setup(self, worker_id):
        """
        Runs one-time `__init__()`-like operations needed to prepare for
        loading.  Normally the code playing this role can go straight into
        `__init__()`, but this is provided for compatibility with the
        `MultiLoader` class, where it is necessary to defer some
        initializations until after starting child loader processes.  It can be
        safely left as a no-op if not needed.
        """
        pass


    def __iter__(self):
        for shard_spec, cpu_tensors in self._preload_iter():
            yield shard_spec, self._load(shard_spec, cpu_tensors)


    def _load(self, shard_spec, cpu_tensors):
        raise NotImplementedError(
            'Subclasses should override `Loader._load()`, which must '
            'return an iterable yielding pairs like `(shard_spec, gpu_data)`, '
            'where `gpu_data` is some collection of gpu-tensors (e.g. tuple of '
            'tensors, dictionary, or single tensor).'
        )


    def _preload_iter(self):
        for shard_spec in self.shard_schedule:
            yield shard_spec, self._preload(shard_spec)


    def _preload(self, shard_spec):
        raise NotImplementedError(
            'Subclasses should override `Loader._preload()`, which must '
            'return an iterable yielding pairs like `(shard_spec, cpu_data)`, '
            'where `cpu_data` is some collection of cpu-tensors (e.g. tuple of '
            'tensors, dictionary, or single tensor).'
        )





class MultiLoader(Loader):

    def __init__(self, shard_schedule, num_loaders, queue_size=1, device=None):
        """
        Like `Loader`, `MultiLoader` is an iterable that yields `(shard_spec,
        gpu_tensors)` pairs.  However, unlike Loader, it runs preloading code
        in separate loader processes, so that it can keep up with GPU demand
        and provide fast loading times.

        This is an abstract base_class.  Concrete subclasses should override
        `_setup`, `_preload`, and `_load`.

        INPUTS
        ``````
        `shard_schedule`    
            An iterator that yields shard_specs, which are
            2-tuples of `SliceObjects`.  These select subsets of the data,
            essentially minibatches, but they represent subsets of rows and
            columns, i.e. "shards" of underlying data matrices.

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
        self.shard_schedule = shard_schedule
        self.num_loaders = num_loaders
        self._start_preloading()
        self.device = device


    def _setup(self, preloader_id):
        """
        Code placed here is deferred `__init__()`-like code, which gets run
        inside child loader processes after the loaded process is started.

        Certain state is not good to transmit to the child process, since it
        involves pickling the state and sending it through a pipe.  For
        instance, if a large amount of data needs to be read from disk, might
        as well *not* do that in `__init__()`, since we will end up copying it,
        pickling it, and transmitting it to the child process.  Instead, by
        placing the reading operations here, the reading can be done in the
        child process.

        Leave this as a no-op if no such initialization is needed.
        """
        pass


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
        self.request_queue = Queue()
        self.result_queue = Queue(maxsize=self.queue_size)

        # Preload the request queue with all of the shards representing one 
        # epoch.
        for shard_spec in self.shard_schedule:
            self.request_queue.put(shard_spec)

        # Start the loader processes.
        self.loading_processes = []
        for loader_id in range(self.num_loaders):
            p = Process(target=self._do_preloading, args=(loader_id,))
            p.start()
            self.loading_processes.append(p)


    def _preload_iter(self):
        # Iterates through preloaded shards, as they become available
        return h.utils.iterate_queue(
            self.result_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=self.num_loaders
        )


    def __iter__(self):
        # Loads and yields shards onto (on gRAM).  This is the public interface
        # of the class which is used by the training loop to request the next
        # shard to be loaded onto the GPU to run forward / backward learning
        # iterations.
        for shard_spec, cpu_tensors in self._preload_iter():
            yield shard_spec, self._load(shard_spec, cpu_tensors)


    def _do_preloading(self, loader_id):
        """
        Target function that runs inside child loader processes.  Using 
        Queues, it simply takes shard specs, generates CPU-loaded shards
        by delegation to `self._preload()`, and places the result onto a queue
        for consumption in the main process.  Usually, don't override this,
        but rather, override the `_preload()` to which it delegates.  Notice
        that before 
        """
        self._setup(loader_id)
        for shard_spec in h.utils.iterate_queue(self.request_queue):
            self.result_queue.put((shard_spec, self._preload(shard_spec)))
        self.result_queue.put(StopIteration())



class BigramLoader(MultiLoader):

    def __init__(
        self, shard_schedule, num_loaders, bigram_path, 
        queue_size=1, device=None
    ):
        """
        Still an abstract class, the `BigramLoader` differentiates from the
        generic multi-processing loader `MultiLoader`, by committing to the
        fact that each shard will be calculated from bigram data, and hence
        that the main work needed in preloading is the preloading of key
        bigram tensors: `Nxx`, `Nx`, `Nxt`, and `N`.

        Concrete classes can override load to provide a more specific 
        set of one-time GPU calculations needed to calculate a desired loss
        function efficiently.  For example, `load()` can pre-calculate various
        terms that would be needed by the loss function.
        """
        self.bigram_path = bigram_path
        super(BigramLoader, self).__init__(
            shard_schedule=shard_schedule, num_loaders=num_loaders, 
            queue_size=queue_size, device=device
        )

    def _setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def _load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        Nxx, Nx, Nxt, N = cpu_data
        return Nxx.to(device), Nx.to(device), Nxt.to(device), N.to(device)

    def _preload(self, shard_spec):
        return self.bigram.load_shard(shard=shard_spec, device='cpu')


