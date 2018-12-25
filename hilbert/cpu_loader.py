import hilbert as h
try:
    import torch
    from torch.multiprocessing import Queue, Process
except ImportError:
    torch = None
    Queue, Process = None, None


def construct_bigram_cpu_loader(shard_schedule, bigram_path, num_workers):
    preloaders = [
        h.preloader.BigramPreloader(bigram_path) for w in range(num_workers)]
    return MultiLoader(shard_schedule, preloaders, bigram_path)



class Loader(object):

    def __init__(self, shard_schedule):
        self.shard_schedule = shard_schedule


    def __iter__(self):
        for shard_spec, cpu_tensors in self.preload_iter():
            yield shard_spec, load(shard_spec, cpu_tensors)


    def load(self, shard_spec, cpu_tensors):
        raise NotImplementedError(
            'Subclasses should override `Loader.load()`, which must '
            'return an iterable yielding pairs like `(shard_spec, gpu_data)`, '
            'where `gpu_data` is some collection of gpu-tensors (e.g. tuple of '
            'tensors, dictionary, or single tensor).'
        )


    def preload_iter(self):
        for shard_spec in shard_schedule:
            yield shard_spec, self.preload(shard_spec)


    def preload(self, shard_spec):
        raise NotImplementedError(
            'Subclasses should override `Loader.preload()`, which must '
            'return an iterable yielding pairs like `(shard_spec, cpu_data)`, '
            'where `cpu_data` is some collection of cpu-tensors (e.g. tuple of '
            'tensors, dictionary, or single tensor).'
        )





class MultiLoader(Loader):

    def __init__(self, shard_schedule, num_loaders, device=None):
        """
        Like `Loader`, `MultiLoader` is an iterable that yields `(shard_spec,
        gpu_tensors)` pairs.  It preloads shards as cpu-tensors, so that the
        gpu-tensors can be loaded in time to keep the GPU busy.

        INPUTS
        ``````
        `shard_schedule`    
            An iterator that yields shard_specs, which are
            2-tuples of `SliceObjects`
        """
        self.shard_schedule = shard_schedule
        self.num_loaders = num_loaders
        self._start_preloading()
        self.device = device


    def setup(self, preloader_id):
        """
        This function is called once when running in a child process.  It allows
        you to execute more __init__-like code, but to do so *after* being
        transmitted through a pipe to a child process.  That's useful if you
        need to load a lot of data, because you might as load it after 
        transmitting through the pipe.  Also, it is useful for things not
        likely to pickle well (e.g. a database connection?). 
        """
        pass


    def _start_preloading(self):
        # The preloaders are responsible for building cpu-loaded shards in
        # background processes, so that they are ready to be loaded to the
        # GPU when they are needed.  This function starts the preloader
        # processes.  First, we make some queues to handle communication.
        # One queue holds all of the jobs to be done (all the shards to be made)
        # Which we fill up with one epoch's worth of shard_ids.  The other
        # collects built shards.  Once the queues are ready, start up the 
        # preloader processes.

        # Make the queues.
        self.request_queue = Queue()
        self.result_queue = Queue(maxsize=1)

        # Load the work.
        for shard_spec in self.shard_schedule:
            self.request_queue.put(shard_spec)

        # Start all the loader processes
        self.loading_processes = []
        for loader_id in range(self.num_loaders):
            p = Process(target=self.do_preloading, args=(loader_id,))
            p.start()
            self.loading_processes.append(p)


    def preload_iter(self):
        # Turn the result_queue into an iterator, and return it
        return h.utils.iterate_queue(
            self.result_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=self.num_loaders
        )


    def __iter__(self):
        for shard_spec, cpu_tensors in self.preload_iter():
            yield shard_spec, self.load(shard_spec, cpu_tensors)


    def do_preloading(self, loader_id):
        self.setup(loader_id)
        for shard_spec in h.utils.iterate_queue(self.request_queue):
            self.result_queue.put((shard_spec, self.preload(shard_spec)))
        self.result_queue.put(StopIteration())





class BigramLoader(MultiLoader):


    def __init__(self, shard_schedule, preloaders, bigram_path, device=None):
        self.bigram_path = bigram_path
        super(MockLoader, self).__init__(shard_schedule, preloaders, device)

    def setup(self, loader_id):
        self.bigram = h.bigram.Bigram.load(self.bigram_path)

    def load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        Nxx, Nx, Nxt, N = cpu_data
        return Nxx.to(device), Nx.to(device), Nxt.to(device), N.to(device)

    def preload(self, shard_spec):
        return self.bigram.load_shard(shard=shard_spec, device='cpu')

