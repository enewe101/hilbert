import hilbert as h
from torch.multiprocessing import Queue, Process


def construct_bigram_cpu_loader(shard_schedule, bigram_path, num_workers):
    preloaders = [
        h.preloader.BigramPreloader(bigram_path) for w in range(num_workers)]
    return MultiCPULoader(shard_schedule, preloaders)



class CPULoader(object):

    def __init__(self, shard_schedule):
        self.shard_schedule = shard_schedule

    def __iter__(self):
        raise NotImplementedError(
            'Subclasses should override `CPULoader.__iter__()`, and they '
            'should do so by yielding `(shard_spec, cpu_data)` 2-tuples.'
        )


class MultiCPULoader(CPULoader):

    def __init__(self, shard_schedule, preloaders):
        """
        This CPULoader yields prepared cpu-loaded shards, which are a precursor
        to gpu-loaded shards.  It prepares the cpu-loaded shards using worker
        processes with which it shares memory.

        INPUTS
        ``````
        `shard_schedule`    
            An iterator that yields shard_specs, which are
            2-tuples of `SliceObjects`

        `preloaders`    
            Instances satisfying the `Preloader` interface, which
            build sets of tensors in shared memory 

        The MultiCPULoader  is an iterator that yields `(shard_id,
        cpu_tensors)` tuples, and hides the fact that workers in the background
        prepare the next items for iteration.

        Before entering into the iteration context, the  master puts shard IDs
        on a queue.  Workers pull shard_IDs and assemble the cpu tensors, then
        put them onto a `result_queue` where they are passed back to the 
        main process by keeping the tensors in shared memory, and passing
        references to them in a queue (thanks to `torch.multiprocessing`)

        (The master process would then normally load them onto the GPU, and do
        any preliminary GPU operations to finish preparing the shard.)
        """
        self.shard_schedule = shard_schedule
        self.num_preloaders = len(preloaders)
        self.preloaders = preloaders
        self._start_preloaders()


    def _start_preloaders(self):
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
        for shard_id in self.shard_schedule:
            self.request_queue.put(shard_id)

        # Start all the loader processes
        self.loading_processes = []
        for loader_id, preloader in enumerate(self.preloaders):
            p = Process(
                target=MultiCPULoader.preload,
                args=(loader_id,self.request_queue,self.result_queue,preloader)
            )
            p.start()
            self.loading_processes.append(p)


    def __iter__(self):
        # Turn the result_queue into an iterator, and return it
        return h.utils.iterate_queue(
            self.result_queue, stop_when_empty=False, sentinal=StopIteration,
            num_sentinals=self.num_preloaders
        )


    @staticmethod
    def preload(loader_id, request_queue, result_queue, preloader):
        preloader.setup(loader_id)
        for shard_id in h.utils.iterate_queue(request_queue):
            result_queue.put((shard_id, preloader[shard_id]))
        result_queue.put(StopIteration())


