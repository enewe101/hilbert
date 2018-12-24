from torch.multiprocessing import Process, Queue
import hilbert as h


class GPULoader(object):

    def __init__(self, cpu_loader, device=None):
        self.device = device
        self.cpu_loader = cpu_loader

    def load(self, shard_id, cpu_data):
        raise NotImplementedError(
            'Subclasses should override `GPUloader.load()`.')

    def __iter__(self):
        for shard_id, cpu_data in self.cpu_loader:
            yield self.load(shard_id, cpu_data)



class PPMISharder(GPULoader):

    def describe(self):
        s = 'PPMI Sharder\n'
        return s


    def load(self, shard_spec, cpu_data):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        self.M = h.corpus_stats.calc_PMI((
            cpu_data['Nxx'].to(device), 
            cpu_data['Nx'].to(device), 
            cpu_data['Nxt'].to(device), 
            cpu_data['N'].to(device)
        ))
        self.M = torch.clamp(self.M, min=0)


    def _get_loss(self, M_hat, shard):
        return self.criterion(M_hat, shard, self.M)


