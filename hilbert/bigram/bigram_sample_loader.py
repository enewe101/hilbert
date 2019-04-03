
import time
import hilbert as h
import torch

class BigramSampleLoader:

    def __init__(
        self, bigram_path, sector_factor, batch_size=1000, device=None
    ):
        Nxx_data, I, J, Nx, Nxt = h.generic_datastructs.get_Nxx_coo(
            bigram_path, sector_factor)

        device = device or h.CONSTANTS.RC['device']
        self.positive_sampler = torch.distributions.Categorical(
            probs=Nxx_data.to(device))

        self.negative_sampler = torch.distributions.Categorical(
            probs=Nx.to(device))
        self.negative_sampler_t = torch.distributions.Categorical(
            probs=Nxt.to(device))

        self.I = I.to(device)
        self.J = J.to(device)

        self.batches_per_epoch = Nx.sum() / batch_size
        self.batch_num = None


    def sample(self, batch_size):

        # Allocate space for the positive and negative samples
        I = torch.empty((2*batch_size,))
        J = torch.empty((2*batch_size,))

        # Randomly draw positive and negative categorical outcomes
        positive_choices = self.positive_sampler.sample(
            sample_shape=(batch_size,))
        negative_I = self.negative_sampler.sample(sample_shape=(batch_size,))
        negative_J = self.negative_sampler.sample_t(sample_shape=(batch_size,))

        # Map outcomes to row and column indices, and store them with positives
        # followed by negatives.
        I[:batch_size] = self.I[positive_choices]
        J[:batch_size] = self.J[positive_choices]
        I[batch_size:] = self.I[negative_I]
        J[batch_size:] = self.J[negative_J]

        return I, J


    def __iter__(self):
        self.batch_num = -1
        return self


    def next(self):
        self.batch_num += 1
        if self.batch_num > self.batches_per_epoch:
            raise StopIteration
        return self.sample(self.batch_size)

        


