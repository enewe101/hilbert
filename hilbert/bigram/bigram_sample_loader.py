from pytorch_categorical import Categorical
import time
import hilbert as h
import torch


class BigramSampleLoader(h.generic_datastructs.Describable):

    def __init__(
        self, bigram_path, sector_factor, temperature=1,
        batch_size=100000, batches_per_epoch=1000, device=None, verbose=True
    ):
        self.bigram_path = bigram_path
        Nxx_data, I, J, Nx, Nxt = h.generic_datastructs.get_Nxx_coo(
            bigram_path, sector_factor, verbose=verbose)

        self.temperature = temperature
        self.device = device or h.CONSTANTS.RC['device']

        # Calculate the probabilities and then temper them.
        # After tempering, probabilities are scores -- they don't sum to one
        # This is okay because the Categorical sampler automatically normalizes.
        Pi = Nx.view((-1,)) / Nx.sum()
        Pi_raised = Pi**(1/temperature - 1)
        Pi_tempered = Pi_raised * Pi

        Pj = Nxt.view((-1,)) / Nx.sum()
        Pj_raised = Pj**(1/temperature - 1)
        Pj_tempered = Pj_raised * Pj

        Nxx_tempered = Nxx_data * Pi_raised[I.long()] * Pj_raised[J.long()]

        self.positive_sampler = Categorical(Nxx_tempered, device=self.device)
        self.negative_sampler = Categorical(Pi_tempered, self.device)
        self.negative_sampler_t = Categorical(Pj_tempered, device=self.device)

        self.I = I.to(self.device)
        self.J = J.to(self.device)

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.batch_num = None


    def sample(self, batch_size):

        # Allocate space for the positive and negative samples.
        # To index using tensor contents, torch requires they be int64.
        IJ_sample = torch.empty(
            (batch_size*2,2), device=self.device, dtype=torch.int64)

        # Randomly draw positive outcomes, and map those outcomes to ij pairs
        positive_choices = self.positive_sampler.sample(
            sample_shape=(batch_size,))
        IJ_sample[:batch_size,0] = self.I[positive_choices]
        IJ_sample[:batch_size,1] = self.J[positive_choices]

        # Randomly draw positive outcomes.  These outcomes are already ij
        # indices, so unlike positive samples don't need to be mapped.
        IJ_sample[batch_size:,0] = self.negative_sampler.sample(
            sample_shape=(batch_size,))
        IJ_sample[batch_size:,1] = self.negative_sampler_t.sample(
            sample_shape=(batch_size,))

        return IJ_sample


    def __len__(self):
        return self.batches_per_epoch


    def __iter__(self):
        self.batch_num = -1
        return self


    def __next__(self):
        self.batch_num += 1
        if self.batch_num > self.batches_per_epoch:
            raise StopIteration
        return self.sample(self.batch_size), None


    def describe(self):
        s = '\tbigram_path = {}\n'.format(self.bigram_path)
        s += '\tbatch_size = {}\n'.format(self.batch_size)
        s += '\ttemperature = {}\n'.format(self.temperature)
        return s
        


