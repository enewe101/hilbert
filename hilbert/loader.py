import os
import torch
from pytorch_categorical import Categorical
import hilbert as h
import scipy


class DenseLoader:
    """
    Base class for any LoaderModel that implements the common functionality,
    being, iteration over the preloaded shards.
    """

    def __init__(
        self,
        cooccurrence_path,
        shard_factor,
        include_unigrams=True,
        undersampling=None,
        smoothing=None,
        device=None,
        verbose=True,
    ):

        # Own your biz.
        self.cooccurrence_path = cooccurrence_path
        self.shard_factor = shard_factor
        self.include_unigrams = include_unigrams
        self.undersampling = undersampling
        self.smoothing = smoothing
        self.verbose = verbose
        self.device = h.utils.get_device(device)

        # these will be used for preloading and loading
        self.cooccurrence_sector = None
        self.preloaded_batches = None
        self.crt_batch_id = None

        # Preload everything into cRAM.
        self._preload()


    def _preload(self):
        """
        Preload iterates over a generator that generates preloaded batches.
        This fills up cRAM with all batches, but a smarter implementation could
        just buffer some batches if they don't fit in cRAM, hence separating
        this iteration over all batches from the generator of batches.
        """
        self.preloaded_batches = []
        if self.verbose:
            print('Preloading all shards...')
        for preload_data in self._preload_iter():
            self.preloaded_batches.append(preload_data)
        if self.verbose:
            print('Preloading complete!')


    def _preload_iter(self, *args, **kwargs):

        sector_factor = h.cooccurrence.CooccurrenceSector.get_sector_factor(
            self.cooccurrence_path)

        for i, sector_id in enumerate(h.shards.Shards(sector_factor)):

            if self.verbose:
                print('loading sector {}'.format(i))

            # Read the sector and transform as desired.
            self.cooccurrence_sector = h.cooccurrence.CooccurrenceSector.load(
                self.cooccurrence_path, sector_id)
            self.cooccurrence_sector.apply_w2v_undersampling(
                self.undersampling)
            self.cooccurrence_sector.apply_unigram_smoothing(
                self.smoothing)

            # Yield cRAM-preloaded shards from this sector
            for shard_id in h.shards.Shards(self.shard_factor):
                cooccurrence_data = (
                    self.cooccurrence_sector.load_relative_shard(
                        shard=shard_id, device='cpu'
                    )
                )
                unigram_data = None
                if self.include_unigrams:
                    unigram_data = (
                        self.cooccurrence_sector.load_relative_unigram_shard(
                            shard=shard_id, device='cpu'
                        )
                    )
                yield shard_id * sector_id, (cooccurrence_data, unigram_data)


    def _load(self, preloaded):
        batch_id, (cooccurrence_data, unigram_data) = preloaded
        cooccurrence_data = tuple(
            tensor.to(self.device) for tensor in cooccurrence_data)
        if self.include_unigrams:
            unigram_data = tuple(
                tensor.to(self.device) for tensor in unigram_data)
        return batch_id, (cooccurrence_data, unigram_data) 


    def __iter__(self):
        self.crt_batch_id = -1
        return self


    def __next__(self):
        self.crt_batch_id += 1
        if self.crt_batch_id >= len(self.preloaded_batches):
            raise StopIteration
        preloaded = self.preloaded_batches[self.crt_batch_id]
        return self._load(preloaded)


    def describe(self):
        return 'CooccurenceLoader'


    def __len__(self):
        return len(self.preloaded_batches)



class GPUSampleLoader:

    def __init__(
        self,
        cooccurrence_path,
        temperature=1,
        batch_size=100000,
        device=None,
        verbose=True
    ):
        self.cooccurrence_path = cooccurrence_path
        Nxx_data, I, J, Nx, Nxt = h.cooccurrence.CooccurrenceSector.load_coo(
            cooccurrence_path, verbose=verbose)

        self.temperature = temperature
        self.device = h.utils.get_device(device)

        # Calculate the probabilities and then temper them.
        # After tempering, probabilities are scores -- they don't sum to one
        # The Categorical sampler will automatically normalize them.
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
        self.yielded = False


    def sample(self, batch_size):

        # Allocate space for the positive and negative samples.
        # To index using tensor contents, torch requires they be int64.
        IJ_sample = torch.empty(
            (batch_size*2,2), device=self.device, dtype=torch.int64)

        # Randomly draw positive outcomes, and map them to ij pairs
        positive_choices = self.positive_sampler.sample(
            sample_shape=(batch_size,))
        IJ_sample[:batch_size,0] = self.I[positive_choices]
        IJ_sample[:batch_size,1] = self.J[positive_choices]

        # Randomly draw negative outcomes.  These outcomes are already ij
        # indices, so unlike positive outcomes they don't need to be mapped.
        IJ_sample[batch_size:,0] = self.negative_sampler.sample(
            sample_shape=(batch_size,))
        IJ_sample[batch_size:,1] = self.negative_sampler_t.sample(
            sample_shape=(batch_size,))

        return IJ_sample


    def __len__(self):
        return 1


    def __iter__(self):
        self.yielded = False
        return self


    def __next__(self):
        if self.yielded:
            raise StopIteration
        self.yielded = True
        return self.sample(self.batch_size), None


    def describe(self):
        s = '\tcooccurrence_path = {}\n'.format(self.cooccurrence_path)
        s += '\tbatch_size = {}\n'.format(self.batch_size)
        s += '\ttemperature = {}\n'.format(self.temperature)
        return s
        


class CPUSampleLoader:

    def __init__(
        self,
        cooccurrence_path,
        temperature=1,
        batch_size=100000,
        device=None,
        verbose=True
    ):

        # Ownage.
        self.cooccurrence_path = cooccurrence_path
        self.batch_size = batch_size
        self.yielded = False

        cooc = h.cooccurrence.Cooccurrence.load(cooccurrence_path)
        Nxx, Nx, Nxt, N = cooc.Nxx, cooc.Nx, cooc.Nxt, cooc.N

        self.temperature = temperature
        self.device = h.utils.get_device(device)

        # Calculate the probabilities and then temper them.
        # After tempering, probabilities are scores -- they don't sum to one
        # The Categorical sampler will automatically normalize them.
        Pi = Nx / Nx.sum()
        Pi_tempered = (Pi ** (1/temperature)).view((-1,))
        Pj = Nxt / Nx.sum()
        Pj_tempered = (Pj ** (1/temperature)).view((-1,))

        # Calculate the exponential of PMI for ij pairs, according to the
        # corpus. These are needed because we are importance-sampling
        # the corpus distribution using the independent distribution.
        self.exp_pmi = Nxx.multiply(
            1/N).multiply(1/Pi.numpy()).multiply(1/Pj.numpy()).tolil()

        # Make samplers for the independent distribution.
        self.I_sampler = Categorical(Pi_tempered, device='cpu')
        self.J_sampler = Categorical(Pj_tempered, device='cpu')


    def sample(self, batch_size):
        # Randomly draw independent outcomes.
        IJ = torch.zeros((batch_size, 2), dtype=torch.int64)
        IJ[:,0] = self.I_sampler.sample(sample_shape=(batch_size,))
        IJ[:,1] = self.J_sampler.sample(sample_shape=(batch_size,))
        exp_pmi = torch.tensor(
            self.exp_pmi[IJ[:,0],IJ[:,1]].toarray().reshape((-1,)),
            dtype=torch.float32, device=self.device
        )
        return IJ, {'exp_pmi':exp_pmi}


    def __len__(self):
        return 1


    def __iter__(self):
        self.yielded = False
        return self


    def __next__(self):
        if self.yielded:
            raise StopIteration
        self.yielded = True
        return self.sample(self.batch_size)


    def describe(self):
        s = '\tcooccurrence_path = {}\n'.format(self.cooccurrence_path)
        s += '\tbatch_size = {}\n'.format(self.batch_size)
        s += '\ttemperature = {}\n'.format(self.temperature)
        return s



class DependencyLoader:

    def __init__(
        self,
        dependency_path,
        batch_size=100000,
        device=None,
        verbose=True
    ):
        self.dependency_path = dependence_path
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.dependency = h.dependency.DependencyCorpus(dependency_path)


    def sample_batch(self, batch_size, pointer):
        device = h.utils.get_device(self.device)
        start = pointer * batch_size
        stop = (pointer + 1) * batch_size
        positives = self.dependency.data[start:stop].to(device)
        negatives = sample_negatives(positives)
        return positives, negatives


    def __iter__(self):
        self.pointer = 0
        return self


    def __next__(self):
        positives, negatives = sample_batch(self.batch_size, self.pointer)
        self.pointer += 1
        return self.pointer, (positives, negatives)


class DependencySampler:

    def __init__(self, embeddings=None, V=None, W=None, architecture='flat'):
        """
        Either use embeddings, and pass a hilbert.embeddings.Embeddings object,
        or use V and W and pass tensors representing vectors and covectors
        respectively, each having one word-embedding per row.
        architecture can be "flat" for a simple flat softmax, or 
        "adaptive" for an adaptive softmax.
        """
        err = False
        if embeddings is not None:
            self.V = embeddings.V
            self.W = embeddings.W
            err = True
        elif V is not None and W is not None:
            self.V = V
            self.W = W
        else:
            err = True
        if err:
            raise ValueError(
                "Must provide either ``embeddings`` OR "
                "V and W, but not embeddings AND V and W."
            )


    def sample(self, positives, mask):

        # Need to add ability to sample the root.  Include it at the end
        # of the sentence.  Include root in dictionary and embeddings.
        assert mask.dtype == torch.uint8

        # Drop tags for now
        positives = positives[:,0:2,:] 
        negatives = torch.zeros_like(positives)
        negatives[:,0,:] = positives[:,0,:]
        sentence_length = len(positives[0][0])
        batch_size = len(positives)

        reflected_mask = (mask.unsqueeze(1) * mask.unsqueeze(2)).byte()
        words = positives[:,0,:]
        vectors = self.V[words]
        covectors = self.W[words]
        energies = torch.bmm(covectors, vectors.transpose(1,2))
        identities_idx = (
            slice(None), range(sentence_length), range(sentence_length))
        energies[identities_idx] = torch.tensor(-float('inf'))
        energies[1-reflected_mask] = torch.tensor(-float('inf'))

        unnormalized_probs = torch.exp(energies)
        unnormalized_probs_2d = unnormalized_probs.view(-1, sentence_length)
        unnormalized_probs_2d[1-mask.reshape(-1),:] = 1
        totals = unnormalized_probs_2d.sum(dim=1, keepdim=True)
        probs = unnormalized_probs_2d / totals

        sample = torch.distributions.Categorical(probs).sample()
        reshaped_sample = sample.reshape(-1, 1, sentence_length)

        idx1 = [i for i in range(batch_size) for j in range(sentence_length)]
        idx2 = [j for i in range(batch_size) for j in range(sentence_length)]

        negatives[idx1,1,idx2] = positives[idx1, 0, sample]

        negatives[:,1,:][1-mask] = 0

        import pdb; pdb.set_trace()

        return negatives



