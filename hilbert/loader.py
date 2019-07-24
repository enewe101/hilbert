import os
import torch
from pytorch_categorical import Categorical
import hilbert as h
import scipy
import numpy as np


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
        verbose=True,
        min_cooccurrence_count=None,
    ):
        self.cooccurrence_path = cooccurrence_path
        Nxx_data, I, J, Nx, Nxt = h.cooccurrence.CooccurrenceSector.load_coo(
            cooccurrence_path, min_cooccurrence_count=min_cooccurrence_count, verbose=verbose)

        self.temperature = temperature
        self.device = h.utils.get_device(device)

        # Calculate the probabilities and then temper them.
        # After tempering, probabilities are scores -- they don't sum to one
        # The Categorical sampler will automatically normalize them.
        Pi = Nx.view((-1,)) / Nx.sum()
        Pi_raised = Pi ** (1 / temperature - 1)
        Pi_tempered = Pi_raised * Pi

        Pj = Nxt.view((-1,)) / Nx.sum()
        Pj_raised = Pj ** (1 / temperature - 1)
        Pj_tempered = Pj_raised * Pj

        Nxx_tempered = Nxx_data * Pi_raised[I.long()] * Pj_raised[J.long()]

        self.positive_sampler = Categorical(Nxx_tempered, device=self.device)
        self.negative_sampler = Categorical(Pi_tempered, device=self.device)
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

class GibbsSampleLoader:
    def __init__(
            self,
            cooccurrence_path,
            learner,
            temperature=1,
            batch_size=1000,
            gibbs_iteration=1,
            get_distr=False,
            device=None,
            verbose=True,
            min_cooccurrence_count=None,
            ):

        # Ownage.
        self.cooccurrence_path = cooccurrence_path
        self.learner = learner
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = h.utils.get_device(device)
        self.yielded = False
        self.adaptive_softmax = False

        self.get_distr = get_distr
        self.gibbs_iteration = gibbs_iteration

        Nxx_data, I, J, Nx, Nxt = h.cooccurrence.CooccurrenceSector.load_coo(
            cooccurrence_path, verbose=verbose, min_cooccurrence_count=min_cooccurrence_count)

        # Calculate the probabilities and then temper them.
        # After tempering, probabilities are scores -- they don't sum to one
        self.Pi = Nx.view((-1,)) / Nx.sum()
        Pi_raised = self.Pi ** (1 / self.temperature - 1)
        # Pi_tempered = Pi_raised * self.Pi

        self.Pj = Nxt.view((-1,)) / Nx.sum()
        Pj_raised = self.Pj ** (1 / self.temperature - 1)
        # Pj_tempered = Pj_raised * self.Pj

        self.Pij = Nxx_data * Pi_raised[I.long()] * Pj_raised[J.long()]

        self.I = I.to(self.device)
        self.J = J.to(self.device)

        self.Pi = self.Pi.to(self.device)
        self.Pj = self.Pj.to(self.device)

        self.positive_sampler = Categorical(self.Pij, device=self.device)

    def get_batch_words(self, batch_id, dictionary):
        # help function for investigating problematic pairs of words
        pos_pairs = []
        neg_pairs = []

        boundary = self.batch_size

        for i, ij in enumerate(batch_id):
            if i < boundary:
                pos_pairs.append((dictionary.get_token(ij[0]), dictionary.get_token(ij[1])))
            else:
                neg_pairs.append((dictionary.get_token(ij[0]), dictionary.get_token(ij[1])))

        return pos_pairs, neg_pairs

    def batch_probs(self, _2dprobs_prenorm, I_flag=True):
        """
        Help function for sampling negative samples by calculating the probabilities of the batch
        :param _2dprobs_prenorm:
        :return: indices of negative samples
        """
        if self.adaptive_softmax:
            # to be implemented
            pass
        else:
            # normalized
            _2dprobs = _2dprobs_prenorm/_2dprobs_prenorm.sum(dim=1)[:,None]
            if torch.isnan(_2dprobs).any():
                raise ValueError("detected nan in probs!")
            negative_sampler = torch.distributions.categorical.Categorical(_2dprobs)
            negative_samples_idx = negative_sampler.sample()    # sample 1 unit from the conditional distribution
            if I_flag:
                negative_samples = self.J[negative_samples_idx]
            else:
                negative_samples = self.I[negative_samples_idx]
        return negative_samples.long()

    def gibbs_stepping(self, condition_on_idx, is_vector=True):
        """
        Run one Gibbs stepping by conditioning on given words.
        e.g. P( j' | i ) = Pj' * exp(<i | j'>)

        :param condition_on_idx: all word vectors conditioning on
        :param is_vector: True if condition on words are samples of vector
        :return: negative samples
        """

        if is_vector:
            # condition on words are from I, computing J given I
            model_pmi = self.learner.V[condition_on_idx] @ self.learner.W.t()
            _2d_posterior_dist = self.Pj * torch.exp(model_pmi)  # without Adaptive softmax
            if torch.isnan(_2d_posterior_dist).any():
                raise ValueError("In gibbs stepping, detected nan in probs!")
            negative_samples = self.batch_probs(_2d_posterior_dist, I_flag=True)
        else:
            # condition on words are from I, computing I given J
            model_pmi = self.learner.W[condition_on_idx] @ self.learner.V.t()
            _2d_posterior_dist = self.Pi * torch.exp(model_pmi)  # without Adaptive softmax
            if torch.isnan(_2d_posterior_dist).any():
                raise ValueError("In gibbs stepping, detected nan in probs!")
            negative_samples = self.batch_probs(_2d_posterior_dist, I_flag=False)

        return negative_samples

    def iterative_gibbs_sampling(self, positive_sample, input_I_flag=True, steps=1, get_distr=False):
        """
        Run Gibbs sampling for number of iterations
        :param positive_sample:
        :param input_I_flag:
        :param steps:
        :param get_distr: If true, return the distribution after number of iterations.
        :return: either negative samples or the distribution of the negative samples depending on the get_distr flag
        """
        # updated_word_choices = None
        # negative_samples = positive_J

        if input_I_flag:
            # update j' given i in the first iteration
            _I = positive_sample
            _J = None
        else:
            # update i' given j in the first iteration
            _J = positive_sample
            _I = None

        for i in range(steps):
            I_flag = (i % 2 == 0) if input_I_flag else ((i+1) % 2 == 0)

            if I_flag:
                _J = self.gibbs_stepping(_I, is_vector=I_flag)
            else:
                _I = self.gibbs_stepping(_J, is_vector=I_flag)

        if get_distr:
            # at the last iteration, get the distribution instead of samples
            j_distr = self.Pj * torch.exp(self.learner.V[_I] @ self.learner.W.t())
            i_distr = self.Pi * torch.exp(self.learner.W[_J] @ self.learner.V.t())
            return (i_distr, j_distr)

        else:
            negative_samples = torch.empty(
                (positive_sample.shape[0], 2), device=self.device, dtype=torch.int64)

            negative_samples[:, 0] = _I
            negative_samples[:, 1] = _J
            return negative_samples

    def distribution_only(self, batch_size, toy):
        """

        :param positive_samples: indices of I that are drawn from the corpus distribution
        :param toy: If True, run one step of Gibbs sampling
        :return: tuple of positive and negative distributions
        """
        positive_choices_idx = self.positive_sampler.sample(
            sample_shape=(batch_size,))
        # actual embeddings of choices
        positive_I = self.I[positive_choices_idx].long()

        negative_sample_distrs = self.iterative_gibbs_sampling(
            positive_I,
            input_I_flag=True,
            steps=self.gibbs_iteration * 2,
            get_distr=True)

        return self.Pij, negative_sample_distrs

    def sample(self, batch_size, toy=False):
        '''

        Gibbs sampler takes positive samples (i, j) drawn from corpus distribution Pij,
        and by fixing i, sample a new j' according to the conditional model distribution pj*e^(i dot j')

        :param batch_size: number of unit in positive sample
        :return: IJ samples or a tuple (Pij in data distribution, Pij in model distribution) where Pij model is
        represented by p(i|j) and p(j|i) after number of Gibbs sampling iterations.
        '''

        # without Adaptive softmax
        # Randomly draw positive outcomes, and map them to ij pairs
        positive_choices_idx = self.positive_sampler.sample(
            sample_shape=(batch_size,))
        # actual embeddings of choices
        positive_samples = (self.I[positive_choices_idx].long(), self.J[positive_choices_idx].long())
        positive_I, positive_J = positive_samples
        IJ_sample = torch.empty(
            (self.batch_size * 2, 2), device=self.device, dtype=torch.int64)


        IJ_negative_samples = self.iterative_gibbs_sampling(positive_I, input_I_flag=True,
                                                            steps=self.gibbs_iteration * 2,
                                                            )
        IJ_sample[:self.batch_size, 0] = positive_I
        IJ_sample[:self.batch_size, 1] = positive_J
        IJ_sample[self.batch_size:, :] = IJ_negative_samples

        return IJ_sample  # they are indices




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
        min_cooccurrence_count = None,
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

class ArcLabelSampleLoader:

    def __init__(
        self,
        cooccurrence_path,
        temperature=1,
        batch_size=100000,
        device=None,
        min_cooccurrence_count=None,
        verbose=True
    ):

        """
        Loads samples by first sampling a dependency arc type.

        `Nk.npy`
            An array (vector) consisting of the counts for the number
            of times each dependency arc type appears in the corpus.
        """

        self.cooccurrence_path = cooccurrence_path
        self.batch_size = batch_size
        self.yielded = False

        Nk = np.load(os.path.join(cooccurrence_path, 'Nk.npy'))
        self.num_labels = Nk.size
        Pk = torch.from_numpy(Nk / Nk.sum())

        self.K_sampler = Categorical(Pk, device='cpu')
        
        self.I_sampler = [None] * self.num_labels
        self.J_sampler = [None] * self.num_labels

        unigram = h.unigram.Unigram.load(cooccurrence_path, verbose=verbose)
        vocab = len(unigram.dictionary.tokens)
        
        self.exp_pmi = [None] * self.num_labels

        for k in range(self.num_labels):
            Nxx_str = "Nxx_" + str(k) + ".npz"
            Nxx_tmp = scipy.sparse.load_npz(os.path.join(cooccurrence_path, Nxx_str)).tolil()
            cooc = h.cooccurrence.Cooccurrence(unigram, Nxx_tmp, marginalize=True, verbose=verbose)
            Nxx, Nx, Nxt, N = cooc.Nxx, cooc.Nx, cooc.Nxt, cooc.N
            Pi = Nx / Nx.sum()
            Pj = Nxt / Nx.sum()
            Pi_tempered = (Pi ** (1/temperature)).view((-1,))
            Pj_tempered = (Pj ** (1/temperature)).view((-1,))
            self.exp_pmi[k] = Nxx.multiply(
                1/N).multiply(1/Pi.numpy()).multiply(1/Pj.numpy()).tolil()
            self.I_sampler[k] = Categorical(Pi_tempered, device='cpu')
            self.J_sampler[k] = Categorical(Pj_tempered, device='cpu')


        self.temperature = temperature
        self.device = h.utils.get_device(device)

    def sample(self, batch_size):
        k = self.K_sampler.sample(sample_shape=(batch_size,))
        counter = np.zeros(self.num_labels)
        IJK = torch.zeros((batch_size,3), dtype=torch.int64)
        exp_pmi = [None] * self.num_labels

        for m in range(batch_size):
            counter[k[m]] += 1

        offset = 0
        for m in range(counter.size):
            num_samples = int(counter[m])
            indices = slice(offset, offset+num_samples)
            IJK[indices,2] = torch.tensor([m]).repeat(num_samples)
            IJK[indices,0] = self.I_sampler[m].sample(sample_shape=(num_samples,))
            IJK[indices,1] = self.J_sampler[m].sample(sample_shape=(num_samples,))
            
            exp_pmi[m] = torch.tensor(
                self.exp_pmi[m][IJK[indices,0],IJK[indices,1]].toarray().reshape((-1,)),
                dtype=torch.float32, device = self.device
            )

            offset += num_samples
        
        #TODO Figure out what this is supposed to look like
        return IJK, {'exp_pmi':exp_pmi}

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

def pad_sentence(sent, length):
    return [pad(sent[0], length), pad(sent[1], length), pad(sent[2], length)]


def pad(lst, length):
    return lst + [h.CONSTANTS.PAD] * (length - len(lst))


class DependencyLoader:

    def __init__(
        self,
        dependency_path,
        batch_size=100000,
        device=None,
        verbose=True
    ):
        self.dependency_path = dependency_path
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.dependency = h.dependency.DependencyCorpus(dependency_path)


    def sample_batch(self, batch_size, pointer):
        device = h.utils.get_device(self.device)
        start = pointer * batch_size
        stop = (pointer + 1) * batch_size

        if start  >= len(self.dependency.sort_idxs):
            raise IndexError(
                "pointing at example {}, but data has only {} examples"
                .format(start, len(self.dependency.sort_idxs))
            )

        idxs = self.dependency.sort_idxs[start:stop]
        sentence_lengths = self.dependency.sentence_lengths[idxs]
        max_length = torch.max(sentence_lengths).item()

        positives = torch.tensor([
            pad_sentence(self.dependency.sentences[idx], max_length)
            for idx in idxs
        ])
        mask = self.generate_mask(sentence_lengths, max_length)

        return positives, mask


    def generate_mask(self, sentence_lengths, max_length):
        mask = torch.zeros((len(sentence_lengths), max_length))
        for row, sentence_length in enumerate(sentence_lengths):
            mask[row, :sentence_length] = 1

        return mask


    def __iter__(self):
        self.pointer = 0
        return self


    def __next__(self):
        try:
            positives, mask = self.sample_batch(self.batch_size, self.pointer)
        except IndexError:
            raise StopIteration()
        self.pointer += 1
        return self.pointer-1, (positives, mask)




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

        negatives[idx1,1,idx2] = sample

        negatives[:,1,:][1-mask] = h.dependency.PAD
        negatives[:,1,0] = h.dependency.PAD

        return negatives



