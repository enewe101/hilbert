try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    torch = None
    sparse = None

import hilbert as h


class FDelta:

    def __init__(self, bigram, M, update_density=1, device=None):
        self.bigram = bigram
        self.M = M
        self.device = device or h.CONSTANTS.MATRIX_DEVICE
        self.update_density = update_density
        self.last_shard = None


    def calc_shard(self, M_hat, shard):
        if shard != self.last_shard:
            self.load_shard(M_hat, shard)
        bit_mask = torch.rand_like(M_hat) > (1 - self.update_density)
        bit_mask = bit_mask.type(torch.float32)
        return self._calc_shard(M_hat, shard) * bit_mask


    def load_shard(self, M_hat, shard):
        pass


    def _calc_shard(self, M_hat, shard):
        raise NotImplementedError('Subclasses must override `_calc_shard`.')



class DeltaMSE(FDelta):
    def _calc_shard(self, M_hat, shard):
        return self.M[shard] - M_hat
        

class NewEpoch(Exception):
    """Marker for end of epoch"""
    def __init__(self, epoch_num, *args, **kwargs):
        self.epoch_num = epoch_num
        super().__init__('Epoch #{}'.format(epoch_num))


class NoMoreSamples(Exception):
    """Signals that there are no more samples left."""


class SampleReader:
    """
    Replays samples based on a trace from running standard word2vec.
    Call next_sample() to get the next sample, consisting of a positive
    example and k negative examples.  Each example is a triple of two token
    ids and a value indicating whether it is a positive (1) or negative (0)
    example.  Treat as an iterable to iterate through all sampless.

    If signal_epochs is True, then a special NewEpoch exception is raised
    whenever a new epoch is started, so that an external process can intervene.

    When all samples are exhausted, raises NoMoreSamples, even if it is being
    used like an iterator (for self-consistency)
    """

    def __init__(
        self,
        sample_path,
        dictionary,
        signal_epochs=True,
        verbose=True
    ):
        self.sample_path = sample_path
        self.dictionary = dictionary
        self.signal_epochs = signal_epochs
        self.verbose = verbose
        self.sample_file = open(self.sample_path)

        # Internal pointers, state tracking, in-memory samples, and
        # partialy-built samples
        self.epoch_num = 0
        self.samples = []
        self.cur_sample = []
        self.sample_pointer = 0
        self.no_more_samples = False


    def __iter__(self):
        while True:
            yield self.next_sample()


    def next_sample(self):

        while True:

            if self.sample_pointer >= len(self.samples):
                if self.no_more_samples:
                    raise NoMoreSamples()
                self.sample_pointer = 0
                self.samples = []
                self.read_some_samples()

            next_item = self.samples[self.sample_pointer]

            # If we hit an epoch marker, signal it, or just step over it.
            if isinstance(next_item, NewEpoch):
                self.sample_pointer += 1
                if self.signal_epochs:
                    raise next_item
                else:
                    continue

            self.sample_pointer += 1
            return next_item


    def read_some_samples(self):
        for i in range(1000):
            try:
                line = next(self.sample_file)

            except StopIteration:
                self.finalize_epoch()
                self.no_more_samples = True
                break

            if line.startswith('Epoch'): 
                self.finalize_epoch()
                continue

            self.read_one_line(line)


    def finalize_epoch(self):
        self.finalize_sample()
        self.samples.append(NewEpoch(self.epoch_num))
        if self.verbose:
            print('Epoch #{}'.format(self.epoch_num))
        self.epoch_num += 1


    def read_one_line(self, line):
        """
        Read one line.  Possibly start a building a new sample.  Add the
        instance from this line to currently-accumulating sample.
        """
        fields = self.parse_one_line(line)
        label = fields[2]
        self.maybe_finalize_sample(label)
        self.cur_sample.append(fields)


    def parse_one_line(self, line):
        fields = line.strip().split('\t')
        return [
            self.dictionary.get_id(fields[0]),
            self.dictionary.get_id(fields[1]),
            int(fields[2])
        ] + fields[3:]


    def maybe_finalize_sample(self, label):
        # Positive samples, having label == 1, mark the start of a new sample.
        if label == 1:
            self.finalize_sample()
        # Meanwhile, validate the value of the label.
        elif label != 0:
            raise ValueError(
                'Label must be either 0 or 1, got {}'.format(label))


    def finalize_sample(self):
        if len(self.cur_sample) > 0:
            self.samples.append(self.cur_sample)
        self.cur_sample = []


    def __del__(self):
        self.sample_file.close()




class DeltaW2VSamples:
    """
    A delta calculator that can "replay" each sample taken during training of
    word2vec, so that they are applied as updates using the HilbertEmbedder
    architecture.  These samples could be taken from a trace of the standard
    word2vec algorithms samples, useful in exploring the simulation of word2vec
    using a HilbertEmbedder.  A single sample consists of one positive example
    along with k negative examples.
    """

    def __init__(self, sample_reader, device=None):
        print('WARNING: DeltaW2VSamples can only be used with shard_factor=1')
        device = device or h.CONSTANTS.MATRIX_DEVICE
        self.sample_reader = sample_reader
        self.delta = torch.zeros(
            len(sample_reader.dictionary), len(sample_reader.dictionary), 
            device=device
        )


    def calc_shard(self, M_hat, shard=None):
        self.delta.fill_(0)
        for fields in self.sample_reader.next_sample():
            t1, t2, val = fields[:3]
            self.delta[t1,t2] += val - sigmoid(M_hat[t1,t2])[0]
        return self.delta



class DeltaW2VSamplesFullCorpus:
    """
    A delta calculator that can take in the positive and negative samples
    that have been produced by some sampler, for example, taken from a 
    trace of the standard word2vec algorithms samples, useful in exploring the
    simulation of word2vec using a HilbertEmbedder.
    """

    def __init__(self, Nxx, Nxx_neg, device=None):
        print(
            'WARNING: DeltaW2VSamplesFullCorpus can only be used with '
            'shard_factor=1'
        )
        device = device or h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        self.multiplier = torch.tensor(Nxx+Nxx_neg, device=device, dtype=dtype)
        self.Nxx = torch.tensor(Nxx, device=device, dtype=dtype)

    def calc_shard(self, M_hat, shard=None):
        return self.Nxx - self.multiplier * sigmoid(M_hat)





class DeltaW2V(FDelta):

    def __init__(self, bigram, M, k, update_density=1, device=None):
        super().__init__(bigram, M, update_density, device)
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        self.k = torch.tensor(k, device=self.device, dtype=dtype)


    def load_shard(self, M_hat, shard):
        self.Nxx, Nx, Nxt, N = self.bigram.load_shard(shard, device=self.device)
        uNx, uNxt, uN = self.bigram.unigram.load_shard(shard,device=self.device)
        self.N_neg = h.M.negative_sample(self.Nxx, Nx, uNxt, uN, self.k)

    def _calc_shard(self, M_hat, shard=None):
        # This simplified form is equivalent to (but a bit cheaper than):
        #      (Nxx + N_Neg) * (sigmoid(M) - sigmoid(M_hat))
        return self.Nxx - (self.Nxx + self.N_neg) * sigmoid(M_hat)



class DeltaGlove:

    def __init__(
        self,
        bigram,
        M,
        X_max=100.0,
        alpha=0.75,
        device=None
    ):
        self.bigram = bigram
        self.M = M
        self.X_max = float(X_max)
        self.alpha = alpha
        self.device=device
        self.precalculate_multiplier()

    def precalculate_multiplier(self):
        self.multiplier = (self.bigram.Nxx / self.X_max).power(self.alpha)
        self.multiplier[self.multiplier>1] = 1
        self.multiplier *= 2

    def calc_shard(self, M_hat, shard=None):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        multiplier = h.utils.load_shard(
            self.multiplier, shard, device=device)
        return multiplier * (self.M[shard] - M_hat)



class DeltaMLE:

    def __init__(self, bigram, M, device=None):
        self.bigram = bigram
        self.M = M
        self.device=device
        self.precalculate_exp_M()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        self.max_multiplier = torch.tensor(
            np.max(bigram.Nx)**2, dtype=torch.float32, device=device)

    def precalculate_exp_M(self):
        pmi_data, I, J = h.corpus_stats.calc_PMI_sparse(self.bigram)
        exp_M_data = np.e**pmi_data
        self.exp_M = sparse.coo_matrix(
            (exp_M_data, (I,J)), self.bigram.Nxx.shape).tocsr()

    def calc_shard(self, M_hat, shard=None, t=1):
        Nxx, Nx, Nxt, N = self.bigram.load_shard(shard)
        multiplier = Nx * Nxt / self.max_multiplier
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        exp_M = h.utils.load_shard(self.exp_M, shard, device=device)
        return multiplier**(1.0/t) * (exp_M - np.e**M_hat)



class DeltaSwivel:

    def __init__(self, bigram, M, device=None):
        self.bigram = bigram
        self.M = M
        self.device = device
        self.sqrtNxx = None
        self.last_shard = None


    def calc_shard(self, M_hat, shard=None):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE

        if shard != self.last_shard:
            self.last_shard = shard
            Nxx, Nx, Nxt, N = self.bigram.load_shard(shard)
            self.sqrtNxx = torch.sqrt(Nxx)

        # Calculate case 1
        difference = self.M[shard] - M_hat
        case1 = self.sqrtNxx * difference

        # Calculate case 2 (only applies where Nxx is zero).
        exp_diff = np.e**difference[self.sqrtNxx==0]
        case2 = exp_diff / (1 + exp_diff)

        # Combine the cases
        case1[self.sqrtNxx==0] = case2

        return case1


def get_delta(name, **kwargs):
    """
    Convenience function to be able to select and instantiate a Delta class by
    name.
    """
    if name.lower() == 'mse':
        return DeltaMSE(**kwargs)
    elif name.lower() == 'mle':
        return DeltaMLE(**kwargs)
    elif name.lower() == 'w2v':
        return DeltaW2V(**kwargs)
    elif name.lower() == 'glove':
        return DeltaGlove(**kwargs)
    elif name.lower() == 'swivel':
        return DeltaSwivel(**kwargs)
    else:
        raise ValueError(
            "``name`` must be one of 'mse', 'mle', 'w2v', 'glove', or "
            "'swivel'. Got {}.".format(repr(name))
        )


def sigmoid(M):
    return 1 / (1 + np.e**(-M))




