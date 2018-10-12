try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    torch = None
    sparse = None

import hilbert as h



class DeltaMSE:

    def __init__(self, bigram, M, device=None):
        self.bigram = bigram
        self.M = M
        self.device=device

    def calc_shard(self, M_hat, shard=None):
        return self.M[shard] - M_hat
        

class DeltaW2V:

    def __init__(self, bigram, M, k, device=None):
        self.bigram = bigram
        self.M = M
        self.device = device
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        dtype = h.CONSTANTS.DEFAULT_DTYPE
        self.k = torch.tensor(k, device=device, dtype=dtype)
        self.last_shard = -1
        self.multiplier = None
        self.exp_M = None


    def calc_shard(self, M_hat, shard=None):
        if self.last_shard != shard:
            self.last_shard = shard
            Nxx, Nx, Nxt, N = self.bigram.load_shard(shard)
            uNx, uNxt, uN = self.bigram.unigram.load_shard(shard)
            N_neg = h.M.negative_sample(Nxx, Nx, uNxt, uN, self.k)
            self.multiplier = Nxx + N_neg
            self.exp_M = sigmoid(self.M[shard])
            return self.multiplier * (self.exp_M - sigmoid(M_hat))

        else:
            return self.multiplier * (self.exp_M - sigmoid(M_hat))



        return self.calculated_shard


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
        self.device=device
        self.sqrtNxx = np.sqrt(bigram.Nxx)

    def calc_shard(self, M_hat, shard=None):

        # Calculate case 1
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        difference = self.M[shard] - M_hat
        sqrtNxx = h.utils.load_shard(self.sqrtNxx, shard, device=device)
        case1 = sqrtNxx * difference

        # Calculate case 2 (only applies where Nxx is zero).
        exp_diff = np.e**difference[sqrtNxx==0]
        case2 = exp_diff / (1 + exp_diff)

        # Combine the cases
        case1[sqrtNxx==0] = case2

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




