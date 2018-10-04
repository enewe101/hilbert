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

    def __init__(self, cooc_stats, M, device=None):
        self.cooc_stats = cooc_stats
        self.M = M
        self.device=device

    def calc_shard(self, M_hat, shard=None):
        return self.M[shard] - M_hat
        

class DeltaW2V:

    def __init__(self, cooc_stats, M, k, device=None):
        self.cooc_stats = cooc_stats
        self.M = M
        self.k = k
        self.device = device
        self.last_shard = -1
        self.multiplier = None
        self.exp_M = None

    def calc_shard(self, M_hat, shard=None):
        if self.last_shard != shard:
            self.last_shard = shard
            Nxx, Nx, Nxt, N = self.cooc_stats.load_shard(shard)
            N_neg = calc_N_neg((Nxx, Nx, Nxt, N), self.k)
            self.multiplier = Nxx + N_neg
            self.exp_M = sigmoid(self.M[shard])
            return self.multiplier * (self.exp_M - sigmoid(M_hat))

        else:
            return self.multiplier * (self.exp_M - sigmoid(M_hat))



        return self.calculated_shard



class DeltaGlove:

    def __init__(
        self,
        cooc_stats,
        M,
        X_max=100.0,
        alpha=0.75,
        device=None
    ):
        self.cooc_stats = cooc_stats
        self.M = M
        self.X_max = float(X_max)
        self.alpha = alpha
        self.device=device
        self.precalculate_multiplier()

    def precalculate_multiplier(self):
        self.multiplier = (self.cooc_stats.Nxx / self.X_max).power(self.alpha)
        self.multiplier[self.multiplier>1] = 1
        self.multiplier *= 2

    def calc_shard(self, M_hat, shard=None):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        multiplier = h.utils.load_shard(
            self.multiplier, shard, from_sparse=True, device=device)
        return multiplier * (self.M[shard] - M_hat)



class DeltaMLE:

    def __init__(self, cooc_stats, M, device=None):
        self.cooc_stats = cooc_stats
        self.M = M
        self.device=device
        self.precalculate_exp_M()
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        self.max_multiplier = torch.tensor(
            np.max(cooc_stats.Nx)**2, dtype=torch.float32, device=device)

    def precalculate_exp_M(self):
        pmi_data, I, J = h.corpus_stats.calc_PMI_sparse(self.cooc_stats)
        exp_M_data = np.e**pmi_data
        self.exp_M = sparse.coo_matrix(
            (exp_M_data, (I,J)), self.cooc_stats.Nxx.shape).tocsr()

    def calc_shard(self, M_hat, shard=None, t=1):
        Nxx, Nx, Nxt, N = self.cooc_stats.load_shard(shard)
        multiplier = Nx * Nxt / self.max_multiplier
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        exp_M = h.utils.load_shard(
            self.exp_M, shard, from_sparse=True, device=device)
        return multiplier**(1.0/t) * (exp_M - np.e**M_hat)



class DeltaSwivel:


    def __init__(self, cooc_stats, M, device=None):
        self.cooc_stats = cooc_stats
        self.M = M
        self.device=device
        self.sqrtNxx = np.sqrt(cooc_stats.Nxx)

    def calc_shard(self, M_hat, shard=None):

        # Calculate case 1
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        difference = self.M[shard] - M_hat
        sqrtNxx = h.utils.load_shard(
            self.sqrtNxx, shard, from_sparse=True, device=device)
        case1 = sqrtNxx * difference

        # Calculate case 2 (only applies where Nxx is zero).
        exp_diff = np.e**difference[sqrtNxx==0]
        case2 = exp_diff / (1 + exp_diff)

        # Combine the cases
        case1[sqrtNxx==0] = case2

        return case1



## TODO: Deprecated, eliminate this
#def get_f_MLE(cooc_stats, M, implementation='torch', device='cuda'):
#    return f_MLE
#
#
## TODO: Deprecated, eliminate this
#def get_f_MSE(cooc_stats, M, implementation='torch', device='cuda'):
#    def f_MSE(M_hat):
#        with np.errstate(invalid='ignore'):
#            return M - M_hat
#    return f_MSE
#
#
## TODO: Deprecated, eliminate this
#def get_f_w2v(cooc_stats, M, k, implementation='torch', device='cuda'):
#    h.utils.ensure_implementation_valid(implementation)
#    N_neg_xx = calc_N_neg(cooc_stats.Nx, k)
#    multiplier = cooc_stats.denseNxx + N_neg_xx
#    sigmoid_M = sigmoid(M)
#    if implementation == 'torch':
#        multiplier = torch.tensor(
#            multiplier, dtype=torch.float32, device=device)
#        sigmoid_M = torch.tensor(
#            sigmoid_M, dtype=torch.float32, device=device)
#
#    def f_w2v(M_hat):
#        return multiplier * (sigmoid_M - sigmoid(M_hat))
#
#    return f_w2v
#
#
## TODO: Deprecated, eliminate this
#def get_f_glove(
#    cooc_stats, M,
#    X_max=100.0,
#    implementation='torch',
#    device='cuda'
#):
#    h.utils.ensure_implementation_valid(implementation)
#    X_max = float(X_max)
#    multiplier = (cooc_stats.denseNxx / X_max) ** (0.75)
#    multiplier[multiplier>1] = 1
#    multiplier *= 2
#    if implementation == 'torch':
#        multiplier = torch.tensor(
#            multiplier, dtype=torch.float32, device=device)
#
#    def f_glove(M_hat):
#        return multiplier * (M - M_hat)
#
#    return f_glove
#
#
## TODO: Deprecated, eliminate this
#def get_f_MLE(cooc_stats, M, implementation='torch', device='cuda'):
#    h.utils.ensure_implementation_valid(implementation)
#    multiplier = cooc_stats.Nx * cooc_stats.Nx.T
#    multiplier = multiplier / np.max(multiplier)
#    exp_M = np.e**M
#    if implementation == 'torch':
#        multiplier = torch.tensor(
#            multiplier, dtype=torch.float32, device=device)
#        exp_M = torch.tensor(exp_M, dtype=torch.float32, device=device)
#
#    def f_MLE(M_hat, t=1):
#        return multiplier**(1.0/t) * (exp_M - np.e**M_hat)
#
#    return f_MLE
#
#
## TODO: Deprecated, eliminate this
#def get_f_swivel(cooc_stats, M, implementation='torch', device='cuda'):
#    h.utils.ensure_implementation_valid(implementation)
#    sqrtNxx = np.sqrt(cooc_stats.denseNxx)
#    if implementation == 'torch':
#        sqrtNxx = torch.tensor(sqrtNxx, dtype=torch.float32, device=device)
#
#    def f_swivel(M_hat):
#
#        # Calculate case 1
#        difference = M - M_hat
#        case1 = sqrtNxx * difference
#
#        # Calculate case 2 (only applies where Nxx is zero).
#        exp_diff = np.e**difference[sqrtNxx==0]
#        case2 = exp_diff / (1 + exp_diff)
#
#        # Combine the cases
#        case1[sqrtNxx==0] = case2
#
#        return case1
#
#    return f_swivel


def calc_N_neg(cooc_stats, k):
    Nxx, Nx, Nxt, N = cooc_stats
    return k * Nx * Nxt / N


def sigmoid(M):
    return 1 / (1 + np.e**(-M))




