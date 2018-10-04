try:
    import numpy as np
    from scipy import sparse
    import torch
except ImportError:
    np = None
    torch = None
    sparse = None

import hilbert_noshard as h

def get_f_MSE(cooc_stats, M, implementation='torch', device='cuda'):
    def f_MSE(M_hat):
        with np.errstate(invalid='ignore'):
            return M - M_hat
    return f_MSE


def get_f_w2v(cooc_stats, M, k, implementation='torch', device='cuda'):
    Nxx, Nx, N = cooc_stats
    Nxt = np.asarray(np.sum(cooc_stats.Nxx, axis=0))
    h.utils.ensure_implementation_valid(implementation)
    N_neg_xx = calc_N_neg_xx((Nxx, Nx, Nxt, N), k)
    multiplier = Nxx + N_neg_xx
    sigmoid_M = sigmoid(M)
    if implementation == 'torch':
        multiplier = torch.tensor(
            multiplier, dtype=torch.float32, device=device)
        sigmoid_M = torch.tensor(
            sigmoid_M, dtype=torch.float32, device=device)
    
    def f_w2v(M_hat):
        return multiplier * (sigmoid_M - sigmoid(M_hat))

    return f_w2v


def get_f_w2v_sh(cooc_stats, M, k, device='cuda'):
    h.utils.ensure_implementation_valid(implementation)
    def f_w2v(M_hat, shard, num_shards):
        Nx = torch.tensor(N_x[shard::num_shards], device=device)
        Nxx = torch.tensor(
            Nxx[shard::num_shards,shard::num_shards], device=device)
        N_neg_xx = k * Nx * Nx.t() / cooc_stats.N
        multiplier = Nxx + N_neg_xx
        sigmoid_M = sigmoid(
            torch.log(Nxx) + torch.log(cooc_stats.N) 
            - torch.log(Nx) - torch.log(Nx.t())
        )
        return multiplier * (sigmoid_M - sigmoid(M_hat))

    return f_w2v




def get_f_glove(
    cooc_stats, M,
    X_max=100.0,
    implementation='torch',
    device='cuda'
):
    h.utils.ensure_implementation_valid(implementation)
    X_max = float(X_max)
    multiplier = (cooc_stats.denseNxx / X_max) ** (0.75)
    multiplier[multiplier>1] = 1
    multiplier *= 2
    if implementation == 'torch':
        multiplier = torch.tensor(
            multiplier, dtype=torch.float32, device=device)

    def f_glove(M_hat):
        return multiplier * (M - M_hat)

    return f_glove


def get_f_MLE(cooc_stats, M, implementation='torch', device='cuda'):
    h.utils.ensure_implementation_valid(implementation)
    multiplier = cooc_stats.Nx * cooc_stats.Nx.T
    multiplier = multiplier / np.max(multiplier)
    exp_M = np.e**M
    if implementation == 'torch':
        multiplier = torch.tensor(
            multiplier, dtype=torch.float32, device=device)
        exp_M = torch.tensor(exp_M, dtype=torch.float32, device=device)

    def f_MLE(M_hat, t=1):
        return multiplier**(1.0/t) * (exp_M - np.e**M_hat)

    return f_MLE


def get_torch_f_MLE_optimized(
    cooc_stats, M, 
    implementation='torch',
    device='cuda'
):
    """
    Mathematically equivalent to `get_torch_f_MLE`, but attempts to minimize
    allocation during the f_MLE calculations.  This turned out to have a 
    negligible effect on runtime.
    """
    h.utils.ensure_implementation_valid(implementation)
    if implementation == 'numpy':
        raise NotImplementedError(
            'get_torch_f_MLE_optimized has only a torch-based implementation.')
    Nx = torch.tensor(cooc_stats.Nx, dtype=torch.float32, device=device)
    M = torch.tensor(M, dtype=torch.float32, device=device)
    multiplier = Nx * Nx.t()
    multiplier = multiplier / torch.max(multiplier)
    exp_M = np.e**M
    tempered_multiplier_ = torch.zeros(M.shape, device=device)
    def f_MLE(M_hat, t=1):
        M_hat_exp = torch.pow(np.e, M_hat, out=M_hat)
        delta = torch.sub(exp_M, M_hat_exp, out=M_hat)
        tempered_multiplier = torch.pow(
            multiplier, 1.0/t, out=tempered_multiplier_)
        return delta.mul_(tempered_multiplier)
    return f_MLE


def get_f_swivel(cooc_stats, M, implementation='torch', device='cuda'):
    h.utils.ensure_implementation_valid(implementation)
    sqrtNxx = np.sqrt(cooc_stats.denseNxx)
    if implementation == 'torch':
        sqrtNxx = torch.tensor(sqrtNxx, dtype=torch.float32, device=device)

    def f_swivel(M_hat):

        # Calculate case 1
        difference = M - M_hat
        case1 = sqrtNxx * difference

        # Calculate case 2 (only applies where Nxx is zero).
        exp_diff = np.e**difference[sqrtNxx==0]
        case2 = exp_diff / (1 + exp_diff)

        # Combine the cases
        case1[sqrtNxx==0] = case2

        return case1

    return f_swivel


def calc_N_neg_xx(cooc_stats, k):
    Nxx, Nx, Nxt, N = cooc_stats
    return k * Nx * Nxt / N


def sigmoid(M):
    return 1 / (1 + np.e**(-M))




