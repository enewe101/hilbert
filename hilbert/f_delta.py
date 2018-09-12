import numpy as np
from scipy import sparse
import torch


def get_f_MSE(cooc_stats, M, implementation='torch', device='cpu'):
    def f_MSE(M_hat):
        with np.errstate(invalid='ignore'):
            return M - M_hat
    return f_MSE


def get_f_w2v(cooc_stats, M, k, implementation='torch', device='cpu'):
    ensure_implementation_valid(implementation)
    N_neg_xx = calc_N_neg_xx(cooc_stats.Nx, k)
    multiplier = cooc_stats.denseNxx + N_neg_xx
    sigmoid_M = sigmoid(M)
    if implementation == 'torch':
        multiplier = torch.tensor(
            multiplier, dtype=torch.float32, device=device)
        sigmoid_M = torch.tensor(
            sigmoid_M, dtype=torch.float32, device=device)
    
    def f_w2v(M_hat):
        return multiplier * (sigmoid_M - sigmoid(M_hat))

    return f_w2v




def get_f_glove(
    cooc_stats, M,
    X_max=100.0,
    implementation='torch',
    device='cpu'
):
    ensure_implementation_valid(implementation)
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
    ensure_implementation_valid(implementation)
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
    ensure_implementation_valid(implementation)
    if implementation == 'numpy':
        raise NotImplementedError(
            'get_torch_f_MLE_optimized has only a torch-based implementation.')
    Nx = torch.tensor(cooc_stats.Nx, dtype=torch.float32, device=device)
    M = torch.tensor(M, dtype=torch.float32, device=device)
    multiplier = Nx * Nx.t()
    multiplier = multiplier / torch.max(multiplier)
    exp_M = np.e**M
    tempered_multiplier_ = torch.zeros(M.shape)
    def f_MLE(M_hat, t=1):
        M_hat_exp = torch.pow(np.e, M_hat, out=M_hat)
        delta = torch.sub(exp_M, M_hat_exp, out=M_hat)
        tempered_multiplier = torch.pow(
            multiplier, 1.0/t, out=tempered_multiplier_)
        return delta.mul_(tempered_multiplier)
    return f_MLE


def calc_M_swivel(cooc_stats):

    with np.errstate(divide='ignore'):
        log_N_xx = np.log(cooc_stats.Nxx.toarray())
        log_N_x = np.log(cooc_stats.Nx.reshape(-1))
        log_N = np.log(cooc_stats.N)

    return np.array([
        [
            log_N + log_N_xx[i,j] - log_N_x[i] - log_N_x[j]
            if cooc_stats.Nxx[i,j] > 0 else log_N - log_N_x[i] - log_N_x[j]
            for j in range(cooc_stats.Nxx.shape[1])
        ]
        for i in range(cooc_stats.Nxx.shape[1])
    ])


def get_f_swivel(cooc_stats, M, implementation='torch', device='cuda'):

    N_xx_sqrt = np.sqrt(cooc_stats.denseNxx)
    selector = cooc_stats.denseNxx==0
    exp_delta = np.zeros(cooc_stats.Nxx.shape)
    exp_delta_p1 = np.zeros(cooc_stats.Nxx.shape)
    temp_result_1 = np.zeros(cooc_stats.Nxx.shape)
    temp_result_2 = np.zeros(cooc_stats.Nxx.shape)

    def f_swivel(M_hat):

        # Calculate cases where N_xx > 0
        np.subtract(M, M_hat, temp_result_1)

        delta = np.multiply(temp_result_1, N_xx_sqrt)

        # Calculate cases where N_xx == 0
        np.power(np.e, temp_result_1, exp_delta)
        np.add(1, exp_delta, exp_delta_p1)
        np.divide(exp_delta, exp_delta_p1, temp_result_2)

        # Combine the results
        delta[selector] = temp_result_2[selector]

        return delta

    return f_swivel


def calc_N_neg_xx(N_x, k):
    N = float(np.sum(N_x))
    return k * N_x * N_x.T / N


def sigmoid(M):
    return 1 / (1 + np.e**(-M))


def ensure_implementation_valid(implementation):
    if implementation != 'torch' and implementation != 'numpy':
        raise ValueError(
            "implementation must be 'torch' or 'numpy'.  Got %s."
            % repr(implementation)
        )



