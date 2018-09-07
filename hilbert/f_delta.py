import numpy as np
from scipy import sparse
import torch


def f_mse(M, M_hat, delta):
    return np.subtract(M, M_hat, delta)


def calc_N_neg_xx(k, N_x):
    N_x = N_x.reshape((-1,1))
    N = float(np.sum(N_x))
    return k * N_x * N_x.T / N


def get_f_w2v(cooc_stats, k):
    N_neg_xx = calc_N_neg_xx(k, cooc_stats.Nx)
    multiplier = cooc_stats.Nxx + N_neg_xx
    sigmoid_M = np.zeros(cooc_stats.Nxx.shape)
    sigmoid_M_hat = np.zeros(cooc_stats.Nxx.shape)
    def f_w2v(M, M_hat, delta):
        sigmoid(M, sigmoid_M)
        sigmoid(M_hat, sigmoid_M_hat)
        np.subtract(sigmoid_M, sigmoid_M_hat, delta)
        return np.multiply(multiplier, delta, delta)
    return f_w2v


def sigmoid(M, sigmoid_M=None):
    if sigmoid_M is None:
        return 1 / (1 + np.e**(-M))
    np.power(np.e, -M, sigmoid_M)
    np.add(1, sigmoid_M, sigmoid_M)
    return np.divide(1, sigmoid_M, sigmoid_M)
    

def get_f_glove(N_xx, X_max=100.0):
    X_max = float(X_max)
    if sparse.issparse(N_xx):
        multiplier = N_xx.todense() / X_max
    else:
        multiplier = N_xx / X_max
    np.power(multiplier, 0.75, multiplier)
    multiplier[multiplier>1] = 1
    np.multiply(multiplier, 2, multiplier)
    def f_glove(M, M_hat, delta):
        with np.errstate(invalid='ignore'):
            np.subtract(M, M_hat, delta)
        delta[multiplier==0] = 0
        return np.multiply(multiplier, delta, delta)
    return f_glove


def get_f_MLE(cooc_stats):

    Nx = cooc_stats.Nx.reshape((-1,1)).astype('float64')
    multiplier = Nx * Nx.T
    multiplier_max = np.max(multiplier)
    np.divide(multiplier, multiplier_max, multiplier)

    tempered_multiplier = np.zeros(cooc_stats.Nxx.shape)
    exp_M = np.zeros(cooc_stats.Nxx.shape)
    exp_M_hat = np.zeros(cooc_stats.Nxx.shape)

    def f_MLE(M, M_hat, delta, t=1):

        np.power(np.e, M, exp_M)
        np.power(np.e, M_hat, exp_M_hat)
        np.subtract(exp_M, exp_M_hat, delta)
        np.power(multiplier, 1.0/t, tempered_multiplier)
        np.multiply(tempered_multiplier, delta, delta)

        return delta

    return f_MLE



def get_torch_f_MLE(cooc_stats, M, device='cpu'):
    Nx = torch.tensor(cooc_stats.Nx, dtype=torch.float32, device=device)
    M = torch.tensor(M, dtype=torch.float32, device=device)
    multiplier = Nx * Nx.view(Nx.shape[0], 1)
    exp_M = np.e**M
    def f_MLE(M, M_hat, t=1):
        tempered_multiplier = multiplier**(1.0/t)
        delta = (exp_M - np.e**M_hat)
        return tempered_multiplier * delta
    return f_MLE


def calc_M_swivel(cooc_stats):

    with np.errstate(divide='ignore'):
        log_N_xx = np.log(cooc_stats.denseNxx)
        log_N_x = np.log(cooc_stats.Nx)
        log_N = np.log(cooc_stats.N)

    return np.array([
        [
            log_N + log_N_xx[i,j] - log_N_x[i] - log_N_x[j]
            if cooc_stats.Nxx[i,j] > 0 else log_N - log_N_x[i] - log_N_x[j]
            for j in range(cooc_stats.Nxx.shape[1])
        ]
        for i in range(cooc_stats.Nxx.shape[1])
    ])


def get_f_swivel(cooc_stats):

    N_xx_sqrt = np.sqrt(cooc_stats.Nxx)
    selector = cooc_stats.Nxx==0
    exp_delta = np.zeros(cooc_stats.Nxx.shape)
    exp_delta_p1 = np.zeros(cooc_stats.Nxx.shape)
    temp_result_1 = np.zeros(cooc_stats.Nxx.shape)
    temp_result_2 = np.zeros(cooc_stats.Nxx.shape)

    def f_swivel(M, M_hat, delta):

        # Calculate cases where N_xx > 0
        np.subtract(M, M_hat, temp_result_1)
        np.multiply(temp_result_1, N_xx_sqrt, delta)

        # Calculate cases where N_xx == 0
        np.power(np.e, temp_result_1, exp_delta)
        np.add(1, exp_delta, exp_delta_p1)
        np.divide(exp_delta, exp_delta_p1, temp_result_2)

        # Combine the results
        delta[selector] = temp_result_2[selector]

        return delta

    return f_swivel




