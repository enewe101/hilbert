import numpy as np
from scipy import sparse


def f_mse(M, M_hat, delta):
    return np.subtract(M, M_hat, delta)


def calc_N_neg_xx(k, N_x):
    N_x = N_x.reshape((-1,1))
    N = float(np.sum(N_x))
    return k * N_x * N_x.T / N


def get_f_w2v(N_xx, N_x, k):
    N_x = N_x.reshape((-1,1))
    N_neg_xx = calc_N_neg_xx(k, N_x)
    multiplier = N_xx + N_neg_xx
    sigmoid_M = np.zeros(N_xx.shape)
    sigmoid_M_hat = np.zeros(N_xx.shape)
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


def get_f_MLE(N_xx, N_x):

    N_x = N_x.reshape((-1,1)).astype('float64')
    multiplier = N_x * N_x.T
    multiplier_max = np.max(multiplier)
    np.divide(multiplier, multiplier_max, multiplier)

    tempered_multiplier = np.zeros(N_xx.shape)
    exp_M = np.zeros(N_xx.shape)
    exp_M_hat = np.zeros(N_xx.shape)

    def f_MLE(M, M_hat, delta, t=1):

        np.power(np.e, M, exp_M)
        np.power(np.e, M_hat, exp_M_hat)
        np.subtract(exp_M, exp_M_hat, delta)
        np.power(multiplier, 1.0/t, tempered_multiplier)
        np.multiply(tempered_multiplier, delta, delta)

        return delta

    return f_MLE


def calc_M_swivel(N_xx, N_x):

    if sparse.issparse(N_xx):
        use_N_xx = N_xx.todense()
    else:
        use_N_xx = N_xx

    with np.errstate(divide='ignore'):
        log_N_xx = np.log(use_N_xx)
        log_N_x = np.log(N_x)
        log_N = np.log(np.sum(N_x))

    return np.array([
        [
            log_N + log_N_xx[i,j] - log_N_x[i] - log_N_x[j]
            if use_N_xx[i,j] > 0 else log_N - log_N_x[i] - log_N_x[j]
            for j in range(use_N_xx.shape[1])
        ]
        for i in range(use_N_xx.shape[0])
    ])


def get_f_swivel(N_xx, N_x):

    if sparse.issparse(N_xx):
        use_N_xx = N_xx.todense()
    else:
        use_N_xx = N_xx
    N_xx_sqrt = np.sqrt(use_N_xx)
    selector = use_N_xx==0
    exp_delta = np.zeros(N_xx.shape)
    exp_delta_p1 = np.zeros(N_xx.shape)
    temp_result_1 = np.zeros(N_xx.shape)
    temp_result_2 = np.zeros(N_xx.shape)

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




