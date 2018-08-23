import hilbert as h
import numpy as np

def load_test_tokens():
    return load_tokens(h.CONSTANTS.TEST_TOKEN_PATH)


def load_tokens(path):
    with open(path) as f:
        return f.read().split()


def get_test_stats(window_size):
    return get_stats(load_test_tokens(), window_size)


def calc_PMI(N_xx):
    N_x = np.sum(N_xx, axis=1).reshape((-1,1))
    N = np.sum(N_x)
    with np.errstate(divide='ignore'):
        return np.log(N) + np.log(N_xx) - np.log(N_x) - np.log(N_x.T)


def calc_positive_PMI(N_xx):
    PMI = calc_PMI(N_xx)
    PMI[PMI<0] = 0
    return PMI


def calc_shifted_w2v_PMI(k, N_xx):
    return calc_PMI(N_xx) - np.log(k)


def get_stats(token_list, window_size):
    unique_tokens = list(set(token_list))
    token_lookup = {token:i for i,token in enumerate(unique_tokens)}
    N_xx = np.zeros((len(unique_tokens), len(unique_tokens)))
    for i in range(len(token_list)):
        focal_word = token_lookup[token_list[i]]
        for j in range(i-window_size, i +window_size+1):
            if i==j or j < 0:
                continue
            try:
                context_word = token_lookup[token_list[j]]
            except IndexError:
                continue
            N_xx[focal_word,context_word] += 1

    N_x = np.sum(N_xx, axis=1)
    return unique_tokens, N_xx


