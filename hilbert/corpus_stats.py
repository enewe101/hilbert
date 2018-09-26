import hilbert as h

try:
    import torch
except ImportError:
    torch = None

def load_test_tokens():
    return load_tokens(h.CONSTANTS.TEST_TOKEN_PATH)


def load_tokens(path):
    with open(path) as f:
        return f.read().split()


def get_test_stats(window_size):
    return get_stats(load_test_tokens(), window_size, verbose=False)


def calc_PMI(cooc_stats):
    Nxx, Nx, N = cooc_stats
    torch_log_Nx = torch.log(Nx)
    return torch.log(N) + torch.log(Nxx) - torch_log_Nx - torch_log_Nx.t()


def calc_positive_PMI(cooc_stats):
    PMI = calc_PMI(cooc_stats)
    PMI[PMI<0] = 0
    return PMI


def calc_shifted_PMI(cooc_stats, k):
    return calc_PMI(cooc_stats) - torch.log(k)


def calc_PMI_star(cooc_stats):
    Nxx, Nx, N = cooc_stats
    useNxx = Nxx.clone()
    useNxx[useNxx==0] = 1
    return calc_PMI((useNxx, Nx, N))


def get_stats(token_list, window_size, verbose=True):
    cooc_stats = h.cooc_stats.CoocStats(verbose=verbose)
    for i in range(len(token_list)):
        focal_word = token_list[i]
        for j in range(i-window_size, i +window_size+1):
            if i==j or j < 0:
                continue
            try:
                context_word = token_list[j]
            except IndexError:
                continue
            cooc_stats.add(focal_word, context_word)
    return cooc_stats


