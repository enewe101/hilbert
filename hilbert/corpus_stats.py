import hilbert as h
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


try:
    import torch
    import numpy as np
    from scipy import sparse
except ImportError:
    torch = None
    np = None
    sparse = None

def load_test_tokens():
    return load_tokens(h.CONSTANTS.TEST_TOKEN_PATH)

def load_tokens(path):
    with open(path) as f:
        return f.read().split()

def get_test_bigram(window_size):
    bigram = get_bigram(load_test_tokens(), window_size, verbose=False) 
    bigram.sort()
    return bigram


def get_test_stats(window_size):
    return get_stats(load_test_tokens(), window_size, verbose=False)


def calc_PMI(bigram):
    Nxx, Nx, Nxt, N = bigram
    return torch.log(N) + torch.log(Nxx) - torch.log(Nx) - torch.log(Nxt)


def histogram(M, predicate=None):

    print('loading')
    M_npy = M.load_all().cpu().numpy()

    if predicate is not None:
        mask = predicate(M)
        M_npy = np.ma.masked_array(M_npy, mask=mask)

    print('accumulating')
    #n, bins = np.histogram(M_npy, bins='auto')

    # the histogram of the data
    n, bins, patches = plt.hist(
	x=M_npy.reshape(-1), bins='auto', density=True, facecolor='green',
        alpha=0.75
    )

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()

    return n, bins




def calc_PMI_smooth(bigram):
    Nxx, Nx, Nxt, N = bigram


    Nxx_exp = Nx * Nxt / N

    Nxx_smooth = torch.tensor([
        [
            Nxx[i,j] if Nxx[i,j] > Nxx_exp[i,j] else
            Nxx_exp[i,j] if Nxx_exp[i,j] > 1 else
            1
            for j in range(Nxx.shape[1])
        ]
        for i in range(Nxx.shape[0])
    ])
    Nx = Nxx_smooth.sum(dim=1, keepdim=True)
    Nxt = Nxx_smooth.sum(dim=0, keepdim=True)
    N = Nxx_smooth.sum()
    return Nxx_smooth, Nx, Nxt, N


    


def calc_PMI_sparse(bigram):
    I, J = bigram.Nxx.nonzero()
    log_Nxx_nonzero = np.log(np.array(bigram.Nxx[I,J]).reshape(-1))
    log_Nx_nonzero = np.log(bigram.Nx[I,0])
    log_Nxt_nonzero = np.log(bigram.Nxt[0,J])
    log_N = np.log(bigram.N)
    pmi_data = log_N + log_Nxx_nonzero - log_Nx_nonzero - log_Nxt_nonzero

    # Here, the default (unrepresented value) in our sparse representation
    # is negative infinity.  scipy sparse matrices only support zero as the
    # unrepresented value, and this would be ambiguous with actual zeros.
    # Therefore, keep data in the (data, (I,J)) format (the same as is used
    # as input to the coo_matrix constructor).
    return pmi_data, I, J


def calc_PMI_star(cooc_stats):
    Nxx, Nx, Nxt, N = cooc_stats
    useNxx = Nxx.clone()
    useNxx[useNxx==0] = 1
    return calc_PMI((useNxx, Nx, Nxt, N))


#def get_stats(token_list, window_size, verbose=True):
#    cooc_stats = h.cooc_stats.CoocStats(verbose=verbose)
#    for i in range(len(token_list)):
#        focal_word = token_list[i]
#        for j in range(i-window_size, i +window_size+1):
#            if i==j or j < 0:
#                continue
#            try:
#                context_word = token_list[j]
#            except IndexError:
#                continue
#            cooc_stats.add(focal_word, context_word)
#    return cooc_stats


def get_bigram(token_list, window_size, verbose=True):
    unigram = h.unigram.Unigram(verbose=verbose)
    for token in token_list:
        unigram.add(token)
    bigram = h.bigram.Bigram(unigram, verbose=verbose)
    for i in range(len(token_list)):
        focal_word = token_list[i]
        for j in range(i-window_size, i +window_size+1):
            if i==j or j < 0:
                continue
            try:
                context_word = token_list[j]
            except IndexError:
                continue
            bigram.add(focal_word, context_word)
    return bigram





