import time
import scipy
import hilbert as h
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


MEAN_PMI = -0.812392711
STD_PMI = 1.2475529909

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
    #bigram.sort()
    return bigram


def get_test_stats(window_size):
    return get_stats(load_test_tokens(), window_size, verbose=False)


def calc_PMI(bigram):
    Nxx, Nx, Nxt, N = bigram
    return torch.log(N) + torch.log(Nxx) - torch.log(Nx) - torch.log(Nxt)


def calc_exp_pmi_stats(bigram):
    Nxx, Nx, Nxt, N = bigram
    pmi = calc_PMI(bigram)

    # Keep only pmis for i,j where Nxx[i,j]>0
    pmi = pmi[Nxx>0]
    exp_pmi = np.e**pmi
    return torch.mean(exp_pmi), torch.std(exp_pmi)


def get_prior_beta_params(Ni, Nj, N, exp_mean, exp_std):
    factor = Ni * Nj / N**2
    mean = exp_mean * factor
    std = exp_std * factor**2
    alpha = mean * ( mean*(1-mean)/std - 1)
    beta = (1-mean) * alpha / mean 
    return alpha.item(), beta.item(), factor.item()


def get_posterior_beta_params(bigram, exp_mean, exp_std, i, j):
    Nxx, Nx, Nxt, N = bigram
    Ni, Nj = Nx[i,0], Nx[j,0]
    alpha, beta, factor = get_prior_beta_params(Ni, Nj, N, exp_mean, exp_std)
    post_alpha = Nxx[i,j] + alpha
    post_beta = N - Nxx[i,j] + beta
    return post_alpha, post_beta, factor


def posterior_pmi_histogram(
    post_alpha, post_beta, factor, a=-20, b=5, delta=0.01
):
    X = np.arange(a,b,delta)
    pdf = [
        (factor * np.e**x)**(post_alpha-1) * (1-factor*np.e**x)**(post_beta-1)
        for x in X
    ]
    Y = pdf / np.sum(pdf)
    plt.plot(X,Y)
    plt.show()


def calculate_all_kls(bigram):
    KL = np.zeros((bigram.vocab, bigram.vocab))
    iters = 0
    start = time.time()
    for i in range(bigram.vocab):
        elapsed = time.time() - start
        start = time.time()
        print(elapsed)
        print(elapsed * 20000 / 60, 'min')
        print(elapsed * 20000 / 60 / 60, 'hrs')
        print(elapsed * 20000 / 60 / 60 / 24, 'days')
        print(100 * iters / 10000**2, '%')
        print('iters', iters)
        for j in range(bigram.vocab):
            iters += 1

            Nij = bigram.Nxx[i,j]
            Ni = bigram.Nx[i,0]
            Nj = bigram.Nx[j,0]
            N = bigram.N
            KL[i,j] = get_posterior_kl(
                MEAN_PMI, STD_PMI, Nij, Ni, Nj, N
            )



def get_posterior_numerically(
    pmi_mean, pmi_stdv, Nij, Ni, Nj, N,
    a=-10, b=10, delta=0.1,
    plot=True
):
    X = np.arange(a, b, delta)
    pmi_pdf = np.array([
        scipy.stats.norm.pdf(x, pmi_mean, pmi_stdv)
        for x in X
    ])

    pmi_pdf = pmi_pdf / np.sum(pmi_pdf)
    factor = Ni * Nj / N**2
    p = [factor * np.e**x for x in X]

    bin_pdf = np.array([
        scipy.stats.binom.pmf(Nij, N, p_)
        for p_ in p
    ])

    post_pdf = bin_pdf * pmi_pdf
    post_pdf = post_pdf / np.sum(post_pdf)

    if plot:
        plt.plot(X, pmi_pdf, label='prior')
        plt.plot(X, post_pdf, label='posterior')
        plt.legend()
        plt.show()

    return X, post_pdf, pmi_pdf


    
def get_posterior_kl(
    pmi_mean, pmi_stdv, Nij, Ni, Nj, N,
    a=-10, b=10, delta=0.1, plot=False
):
    X, posterior, prior = get_posterior_numerically(
        pmi_mean, pmi_stdv, Nij, Ni, Nj, N, a=a, b=b, delta=delta, plot=plot)
    return kl(posterior, prior)



def kl(pdf1, pdf2):
    # strip out cases where pdf1 is zero
    pdf2 = pdf2[pdf1!=0]
    pdf1 = pdf1[pdf1!=0]

    return np.sum(pdf1 * np.log(pdf1 / pdf2))


def plot_beta(alpha, beta):
    X = np.arange(0, 1, 0.01)
    Y = scipy.stats.beta.pdf(X, alpha, beta)
    plt.plot(X, Y)
    plt.show()





def histogram(values, plot=True):
    values = values.reshape(-1)

    n, bins = np.histogram(values, bins='auto')
    bin_centers = [ 0.5*(bins[i]+bins[i+1]) for i in range(len(n))]

    if plot:
        plt.plot(bin_centers, n)
        plt.show()
    return bin_centers, n




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
    log_Nxx_nonzero = np.log(np.array(bigram.Nxx.tocsr()[I,J]).reshape(-1))
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





