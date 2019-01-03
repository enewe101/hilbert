import os
import time
import scipy
import hilbert as h
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


PMI_MEAN = -0.812392711
PMI_STD = 1.2475529909

try:
    import torch
    import numpy as np
    from scipy import sparse
except ImportError:
    torch = None
    np = None
    sparse = None


####
#
# CONVENIENCE LOADERS FOR GETTING AHOLD OF CORPUS-RELATED TEST DATA STRUCTURES.
#
####

def load_test_tokens():
    return load_tokens(h.CONSTANTS.TEST_TOKEN_PATH)


def load_tokens(path):
    with open(path) as f:
        return f.read().split()


def get_test_stats(window_size):
    return get_stats(load_test_tokens(), window_size, verbose=False)


def get_test_bigram_mutable(window_size):
    bigram = get_bigram_mutable(load_test_tokens(), window_size, verbose=False) 
    return bigram


#def get_test_bigram(window_size):
#    bigram = get_bigram(load_test_tokens(), window_size, verbose=False) 
#    #bigram.sort()
#    return bigram


def get_test_bigram_base(device=None, verbose=True):
    """
    For testing purposes, builds a bigram_base from constituents (not using it's
    own load function) and returns the bigram_base along with the constituents
    used to make it.
    """
    path = os.path.join(h.CONSTANTS.TEST_DIR, 'bigram')
    unigram = h.unigram.Unigram.load(path, device=device, verbose=verbose)
    Nxx = sparse.load_npz(os.path.join(path, 'Nxx.npz')).tolil()
    bigram_base = h.bigram.BigramBase(
        unigram, Nxx, device=device, verbose=verbose)

    return bigram_base, unigram, Nxx


def get_test_bigram_sector(sector):
    """
    For testing purposes, builds a `BigramSector` starting from a `BigramBase`
    (not using `BigramBase`'s load function) and returns both.
    """
    bigram_base = h.bigram.BigramBase.load(
        os.path.join(h.CONSTANTS.TEST_DIR, 'bigram'))
    args = {
        'unigram':bigram_base.unigram,
        'Nxx':bigram_base.Nxx[sector],
        'Nx':bigram_base.Nx,
        'Nxt':bigram_base.Nxt,
        'sector':sector
    }
    bigram_sector = h.bigram_sector.BigramSector(**args)
    return bigram_sector, bigram_base



#############
#
# Operations on Corpus Statistics used by models.
#
#############

def w2v_prob_keep(uNx, uN, t=1e-5):
    freqs = uNx / uN
    drop_probs = torch.clamp((freqs - t)/freqs - torch.sqrt(t/freqs), 0, 1)
    keep_probs = 1 - drop_probs
    return keep_probs


def calc_PMI(bigram_shard):
    Nxx, Nx, Nxt, N = bigram_shard
    return torch.log(N) + torch.log(Nxx) - torch.log(Nx) - torch.log(Nxt)


def calc_prior_beta_params(bigram, exp_mean, exp_std, Pxx_independent):
    _, Nx, Nxt, N = bigram
    mean = exp_mean * Pxx_independent
    std = exp_std * Pxx_independent
    alpha = mean * (mean*(1-mean)/std**2 - 1)
    beta = (1-mean) * alpha / mean 
    return alpha, beta


def calc_exp_pmi_stats(bigram):
    Nxx, _, _, _ = bigram
    pmi = h.corpus_stats.calc_PMI(bigram)
    # Keep only pmis for i,j where Nxx[i,j]>0
    pmi = pmi[Nxx>0]
    exp_pmi = np.e**pmi
    return torch.mean(exp_pmi), torch.std(exp_pmi)



#############
#
#    Stuff below here is scratch used for real-time analysis, but not
#    necessarily good outside use.  Keeping it for now.
#
#############

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


def get_posterior_numerically(
    Nij, Ni, Nj, N, pmi_mean=PMI_MEAN, pmi_std=PMI_STD, 
    a=-10, b=10, delta=0.1,
    plot=True
):
    X = np.arange(a, b, delta)
    pmi_pdf = np.array([
        scipy.stats.norm.pdf(x, pmi_mean, pmi_std)
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


def calculate_all_kls(bigram):
    assert bigram.sector == h.shards.whole, "expecting whole bigram"
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
                MEAN_PMI, PMI_STD, Nij, Ni, Nj, N
            )

    
def get_posterior_kl(
    pmi_mean, pmi_std, Nij, Ni, Nj, N,
    a=-10, b=10, delta=0.1, plot=False
):
    X, posterior, prior = get_posterior_numerically(
        pmi_mean, pmi_std, Nij, Ni, Nj, N, a=a, b=b, delta=delta, plot=plot)
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



def calc_PMI_star(cooc_stats):
    Nxx, Nx, Nxt, N = cooc_stats
    useNxx = Nxx.clone()
    useNxx[useNxx==0] = 1
    return calc_PMI((useNxx, Nx, Nxt, N))



# There should be some code that generates a BigramMutable by sampling text
# It should exhibit the different samplers too.  For now this stub is a
# reminder.
def get_bigram_mutable(token_list, window_size, verbose=True):
    unigram = h.unigram.Unigram(verbose=verbose)
    for token in token_list:
        unigram.add(token)
    bigram = h.bigram_mutable.BigramMutable(unigram, verbose=verbose)
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






#def calc_PMI_sparse(bigram):
#    I, J = bigram.Nxx.nonzero()
#    log_Nxx_nonzero = np.log(np.array(bigram.Nxx.tocsr()[I,J]).reshape(-1))
#    log_Nx_nonzero = np.log(bigram.Nx[I,0])
#    log_Nxt_nonzero = np.log(bigram.Nxt[0,J])
#    log_N = np.log(bigram.N)
#    pmi_data = log_N + log_Nxx_nonzero - log_Nx_nonzero - log_Nxt_nonzero
#
#    # Here, the default (unrepresented value) in our sparse representation
#    # is negative infinity.  scipy sparse matrices only support zero as the
#    # unrepresented value, and this would be ambiguous with actual zeros.
#    # Therefore, keep data in the (data, (I,J)) format (the same as is used
#    # as input to the coo_matrix constructor).
#    return pmi_data, I, J

