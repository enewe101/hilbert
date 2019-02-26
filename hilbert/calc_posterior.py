import random
import time
import hilbert as h
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
from pylab import text
import pdb

PMI_MEAN = -0.812392711
PMI_STD = 1.2475529909
N_10k = 2789746299

try:
    import torch
    import numpy as np
    from scipy import sparse, stats, special
except ImportError:
    torch = None
    np = None
    sparse, stats, special = None, None, None


def calc_PMI(bigram):
    Nxx, Nx, Nxt, N = bigram
    return torch.log(N) + torch.log(Nxx) - torch.log(Nx) - torch.log(Nxt)


class BayesianPMI:

    def __init__(self, bigram, device=None):

        self.Nxx, self.Nx, self.Nxt, self.N = bigram
        self.device = device
        self.visible_pmi = calc_visible_pmi((
            self.Nxx, self.Nx, self.Nxt, self.N
        ))
        self.mean_exp_pmi, self.std_exp_pmi = calc_exp_pmi_stats(
            self.visible_pmi)


    def prior_beta_params(self, i,j):
        pij_expected = self.pij_expected(i,j)
        return prior_beta_params(
            pij_expected, self.mean_exp_pmi, self.std_exp_pmi
        )


    def pij_expected(self, i, j):
        return self.Nij_expected(i, j) / self.N


    def Nij_expected(self, i, j):
        return self.Nx[i,0] * self.Nx[j,0] / self.N


    def posterior_beta_params(self, i, j):
        pij_expected = self.pij_expected(i,j)
        return posterior_beta_params(
            pij_expected, self.Nxx[i,j], self.N,
            self.mean_exp_pmi, self.std_exp_pmi
        )

    def posterior_pij_pdf(self, X_pij, i, j, plot=False, device=None):
        device = device or self.device
        post_alpha, post_beta = self.posterior_beta_params(i,j)

        # TODO: Double check the derivation, in addition to testing the
        # implementation.
        map_pij_numerator = self.Nij + post_alpha
        map_pij_denominator = self.N + post_alpha + post_beta - 1
        map_pij = map_pij_numerator / map_pij_denominator

        posterior_pij_pdf = stats.beta.pdf(X_pij, post_alpha, post_beta)

        if plot:
            plt.plot(X_pij, posterior_pij_pdf)
            plt.show()
        return torch.tensor(
            posterior_pij_pdf, device=device, dtype=h.CONSTANTS.DEFAULT_DTYPE
        ), map_pij


    def posterior_pmi_pdf(self, X_pmi, i, j, plot=False, device=None):

        X_pij = self.pij_expected(i,j) * torch.exp(X_pmi)
        post_alpha, post_beta = self.posterior_beta_params(i,j)
        posterior_pij_pdf = stats.beta.pdf(X_pij, post_alpha, post_beta)
        posterior_pmi_pdf = posterior_pij_pdf * X_pij

        # Calculate MAP PMI
        term1 = np.log(self.Nxx[i,j] + post_alpha)
        term2 = -np.log(self.N + post_alpha + post_beta - 1)
        term3 = -np.log(self.pij_expected(i, j))
        map_pmi = term1 + term2 + term3

        if plot:
            X_prior, prior_pmi_pdf = self.prior_pmi_histogram(plot=False)
            delta = X_prior[1] - X_prior[0]
            total_mass = sum(prior_pmi_pdf) * delta
            prior_pmi_pdf = prior_pmi_pdf / total_mass
            h.corpus_stats.vert_line_at(map_pmi)
            plt.plot(X_prior, prior_pmi_pdf)

            plt.plot(X_prior, self.prior_pmi_pdf_beta_prim(np.array(X_prior)))
            plt.plot(X_prior, self.prior_pmi_pdf_norm(np.array(X_prior)))

            plt.plot(X_pmi.numpy(), posterior_pmi_pdf.numpy())
            plt.show()

        return posterior_pmi_pdf, map_pmi


    def posterior_pij_histogram(self, i, j, plot=True):
        return h.corpus_stats.histogram(self.visible_pmi, plot=plot)


    def prior_pmi_histogram(self, plot=True):
        return h.corpus_stats.histogram(self.visible_pmi, plot=plot)


    def prior_exp_pmi_histogram(self, plot=True, a=None, b=None):
        exp_pmi = torch.exp(self.visible_pmi)
        return h.corpus_stats.histogram(exp_pmi, plot=plot, a=a, b=b)


    def prior_pmi_pdf(self, X, plot=False):
        factor = self.mean_exp_pmi * (1-self.mean_exp_pmi) / self.std_exp_pmi
        alpha = (self.mean_exp_pmi*(factor + 1)).item()
        beta = (factor + 2).item()
        import pdb; pdb.set_trace()
        numerator = np.exp(X * alpha)
        denominator = (1+np.exp(X))**(alpha+beta) * special.beta(alpha,beta)
        prior_pmi_pdf = numerator / denominator

        mean_pmi = torch.mean(self.visible_pmi).item()
        std_pmi = torch.std(self.visible_pmi).item()
        norm_pdf = stats.norm.pdf(X, loc=mean_pmi, scale=std_pmi)

        bin_centers, n = self.prior_pmi_histogram(plot=False)
        n = n / np.sum(n)

        if plot:
            plt.plot(X, prior_pmi_pdf, color='b', label='beta prime')
            plt.plot(X, norm_pdf, color='g', label='normal')
            plt.plot(bin_centers, n, color='r', label='data')
            plt.legend()
            plt.show()
        return prior_pmi_pdf

#
#    def prior_pmi_pdf(self, X, plot=False):
#        factor = self.mean_exp_pmi * (1-self.mean_exp_pmi) / self.std_exp_pmi
#        alpha = (self.mean_exp_pmi*(factor + 1)).item()
#        beta = (factor + 2).item()
#        import pdb; pdb.set_trace()
#        numerator = np.exp(X * alpha)
#        denominator = (1+np.exp(X))**(alpha+beta) * special.beta(alpha,beta)
#        prior_pmi_pdf = numerator / denominator
#
#        mean_pmi = torch.mean(self.visible_pmi).item()
#        std_pmi = torch.std(self.visible_pmi).item()
#        norm_pdf = stats.norm.pdf(X, loc=mean_pmi, scale=std_pmi)
#
#        if plot:
#            plt.plot(X, prior_pmi_pdf, linestyle=':')
#            plt.show()
#        return prior_pmi_pdf




#bigram = h.bigram.Bigram.load(bigram_path)
def plot_visible_pmi(path):
    bin_centers, n = read_pmi_histogram(path)
    plt.plot(bin_centers, n)
    plt.show()
    return bin_centers, n


# Replicated in H.E./analysis/analyze_glove_bias.py
def read_pmi_histogram(path):
    bin_centers = []
    n = []
    with open(path) as in_file:
        for line in in_file:
            line = line.strip()
            if line == '':
                continue
            bin_center_str, n_str = line.split()
            bin_centers.append(float(bin_center_str))
            n.append(int(n_str))
    return bin_centers, n



def calc_visible_pmi_histogram(bigram_or_path, out_path=None, device=None):
    if isinstance(bigram_or_path, str):
        print('loading bigram...')
        bigram_path = bigram_or_path
        bigram = h.bigram.BigramBase.load(bigram_path, device=device)
    elif isinstance(bigram_or_path, h.bigram.BigramBase):
        bigram = bigram_or_path
    else: raise ValueError(
        "First to argument to calc_visible_pmi_histogram must be a bigram "
        "or path to bigram data"
    )

    print('calculating visible pmi values...')
    visible_pmi = calc_visible_pmi(bigram).reshape(-1)
    n, bins = np.histogram(visible_pmi, bins='auto')
    bin_centers = [ 0.5*(bins[i]+bins[i+1]) for i in range(len(n))]
    if out_path is None:
        return bin_centers, n

    print('writing results to disk...')
    with open(out_path, 'w') as out_file:
        out_file.write(''.join([
            '{}\t{}\n'.format(bin_center, n_item) 
            for bin_center, n_item in zip(bin_centers, n)
        ]))







def calc_visible_pmi(bigram):
    """
    Calculate PMI values, then discard the PMI values corresponding to 
    unobserved pairs.
    """
    Nxx, Nx, Nxt, N = bigram
    pmi = calc_PMI(bigram)
    visible_pmi = pmi[Nxx>0]
    return visible_pmi


def calc_exp_pmi_stats(visible_pmi):
    exp_pmi = np.e**visible_pmi
    return torch.mean(exp_pmi), torch.std(exp_pmi)


def prior_beta_params(pij_expected, exp_mean, exp_std):
    """
    pij_expected = Ni * Nj / N**2
    """
    mean = exp_mean * pij_expected
    std = exp_std * pij_expected**2
    alpha = mean * ( mean*(1-mean)/std - 1)
    beta = (1-mean) * alpha / mean 
    return alpha, beta


def posterior_beta_params(pij_expected, Nij_observed, N, exp_mean, exp_std):
    prior_alpha, prior_beta = prior_beta_params(
        pij_expected, exp_mean, exp_std)
    post_alpha = Nij_observed + prior_alpha
    post_beta = N - Nij_observed + prior_beta
    return post_alpha, post_beta


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


def get_Nij_vals(expected_occurrences):
    #Nij_min = max(Ni * Nj / N * np.e**-6, 0)
    Nij_max = expected_occurrences * np.e**3
    #if Nij_max < 10:
    #    Nij_max = 4
    steps = 10

    Nij = [
        int(np.floor(x))
        for x in np.logspace(0, np.log(Nij_max + 1), steps, base=np.e)
    ]

    if Nij[0] < 10:
        Nij[0] = 0

    return sorted(list(set(Nij)))



def vert_line_at(x, label, y_min=0, y_max=1, color=None):
    if color is None:
        color = 'r'

    plt.plot(
        [x, x], [y_min, y_max],
        linestyle=':', label=label, color=color
    )


def compute_posterior(
    X, expected_occurrence, Nij,
    N=N_10k, pmi_mean=PMI_MEAN, pmi_std=PMI_STD, 
):

    # Calculate and possibly plot the prior
    prior_log_pdf = get_pmi_log_pdf(X)

    # Calculate the cooccurrence probabilities that would obtain for various
    # possible posterior PMI values.
    factor = expected_occurrence / N
    p = factor * np.e**X
    p[p>1] = 1

    # Calculate the likelihood of each posterior PMI value
    bin_log_pdf = np.array([
        stats.binom.logpmf(Nij, N, p_)
        for p_ in p
    ])

    # Calculate the posterior distribution
    post_log_pdf = bin_log_pdf + prior_log_pdf

    #tol = -5
    #post_support_size = sum(post_log_pdf>tol)
    #print('support', post_support_size)
    #if post_support_size < 500:
    #    raise ValueError('bin size too small to calculate KL')

    prior = np.e**prior_log_pdf
    posterior = np.e**post_log_pdf

    normed_posterior = posterior / sum(posterior)
    print('max posterior', max(normed_posterior))

    if max(normed_posterior) > 0.01:
        raise RangeTooLargeError()
    elif normed_posterior[0] > 1e-6 and normed_posterior[-1] > 1e-6:
        raise RangeTooSmallError()
    elif normed_posterior[0] > 1e-6:
        raise RangeTooFarRight()
    elif normed_posterior[-1] > 1e-6:
        raise RangeTooFarLeft()

    return posterior, prior, post_log_pdf, prior_log_pdf


class RangeTooLargeError(Exception):
    pass
class RangeTooSmallError(Exception):
    pass
class RangeTooFarRight(Exception):
    pass
class RangeTooFarLeft(Exception):
    pass


def lognormalize(x, total_desired_weight=1):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a + np.log(total_desired_weight))


def analyze_ij(
    expected_occurrence, empirical_PMI, spread=5,
    N=N_10k, pmi_mean=PMI_MEAN, pmi_std=PMI_STD, plot=True
):

    # Determine the cooccurrence count based on the expected_cooccurrences and
    # the empirical_PMI
    Nij = int(np.floor(expected_occurrence * np.e**empirical_PMI))

    posterior_calculated = False
    while not posterior_calculated:
        low = empirical_PMI - spread/2
        high = empirical_PMI + spread/2
        posterior_delta = (high - low) / 1000
        posterior_X = np.arange(low, high, posterior_delta)
        try:
            posterior, prior, log_post, log_prior = compute_posterior(
                posterior_X, expected_occurrence, Nij,
                N=N, pmi_mean=pmi_mean, pmi_std=pmi_std
            )
        except RangeTooSmallError:
            print('bins too large!')
            spread *= 2
        except RangeTooLargeError:
            print('bins too small!')
            spread /= 2
        except RangeTooFarRight:
            raise NotImplementedError()
        except RangeTooFarLeft:
            raise NotImplementedError()
        else:
            posterior_calculated = True


    total_prior_mass = abs(
        stats.norm.cdf(posterior_X[0]) - stats.norm.cdf(posterior_X[-1]))
    prior_Z = sum(prior) / total_prior_mass
    post_Z = sum(posterior)

    prior = prior / prior_Z
    posterior = posterior / post_Z

    log_prior = log_prior - np.log(prior_Z)
    #log_prior = lognormalize(log_prior, total_prior_mass)
    log_post = log_post - np.log(post_Z)

    KL = kl(posterior, prior, log_post, log_prior)
    MAP_PMI = posterior_X[np.argmax(posterior)]

    # Possibly plot the posterior
    if plot:

        # Visual ajustment of prior because it is plotted using a different
        # bin-size
        visual_range = np.arange(-10, 10, 0.01)
        visual_prior = get_pmi_pdf(visual_range)

        plt.fill_between(
            visual_range, 0, visual_prior, label='prior'
        )
        vert_line_at(empirical_PMI, label='empirical PMI')
        vert_line_at(low, label='lower limit of posterior support')
        vert_line_at(high, label='upper limit of posterior support')
        
        found_prior_value = prior[0]
        non_normalized_prior_value = stats.norm.pdf(
            posterior_X[0], pmi_mean, pmi_std)
        multiplier = non_normalized_prior_value / found_prior_value

        plt.plot(
            posterior_X, posterior * multiplier, 
            label='posterior, NiNj/N={}, Nij={}, emp_PMI={}'
            .format(expected_occurrence, Nij, empirical_PMI),
            color='c'
        )
        plt.plot(
            posterior_X, prior * multiplier, color='y'
        )
        vert_line_at(MAP_PMI, label='MAP PMI', color='c')
        plt.xlim(empirical_PMI - 3*spread, empirical_PMI + 3*spread)

        ax = plt.gca()
        text(
            0.04, 0.7, 
            '$N_{ij}\\;=\\;%d$\n'
            '$\\frac{N_iN_j}{N}\\;=\\;%.2f$\n'
            '$\\mathrm{PMI}_\\mathrm{MLE}\\;=\\;%.2f$\n' 
            '$D_\\mathrm{KL}\\; = \\; %.2f$' % (
                Nij, expected_occurrence, empirical_PMI, KL),
            transform=ax.transAxes
        )

        plt.show()

    return KL, MAP_PMI


def get_pmi_pdf(X, pmi_mean=PMI_MEAN, pmi_std=PMI_STD):
    return stats.norm.pdf(X, pmi_mean, pmi_std)
    


def get_pmi_log_pdf(X, pmi_mean=PMI_MEAN, pmi_std=PMI_STD):
    pmi_log_pdf = stats.norm.logpdf(X, pmi_mean, pmi_std)
    return pmi_log_pdf



COLORS = ['r', 'b', 'g', 'y']
def plot_possible_posteriors(
    expected_occurrences, N=N_10k, pmi_mean=PMI_MEAN, pmi_std=PMI_STD, 
    a=-10, b=10, delta=0.01,
    plot=True
):

    if not isinstance(expected_occurrences, list):
        expected_occurrences = [expected_occurrences]

    X = np.arange(a, b, delta)
    pmi_pdf = get_pmi_pdf(X)

    if plot:
        plt.fill_between(X, 0, pmi_pdf, label='prior')

    for i, expected_occurrence in enumerate(expected_occurrences):

        factor = expected_occurrence/ N
        p = factor * np.e**X
        p[p>1] = 1
        Nij = get_Nij_vals(expected_occurrence)

        for nij in Nij:
            bin_pdf = np.array([
                stats.binom.pmf(nij, N, p_)
                for p_ in p
            ])

            post_pdf = bin_pdf * pmi_pdf
            post_pdf = post_pdf / np.sum(post_pdf)

            if plot:
                plt.plot(
                    X, post_pdf, 
                    label='posterior, Nij={}, X={}'
                    .format(nij, nij/expected_occurrence),
                    color=COLORS[i]
                )
                #plt.legend()

    plt.ylim([0,0.03])
    plt.show()

    return X, post_pdf, pmi_pdf


def get_posterior_numerically(
    Nij, Ni, Nj, N, pmi_mean=PMI_MEAN, pmi_std=PMI_STD, 
    a=-10, b=10, delta=0.1,
    plot=True
):
    X = np.arange(a, b, delta)
    pmi_pdf = np.array([
        stats.norm.pdf(x, pmi_mean, pmi_std)
        for x in X
    ])

    pmi_pdf = pmi_pdf / np.sum(pmi_pdf)
    factor = Ni * Nj / N**2
    p = factor * np.e**X
    p[p>1] = 1

    if not isinstance(Nij, list):
        Nij = [Nij]

    for nij in Nij:
        bin_pdf = np.array([
            stats.binom.pmf(nij, N, p_)
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



def kl(pdf1, pdf2, log_pdf1, log_pdf2):

    # strip out cases where pdf1 is zero
    pdf2 = pdf2[pdf1!=0]
    log_pdf1 = log_pdf1[pdf1!=0]
    log_pdf2 = log_pdf2[pdf1!=0]

    pdf1 = pdf1[pdf1!=0]

    return np.sum(pdf1 * (log_pdf1 - log_pdf2))
    #return np.sum(pdf1 * np.log(pdf1 / pdf2))


def plot_beta(alpha, beta):
    X = np.arange(0, 1, 0.01)
    Y = stats.beta.pdf(X, alpha, beta)
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






