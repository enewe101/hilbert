import hilbert_noshard as h
try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


def apply_effects(base):
    def effected_base(
        cooc_stats, 
        t_undersample=None,
        shift_by=None,
        neg_inf_val=None,
        clip_thresh=None,
        diag=None,
        implementation='torch',
        device='cuda',
        **kwargs
    ):

        # Apply undersampling if desired 
        cooc_stats = undersample(cooc_stats, t_undersample)

        # Calculate the base values for M
        M = base(cooc_stats, **kwargs)

        # Optionally apply a variety of effects
        shift(M, shift_by)
        set_neg_inf(M, neg_inf_val) 
        clip_below(M, clip_thresh) 
        set_diag(M, diag)

        # Cast to the correct datastructure and device before returning.
        return use_implementation(M, implementation, device)

    return effected_base


@apply_effects
def calc_M_pmi(cooc_stats):
    return h.corpus_stats.calc_PMI(cooc_stats)


@apply_effects
def calc_M_logNxx(cooc_stats):
    Nxx, Nx, N = cooc_stats
    with np.errstate(divide='ignore'):
        return np.log(Nxx)


@apply_effects
def calc_M_pmi_star(cooc_stats):
    return h.corpus_stats.calc_PMI_star(cooc_stats)


@apply_effects
def calc_M_neg_samp(
    cooc_stats,
    k_samples=15,
    k_weight=None,
    alpha=0.75
):
    """
    Returns a PMI-like matrix based on negative sampling, simulating sampling 
    used in Mikolov's word2vec.
    ``k_samples`` (int): number of times noise distribution is sampled, per
        corpus sample.  Default is to sample 15 times from noise, for every 1
        sample from the corpus, because it was found to be good in Mikolov's
        2013 paper.
    ``k_weight`` (float): total weight of noise distribution.  If this isn't set
        then the total weight is just equal to the number of negative samples
        per corpus sample.  But if the value is provided, then that's what 
        is used as the weight.
    ``alpha`` (float): exponent applied to unigram distribution, which distorts
        it.  As alpha becomes < 1, it makes the distribution flatter.
        Conceptually, alpha is inverse temperature.
    """
    # Unpack args and apply defaults.
    k_weight = k_samples if k_weight is None else k_weight
    Nxx, Nx, N = cooc_stats

    # Apply unigram distortion
    if alpha is not None:
        distorted_Nx = Nx**alpha
        distorted_N = np.sum(distorted_Nx)
    else:
        distorted_Nx = Nx
        distorted_N = N

    # Draw negative samples
    distorted_px = distorted_Nx / distorted_N
    samples = sample_multi_multinomial(k_samples * Nx, distorted_px)

    # Set negative sample wieght, if provided
    # Note that if k_weight is None, effective weight is k_samples
    if k_weight is not None:
        samples = k_weight * (samples / k_samples)
    # Return the negative sample objective as defined.
    with np.errstate(divide='ignore'):
        return np.log(Nxx) - np.log(samples)


def sample_multi_multinomial(kNx, px):
    kNx = kNx.reshape(-1)
    px = px.reshape(-1)
    samples = np.zeros((len(kNx), len(kNx)))
    for i in range(len(kNx)):
        samples[i,:] = np.random.multinomial(kNx[i], px)
    return samples


def set_diag(M, val=None):
    if val is not None:
        np.fill_diagonal(M, val)


def clip_below(M, thresh=None):
    if thresh is not None:
        M[M<thresh] = thresh


def set_neg_inf(M, val=None):
    if val is not None:
        M[M==-np.inf] = val


def shift(M, val=None):
    if val is not None:
        M += val


def undersample(cooc_stats, t=None):
    if t is None:
        return cooc_stats

    Nxx, Nx, N = cooc_stats

    # Probability that token of a given type is kept.
    p_x = np.sqrt(t * cooc_stats.N / cooc_stats.Nx)
    p_x[p_x > 1] = 1

    # Probability that, for a particular cooccurring pair (i, j)
    # both i and j are kept
    p_xx = p_x * p_x.T

    # Expected number of cooccurrences after applying, undersampling.
    use_Nxx = cooc_stats.denseNxx * p_xx

    # Recalculate unigram distribution, given undersampling
    use_Nx = np.sum(use_Nxx, axis=1, keepdims=True)

    # Recalculate total number of tokens, given undersampling
    use_N = np.sum(use_Nx)

    return use_Nxx, use_Nx, use_N


def use_implementation(M, implementation='torch', device='cuda'):
    h.utils.ensure_implementation_valid(implementation)
    if implementation == 'torch':
        return torch.tensor(M, dtype=torch.float32, device=device)
    elif implementation == 'numpy':
        return M


def calc_M(
    cooc_stats,
    base,
    *args,
    **kwargs
):
    if base == 'pmi':
        return calc_M_pmi(cooc_stats, *args, **kwargs)
    elif base == 'logNxx':
        return calc_M_logNxx(cooc_stats, *args, **kwargs)
    elif base == 'pmi-star':
        return calc_M_pmi_star(cooc_stats, *args, **kwargs)
    elif base == 'neg-samp':
        return calc_M_neg_samp(cooc_stats, *args, **kwargs)
    else:
        raise ValueError(
            "Unexpected base for calculating M: %s.  "
            "Expected one of: 'pmi', 'logNxx', 'pmi-star', or 'neg-samp'."
        )

