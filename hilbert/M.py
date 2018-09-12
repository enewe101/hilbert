import hilbert as h
import numpy as np
import torch


def calc_M(
    cooc_stats,
    base,
    shift=None,
    no_neg_inf=False,
    positive=False,
    diag=None,
    implementation='torch',
    device='cuda'
):
    """
    Get the matrix of target dot-products.
    `base` ('pmi' | 'logNxx' | 'swivel'): 
        Determines the type of cooccurrence statistic to use as a base for M.
    `shift` (None | float):
        If `shift` is not None, add `log(shift)` to all values
    `no_neg_inf` (False | True):
        If True, set any negative infinity values to zero.
    `positive` (False | True):
        If `positive` is True, set all negative values to 0.
    `diag` (None | float):
        If `diag` is not None, set all diagonal cells to to the value of `diag`.
    """

    # First, get the basic values for M
    if base == 'pmi':
        M = h.corpus_stats.calc_PMI(cooc_stats)
    elif base == 'logNxx':
        with np.errstate(divide='ignore'):
            M = np.log(cooc_stats.denseNxx)
    elif base == 'swivel':
        M = h.corpus_stats.calc_PMI_star(cooc_stats)
    else:
        raise ValueError('Unexpected `base` in get_M: %s' % repr(base))

    if shift is not None:
        M += shift

    if no_neg_inf:
        M[M==-np.inf] = 0

    if positive:
        M[M<0] = 0

    if diag is not None:
        np.fill_diagonal(M, diag)

    if implementation == 'torch':
        return torch.tensor(M, dtype=torch.float32, device=device)
    elif implementation == 'numpy':
        return M
    else:
        raise ValueError(
            'Unexpected `implementation` in get_M: %s.  Should be "torch" or '
            '"numpy".' % repr(base)
        )

