import hilbert as h
try:
    from scipy import sparse, stats
    import numpy as np
    import torch
except ImportError:
    stats = None
    sparse = None
    np = None
    torch = None


#def calc_M(
#    cooc_stats,
#    base,
#    t_undersample=None,
#    shift_by=None,
#    neg_inf_val=None,
#    clip_thresh=None,
#    diag=None,
#    **kwargs
#):
#    cooc_stats = undersample(cooc_stats, t_undersample)
#    base = _get_base(base)
#    M = base(cooc_stats, **kwargs)
#    return apply_effects(M, shift_by, neg_inf_val, clip_thresh, diag)


class M:

    def __init__(
        self,
        cooc_stats, 
        base,
        t_undersample=None,
        shift_by=None,
        neg_inf_val=None,
        clip_thresh=None,
        diag=None,
        **kwargs
    ):
        self.cooc_stats = undersample(cooc_stats, t_undersample)
        self.base = _get_base(base)
        self.t_undersample = t_undersample
        self.shift_by = shift_by
        self.neg_inf_val = neg_inf_val
        self.clip_thresh = clip_thresh
        self.diag = diag
        self.device = kwargs.pop('device', h.CONSTANTS.MATRIX_DEVICE)
        self.base_args = kwargs


    # TODO: For logNxx base, pre-calculate logNxx for only non-zero elements,
    #   which will be sparse
    def __getitem__(self, shard):
        Nxx, Nx, Nxt, N = self.cooc_stats.load_shard(shard, device=self.device)
        # Calculate the basic elements of M.
        M_shard = self.base((Nxx, Nx, Nxt, N), **self.base_args)
        # Apply effects to M.  Only apply diagonal value for diagonal shards.
        use_diag = self.diag if h.shards.on_diag(shard) else None
        return apply_effects(
            M_shard, self.shift_by, self.neg_inf_val,
            self.clip_thresh, use_diag
        )

    def load_all(self):
        return self[h.shards.whole]



def _get_base(base_or_name, **base_kwargs):
    """
    Allows the base to be specified by name using a string, or by providing
    a callable that expects a CoocStats-like.  If ``base_or_name`` is 
    string-like, then it is treated as a name, otherwise it is assumed to be
    the callable base itself.
    """
    # Internal usage in the hilbert module is always by providing a string.
    # Allowing to pass a callable is for extensibility, should you want to 
    # define a new callable

    if not isinstance(base_or_name, str):
        return base_or_name

    if base_or_name == 'pmi':
        return calc_M_pmi
    elif base_or_name == 'logNxx':
        return calc_M_logNxx
    elif base_or_name == 'pmi-star':
        return calc_M_pmi_star
    #elif base_or_name == 'neg-samp':
    #    return calc_M_neg_samp
    else:
        raise ValueError(
            "Unexpected base for calculating M: %s.  "
            "Expected one of: 'pmi', 'logNxx', 'pmi-star', or 'neg-samp'."
        )


def apply_effects(
    M,
    shift_by=None,
    neg_inf_val=None,
    clip_thresh=None,
    diag=None
):
    # Optionally apply a variety of effects
    shift(M, shift_by)
    set_neg_inf(M, neg_inf_val) 
    clip_below(M, clip_thresh) 
    set_diag(M, diag)
    return M


## BASES ##


def calc_M_pmi(cooc_stats):
    return h.corpus_stats.calc_PMI(cooc_stats)


def calc_M_logNxx(cooc_stats):
    Nxx, Nx, Nxt, N = cooc_stats
    return torch.log(Nxx)


def calc_M_pmi_star(cooc_stats):
    return h.corpus_stats.calc_PMI_star(cooc_stats)


### PRE-EFFECTS ###

def undersample(cooc_stats, t=None):
    """
    Given a true h.cooc_stats.CoocStats instance, produces a new
    h.cooc_stats.CoocStats instance in which common words have been 
    undersampled simulating the rejection of common words in word2vec.

    Returns a true h.cooc_stats.CoocStats instance.
    (In many places, a tuple of (Nxx, Nx, Nxt, N) tensors are treated in a way that
    is equivalent to a h.cooc_stats.CoocStats instance).
    """
    if t is None:
        return cooc_stats

    # Probability that token of a given type is kept.
    p_x = np.sqrt(t * cooc_stats.N / cooc_stats.Nx)
    p_x[p_x > 1] = 1

    # Calculate the elements of p_x * p_x.T that correspond to nonzero elements
    # of Nxx.  That way we keep it sparse.
    p_x = sparse.csr_matrix(p_x)
    I, J = cooc_stats.Nxx.nonzero()
    nonzero_mask = sparse.coo_matrix(
        (np.ones(I.shape),(I,J)),cooc_stats.Nxx.shape).tocsr()
    p_xx = nonzero_mask.multiply(p_x).multiply(p_x.T)

    # Now interpret the non-zero elements of Nxx and corresponding elements
    # as p_xx each as the number of trials and probability of success in 
    # a series of binomial distributions.
    kept_Nxx = stats.binom.rvs(cooc_stats.Nxx[I,J], p_xx[I,J])
    use_Nxx = sparse.coo_matrix((kept_Nxx, (I,J)), cooc_stats.Nxx.shape).tocsr()

    # Expected number of cooccurrences after applying, undersampling.
    use_dictionary = h.dictionary.Dictionary(cooc_stats.dictionary.tokens)

    return h.cooc_stats.CoocStats(dictionary=use_dictionary, Nxx=use_Nxx)



### POST-EFFECTS ###

def set_diag(M, val=None):
    if val is not None:
        h.utils.fill_diagonal(M, val)


def clip_below(M, thresh=None):
    if thresh is not None:
        M[M<thresh] = thresh


def set_neg_inf(M, val=None):
    if val is not None:
        M[M==-np.inf] = val


def shift(M, val=None):
    if val is not None:
        M += val



