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


def get_expectation_M_w2v(cooc_stats, k, t, alpha):
    return M(
        cooc_stats, 'pmi', t_undersample=t, undersample_method='expectation',
        unigram_exponent=alpha, shift_by=-np.log(k)
    )

def get_sample_M_w2v(cooc_stats, k, t, alpha):
    return M(
        cooc_stats, 'pmi', t_undersample=t, undersample_method='sample',
        unigram_exponent=alpha, shift_by=-np.log(k)
    )


class M:

    def __init__(
        self,
        cooc_stats, 
        base,
        t_undersample=None,
        undersample_method=None,
        unigram_exponent=None,
        shift_by=None,
        neg_inf_val=None,
        clip_thresh=None,
        diag=None,
        **kwargs
    ):

        # First do undersampling on cooc_stats if desired
        if t_undersample is None:
            self.cooc_stats = cooc_stats
        else:
            if undersample_method == 'sample':
                self.cooc_stats = h.cooc_stats.w2v_undersample(
                    cooc_stats, t_undersample)
            elif undersample_method == 'expectation':
                self.cooc_stats = h.cooc_stats.expectation_w2v_undersample(
                    cooc_stats, t_undersample)
            else:
                raise ValueError(
                    'Undersample method must be either "sample" or '
                    '"expectation".'
                )

        # Applies unigram_distortion if unigram_exponent is not None
        self.cooc_stats = h.cooc_stats.smooth_unigram(
            self.cooc_stats, unigram_exponent)

        self.base = _get_base(base)
        self.shift_by = shift_by
        self.neg_inf_val = neg_inf_val
        self.clip_thresh = clip_thresh
        self.diag = diag
        self.device = kwargs.pop('device', h.CONSTANTS.MATRIX_DEVICE)
        self.base_args = kwargs

        self.shape = self.cooc_stats.Nxx.shape


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



### EFFECTS ###


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


