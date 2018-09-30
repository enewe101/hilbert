import hilbert as h
import time
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None


def sim(word, context, embedder, dictionary):
    word_id = dictionary.get_id(word)
    context_id = dictionary.get_id(context)
    word_vec, context_vec = embedder.W[word_id], embedder.W[context_id]
    product = torch.mm(word_vec, context_vec) 
    word_norm = torch.norm(word_vec)
    context_norm =  torch.norm(context_vec)
    return product / (word_norm * context_norm)


def get_w2v_embedder(
    # Required
    cooc_stats, 

    # Theory options
    k=15, alpha=3./4, t=1e-5, 
    d=300,                     # embedding dimension
    constrainer=None,   # constrainer instances can mutate values after update
    one_sided=False,

    # Implementation options
    solver='sgd',
    undersample_method='expectation', # None | 'sample' | 'expectation'
    shard_factor=10,
    learning_rate=1e-6,
    momentum_decay=0.9,
    verbose=True,
    device='cuda'
):

    # Undersample cooc_stats.
    if t is None:
        print('WARNING: not using common-word undersampling')
    else:
        if undersample_method == 'sample':
            cooc_stats = h.cooc_stats.w2v_undersample(cooc_stats, t)
        elif undersample_method == 'expectation':
            cooc_stats = h.cooc_stats.expectation_w2v_undersample(cooc_stats, t)
        else:
            raise ValueError(
                'If `t` is not `None`, `undersample_method` must be either '
                '"sample" or "expectation".'
            )

    # Smooth unigram distribution.
    cooc_stats = h.cooc_stats.smooth_unigram(cooc_stats, alpha)

    # Make M, delta, and the embedder.
    M = h.M.M(cooc_stats, 'pmi', shift_by=-np.log(k), device=device)
    delta = h.f_delta.DeltaW2V(cooc_stats, M, k, device=device)
    embedder = h.embedder.HilbertEmbedder(
        delta=delta, d=d, learning_rate=learning_rate,
        shard_factor=shard_factor, one_sided=one_sided,
        constrainer=constrainer, verbose=verbose, device=device,
    )

    return embedder

    


# TODO: test that all options are respected
def get_embedder(
    cooc_stats,
    f_delta,            # 'mse' | 'w2v' | 'glove' | 'swivel' | 'mse'
    base,               # 'pmi' | 'logNxx' | 'pmi-star' | ('neg-samp')

    solver='sgd',       # 'sgd' | 'momentum' | 'nesterov' | 'slosh'

    # Options for f_delta
    X_max=None,         # denominator of multiplier (glove only)
    k=None,             # weight of negative samples (w2v only)

    # Options for M
    undersample=None,   # None|'sample'|'expectation' Whether/how to undersample
    t=None,             # Tokens more common than this are undersampled.
    smooth_unigram=None,# Exponent used to smooth unigram.  Use 0.75 for w2v.
    shift_by=None,      # None | float -- shift all vals e.g. -np.log(k)
    neg_inf_val=None,   # None | float -- set any negative infinities to given
    clip_thresh=None,   # None | float -- clip values below thresh to thresh.
    diag=None,          # None | float -- set main diagonal to given value.

    # Options for M if base is 'neg-samp':
    k_samples=1,
    k_weight=None,
    alpha=1.0,

    # Options for embedder
    d=300,              # embedding dimension
    shard_factor=10,
    learning_rate=1e-6,
    one_sided=False,    # whether vectors share parameters with covectors
    constrainer=None,   # constrainer instances can mutate values after update

    # Options for solver
    momentum_decay=0.9,

    # Implementation details
    verbose=True,
    device='cuda'
):

    if undersample is 'sample':
        cooc_stats = w2v_undersample(cooc_stats, t)
    elif undersample is 'expectation':
        cooc_stats = expectation_w2v_undersample(cooc_stats, t)
    elif undersample is not None:
        raise ValueError(
            "Expected None, 'sample', or 'expectation' as values for "
            "undersample.  Found %s" % repr(undersample)
        )

    M_args = {
        'cooc_stats':cooc_stats, 'base':base, #'t_undersample':t_undersample,
        'shift_by':shift_by, 'neg_inf_val':neg_inf_val,
        'clip_thresh':clip_thresh, 'diag':diag,
        'device':device
    }

    ###
    #
    #   The negative sampling base is not currently available.
    #
    #if base == 'neg-samp':
    #    M_args.update({
    #        'k_samples':k_samples, 'k_weight':k_weight, 'alpha':alpha})
    #
    ###
    if base == 'neg-samp':
        raise NotImplemented(
            'The negative sampling base is not currently available.')

    M = h.M.M(**M_args)

    DeltaClass = (
        h.f_delta.DeltaMSE if f_delta=='mse' else
        h.f_delta.DeltaW2V if f_delta=='w2v' else
        h.f_delta.DeltaGlove if f_delta=='glove' else
        h.f_delta.DeltaSwivel if f_delta=='swivel' else
        h.f_delta.DeltaMLE if f_delta=='mle' else
        None
    )

    if DeltaClass is None: 
        raise ValueError('Unexpected value for f_delta: %s' % repr(f_delta))

    f_options = {}
    if f_delta == 'w2v':
        if k is None: 
            raise ValueError(
                'A negative sample weight `k` must be a given when f_delta is '
                '"w2v".'
            )
        f_options['k'] = k
    elif k is not None:
        raise ValueError(
            'Negative sample weight `k` can only be given when f_delta is '
            '"w2v".'
        )

    if f_delta == 'glove':
        if X_max is None: 
            raise ValueError(
                'Multiplier denominator `X_max` must be a given when f_delta '
                'is "glove".'
            )
        f_options['X_max'] = X_max
    elif X_max is not None:
        raise ValueError(
            'Multiplier denominator `X_max` can only be given when f_delta is '
            '"glove".'
        )

    f_delta = DeltaClass(
        cooc_stats, M, device=device, **f_options
    )

    # TODO: delegate validation of option combinations to a validation
    #   subroutine
    embedder = h.embedder.HilbertEmbedder(
        delta=f_delta, d=d, learning_rate=learning_rate, 
        shard_factor=shard_factor,
        one_sided=one_sided, constrainer=constrainer,
        verbose=verbose,
        device=device,
    )

    solver_instance = (
        embedder if solver=='sgd' else
        h.solver.MomentumSolver(embedder, learning_rate, momentum_decay,
            device) if solver=='momentum' else
        h.solver.NesterovSolverOptimized(embedder, learning_rate,
            momentum_decay, device) if solver=='nesterov' else
        h.solver.NesterovSolverCautious(embedder, learning_rate,
            momentum_decay, device) if solver=='slosh' else
        None
    )
    if solver_instance is None:
        raise ValueError('Solver must be one of "sgd", "momentum", "nesterov", '
            'or "slosh".  Got %s' % repr(solver))

    return solver_instance



class HilbertEmbedder(object):

    def __init__(
        self,
        delta,
        d=300,
        num_vecs=None,
        num_covecs=None,
        learning_rate=1e-6,
        one_sided=False,
        constrainer=None,
        shard_factor=10,
        verbose=True,
        device='cuda',
    ):

        self.delta = delta
        self.d = d
        self.num_vecs = num_vecs or self.delta.M.shape[1]
        self.num_covecs = num_covecs or self.delta.M.shape[0]
        self.learning_rate = learning_rate
        self.one_sided = one_sided
        self.constrainer = constrainer
        self.shard_factor = shard_factor
        self.verbose = verbose
        self.device = device

        self.num_pairs = self.num_covecs * self.num_vecs
        if self.one_sided and self.num_covecs != self.num_vecs:
            raise ValueError(
                'A one-sided embedder must have the same number of vectors '
                'and covectors.'
            )
        self.reset()


    def sample_sphere(self):
        sample = torch.rand(
            (self.num_vecs, self.d), device=self.device
        ).mul_(2).sub_(1)
        return sample.div_(torch.norm(sample, 2, dim=1,keepdim=True))


    def reset(self):
        self.V = self.sample_sphere()
        if self.one_sided:
            self.W = self.V
        else:
            self.W = self.sample_sphere()
        self.badness = None


    def get_gradient(self, offsets=None, pass_args=None, verbose=None):
        """ 
        Calculate and return the current gradient.  
            `offsets`: 
                Allowed values: None, dV, (dV, dW) where dV and dW are are
                V.shape and W.shape numpy arrays. Temporarily applies self.V +=
                dV and self.W += dW before calculating the gradient.
            `pass_args`:
                Allowed values: dict of keyword arguments.  Supplies the
                keyword arguments to delta.
        """
        if verbose is None:
            verbose = self.verbose

        pass_args = pass_args or {}
        # Determine the prediction for current embeddings.  Allow an offset to
        # be specified for solvers like Nesterov Accelerated Gradient.
        if offsets is not None:
            if not self.one_sided:
                dV, dW = offsets
                use_V = self.V + dV
                use_W = self.W + dW
            else:
                dV = offsets
                use_V = self.V + dV
                use_W = use_V
        else:
            use_W, use_V = self.W, self.V

        # Determine the gradient, one shard at a time
        nabla_V = torch.zeros_like(self.V)
        if not self.one_sided:
            nabla_W = torch.zeros_like(self.W)
        self.badness = 0
        shards = h.shards.Shards(self.shard_factor)
        for i, shard in enumerate(shards):
            #if self.verbose:
            #    print('Shard ', i)
            # Determine the errors.
            M_hat = torch.mm(use_W[shard[0]], use_V[shard[1]].t())
            start = time.time()
            delta = self.delta.calc_shard(M_hat, shard, **pass_args)
            #if self.verbose:
            #    print('delta calc time: ', time.time() - start)
            self.badness += torch.sum(abs(delta))
            nabla_V[shard[1]] += torch.mm(delta.t(), use_W[shard[0]])
            if not self.one_sided:
                nabla_W[shard[0]] += torch.mm(delta, use_V[shard[1]])

        self.badness /= self.num_pairs

        if self.one_sided:
            return nabla_V
        return nabla_V, nabla_W


        ## Determine the errors.
        #M_hat = torch.mm(use_W, use_V)

        #delta = self.f_delta(M_hat, **pass_args)
        #self.badness = torch.sum(abs(delta)) / (
        #    self.M.shape[0] * self.M.shape[1])

        #
        #nabla_V = torch.mm(use_W.t(), delta)
        #if self.one_sided:
        #    return nabla_V

        #nabla_W = torch.mm(delta, use_V.t())
        #return nabla_V, nabla_W


    def update(self, delta_V=None, delta_W=None):
        if self.one_sided and delta_W is not None:
            raise ValueError(
                "Cannot update covector embeddings (W) for a one-sided model. "
                "Update V instead."
            )
        if delta_V is not None:
            self.V += delta_V
        if delta_W is not None:
            self.W += delta_W
        self.apply_constraints()


    def update_self(self, pass_args=None):
        if self.one_sided:
            nabla_V = self.get_gradient(pass_args=pass_args)
            self.V += nabla_V * self.learning_rate
        else:
            nabla_V, nabla_W = self.get_gradient(pass_args=pass_args)
            self.V += nabla_V * self.learning_rate
            self.W += nabla_W * self.learning_rate


    def apply_constraints(self):
        if self.constrainer is not None:
            self.constrainer(self.W, self.V)


    def cycle(self, times=1, print_badness=True, pass_args=None):
        for i in range(times):
            self.update_self(pass_args)
            self.apply_constraints()
            if print_badness:
                print(self.badness)




