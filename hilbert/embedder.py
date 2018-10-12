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
    bigram, 

    # Theory options
    k=15,               # negative sample weight
    alpha=3./4,         # unigram smoothing exponent
    d=300,              # embedding dimension

    # Implementation options
    solver='sgd',
    shard_factor=1,
    learning_rate=1e-6,
    momentum_decay=0.9,
    verbose=True,
    device=None
):

    # Smooth unigram distribution.
    bigram.unigram.apply_smoothing(alpha)

    # Make M, delta, and the embedder.
    M = h.M.M_w2v(bigram, k=k)
    delta = h.f_delta.DeltaW2V(bigram, M, k, device=device)
    embedder = h.embedder.HilbertEmbedder(
        delta=delta, d=d, learning_rate=learning_rate,
        shard_factor=shard_factor,
        verbose=verbose, device=device
    )

    return embedder

    


# TODO: test that all options are respected
def get_embedder(
    bigram,
    delta,            # 'mse' | 'mle' | 'w2v' | 'glove' | 'swivel'
    base,               # 'pmi' | 'w2v' | 'logNxx' | 'pmi_star'  
    solver='sgd',       # 'sgd' | 'momentum' | 'nesterov' | 'slosh'

    # Glove-specific options
    X_max=None,         # denominator of multiplier (glove only)

    # w2v-specific options
    k=None,             # weight of negative samples (w2v only)

    # Options for M
    smooth_unigram=None,# Exponent used to smooth unigram.  Use 0.75 for w2v.
    shift_by=None,      # None | float -- shift all vals e.g. -np.log(k)
    neg_inf_val=None,   # None | float -- set any negative infinities to given
    clip_thresh=None,   # None | float -- clip values below thresh to thresh.
    diag=None,          # None | float -- set main diagonal to given value.

    # Options for embedder
    d=300,              # embedding dimension
    shard_factor=10,
    learning_rate=1e-6,
    one_sided=False,    # whether vectors share parameters with covectors
    constrainer=None,   # constrainer instances can mutate values after update

    # Options for solver
    momentum_decay=0.9,

    # Misc.
    verbose=True,
    device=None
):

    # Smooth unigram distribution.
    bigram.unigram.apply_smoothing(smooth_unigram)

    # Create the M instance.
    M_args = {}
    if base == 'w2v': M_args['k'] = k
    M = h.M.get_M(
        base, bigram=bigram, shift_by=shift_by, neg_inf_val=neg_inf_val, 
        clip_thresh=clip_thresh, diag=diag, **M_args)

    # Create the delta instance.
    delta_args = {'bigram':bigram, 'M':M, 'device':device}
    if delta == 'w2v': delta_args['k'] = k
    elif delta == 'glove': delta_args['X_max'] = X_max
    delta = h.f_delta.get_delta(delta, **delta_args)

    # Create the embedder.
    embedder = h.embedder.HilbertEmbedder(
        delta=delta, d=d, learning_rate=learning_rate, 
        shard_factor=shard_factor,
        one_sided=one_sided, constrainer=constrainer,
        verbose=verbose,
        device=device,
    )

    # Create the solver, if desired.
    solver_instance = (
        embedder if solver=='sgd' else
        h.solver.MomentumSolver(embedder, learning_rate, momentum_decay,
            device=device) if solver=='momentum' else
        h.solver.NesterovSolverOptimized(embedder, learning_rate,
            momentum_decay, device=device) if solver=='nesterov' else
        h.solver.NesterovSolverCautious(embedder, learning_rate,
            momentum_decay, device=device) if solver=='slosh' else
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
        device=None,
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


    def sample_sphere(self, device=None):
        device = device or self.device or h.CONSTANTS.MATRIX_DEVICE
        sample = torch.rand(
            (self.num_vecs, self.d), device=device
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
                print('badness\t{}'.format(self.badness.item()))




