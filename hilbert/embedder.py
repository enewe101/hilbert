import hilbert as h
import sys
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
    t_clean=None,       # clean (post-sample) common-word undersampling
    d=300,              # embedding dimension
    init_vecs=None,

    # Implementation options
    update_density=1,
    solver='sgd',
    shard_factor=1,
    learning_rate=1e-6,
    momentum_decay=0.9,
    verbose=True,
    device=None
):

    # TODO: Test!
    # Possibly apply clean common-word undersampling (in expectation).
    if t_clean is not None:
        bigram.apply_w2v_undersampling(t_clean)

    # Smooth unigram distribution.
    bigram.unigram.apply_smoothing(alpha)

    # Note, DeltaW2V no longer needs M!
    #M = h.M.M_w2v(bigram, k=k, device=device)
    M = None
    delta = h.f_delta.DeltaW2V(
        bigram, M, k, update_density=update_density, device=device)

    embedder = h.embedder.HilbertEmbedder(
        delta=delta, d=d, 
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        shape=(bigram.vocab, bigram.vocab),
        shard_factor=shard_factor,
        verbose=verbose, device=device
    )

    solver = h.solver.get_solver(solver, embedder, learning_rate=learning_rate)

    return embedder, solver


def get_w2v_sample_embedder(
    dictionary,
    d=300,
    solver='sgd',
    shard_factor=1,
    learning_rate=1e-3,
    verbose=True,
    device=None
):

    M = h.M.M_w2v(bigram, k=k)
    delta = h.f_delta.DeltaW2V(sample_path, dictionary)
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
    init_vecs=None,
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
    shape = (bigram.vocab,) if one_sided else (bigram.vocab, bigram.vocab)
    embedder = h.embedder.HilbertEmbedder(
        delta=delta, d=d, learning_rate=learning_rate, 
        init_vecs=init_vecs,
        shape=shape,
        shard_factor=shard_factor,
        one_sided=one_sided, constrainer=constrainer,
        verbose=verbose,
        device=device
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



class W2VReplica:

    def __init__(
        self, 
        sampler,
        d=300,
        learning_rate=1e-3,
        delay_update=True,  # Whether to update vectors once at end of sample
        vocab=None,
        verbose=True,
        device=None
    ):
        self.sampler = sampler
        self.d = d
        self.learning_rate = learning_rate
        self.delay_update = delay_update
        if vocab is None:
            vocab = len(sampler.dictionary)
        self.vocab = vocab
        self.verbose = verbose
        self.device = device
        self.reset()


    def reset(self):
        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        self.V = torch.rand((self.vocab, self.d), device=device) - 0.5
        self.W = torch.rand((self.vocab, self.d), device=device) - 0.5



    # TODO: test delay_update
    def cycle(self, times=1):

        device = self.device or h.CONSTANTS.MATRIX_DEVICE
        delta_V = torch.zeros(self.d, device=device)
        cycles_completed = 0

        while times is None or cycles_completed < times:

            delta_V.fill_(0)
            last_token_id1 = None
            for fields in self.sampler.next_sample():

                token_id1, token_id2, val = fields[0], fields[1], fields[2]

                # One sample should provide many word pairs having the
                # same token_id1, but different token_id2's
                if last_token_id1 is not None:
                    assert token_id1 == last_token_id1

                dot = torch.dot(self.V[token_id1], self.W[token_id2])
                g = (val - h.f_delta.sigmoid(dot)) * self.learning_rate
                #print(g, fields[3])

                self.W[token_id2] += g * self.V[token_id1]

                if self.delay_update:
                    delta_V += g * self.W[token_id2]
                else:
                    self.V[token_id1] += g * self.W[token_id2]

                last_token_id1 = token_id1

            if self.delay_update:
                self.V[last_token_id1] += delta_V

            cycles_completed += 1





class HilbertEmbedder(object):

    def __init__(
        self,
        delta,
        d=300,
        learning_rate=1e-6,
        init_vecs = None,
        shape = None,
        one_sided=False,
        constrainer=None,
        shard_factor=10,
        verbose=True,
        device=None,
    ):

        self.delta = delta
        self.d = d
        self.learning_rate = learning_rate
        self.constrainer = constrainer
        self.shard_factor = shard_factor
        self.verbose = verbose
        self.device = device

        # Work out the vector initialization and shape
        self.one_sided = one_sided
        self.V = None       #|
        self.W = None       #| these are set in initialize_vectors().
        self.shape = None   #|
        self.initialize_vectors(shape, init_vecs)


    def initialize_vectors(self, shape, init_vecs):

        if shape is None and init_vecs is None:
            raise ValueError("Provide `shape` or `init_vecs`.")

        if init_vecs is not None:

            # Unpack the vectors
            if isinstance(init_vecs, h.embeddings.Embeddings):
                self.V = init_vecs.V
                if not one_sided:
                    self.W = init_vecs.W
            else:
                if self.one_sided:
                    self.V = init_vecs
                else:
                    self.V, self.W = init_vecs

            if self.one_sided:
                self.shape = (self.V.shape[0],)
            else:
                self.shape = (self.V.shape[0], self.W.shape[0])

        # If  not initial vectors are given, get random ones
        else:
            self.shape = shape
            self.validate_shape()
            self.sample_vectors()


    def validate_shape(self):

        if self.one_sided and len(self.shape) != 1:
            raise ValueError(
                "For one-sided embeddings `shape` should be a "
                "tuple containing a single int, e.g. `(10000,)`."
            )

        if not self.one_sided and len(self.shape) != 2:
            raise ValueError(
                "For two-sided embeddings `shape` should be a "
                "tuple containing two ints, e.g. `(10000,10000)`."
            )


    def sample_vectors(self):

        device = self.device or h.CONSTANTS.MATRIX_DEVICE

        #self.V = h.utils.sample_sphere(self.shape[0], self.d, self.device)
        self.V = (
            torch.rand((self.shape[0], self.d), device=device) - .5
        ) / self.d

        if self.one_sided:
            self.W = self.V
        else:
            #self.W = h.utils.sample_sphere(
            #    self.shape[0], self.d, self.device)
            self.W = (
                torch.rand((self.shape[1], self.d), device=device) - .5
            ) / self.d


        ## Use initialized Vectors if provided...
        #if init_V is not None:
        #    self.V = init_V
        #    if self.V.shape[0] != self.num_vecs:
        #        raise ValueError(
        #            'Incorrect number of vectors provided for initialization.'
        #            'Got {}, expected {}'.format(self.V.shape[0], self.num_vecs)
        #        )

        ## ... or generate them randomly.
        #else:
        #    #self.V = h.utils.sample_sphere(self.num_vecs, self.d, self.device)
        #    self.V = (
        #        torch.rand((self.num_vecs, self.d), device=device) - .5
        #    ) / self.d

        ## Use initialized covectors if provided...
        #if init_W is not None:
        #    if self.one_sided:
        #        raise ValueError(
        #            'One-sided embedder should not have covector '
        #            'initialization'
        #        )
        #    self.W = init_W
        #    if self.W.shape[0] != self.num_covecs:
        #        raise ValueError(
        #            'Incorrect number of covectors provided for '
        #            'initialization. Got {}, expected {}.'
        #            .format(self.W.shape[0], self.num_covecs)
        #        )

        ## ... or generate them randomly.
        #else:
        #    if self.one_sided:
        #        self.W = self.V
        #    else:
        #        #self.W = h.utils.sample_sphere(
        #        #    self.num_vecs, self.d, self.device)
        #        self.W = (
        #            torch.rand((self.num_vecs, self.d), device=device) - .5
        #        ) / self.d

        #self.badness = None


    def get_gradient(
        self,
        shard=None,
        offsets=None,
        pass_args=None,
        verbose=None
    ):
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
        if shard is None:
            shard = h.shards.whole

        if verbose is None:
            verbose = self.verbose

        pass_args = pass_args or {}
        # Determine the prediction for current embeddings.  Allow an offset to
        # be specified for solvers like Nesterov Accelerated Gradient.
        if offsets is not None:
            if not self.one_sided:
                dV, dW = offsets
                use_V = self.V[shard[1]] + dV
                use_W = self.W[shard[0]] + dW
            else:
                dV = offsets
                use_V = self.V[shard[1]] + dV
                use_W = use_V
        else:
            use_W, use_V = self.W[shard[0]], self.V[shard[1]]

        # Determine the gradient for this shard
        M_hat = torch.mm(use_W, use_V.t())
        delta = self.delta.calc_shard(M_hat, shard, **pass_args)
        self.badness = torch.sum(abs(delta)) / (delta.shape[0] * delta.shape[1])
        nabla_V = torch.mm(delta.t(), use_W)
        nabla_W = torch.mm(delta, use_V)

        return nabla_V, nabla_W


    # TODO: adapt handlingn of one-sided: Now updates are being applied on a
    # shard-by-shard basis, which means that there are distinct V and W
    # updates, even for one-sided embedders, because V and W will represent
    # different parts of V.
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


    def update_self(self, shard=None, pass_args=None):
        nabla_V, nabla_W = self.get_gradient(shard=shard, pass_args=pass_args)
        self.W[shard[0]] += nabla_W * self.learning_rate
        self.V[shard[1]] += nabla_V * self.learning_rate


    def apply_constraints(self):
        if self.constrainer is not None:
            self.constrainer(self.W, self.V)


    # TODO: rename times => epochs
    # TODO: test making times able to be zero for unlimited cycling.
    def cycle(self, times=1, shard_times=1, pass_args=None):
        cycles_completed = 0
        while times is None or cycles_completed < times:
            for shard in h.shards.Shards(self.shard_factor):
                for shard_time in range(shard_times):
                    self.update_self(shard, pass_args)
                    self.apply_constraints()

            if self.verbose:
                print('badness\t{}'.format(self.badness.item()))

            cycles_completed += 1




