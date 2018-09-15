import hilbert as h
try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


def sim(word, context, embedder, dictionary):
    word_id = dictionary.get_id(word)
    context_id = dictionary.get_id(context)
    word_vec, context_vec = embedder.W[word_id], embedder.W[context_id]
    product = np.dot(word_vec, context_vec) 
    word_norm = np.linalg.norm(word_vec)
    context_norm =  np.linalg.norm(context_vec)
    return product / (word_norm * context_norm)


def get_embedder(
    cooc_stats,
    f_delta,            # 'mse' | 'w2v' | 'glove' | 'swivel' | 'mse'
    base,               # 'pmi' | 'logNxx' | 'swivel'

    solver='sgd',       # 'sgd' | 'momentum' | 'nesterov' | 'slosh'

    # Options for f_delta
    X_max=None,         # denominator of multiplier (glove only)
    k=None,             # weight of negative samples (w2v only)

    # Options for M
    shift=None,         # None | float -- shift all vals e.g. -np.log(k)
    no_neg_inf=False,   # whether to set -np.inf values to zero.
    positive=False,     # whether to clip negative values to zero.
    diag=None,          # None | float -- set diagonals to this val

    # TODO: Add use noise samples, use noise smoothing

    # Options for embedder
    d=300,              # embedding dimension
    learning_rate=1e-5,
    one_sided=False,    # whether vectors share parameters with covectors
    constrainer=None,   # constrainer instances can mutate values after update
    pass_args={},       # kwargs to pass into f_delta when it is called

    # Options for solver
    momentum_decay=0.9,

    # Implementation details
    implementation='torch',
    device='cuda'
):

    h.utils.ensure_implementation_valid(implementation)
    M = h.M.calc_M(
        cooc_stats, base, shift, no_neg_inf, positive, diag, 
        implementation, device
    )
    f_getter = (
        h.f_delta.get_f_MSE if f_delta=='mse' else
        h.f_delta.get_f_w2v if f_delta=='w2v' else
        h.f_delta.get_f_glove if f_delta=='glove' else
        h.f_delta.get_f_swivel if f_delta=='swivel' else
        h.f_delta.get_f_MLE if f_delta=='mle' else
        None
    )

    if f_getter is None: 
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

    f_delta = f_getter(
        cooc_stats, M, implementation=implementation,
        device=device, **f_options
    )

    if implementation == 'torch':
        embedder = h.torch_embedder.TorchHilbertEmbedder(
            M, f_delta, d, learning_rate, one_sided, constrainer, pass_args,
            device,
        )
    else:
        embedder = HilbertEmbedder(
            M, f_delta, d, learning_rate, one_sided, constrainer, pass_args)

    solver_instance = (
        embedder if solver=='sgd' else
        h.solver.MomentumSolver(embedder, learning_rate, momentum_decay,
            implementation, device) if solver=='momentum' else
        h.solver.NesterovSolverOptimized(embedder, learning_rate,
            momentum_decay, implementation, device) if solver=='nesterov' else
        h.solver.NesterovSolverCautious(embedder, learning_rate,
            momentum_decay, implementation, device) if solver=='slosh' else
        None
    )
    if solver_instance is None:
        raise ValueError('Solver must be one of "sgd", "momentum", "nesterov", '
            'or "slosh".  Got %s' % repr(solver))

    return solver_instance





# TODO: enable sharding
class HilbertEmbedder(object):

    def __init__(
        self,
        M,
        f_delta,
        d=300,
        learning_rate=1e-6,
        one_sided=False,
        constrainer=None,
        pass_args={}
    ):
        self.M = M
        self.d = d
        self.f_delta = f_delta
        self.learning_rate = learning_rate
        self.one_sided = one_sided
        self.constrainer = constrainer

        self.num_covecs, self.num_vecs = self.M.shape
        self.num_pairs = self.num_covecs * self.num_vecs
        if self.one_sided and self.num_covecs != self.num_vecs:
            raise ValueError('M must be square for a one-sided embedder.')
        self.reset()
        #self.measure(**pass_args)


    def sample_sphere(self):
        sample = np.random.random(
            (self.d, self.num_vecs)) * 2 - 1
        norms = np.linalg.norm(sample, axis=1).reshape((-1,1))
        return np.divide(sample, norms, sample)


    def reset(self):
        self.V = self.sample_sphere()
        self.temp_V = np.zeros(self.V.shape)
        self.nabla_V = np.zeros(self.V.shape)
        self.update_V = np.zeros(self.V.shape)
        if self.one_sided:
            self.W = self.V.T
            self.temp_W = self.temp_V.T
            self.nabla_W = self.nabla_V.T
            self.update_W = self.update_V.T
        else:
            self.W = self.sample_sphere().T
            self.temp_W = np.zeros(self.W.shape)
            self.nabla_W = np.zeros(self.W.shape)
            self.update_W = np.zeros(self.W.shape)
        self.M_hat = np.zeros(self.M.shape)
        self.delta = np.zeros(self.M.shape)
        self.badness = None


    def calc_badness(self):
        total_absolute_error = np.sum(abs(self.delta))
        num_cells = (self.M.shape[0] * self.M.shape[1])
        self.badness = total_absolute_error / num_cells
        return self.badness


    # TODO: Test. (esp. that offsets work.)
    def get_gradient(self, offsets=None, pass_args=None):
        """ 
        Calculate and return the current gradient.  
            offsets: 
                Allowed values: None, (dV, dW)
                    where dV and dW are is a V.shape and W.shape numpy arrays
                Temporarily applies self.V += dV and self.W += dW before 
                calculating the gradient.
            pass_args:
                Allowed values: dict of keyword arguments.
                Supplies the keyword arguments to f_delta.
        """

        pass_args = pass_args or {}
        # Determine the prediction for current embeddings.  Allow an offset to
        # be specified for solvers like Nesterov Accelerated Gradient.
        if offsets is not None:
            use_W, use_V = self.temp_W, self.temp_V
            if not self.one_sided:
                dV, dW = offsets
                np.add(self.V, dV, use_V)
                np.add(self.W, dW, use_W)
            else:
                dV = offsets
                np.add(self.V, dV, use_V)
        else:
            use_W, use_V = self.W, self.V

        np.dot(use_W, use_V, self.M_hat)

        # Determine the errors.
        self.delta = self.f_delta(self.M_hat, **pass_args)

        # Determine the gradient
        np.dot(use_W.T, self.delta, self.nabla_V)
        
        if self.one_sided:
            return self.nabla_V

        np.dot(self.delta, use_V.T, self.nabla_W)
        return self.nabla_V, self.nabla_W



    def update(self, delta_V=None, delta_W=None):
        if self.one_sided and delta_W is not None:
            raise ValueError(
                "Cannot update covector embeddings (W) for a one-sided model. "
                "Update V instead."
            )
        if delta_V is not None:
            np.add(delta_V, self.V, self.V)
        if delta_W is not None:
            np.add(delta_W, self.W, self.W)
        self.apply_constraints()


    def update_self(self, pass_args=None):
        self.get_gradient(pass_args=pass_args)
        np.multiply(self.learning_rate, self.nabla_V, self.update_V)
        np.add(self.V, self.update_V, self.V)
        if not self.one_sided:
            np.multiply(self.learning_rate, self.nabla_W, self.update_W)
            np.add(self.W, self.update_W, self.W)


    def apply_constraints(self):
        if self.constrainer is not None:
            self.constrainer(self.W, self.V)


    def cycle(self, times=1, print_badness=True, pass_args=None):
        pass_args = pass_args or {}
        for i in range(times):
            self.update_self(pass_args)
            self.apply_constraints()
            if print_badness:
                print(self.calc_badness())


    def project(self, new_d):

        delta_dim = abs(self.d - new_d)
        if delta_dim == 0:
            print('warning: no change during projection.')
            return

        elif new_d < self.d:
            mass = 1.0 / new_d
            random_projector = np.random.random((delta_dim, new_d)) * mass
            downsampler = np.append(np.eye(new_d), random_projector, axis=0)
            self.W = np.dot(self.W, downsampler)
            self.V = np.dot(downsampler.T, self.V)

        else:
            old_mass = float(self.d) / new_d
            new_mass = float(delta_dim) / new_d
            covector_extension = (np.random.random((
                self.num_covecs, delta_dim)) * 2 - 1) * new_mass
            self.W = np.append(self.W * old_mass, covector_extension, axis=1)
            vector_extension = (np.random.random((
                delta_dim, self.num_vecs)) * 2 - 1) * new_mass
            self.V = np.append(self.V * old_mass, vector_extension, axis=0)



