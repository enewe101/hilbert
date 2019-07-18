import os
import numpy as np
import torch
import hilbert as h
import warnings


def get_constructor(model_str):
    return {
        'mle': build_mle_solver,
        'glove': build_glove_solver,
        'sgns': build_sgns_solver,
        'mle_sample': build_mle_sample_solver
    }[model_str]


def get_optimizer(opt_str, learner, learning_rate):
    optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adagrad': torch.optim.Adagrad,
    }
    if opt_str not in optimizers:
        valid_opt_strs = ["{}".format(k) for k in optimizers.keys()]
        valid_opt_strs[-1] = "or " + valid_opt_strs[-1]
        raise ValueError("Optimizer choice be one of '{}'. Got '{}'.".format(
            ', '.join(valid_opt_strs), opt_str
        ))

    return ResettableOptimizer(optimizers[opt_str], learner, learning_rate)


class ResettableOptimizer:
    # Create an underlying optimizer, and memorize the constructor arguments
    def __init__(self, opt_class, learner, lr):
        self.opt_class = opt_class
        self.learner = learner
        self.lr = lr
        self.reset()
    # Delegate everything not found here to the underlying optimizer
    def __getattr__(self, attr):
        return self.opt.__getattribute__(attr)
    # Create the underlying optimizer
    def reset(self, lr=None):
        self.lr = self.lr if lr is None else lr
        self.opt = self.opt_class(self.learner.parameters(), lr=self.lr)


def get_lr_scheduler(scheduler_str, optimizer, start_lr, num_updates, end_learning_rate=None, lr_scheduler_constant_fraction=1, verbose=False):

    if scheduler_str == 'None':
        return []

    scheduler_options = {
        'linear': h.scheduler.LinearLRScheduler,
        'inverse': h.scheduler.InverseLRScheduler,

    }
    # error handling for unrecognized learning rate scheduler string.
    if scheduler_str not in scheduler_options:
        valid_scheduler_strs = ["{}".format(k) for k in scheduler_options.keys()]
        valid_scheduler_strs[-1] = "or " + valid_scheduler_strs[-1]
        raise ValueError("Scheduler choice be one of '{}'. Got '{}'.".format(
            ', '.join(valid_scheduler_strs), scheduler_str
        ))

    if start_lr < 0:
        raise ValueError("Learning rate must be non-negative, got start_lr={} < 0".format(start_lr))

    if end_learning_rate < 0:
        end_learning_rate = 0

    scheduler = scheduler_options[scheduler_str]

    if scheduler_str == 'linear':
        assert end_learning_rate is not None, "End learning rate for linear learning rate scheduler is None."
        return [scheduler(optimizer, start_lr, num_updates, end_learning_rate)]

    elif scheduler_str == 'inverse':
        # num_updates is the total number of updates, inverse scheduler keep start constant learning rate for
        # num_updates * fraction updates

        if lr_scheduler_constant_fraction == 1:
            if verbose:
                warnings.warn("Using inverse learning rate scheduler without setting the constant fraction. Learning rate "
                              "keep constant for all updates.")
        elif lr_scheduler_constant_fraction > 1 or lr_scheduler_constant_fraction <= 0:
            lr_scheduler_constant_fraction = 1
            if verbose:
                print("Constant fraction should be a number betweeen 0 and 1. \n Setting constant fraction to 1..")
        else:
            pass
        return [scheduler(optimizer, start_lr, num_updates * lr_scheduler_constant_fraction)]
    else:
        raise ValueError("Scheduler string not found!")





def get_init_embs(path, device):
    if path is None:
        return None
    device = h.utils.get_device(device)
    inits = h.embeddings.Embeddings.load(path)
    return [
        None if p is None else p.to(device)
        for p in (inits.V, inits.W, inits.vb, inits.wb)
    ]


def yields_recallable(f):
    """
    Given a function f, define a *recallable* function g, such that g returns
    always the same value as f, plus it returns another function h (the
    recall function).  The recall function is defined as follows:

        Given that h is the recall function produced at x:
        g(x) = f(x), h
        Then calling h with no parameters is like calling f at x:
        h() = f(x)

        I.e. h is like f curried with arguments x.

    h can nevertheless be supplied overriding kwargs for arguments initially
    supplied as kwargs to g, and this will evaluate f at the original set of
    arguments, but with any newly supplied kwargs overriding original values.
    """
    def recallable(*args, **kwargs):
        result = f(*args, **kwargs)
        def recall(**kwargs_mod):
            new_kwargs = {**kwargs, **kwargs_mod}
            return f(*args, **new_kwargs)
        return result, recall

    return recallable


def build_mle_sample_solver(
        cooccurrence_path,
        temperature=2,            # MLE option
        batch_size=10000,
        balanced=True,
        gibbs=False,
        gibbs_iteration=1,
        get_distr=False,
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        learning_rate=0.01,
        opt_str='adam',
        scheduler_str=None,
        lr_scheduler_constant_fraction=1,
        end_learning_rate=0,
        num_updates=1,
        min_cooccurrence_count=None,
        seed=1917,
        device=None,
        verbose=True,
        gradient_accumulation=1,
        gradient_clipping=None,
    ):
    """
    Similar to build_mle_solver, but it is based on 
    approximating the loss function using sampling.

    min_cooccurrence_count: A small number threshold of cooc counts to be removed to fit into the
    GPU memory.
    """

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    if balanced:
        print('Keep your balance.')
        loss = h.loss.BalancedSampleMLELoss()
    elif gibbs:
        print('Use Gibbs loss.')
        loss = h.loss.GibbsSampleMLELoss()
    else:
        loss = h.loss.SampleMLELoss()

    learner = h.learner.SampleLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
        device=device
    )
    if gibbs:
        print("Using Gibbs sampling.")
        loader = h.loader.GibbsSampleLoader(
            cooccurrence_path=cooccurrence_path,
            learner=learner,
            gibbs_iteration=gibbs_iteration,
            get_distr=get_distr,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            min_cooccurrence_count=min_cooccurrence_count,
        )
    else:
        if balanced:
            print('CPU loader for balanced samples.')
            loader_class = h.loader.CPUSampleLoader
        else:
            loader_class = h.loader.GPUSampleLoader

        loader = loader_class(
            cooccurrence_path=cooccurrence_path,
            temperature=temperature,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            min_cooccurrence_count=min_cooccurrence_count,
        )


    optimizer = get_optimizer(opt_str, learner, learning_rate)

    if scheduler_str is not None:
        lr_scheduler = get_lr_scheduler(
            scheduler_str=scheduler_str,
            optimizer=optimizer,
            start_lr=learning_rate,
            num_updates=num_updates,
            end_learning_rate=end_learning_rate,
            lr_scheduler_constant_fraction=lr_scheduler_constant_fraction,
            verbose=verbose)


    else:
        lr_scheduler = []

    solver = h.solver.Solver(
        loader=loader,
        loss=loss,
        learner=learner,
        optimizer=optimizer,
        schedulers=lr_scheduler,
        dictionary=dictionary,
        verbose=verbose,
    )

    return solver


def build_multisense_solver(
        cooccurrence_path,
        temperature=2,            # MLE option
        batch_size=10000,
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        num_senses=5,
        learning_rate=0.01,
        opt_str='adam',
        seed=1917,
        device=None,
        verbose=True
    ):
    """
    Similar to build_mle_solver, but it is based on 
    approximating the loss function using sampling.
    """

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    loss = h.loss.BalancedSampleMLELoss()

    learner = h.learner.MultisenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        num_senses=num_senses,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
        device=device
    )

    loader = h.loader.CPUSampleLoader(
        cooccurrence_path=cooccurrence_path, 
        temperature=temperature,
        batch_size=batch_size,
        device=device, 
        verbose=verbose
    )

    optimizer = get_optimizer(opt_str, learner, learning_rate)

    solver = h.solver.Solver(
        loader=loader,
        loss=loss,
        learner=learner,
        optimizer=optimizer,
        schedulers=lr_scheduler,
        dictionary=dictionary,
        verbose=verbose,
    )

    return solver



def build_mle_solver(
        cooccurrence_path,
        temperature=2,      # MLE option
        shard_factor=1,     # Dense option
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        learning_rate=0.01,
        opt_str='adam',
        seed=1917,
        device=None,
        verbose=True,
    ):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    loss = h.loss.MLELoss(ncomponents=len(dictionary)**2)

    learner = h.learner.DenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
        device=device
    )

    loader = h.loader.DenseLoader(
        cooccurrence_path,
        shard_factor,
        include_unigrams=loss.REQUIRES_UNIGRAMS,
        device=device,
        verbose=verbose,
    )

    optimizer = get_optimizer(opt_str, learner, learning_rate)

    solver = h.solver.Solver(
        loader=loader,
        loss=loss,
        learner=learner,
        optimizer=optimizer,
        schedulers=[],
        dictionary=dictionary,
        verbose=verbose,
    )
    return solver


def build_sgns_solver(
        cooccurrence_path,
        k=15,                   # SGNS option
        undersampling=2.45e-5,  # SGNS option
        smoothing=0.75,         # SGNS option
        shard_factor=1,         # Dense option
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        learning_rate=0.01,
        opt_str='adam',
        seed=1917,
        device=None,
        verbose=True,
    ):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    loss = h.loss.SGNSLoss(ncomponents=len(dictionary)**2, k=k)

    learner = h.learner.DenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
        device=device
    )

    loader = h.loader.DenseLoader(
        cooccurrence_path,
        shard_factor,
        include_unigrams=loss.REQUIRES_UNIGRAMS,
        undersampling=undersampling,
        smoothing=smoothing,
        device=device,
        verbose=verbose,
    )

    optimizer = get_optimizer(opt_str, learner, learning_rate)

    solver = h.solver.Solver(
        loader=loader,
        loss=loss,
        learner=learner,
        optimizer=optimizer,
        schedulers=[],
        dictionary=dictionary,
        verbose=verbose,
    )
    return solver



def build_glove_solver(
        cooccurrence_path,
        X_max=100,      # Glove option
        alpha=3/4,      # Glove option
        shard_factor=1, # Dense option
        bias=True,
        init_embeddings_path=None,
        dimensions=300,
        learning_rate=0.01,
        opt_str='adam',
        seed=1917,
        device=None,
        verbose=True,
    ):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    learner = h.learner.DenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
        device=device
    )

    loss = h.loss.GloveLoss(
        ncomponents=len(dictionary)**2, X_max=100, alpha=3/4)

    loader = h.loader.DenseLoader(
        cooccurrence_path,
        shard_factor,
        include_unigrams=loss.REQUIRES_UNIGRAMS,
        device=device,
        verbose=verbose,
    )

    optimizer = get_optimizer(opt_str, learner, learning_rate)

    solver = h.solver.Solver(
        loader=loader,
        loss=loss,
        learner=learner,
        optimizer=optimizer,
        schedulers=[],
        dictionary=dictionary,
        verbose=verbose,
    )
    return solver

