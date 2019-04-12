import os
import numpy as np
import torch
import hilbert as h


def get_optimizer(opt_str, parameters, learning_rate):
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
    return optimizers[opt_str](parameters, lr=learning_rate)


def get_init_embs(path, device):
    if path is None:
        return None
    device = h.utils.get_device(device)
    inits = h.embeddings.Embeddings.load(path)
    return [
        None if p is None else p.to(device)
        for p in (inits.V, inits.W, inits.vb, inits.wb)
    ]


def build_sgns_solver(
        cooccurrence_path,
        init_embeddings_path=None,
        d=300,
        bias=False,
        k=15,
        undersampling=None,
        smoothing=0.75,
        learning_rate=0.01,
        opt_str='adam',
        batch_size=1000,
        shard_factor=1,
        seed=1917,
        device=None,
        verbose=True
    ):

    # Apply seed
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    preloader = h.cooccurrence.DenseShardPreloader(
        cooccurrence_path=cooccurrence_path,
        shard_factor=shard_factor,
        undersampling=undersampling,
        smoothing=smoothing,
        verbose=verbose
    )

    loader = h.loaders.Word2vecLoader(
        preloader, verbose=verbose, device=device, k=k)

    dictionary = h.dictionary.Dictionary.load(
           os.path.join(cooccurrence_path, 'dictionary'))

    learner = h.solver.DenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=d,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
    )

    loss = h.loss.Word2vecLoss(ncomponents=len(dictionary)**2)

    optimizer = get_optimizer(opt_str, learner.parameters(), learning_rate)

    solver = h.solver.Solver(
        loader=loader,
        optimizer=optimizer,
        loss=loss,
        learner=learner,
        schedulers=[],
        dictionary=dictionary,
        verbose=verbose
    )

    return solver


### GLOVE ###
def build_glove_solver(
        cooccurrence_path,
        init_embeddings_path=None,
        d=300,
        alpha=0.75,
        X_max=100,
        undersampling=None,
        smoothing=None,
        learning_rate=0.01,
        opt_str='adam',
        seed=1,
        bias=True,
        batch_size=10000,
        shard_factor=1,
        device=None,
        verbose=True
    ):
    if not bias:
        print('NOTE: running GloVe without biases!')

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    preloader = h.cooccurrence.DenseShardPreloader(
        cooccurrence_path,
        shard_factor=shard_factor,
        undersampling=undersampling,
        smoothing=smoothing,
        verbose=verbose
    )

    loader = h.loaders.GloveLoader(
        preloader,
        verbose=verbose,
        device=device,
        X_max=X_max,
        alpha=alpha,
    )

    dictionary = h.dictionary.Dictionary.load(
           os.path.join(cooccurrence_path, 'dictionary'))

    learner = h.solver.DenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=d,
        bias=bias,
        init=get_init_embs(init_embeddings_path, device),
    )

    loss = h.loss.MSELoss(ncomponents=len(dictionary)**2)

    optimizer = get_optimizer(opt_str, learner.parameters(), learning_rate)

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


def build_mle_sample_solver(
    cooccurrence_path,
    init_embeddings_path=None,
    d=300,
    temperature=1,
    learning_rate=0.01,
    opt_str='adam',
    num_writes=10000,
    num_updates=1000,
    batch_size=10000,
    shard_factor=1,
    bias=False,
    seed=1,
    device=None,
    verbose=True
):
    """
    Similar to build_mle_solver, but it is based on 
    approximating the loss function using sampling.
    """

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Make cooccurrence loader
    loader = h.cooccurrence.SampleLoader(
        cooccurrence_path=cooccurrence_path, 
        temperature=temperature,
        batch_size=batch_size,
        device=device, 
        verbose=verbose
    )

    # Make the loss.  This sample based loss is simpler than the others:
    # it doesn't need to know the vocabulary size nor does it make use of
    # update_density.
    loss = h.loss.SampleMaxLikelihoodLoss()

    # initialize the vectors
    init_vecs = get_init_embs(init_embeddings_path, device)
    shape = None
    dictionary_path = os.path.join(cooccurrence_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    if init_vecs is None:
        vocab = len(dictionary)
        shape = (vocab, vocab)

    # get the solver and we good!
    embsolver = h.solver.Solver(
        loader=loader,
        loss=loss,
        optimizer_constructor=get_opt(opt_str),
        d=d,
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        dictionary=dictionary,
        shape=shape,
        one_sided=False,
        learn_bias=False,
        seed=seed,
        device=device,
        learner='sparse',
        verbose=verbose
    )
    return embsolver



def _build_tempered_solver(
    loader_class,
    loss_class,
    cooccurrence_path,
    init_embeddings_path=None,
    d=300,
    temperature=1,
    undersamplint=None,
    smoothing=None,
    learning_rate=0.01,
    opt_str='adam',

    shard_factor=1,
    bias=False,
    batch_size=10000,

    seed=1,
    device=None,
    verbose=True
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make the preloader
    preloader = build_preloader(
        cooccurrence_path,
        shard_factor=shard_factor,
        smoothing=smoothing,
        undersamplint=undersamplint,
        is_w2v=False,
        device=device
    )

    # Now make the loader.
    loader = loader_class(
        preloader,
        verbose=verbose,
        device=device,
    )

    # Make the loss
    dictionary_path = os.path.join(cooccurrence_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = loss_class(
        ncomponents=vocab**2,
        temperature=temperature
    )

    # Get initial embeddings.
    init_vecs = get_init_embs(init_embeddings_path, device)
    shape = (vocab, vocab) if init_vecs is None else None

    # Build the main daddyboi!
    embsolver = h.solver.Solver(
        loader=loader,
        loss=loss,
        optimizer_constructor=get_opt(opt_str),
        d=d,
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        dictionary=dictionary,
        shape=shape,
        one_sided=False,
        learn_bias=False,
        seed=seed,
        device=device,
        verbose=verbose
    )
    return embsolver


def build_mle_solver(*args, verbose=True, **kwargs):
    """
    This factory accepts the same set of arguments as
    _build_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    simple_loss = kwargs.pop('simple_loss', False)
    if simple_loss:
        loss = h.loss.SimpleMaxLikelihoodLoss
        print("USING SIMPLE!")
    else:
        print("Nothing in life is simple...")
        loss = h.loss.MaxLikelihoodLoss

    solver = _build_tempered_solver(
        h.loaders.MaxLikelihoodLoader, loss,
        *args, verbose=verbose, **kwargs
    )
    if verbose:
        print('finished loading max-likelihood bad boi!')
    return solver

