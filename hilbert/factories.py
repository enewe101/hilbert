import os
import hilbert as h
import numpy as np
import torch
import time
import torch.optim as op

def get_base_loader(base_loader_str):
    return {
        'parallel': h.loader.MultiLoader,
        'series': h.loader.Loader,
        'buffered': h.loader.BufferedLoader,
    }[base_loader_str]


def get_opt(string):
    s = string.lower()
    d = {
        'sgd': op.SGD,
        'adam': op.Adam,
        'adagrad': op.Adagrad,
    }
    return d[s]


def get_bigram(pth):
    start = time.time()
    bigram = h.bigram.Bigram.load(pth)
    if verbose:
        print('bigrams loading time {}'.format(time.time() - start))
    return bigram


def get_init_embs(pth):
    if pth is None:
        return None
    init_embeddings = h.embeddings.Embeddings.load(pth)
    return init_embeddings.V, init_embeddings.W


def construct_w2v_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        k=15,
        t_clean_undersample=None,
        alpha_unigram_smoothing=0.75,
        update_density=1.,
        mask_diagonal=False,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        num_loaders=1,
        queue_size=1,
        loader_policy='parallel',
        seed=1,
        device=None,
        verbose=True
    ):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    base_loader_class = get_base_loader(loader_policy)

    # Make the loader
    loader = h.bigram_loader.get_loader(
        h.bigram_loader.Word2vecLoader, base_loader_class,
        bigram_path=bigram_path, sector_factor=sector_factor,
        shard_factor=shard_factor, num_loaders=num_loaders,
        k=k, t_clean_undersample=t_clean_undersample,
        alpha_unigram_smoothing=alpha_unigram_smoothing,
        queue_size=queue_size, device=device, verbose=verbose
    )

    # Make the loss.  
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = h.hilbert_loss.Word2vecLoss(
        keep_prob=update_density, ncomponents=vocab**2, 
        mask_diagonal=mask_diagonal
    )

    # get initial embeddings (if any)
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # build the main daddyboy
    embsolver = h.autoembedder.HilbertEmbedderSolver(
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
        verbose=verbose,
        device=device
    )
    if verbose:
        print('finished loading w2v bad boi!')
    return embsolver


def construct_glv_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        alpha=0.75,
        xmax=100,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
        update_density=1.,
        mask_diagonal=False,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        num_loaders=1,
        queue_size=1,
        loader_policy='parallel',
        seed=1,
        device=None,
        verbose=True
    ):

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)


    # Make bigram loader
    base_loader_class = get_base_loader(loader_policy)
    loader = h.bigram_loader.get_loader(
        h.bigram_loader.GloveLoader, base_loader_class,
        bigram_path=bigram_path, sector_factor=sector_factor, 
        shard_factor=shard_factor, num_loaders=num_loaders,
        X_max=xmax, alpha=alpha, t_clean_undersample=t_clean_undersample,
        alpha_unigram_smoothing=alpha_unigram_smoothing, 
        queue_size=queue_size, device=device, verbose=verbose
    )

    # Make the loss
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = h.hilbert_loss.MSELoss(
        keep_prob=update_density, ncomponents=vocab**2, 
        mask_diagonal=mask_diagonal
    )

    # initialize the vectors
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # get the solver and we good!
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        loader=loader,
        loss=loss,
        optimizer_constructor=get_opt(opt_str),
        d=d,
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        dictionary=dictionary,
        shape=shape,
        one_sided=False,
        learn_bias=True,
        seed=seed,
        device=device,
        verbose=verbose
    )
    return embsolver



def construct_max_likelihood_solver(*args, verbose=True, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(
        h.bigram_loader.MaxLikelihoodLoader, h.hilbert_loss.MaxLikelihoodLoss, 
        *args, verbose=verbose, **kwargs
    )
    if verbose:
        print('finished loading max-likelihood bad boi!')
    return solver

def construct_max_posterior_solver(*args, verbose=True, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(
        h.bigram_loader.MaxPosteriorLoader, h.hilbert_loss.MaxPosteriorLoss, 
        *args, verbose=verbose, **kwargs
    )
    if verbose:
        print('finished loading max-posterior bad boi!')
    return solver


def construct_KL_solver(*args, verbose=True, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(
        h.bigram_loader.KLLoader, h.hilbert_loss.KLLoss, 
        *args, verbose=verbose, **kwargs
    )
    if verbose:
        print('finished loading KL bad boi!')
    return solver


def _construct_tempered_solver(
    loader_class,
    loss_class,
    bigram_path,
    init_embeddings_path=None,
    d=300,
    temperature=1,
    t_clean_undersample=None,
    alpha_unigram_smoothing=None,
    update_density=1.,
    mask_diagonal=False,
    learning_rate=0.01,
    opt_str='adam',
    sector_factor=1,
    shard_factor=1,
    num_loaders=1,
    queue_size=1,
    loader_policy='parallel',
    seed=1,
    device=None,
    verbose=True
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Now make the loader.
    base_loader_class = get_base_loader(loader_policy)
    loader = h.bigram_loader.get_loader(
        loader_class, base_loader_class, bigram_path=bigram_path,
        sector_factor=sector_factor, shard_factor=shard_factor,
        num_loaders=num_loaders, t_clean_undersample=t_clean_undersample,
        alpha_unigram_smoothing=alpha_unigram_smoothing, queue_size=queue_size,
        device=device, verbose=verbose
    )

    # Make the loss
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = loss_class(
        keep_prob=update_density, ncomponents=vocab**2, 
        mask_diagonal=mask_diagonal, temperature=temperature
    )

    # Get initial embeddings.
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # Build the main daddyboi!
    embsolver = h.autoembedder.HilbertEmbedderSolver(
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

