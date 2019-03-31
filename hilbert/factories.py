import os
import numpy as np
import torch
import torch.optim as op
import hilbert as h
from hilbert.bigram import DenseShardPreloader, SparsePreloader


def get_opt(string):
    s = string.lower()
    d = {
        'sgd': op.SGD,
        'adam': op.Adam,
        'adagrad': op.Adagrad,
    }
    return d[s]


def get_init_embs(pth):
    if pth is None:
        return None
    init_embeddings = h.embeddings.Embeddings.load(pth)
    return init_embeddings.V, init_embeddings.W


def build_preloader(
        bigram_path,
        sector_factor=1,
        shard_factor=1,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
        sparse=False,
        is_w2v=False,
        is_mle=False,
        device=None
    ):
    if sparse:
        # if not is_mle:
        #     raise NotImplementedError('Sparse only works for MLE (for now)!')

        preloader = SparsePreloader(
            bigram_path, zk=1000,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
            filter_repeats=False,
            include_unigram_data=is_w2v,
            device=device,
        )
    else:
        preloader = DenseShardPreloader(
            bigram_path, sector_factor, shard_factor,
            t_clean_undersample=t_clean_undersample,
            alpha_unigram_smoothing=alpha_unigram_smoothing,
        )
    return preloader


### Word2vec ###
def construct_w2v_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        k=15,
        t_clean_undersample=None,
        alpha_unigram_smoothing=0.75,
        update_density=1.,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        seed=1,
        sparse=False,
        device=None,
        verbose=True
    ):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make the preloader
    preloader = build_preloader(
        bigram_path,
        sector_factor=sector_factor,
        shard_factor=shard_factor,
        alpha_unigram_smoothing=alpha_unigram_smoothing,
        t_clean_undersample=t_clean_undersample,
        sparse=sparse,
        is_w2v=True,
        device=device
    )

    # Make the loader
    loader = h.model_loaders.Word2vecLoader(
        preloader,
        verbose=verbose,
        device=device,
        k=k,
    )

    # Make the loss.  
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = h.hilbert_loss.Word2vecLoss(
        keep_prob=update_density, ncomponents=vocab**2, 
    )

    # get initial embeddings (if any)
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # build the main daddyboy
    embsolver = h.embedder.HilbertEmbedderSolver(
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
        learner='sparse' if sparse else 'dense',
        device=device
    )
    if verbose:
        print('finished loading w2v bad boi!')
    return embsolver


### GLOVE ###
def construct_glv_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        alpha=0.75,
        X_max=100,
        t_clean_undersample=None,
        alpha_unigram_smoothing=None,
        update_density=1.,
        learning_rate=0.01,
        opt_str='adam',
        sector_factor=1,
        shard_factor=1,
        seed=1,
        device=None,
        nobias=False,
        sparse=False,
        verbose=True
    ):
    if nobias:
        print('NOTE: running GloVe without biases!')

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make the preloader
    preloader = build_preloader(
        bigram_path,
        sector_factor=sector_factor,
        shard_factor=shard_factor,
        alpha_unigram_smoothing=alpha_unigram_smoothing,
        t_clean_undersample=t_clean_undersample,
        sparse=sparse,
        is_w2v=False,
        device=device
    )

    # Make bigram loader
    loader = h.model_loaders.GloveLoader(
        preloader,
        verbose=verbose,
        device=device,
        X_max=X_max,
        alpha=alpha,
    )

    # Make the loss
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = h.hilbert_loss.MSELoss(
        keep_prob=update_density, ncomponents=vocab**2, 
    )

    # initialize the vectors
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # get the solver and we good!
    embsolver = h.embedder.HilbertEmbedderSolver(
        loader=loader,
        loss=loss,
        optimizer_constructor=get_opt(opt_str),
        d=d,
        learning_rate=learning_rate,
        init_vecs=init_vecs,
        dictionary=dictionary,
        shape=shape,
        one_sided=False,
        learn_bias=not nobias,
        seed=seed,
        device=device,
        learner='sparse' if sparse else 'dense',
        verbose=verbose
    )
    return embsolver


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
    learning_rate=0.01,
    opt_str='adam',
    sector_factor=1,
    shard_factor=1,
    seed=1,
    device=None,
    sparse=False,
    verbose=True
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make the preloader
    preloader = build_preloader(
        bigram_path,
        sector_factor=sector_factor,
        shard_factor=shard_factor,
        alpha_unigram_smoothing=alpha_unigram_smoothing,
        t_clean_undersample=t_clean_undersample,
        sparse=sparse,
        is_w2v=False,
        is_mle=True,
        device=device
    )

    # Now make the loader.
    loader = loader_class(
        preloader,
        verbose=verbose,
        device=device,
    )

    # Make the loss
    dictionary_path = os.path.join(bigram_path, 'dictionary')
    dictionary = h.dictionary.Dictionary.load(dictionary_path)
    vocab = len(dictionary)
    loss = loss_class(
        keep_prob=update_density,
        ncomponents=vocab**2,
        temperature=temperature
    )

    # Get initial embeddings.
    init_vecs = get_init_embs(init_embeddings_path)
    shape = None
    if init_vecs is None:
        shape = (vocab, vocab)

    # Build the main daddyboi!
    embsolver = h.embedder.HilbertEmbedderSolver(
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
        learner='sparse' if sparse else 'dense',
        verbose=verbose
    )
    return embsolver


def construct_max_likelihood_solver(*args, verbose=True, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    simple_loss = kwargs.pop('simple_loss', False)
    if simple_loss:
        loss = h.hilbert_loss.SimpleMaxLikelihoodLoss
        print("USING SIMPLE!")
    else:
        print("Nothing in life is simple...")
        loss = h.hilbert_loss.MaxLikelihoodLoss

    solver = _construct_tempered_solver(
        h.model_loaders.MaxLikelihoodLoader, loss,
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
        h.model_loaders.MaxPosteriorLoader, h.hilbert_loss.MaxPosteriorLoss,
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
        h.model_loaders.KLLoader, h.hilbert_loss.KLLoss,
        *args, verbose=verbose, **kwargs
    )
    if verbose:
        print('finished loading KL bad boi!')
    return solver
