import hilbert as h
import numpy as np
import torch
import time
import torch.optim as op


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
    print('bigrams loading time {}'.format(time.time() - start))
    return bigram


def get_init_embs(pth):
    if pth is not None:
        init_embeddings = h.embeddings.Embeddings.load(pth)
        return init_embeddings.V, init_embeddings.W
    else:
        raise NotImplementedError('Need initial embeddings now!')


def construct_w2v_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        k=15,
        t_clean_undersample=None,
        alpha_smoothing=0.75,
        update_density=1.,
        mask_diagonal=False,
        learning_rate=0.01,
        opt_str='adam',
        shard_factor=1,
        seed=1,
        device=None,
    ):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # first make the bigram then do special things!
    bigram = get_bigram(bigram_path)
    if t_clean_undersample is not None:
        bigram.apply_w2v_undersampling(t_clean_undersample)
    bigram.unigram.apply_smoothing(alpha_smoothing)

    # now make the sharder
    sharder = h.msharder.Word2vecSharder(
        bigram=bigram, k=k, update_density=update_density,
        mask_diagonal=mask_diagonal, device=device
    )

    # get initial embeddings (if any)
    init_vecs = get_init_embs(init_embeddings_path)

    # build the main daddyboy
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        sharder, get_opt(opt_str), d=d, learning_rate=learning_rate,
        init_vecs=init_vecs, shape=None, one_sided=False, learn_bias=False,
        shard_factor=shard_factor, seed=seed, device=device,
    )
    print('finished loading w2v bad boi!')
    return embsolver


def construct_glv_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        alpha=0.75,
        xmax=100,
        update_density=1.,
        mask_diagonal=False,
        learning_rate=0.01,
        opt_str='adam',
        shard_factor=1,
        seed=1,
        device=None,
        nobias=False,
    ):
    if nobias:
        print('NOTE: running GloVe without biases!')

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make bigram and that's all we need for glove
    bigram = get_bigram(bigram_path)
    sharder = h.msharder.GloveSharder(bigram, X_max=xmax, alpha=alpha,
        update_density=update_density, mask_diagonal=mask_diagonal, 
        device=device
    )

    # initialize the vectors
    init_vecs = get_init_embs(init_embeddings_path)

    # get the solver and we good!
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        sharder, get_opt(opt_str), d=d, learning_rate=learning_rate,
        init_vecs=init_vecs, shape=None, one_sided=False, learn_bias=not nobias,
        shard_factor=shard_factor, seed=seed, device=device,
    )
    return embsolver



def construct_max_likelihood_solver(*args, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(
        h.msharder.MaxLikelihoodSharder, *args, **kwargs)
    print('finished loading max-likelihood bad boi!')
    return solver


def construct_max_posterior_solver(*args, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(
        h.msharder.MaxPosteriorSharder, *args, **kwargs)
    print('finished loading max-posterior bad boi!')
    return solver


def construct_KL_solver(*args, **kwargs):
    """
    This factory accepts the same set of arguments as
    _construct_tempered_solver, except for sharder_class (which should not be
    provided here).
    """
    solver = _construct_tempered_solver(h.msharder.KLSharder, *args, **kwargs)
    print('finished loading KL bad boi!')
    return solver


def _construct_tempered_solver(
    sharder_class,
    bigram_path,
    init_embeddings_path=None,
    d=300,
    temperature=1,
    update_density=1.,
    mask_diagonal=False,
    learning_rate=0.01,
    opt_str='adam',
    shard_factor=1,
    seed=1,
    device=None,
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Make the bigram.
    bigram = get_bigram(bigram_path)

    # Now make the sharder.
    sharder = sharder_class(
        bigram=bigram, temperature=temperature, update_density=update_density,
        mask_diagonal=mask_diagonal, device=device
    )

    # Get initial embeddings.
    init_vecs = get_init_embs(init_embeddings_path)

    # Build the main daddyboi!
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        sharder, get_opt(opt_str), d=d, learning_rate=learning_rate,
        init_vecs=init_vecs, shape=None, one_sided=False,
        learn_bias=False, shard_factor=shard_factor, seed=seed, device=device,
    )
    return embsolver

