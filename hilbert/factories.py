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
    sharder = h.msharder.Word2vecSharder(bigram, k, update_density, device)

    # get initial embeddings (if any)
    init_vecs = get_init_embs(init_embeddings_path)

    # build the main daddyboy
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        sharder, get_opt(opt_str), d=d, learning_rate=learning_rate,
        init_vecs=init_vecs, shape=None, one_sided=False, learn_bias=False,
        shard_factor=shard_factor, seed=seed, device=device,
    )
    print('finished loading w2v bad boy!')
    return embsolver


def construct_glv_solver(
        bigram_path,
        init_embeddings_path=None,
        d=300,
        alpha=0.75,
        xmax=100,
        update_density=1.,
        learning_rate=0.01,
        opt_str='adam',
        shard_factor=1,
        seed=1,
        device=None,
    ):

    # repeatability
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # make bigram and that's all we need for glove
    bigram = get_bigram(bigram_path)
    sharder = h.msharder.GloveSharder(bigram, X_max=xmax, alpha=alpha,
        update_density=update_density, device=device)

    # initialize the vectors
    init_vecs = get_init_embs(init_embeddings_path)

    # get the solver and we good!
    embsolver = h.autoembedder.HilbertEmbedderSolver(
        sharder, get_opt(opt_str), d=d, learning_rate=learning_rate,
        init_vecs=init_vecs, shape=None, one_sided=False, learn_bias=True,
        shard_factor=shard_factor, seed=seed, device=device,
    )
    return embsolver

