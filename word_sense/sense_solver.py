from hilbert.factories import get_constructor, get_optimizer, get_init_embs, yields_recallable
import torch
import hilbert as h
import numpy as np
import os
from word_sense import sense_learner

# sense factory


def build_sense_mle_solver(
        cooccurrence_path,
        simple_loss=False,  # MLE option
        temperature=2,      # MLE option
        shard_factor=1,     # Dense option
        bias=False,
        init_embeddings_path=None,
        dimensions=300,
        sense=5,
        learning_rate=0.01,
        opt_str='adam',
        seed=616,
        device=None,
        verbose=True,
    ):

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dictionary = h.dictionary.Dictionary.load(
        os.path.join(cooccurrence_path, 'dictionary'))

    if simple_loss:
        loss = h.loss.SimpleMLELoss(ncomponents=len(dictionary)**2)
    else:
        loss = h.loss.MLELoss(ncomponents=len(dictionary)**2)

    learner = sense_learner.SenseLearner(
        vocab=len(dictionary),
        covocab=len(dictionary),
        d=dimensions,
        bias=bias,
        num_sense=sense,
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
