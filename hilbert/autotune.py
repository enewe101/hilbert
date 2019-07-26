import argparse

import torch
from math import log10, floor

import hilbert as h

"""
This program will take any hilbert embedder and automatically
tune the learning rate using a binary search. Its objective is
to maximize the learning rate to be as high as possible without
allowing model divergence. I (Kian) recommend that this be used
over at least 2 iterations if you have no idea what the LR 
should be. This is as follows:
(1) Tune starting at head_lr = 10000, n_iters = 100. This will
    give you a soft upper bound on what the LR should be very
    quickly.
(2) Tune starting with head_lr = "the minimum diverging LR of (1)".
    Then, set n_iters = 1000. This will then give you the best
    learning rate that will (most likely) converge for real.

Note that Glove only implicitly diverges due to the fact that 
it does not use exponents, so we "loss_check" just in case.
"""


# https://stackoverflow.com/questions/3410976/
# how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=4):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def next_lr(divs, unmovings, goods, crt, D=False):
    if D and len(goods) == 0 and len(unmovings) == 0:
        val = crt / 10
    else:
        lowest_div = min(divs)
        highest_conv = max(goods + unmovings)
        val = (lowest_div + highest_conv) / 2

    oriented_lr = round_sig(val)
    if oriented_lr == crt:  # we have converged the learning rate
        return None
    return oriented_lr


def loss_check(losses):
    # check if the loss is going to diverge soon
    going_crazy = True
    for i in range(1, 11):
        if losses[-i] < losses[0]:
            going_crazy = False
            break
    if going_crazy:
        raise h.exceptions.DivergenceError('Diverged!')


def loss_is_stationary(loss_values, minimum_rate=0.1):
    # relative improvement in loss must be at least at
    # a ratio corresponding to the minimum_rate level (10% when it=0.1)
    start = loss_values[0]  # not the max
    end = min(loss_values)
    improvement_rate = abs(1 - (end / start))
    return improvement_rate < minimum_rate


def double_check(solver, obtained_losses, n_iters):
    if len(obtained_losses) < n_iters:
        return []

    # check if the embeddings are blowing up to be too big
    avg_vnorm = torch.mean(solver.learner.V.norm(dim=1))
    avg_wnorm = torch.mean(solver.learner.W.norm(dim=1))
    if avg_vnorm > 50 or avg_wnorm > 50:
        return []

    return obtained_losses


def autotune(
        constructor,
        constr_kwargs,
        n_iters=100,
        head_lr=1e5,
        n_goods=10
):
    div_lrs = []
    stationary_lrs = []
    good_lrs = []

    # we are gonna start by working backwards
    crt_lr = head_lr

    while len(good_lrs) < n_goods and crt_lr is not None:
        print('\nNext test...')

        kwargs = {**constr_kwargs, 'learning_rate': crt_lr}
        solver = constructor(**kwargs)
        print(solver.describe())
        losses = []
        for i in range(n_iters):
            try:
                losses.append(solver.cycle(1))
                if len(losses) > 11:
                    loss_check(losses)
            except h.exceptions.DivergenceError:
                break
        losses = double_check(solver, losses, n_iters)

        if len(losses) < n_iters:
            print('Diverged at lr = {}...'.format(crt_lr))
            div_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=True)

        elif loss_is_stationary(losses):
            print('Stationary at lr = {}...'.format(crt_lr))
            stationary_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=False)

        else:
            print('Learning well at lr = {}...'.format(crt_lr))
            good_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=False)

    print('The good boys:')
    print(good_lrs)
    print('\nDiverging boys:')
    print(div_lrs)
    print('\nToo slow boys:')
    print(stationary_lrs)
    return good_lrs


def make_parser():
    # Make an argument parser; add some arguments that are always applicable.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-iters', type=int, default=1000,
        help='how many cycles to run for each LR tested'
    )
    parser.add_argument(
        '--head-lr', type=float, default=1e5,
        help='estimate of what the maximum possible LR could be'
    )

    # Add the runners for all models
    subparsers = parser.add_subparsers(dest='model')
    mle_parser = subparsers.add_parser('mle')
    h.runners.run_mle.add_model_args(mle_parser)
    glove_parser = subparsers.add_parser('glove')
    h.runners.run_glove.add_model_args(glove_parser)
    sgns_parser = subparsers.add_parser('sgns')
    h.runners.run_sgns.add_model_args(sgns_parser)
    mle_sample_parser = subparsers.add_parser('mle_sample')
    h.runners.run_mle_sample.add_model_args(mle_sample_parser)

    return parser


def main():
    # Build a parser based on the command-line parsers for each model.
    parser = make_parser()

    # Separate the autotune args from model-specific args
    args = vars(parser.parse_args())
    n_iters = args.pop('n_iters')
    head_lr = args.pop('head_lr')
    model = args.pop('model')

    # Apply rc options for directories 
    h.utils.cooc_path(args, 'cooccurrence_path')
    h.utils.emb_path(args, 'init_embeddings_path')

    print('Autotuning model {}!'.format(model))
    constructor = h.factories.get_constructor(model)
    autotune(
        constructor,
        args,
        head_lr=head_lr,
        n_iters=n_iters
    )


if __name__ == '__main__':
    main()
