import torch
import hilbert.factories as proletariat
from hilbert.runners import get_base_argparser
from hilbert.embedder import DivergenceError
from progress.bar import IncrementalBar
from math import log10, floor

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
    if D and len(goods) == 0 and len(unmovings)==0:
        val = crt / 10
    else:
        lowest_div = min(divs)
        highest_conv = max(goods + unmovings)
        val = (lowest_div + highest_conv) / 2

    oriented_lr = round_sig(val)
    if oriented_lr == crt: # we have converged the learning rate
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
        raise DivergenceError('Diverged!')


def loss_is_stationary(loss_values, minimum_rate=0.1):
    # relative improvement in loss must be at least at
    # a ratio corresponding to the minimum_rate level (10% when it=0.1)
    start = loss_values[0] # not the max
    end = min(loss_values)
    improvement_rate = abs(1 - (end / start))
    return improvement_rate < minimum_rate


def double_check(embsolver, obtained_losses, n_iters):
    if len(obtained_losses) < n_iters:
        return []

    # check if the embeddings are blowing up to be too big
    avg_vnorm = torch.mean(embsolver.V.norm(dim=1))
    avg_wnorm = torch.mean(embsolver.W.norm(dim=1))
    if avg_vnorm > 50 or avg_wnorm > 50:
        return []

    return obtained_losses


def autotune(constructor, constr_kwargs, n_iters=100, head_lr=1e5, n_goods=10):
    embsolver = constructor(**constr_kwargs)
    embsolver.verbose = False
    div_lrs = []
    stationary_lrs = []
    good_lrs = []

    # we are gonna start by working backwards
    crt_lr = head_lr

    while len(good_lrs) < n_goods and crt_lr is not None:
        print('\nNext test...')
        embsolver.learning_rate = crt_lr
        embsolver.restart()
        bar = IncrementalBar('Cycling {:10}'.format(crt_lr), max=n_iters)
        losses = []
        for i in range(n_iters):
            try:
                losses += embsolver.cycle(1)
                bar.next()
                if len(losses) > 11:
                    loss_check(losses)
            except DivergenceError:
                break
        bar.finish()
        losses = double_check(embsolver, losses, n_iters)

        if len(losses) < n_iters:
            print(f'Diverged at lr = {crt_lr}...')
            div_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=True)

        elif loss_is_stationary(losses):
            print(f'Stationary at lr = {crt_lr}...')
            stationary_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=False)

        else:
            print(f'Learning well at lr = {crt_lr}...')
            good_lrs.append(crt_lr)
            crt_lr = next_lr(div_lrs, stationary_lrs, good_lrs, crt_lr, D=False)

    print('The good boys:')
    print(good_lrs)
    print('\nDiverging boys:')
    print(div_lrs)
    print('\nToo slow boys:')
    print(stationary_lrs)
    return good_lrs


def main():
    base_parser = get_base_argparser()
    base_parser.add_argument(
        '--model', '-m', type=str, default='mle',
        help="what model to autotune: mle, glv, w2v?"
    )
    base_parser.add_argument(
        '--n_iters', type=int, default=100,
        help='how many cycles to run for each LR tested'
    )
    base_parser.add_argument(
        '--head_lr', type=float, default=1e4,
        help='estimate of what the maximum possible LR could be'
    )

    # MLE hyper
    base_parser.add_argument(
        '--temperature', '-T', type=float, default=1, dest='temperature',
        help="equalizes weighting for loss from individual token pairs."
    )

    # GLV hypers
    base_parser.add_argument(
        '--X-max', '-x', type=float, default=100, dest='X_max',
        help="xmax in glove weighting function"
    )
    base_parser.add_argument(
        '--alpha', '-a', type=float, default=3 / 4,
        help="exponent in the weighting function for glove"
    )

    # W2V hypers
    base_parser.add_argument(
        '--t-clean', '-t', type=float, default=2.45e-5, dest='t_clean_undersample',
        help="Post-sampling (clean) Common word undersampling threshold"
    )
    base_parser.add_argument(
        '--neg-samples', '-k', type=int, default=15, dest='k',
        help="number of negative samples"
    )
    base_parser.add_argument(
        '--alpha-smoothing', '-A', type=float, default=0.75, dest='alpha_unigram_smoothing',
        help='context distribution smoothing of PMI'
    )

    # now gotta filter out the kwargs appropriately
    filter_kwargs = {'model', 'n_iters', 'head_lr',
                     'temperature', 'X_max', 'alpha', 'k',
                     'alpha_unigram_smoothing', 't_clean_undersample',
                     'epochs', 'iters_per_epoch',
                     'save_embeddings_dir', 'shard_times'}
    bp_namespace = base_parser.parse_args()

    if bp_namespace.model == 'mle':
        filter_kwargs.remove('temperature')
        constructor = proletariat.construct_max_likelihood_solver

    elif bp_namespace.model == 'glv':
        filter_kwargs.remove('X_max')
        filter_kwargs.remove('alpha')
        constructor = proletariat.construct_glv_solver

    elif bp_namespace.model == 'w2v':
        filter_kwargs.remove('k')
        filter_kwargs.remove('alpha_unigram_smoothing')
        filter_kwargs.remove('t_clean_undersample')
        constructor = proletariat.construct_w2v_solver

    else:
        raise NotImplementedError(f'Model {bp_namespace.model} not implemented!')

    # filter out those kwargs
    constr_kwargs = {**vars(bp_namespace)}
    for kw in filter_kwargs:
        del constr_kwargs[kw]

    # now autotune like Kanye
    print(f'Autotuning model {bp_namespace.model}!')
    autotune(constructor, constr_kwargs,
             head_lr=bp_namespace.head_lr,
             n_iters=bp_namespace.n_iters)


if __name__ == '__main__':
    main()


"""
Example:
python autotune.py -m mle --seed 1 -s adam -b 5w-dynamic-v10k -T 2.0 -o /dev/null
/home/rldata/hilbert-embeddings/cooccurrence/5w-dynamic-v10k
"""