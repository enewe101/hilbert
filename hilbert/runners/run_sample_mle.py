import hilbert.factories as proletariat
from hilbert.runners.run_base import init_and_run, modify_args, get_base_argparser, kw_filter


def run_sample_mle(
        cooccurrence_path,
        save_embeddings_dir,
        temperature=1.,
        **kwargs
    ):

    embsolver = proletariat.construct_max_likelihood_sample_based_solver(
        cooccurrence_path=cooccurrence_path, temperature=temperature, **kw_filter(kwargs)
    )
    init_and_run(
        embsolver,
        kwargs['epochs'],
        kwargs['iters_per_epoch'],
        shard_times=1,
        save_embeddings_dir=save_embeddings_dir
    )


def validate_args(all_args):
    ignored_args = [
        #'opt_str',   This can probably also be ignored, but keep for now.
        'update_density', 'shard_factor', 'shard_times',
        'datamode', 'tup_n_batches', 'zk'
    ]
    for arg in ignored_args:
        if arg in all_args:
            print('Ignoring inapplicable argument "{}"'.format(arg))
            del all_args[arg]


if __name__ == '__main__':

    base_parser = get_base_argparser()
    base_parser.add_argument(
        '--temperature', '-t', type=float, default=1, dest='temperature',
        help=(
            "equalizes weighting for loss from individual token pairs.  "
            "Use temperature > 1 for more equal weights."
        )
    )
    base_parser.add_argument(
        '--batch-size', '-p', type=int, default=10000,
        help=(
            "Size of sampled batches of (i,j)-pairs used for each update. "
            "Each batch has positive and negative samples, so will have "
            "2*batch_size samples in total."
        )
    )
    base_parser.add_argument(
        '--batches-per-epoch', '-P', type=int, default=1000,
        help=(
            "Number of batches that will be considered as one epoch. "
            "The total number of updates is always batch_size * "
            "batches_per_epoch, and only this product matters.  However, this "
            "will affect how the training routine displays and saves "
            "intermediate results."
        )
    )
    all_args = vars(base_parser.parse_args())
    modify_args(all_args)
    # Some of the standard args are not accepted by the sampling based approach
    validate_args(all_args)
    run_sample_mle(**all_args)



"""
Example command:
python run_sample_mle.py -l 0.01 -s adam -I 100 --init
    std-w2v-s1-t1-v10k-iter5/vectors-init --epochs 150 --seed 1 --cooccurrence
    5w-dynamic-10k/thresh1 -t 2 -o testmle
"""

