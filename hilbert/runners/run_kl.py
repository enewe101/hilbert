import hilbert.factories as proletariat
from hilbert.runners.run_base import init_and_run, modify_args, get_base_argparser, kw_filter


def run_kl(
        cooccurrence_path,
        save_embeddings_dir,
        temperature=1.,
        **kwargs
    ):

    embsolver = proletariat.construct_KL_solver(
        cooccurrence_path=cooccurrence_path, temperature=temperature,
        **kw_filter(kwargs)
    )
    init_and_run(embsolver,
                 kwargs['epochs'],
                 kwargs['iters_per_epoch'],
                 kwargs['shard_times'],
                 save_embeddings_dir)


if __name__ == '__main__':
    base_parser = get_base_argparser()
    base_parser.add_argument(
        '--temperature', '-t', type=float, default=1, dest='temperature',
        help=(
            "equalizes weighting for loss from individual token pairs.  "
            "Use temperature > 1 for more equal weights."
        )
    )
    all_args = vars(base_parser.parse_args())
    modify_args(all_args)
    run_kl(**all_args)


"""
Example command:
python run_kl.py -u 1.0 -l 0.01 -s adam -I 100 --init
    std-w2v-s1-t1-v10k-iter5/vectors-init --epochs 150 --seed 1 --cooccurrence
    5w-dynamic-10k/thresh1 -t 1 -o testkl
"""

