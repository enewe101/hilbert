"""
Example command:
python run_sample_mle.py -l 0.01 -s adam -I 100 --init
    std-w2v-s1-t1-v10k-iter5/vectors-init --epochs 150 --seed 1 --cooccurrence
    5w-dynamic-10k/thresh1 -t 2 -o testmle
"""


import hilbert as h


if __name__ == '__main__':
    parser = h.runner.run_base.get_base_argparser()
    parser.add_argument(
        '--temperature', '-t', type=float, default=1, dest='temperature',
        help=(
            "equalizes weighting for loss from individual token pairs.  "
            "Use temperature > 1 for more equal weights."
        )
    )
    parser.add_argument(
        '--batch-size', '-p', type=int, default=10000,
        help=(
            "Size of sampled batches of (i,j)-pairs used for each update. "
            "Each batch has positive and negative samples, so will have "
            "2*batch_size samples in total."
        )
    )
    args = parser.parse_args()
    solver = h.factories.construct_max_likelihood_sample_based_solver(
        **h.runners.run_base.factory_args(args)
    )
    h.runners.run_base.init_and_run(solver, **args)


