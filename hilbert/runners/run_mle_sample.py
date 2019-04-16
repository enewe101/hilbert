import hilbert as h


if __name__ == '__main__':
    parser = h.runner.run_base.get_base_argparser()
    parser.add_argument(
        '--temperature', '-t', type=float, default=2, dest='temperature',
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
    solver = h.factories.build_mle_sample_based_solver(
        **h.runners.run_base.factory_args(args)
    )
    h.runners.run_base.init_and_run(solver, **args)


