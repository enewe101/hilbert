

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
        '--simple-loss', '-j', action='store_true', 
        help=(
            "Whether to use the simpler loss function, obtained by neglecting "
            "the denominator of the full loss function after differentiation."
        )
    )
    parser.add_argument(
        '--bias', action='store_true', dest='bias'
        help=(
            "Set this flag to include biases in the model for each vector and "
            "covector"
        )
    )
    args = parser.parser_args()
    solver = h.factories.build_mle_solver(
        **h.runners.run_base.factory_args(args)
    )
    h.runners.run_base.init_and_run(solver, **args)

