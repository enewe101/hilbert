

import hilbert as h

def add_model_args(parser):
    h.runners.run_base.add_common_constructor_args(parser)
    h.runners.run_base.add_temperature_arg(parser)
    h.runners.run_base.add_bias_arg(parser)
    h.runners.run_base.add_shard_factor_arg(parser)
    parser.add_argument(
        '--simple-loss', '-j', action='store_true', help=(
            "Whether to use the simpler loss function, obtained by neglecting "
            "the denominator of the full loss function after differentiation."
        )
    )
    return parser


def get_argparser():
    parser = h.runners.run_base.get_argparser()
    add_model_args(parser)
    return parser


def run(**args):
    solver = h.factories.build_mle_solver(
        **h.runners.run_base.factory_args(args))
    h.runners.run_base.init_and_run(solver, **args)


if __name__ == '__main__':
    run(get_argparser().parse_args())
