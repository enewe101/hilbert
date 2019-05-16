import hilbert as h


def add_model_args(parser):
    h.runners.run_base.add_common_constructor_args(parser)
    h.runners.run_base.add_shard_factor_arg(parser)
    parser.add_argument(
        '--X-max', '-x', type=float, default=100, dest='X_max',
        help="xmax in glove weighting function"
    )
    parser.add_argument(
        '--alpha', '-a', type=float, default=3/4,
        help="exponent in the weighting function for glove"
    )
    parser.add_argument(
        '--nobias', action='store_false', dest='bias',
        help='Set this flag to *remove* bias learning.' 
    )
    return parser


def get_argparser():
    parser = h.runners.run_base.get_argparser()
    add_model_args(parser)
    return parser


def run(**args):
    h.runners.run_base.run(h.factories.build_glove_solver, **args)


if __name__ == '__main__':
    run(**get_argparser().parse_args())
