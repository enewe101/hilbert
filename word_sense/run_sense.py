from hilbert.runners import run_base
import hilbert as h
from word_sense import sense_solver


def add_model_args(parser):
    h.runners.run_base.add_common_constructor_args(parser)
    h.runners.run_base.add_shard_factor_arg(parser)
    parser.add_argument('--sense', '-k', default=5, type=int, help="number of senses for each word.")
    parser.add_argument(
        '--simple-loss', '-j', action='store_true', help=(
            "Whether to use the simpler loss function, obtained by neglecting "
            "the denominator of the full loss function after differentiation."
        )
    )
    return parser


def get_argparser():
    parser = run_base.get_argparser()
    add_model_args(parser)
    return parser


def run(**args):
    run_base.run(sense_solver.build_sense_mle_solver, **args)


if __name__ == '__main__':
    run(**get_argparser().parse_args())
