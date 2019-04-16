import hilbert as h


if __name__ == '__main__':
    parser = h.runners.run_base.get_base_argparser()
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
    args = parser.parse_args()
    solver = h.factories.build_glove_solver(
        **h.runners.run_base.factory_args(args))
    h.runners.run_base.init_and_run(solver, **args)
