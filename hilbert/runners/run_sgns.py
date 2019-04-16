import hilbert as h


if __name__ == '__main__':
    parser = h.runners.run_base.get_base_argparser()
    parser.add_argument(
        '--undersampling', '-t', type=float, default=2.45e-5,
        dest='undersampling',
        help=(
            "Carry out common word undersampling as post-processing of the "
            "coccurrence data.  This is performed by first calculating the "
            "probability of dropping a given token using the threshold "
            "value (UNDERSAMPLING) provided employed in the same equation "
            "as described in the SGNS paper.  However it differs from the "
            "undersampling described in the SGNS paper two ways. "
            "(1) it calculates the *expected* change in bigram counts rather "
            "than stochastically dropping counts. (2) Because undersampling "
            "is done as a post-processing step, it amonts to what Goldberg "
            "called 'clean' undersampling, which does not increase the "
            "effective cooccurrence window size."
        )
    )
    parser.add_argument(
        '--neg-samples', '-k', type=int, default=15, dest='k',
        help="number of negative samples"
    )
    parser.add_argument(
        '--smoothing', '-a', type=float, default=0.75,
        dest='smoothing', help='Apply smoothing the unigram distribution.'
    )
    parser.add_argument(
        '--bias', action='store_true', dest='bias',
        help=(
            "Set this flag to include biases in the model for each vector and "
            "covector"
        )
    )
    args = parser.parse_args()
    solver = h.factories.build_sgns_solver(
        **h.runners.run_base.factory_args(args))
    h.runners.run_base.init_and_run(solver, **args)

