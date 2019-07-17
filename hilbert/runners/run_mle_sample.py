import hilbert as h

def add_model_args(parser):
    h.runners.run_base.add_common_constructor_args(parser)
    h.runners.run_base.add_temperature_arg(parser)
    h.runners.run_base.add_batch_size_arg(parser)
    h.runners.run_base.add_balanced_arg(parser)
    h.runners.run_base.add_gibbs_arg(parser)
    h.runners.run_base.add_bias_arg(parser)
    h.runners.run_base.add_remove_cooc_arg(parser)
    return parser


def get_argparser():
    parser = h.runners.run_base.get_argparser()
    add_model_args(parser)
    return parser


def run(**args):
    h.runners.run_base.run(h.factories.build_mle_sample_solver, **args)


if __name__ == '__main__':
    run(**get_argparser().parse_args())

