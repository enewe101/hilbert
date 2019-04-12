import os
import hilbert as h
from argparse import ArgumentParser


def factory_args(args):
    ignore = {'save_embeddings_dir', 'num_writes', 'num_updates'}
    return {key:args[key] for key in args if key not in ignore}




def init_and_run(solver, **args):
    """
    Use run the solver for many updates, and periodically write the model 
    parameters to disk.
    """

    # Do some unpacking
    num_writes = args['num_writes']
    num_updates = args['num_updates']
    save_dir = args['save_embeddings_dir']
    verbose = args['verbose']

    # Make sure the output dir exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Make the tracer (whichs helps us print / log), and generate a preamble.
    trace_path = os.path.join(save_dir, 'trace.txt')
    tracer = h.tracer.Tracer(solver, trace_path, verbose)
    tracer.start(args)

    # Train train train!  Write to disk once in awhile!
    updates_per_write = int(num_updates / num_writes)
    for write_num in range(num_writes):
        solver.cycle(updates_per_write)
        num_updates = updates_per_write * (write_num+1)
        save_path = os.path.join(save_dir, '{}'.format(num_updates))
        solver.get_embeddings().save(save_path)
        tracer.step()


class ModelArgumentParser(ArgumentParser):
    def parse_args(self):
        args = vars(super(ModelArgumentParser, self).parse_args())
        args['save_embeddings_dir'] = os.path.join(
            h.CONSTANTS.RC['embeddings_dir'], args['save_embeddings_dir'])
        args['cooccurrence_path'] = os.path.join(
            h.CONSTANTS.RC['cooccurrence_dir'], args['cooccurrence_path'])
        if args['init_embeddings_path'] is not None:
            args['init_embeddings_path'] = os.path.join(
                h.CONSTANTS.RC['embeddings_dir'], args['init_embeddings_path'])
        args['verbose'] = not args.pop('quiet')
        return args


def get_base_argparser():
    """
    Create an argument parser that includes all of the common options and
    knows how to relativise paths using the RC file.  More specialized 
    paramters can be added by the caller.
    """

    parser = ModelArgumentParser()
    parser.add_argument(
        '--cooccurrence', '-b', required=True, dest='cooccurrence_path',
        help=(
            "Name of the cooccurrence subdirectory containing cooccurrence "
            "statistics"
        )
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, dest='save_embeddings_dir',
        help="Name of embeddings subdirectory in which to store embeddings"
    )

    parser.add_argument(
        '--optimizer', '-s', default='adam', help="Type of optimizer to use",
        dest='opt_str'
    )
    parser.add_argument(
        '--learning-rate', '-l', type=float, default=0.01,
        help="Learning rate",
    )


    parser.add_argument(
        '--writes', '-e', dest='num_writes', type=int, default=100,
        help="Number of times to write intermediate model state to disk."
    )
    parser.add_argument(
        '--updates', dest='num_updates', type=int, default=100000,
        help="Total number of training updates to run."
    )
    parser.add_argument(
        '--batch-size', type=int, default=1000,
        help=(
            "Number of examples used to create one batch.  The preciece "
            "interpretation of an 'example' depends on the type of model and "
            "loader being used."
        )
    )
    parser.add_argument(
        '--shard-factor', type=int, default=1,
        help= "Divide sectors by shard_factor**2 to make it fit on GPU."
    )



    #parser.add_argument(
    #    '--dtype', choices=(64, 32, 16), default=32, dest='dtype',
    #    help="Bit depth of floats used in the model."
    #)

    parser.add_argument(
        '--device', default='cuda:0', dest='device',
        help="Name of the processor we want to use for math (default is cuda:0)"
    )


    parser.add_argument(
        '--init', '-i', dest="init_embeddings_path", default=None,
        help="Name of embeddings subdirectory to use as initialization"
    )
    parser.add_argument(
        '--seed', '-S', type=int, default=1917, help="Random seed"
    )
    parser.add_argument(
        '--dimensions', '-d', type=int, default=300, dest='d',
        help='desired dimensionality of the embeddings being produced'
    )
    parser.add_argument(
        '--quiet', '-q', default=False, action='store_true',
        help="Don't print the trace to stdout."
    )

    return parser
