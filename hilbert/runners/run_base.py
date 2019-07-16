import os
import hilbert as h
from hilbert.tracer import tracer
from argparse import ArgumentParser
import time


def factory_args(args, keep_updates=True):
    if keep_updates:
        ignore = {
            'save_embeddings_dir', 'num_writes',
            'monitor_closely', 'debug'
        }
    else:
        ignore = {
            'save_embeddings_dir', 'num_writes', 'num_updates',
            'monitor_closely', 'debug'
        }
    return {key:args[key] for key in args if key not in ignore}


def run(solver_factory, **args):
    """
    Use run the solver for many updates, and periodically write the model 
    parameters to disk.
    """

    # Do some unpacking
    monitor_closely = args['monitor_closely']
    num_writes = args['num_writes']
    num_updates = args['num_updates']
    save_dir = args['save_embeddings_dir']
    verbose = args['verbose']

    # Make sure the output dir exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Make the tracer (which helps us print / log), and generate a preamble.
    tracer.open(os.path.join(save_dir, 'trace.txt'))
    tracer.verbose = verbose

    # Make a little preamble in the trace.  Todays date, and the exact
    # command used
    tracer.today()
    tracer.command()
    tracer.declare_many(
        {'solver_factory':solver_factory.__name__, **args}
    )

    if solver_factory.__name__ == 'build_mle_sample_solver':
        solver = solver_factory(**h.runners.run_base.factory_args(args))
    else:
        solver = solver_factory(**h.runners.run_base.factory_args(args, keep_updates=False))
    solver.describe()

    # Trigger interactive debugger to explore solver behavior if desired.
    if args['debug']:
        import pdb; pdb.set_trace()

    # Train train train!  Write to disk once in awhile!
    updates_per_write = int(num_updates / num_writes)
    for write_num in range(num_writes):
        try:
            solver.cycle(updates_per_write, monitor_closely)
        except h.exceptions.DivergenceError:
            print("\n\nDied at {} epoch.".format(write_num))
            tracer.declare(key='wrote_num', value=write_num+1)
            raise h.exceptions.DivergenceError("Model has diverged")
        num_updates = updates_per_write * (write_num+1)
        save_path = os.path.join(save_dir, '{}'.format(num_updates))
        solver.get_embeddings().save(save_path)

    tracer.today()

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
        return args


def add_bias_arg(parser):
    parser.add_argument(
        '--bias', action='store_true', dest='bias',
        help=(
            "Set this flag to include biases in the model for each vector and "
            "covector"
        )
    )


def add_temperature_arg(parser):
    parser.add_argument(
        '--temperature', '-t', type=float, default=2, dest='temperature',
        help=(
            "equalizes weighting for loss from individual token pairs.  "
            "Use temperature > 1 for more equal weights."
        )
    )


def add_balanced_arg(parser):
    parser.add_argument(
        '--balanced', '-B', action='store_true',
        help=(
            "Sample positive and negative samples together, using importance "
            "sampling with the independence distribution as proposal."
        )
    )

def add_gibbs_arg(parser):
    parser.add_argument(
        '--gibbs', '-G', action='store_true',
        help=(
            "Sample positive and negative samples together, using Gibbs sampler to sample "
            "negative samples from the model distribution. Get actual samples by default."
        )
    )
    parser.add_argument(
        '--gibbs_iteration', type=int, default=1,
        help=(
            "Number of Gibbs iteration to run before drawing negative samples."
        )
    )
    parser.add_argument(
        '--get_dist', action='store_true',
        help=(
            "Rather than getting actual samples drawn from the model distribution, "
            "get the model distribution instead."
        )
    )

def add_num_senses_arg(parser):
    parser.add_argument(
        '--num_senses', '-K', type=int, required=True,
        help=(
            "Number of sense vectors to allocate per vocabulary item."
        )
    )

def add_batch_size_arg(parser):
    parser.add_argument(
        '--batch-size', '-p', type=int, default=10000,
        help=(
            "Size of sampled batches of (i,j)-pairs used for each update. "
            "Each batch has positive and negative samples, so will have "
            "2*batch_size samples in total."
        )
    )


def add_shard_factor_arg(parser):
    parser.add_argument(
        '--shard-factor', type=int, default=1,
        help="Divide sectors by shard_factor**2 to make it fit on GPU."
    )

def add_remove_cooc_arg(parser):
    parser.add_argument(
        '--remove-threshold', '-thres', type=int, default=10, dest='remove_threshold',
        help="A small number threshold of cooc counts to be removed to fit into the "
             "GPU memory."
    )

def add_gradient_clipping_arg(parser):
    parser.add_argument(
        '--clipping', '-C', type=float, default=None, dest='gradient_clipping',
        help="gradient clipping value."
    )

def add_LR_scheduler_arg(parser):
    # can't have both LR and LR scheduler??
    parser.add_argument(
        '--LR-scheduler', '-scheduler', default=None, choices=['linear', 'inverse', 'None'], dest='scheduler_str',
        help="Type of learning rate scheduler"
    )
    parser.add_argument(
        '--LR-scheduler-endLR', '-le', type=float, default=0.0, dest='end_lr',
        help="The end learning rate for linear learning rate scheduler"
    )
    parser.add_argument(
        '--constant-fraction', '-frac', type=float, default=0.1, dest='constant_fraction',
        help="Required for inverse LR scheduler. Control the number of updates for which learning rate stays constant."
    )

def add_gradient_accumulation_arg(parser):
    parser.add_argument(
        '--gradient-accumulation', '-ga', type=int, default=1, dest='gradient_accumulation',
        help="Accumulate gradients for number of updates."
    )



def add_common_constructor_args(parser):
    """
    Add the arguments that are common to all model constructors.
    """
    parser.add_argument(
        '--cooccurrence', '-b', required=True, dest='cooccurrence_path',
        help=(
            "Name of the cooccurrence subdirectory containing cooccurrence "
            "statistics"
        )
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
        '--dimensions', '-d', type=int, default=300, dest='dimensions',
        help='desired dimensionality of the embeddings being produced'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_false', dest='verbose',
        help="Don't print the trace to stdout."
    )




def get_argparser(**kwargs):
    """
    Create an argument parser that includes all of the common options and
    knows how to relativise paths using the RC file.  More specialized 
    paramters can be added by the caller.
    """
    parser = ModelArgumentParser(**kwargs)
    parser.add_argument(
        '--out-dir', '-o', required=True, dest='save_embeddings_dir',
        help="Name of embeddings subdirectory in which to store embeddings"
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
        '--monitor-closely', '-M', action='store_true',
        help="Get the loss after every single model batch"
    )
    parser.add_argument(
        '--debug', '-D', action='store_true',
        help="After making the solver, go into interactive debugger"
    )

    #parser.add_argument(
    #    '--dtype', choices=(64, 32, 16), default=32, dest='dtype',
    #    help="Bit depth of floats used in the model."
    #)
    return parser
