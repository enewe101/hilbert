import os
import hilbert as h
from argparse import ArgumentParser
try:
    import shared
except ImportError:
    shared = None

COOCCURRENCE_DIR = (
    shared.CONSTANTS.COOCCURRENCE_DIR 
    if shared is not None else h.CONSTANTS.COOCCURRENCE_DIR
)
EMBEDDINGS_DIR = (
    shared.CONSTANTS.EMBEDDINGS_DIR if shared is not None else
    h.CONSTANTS.EMBEDDINGS_DIR
)


# Main thing that is imported
def init_and_run(embsolver, epochs, iters_per_epoch, shard_times, save_embeddings_dir):

    # special things for initialization.
    print(embsolver.describe())
    init_workspace(embsolver, save_embeddings_dir)
    trace_path = os.path.join(save_embeddings_dir, 'trace.txt')

    # iterate over each epoch, after which we write results
    for epoch in range(1, epochs+1):
        print('epoch\t{}'.format(epoch))

        # cycle the solver, this is a big boy that backprops gradients.
        losses = embsolver.cycle(iters=iters_per_epoch, shard_times=shard_times)

        # saving data intermediately
        save_embeddings(embsolver, save_embeddings_dir, iters_per_epoch * epoch)
        write_trace(trace_path, (epoch - 1) * iters_per_epoch, losses)


# Helper functions.
def init_workspace(embsolver, save_embeddings_dir):

    # Work out the path at which embeddings will be saved.
    if not os.path.exists(save_embeddings_dir):
        os.makedirs(save_embeddings_dir)

    # Write a description of this run within the embeddings save directory
    trace_path = os.path.join(save_embeddings_dir, 'trace.txt')
    with open(trace_path, 'w') as trace_file:
        trace_file.write(embsolver.describe())


def save_embeddings(embsolver, save_embeddings_dir, count):
    embeddings = h.embeddings.Embeddings(
        V=embsolver.V, W=embsolver.W,
        dictionary=embsolver.get_dictionary())

    embeddings_save_path = os.path.join(
        save_embeddings_dir,
        'iter-{}'.format(count))

    embeddings.save(embeddings_save_path)


def write_trace(trace_path, crt_iter, losses):
    with open(trace_path, 'a') as trace_file:
        for i, loss in enumerate(losses):
            trace_file.write('iter-{}: loss={}\n'.format(
                i + crt_iter, loss))


# For convenience, paths are relative to dedicated subdirectories in the
# hilbert data folder.
def modify_args(args):
    args['save_embeddings_dir'] = os.path.join(
        EMBEDDINGS_DIR, args['save_embeddings_dir'])

    args['bigram_path'] = os.path.join(
        COOCCURRENCE_DIR, args['bigram_path'])

    if args['init_embeddings_path'] is not None:
        args['init_embeddings_path'] = os.path.join(
            EMBEDDINGS_DIR, args['init_embeddings_path'])


# Argparser common across everything
def get_base_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        '--bigram', '-b', required=True, dest='bigram_path',
        help="Name of the bigrams subdirectory containing bigram statistics"
    )
    parser.add_argument(
        '--out-dir', '-o', required=True, dest='save_embeddings_dir',
        help="Name of embeddings subdirectory in which to store embeddings"
    )
    parser.add_argument(
        '--device', default='cuda:0', dest='device',
        help="Index of the GPU we want to use (default is cuda:0)"
    )
    parser.add_argument(
        '--init', '-i', dest="init_embeddings_path", default=None,
        help="Name of embeddings subdirectory to use as initialization"
    )
    parser.add_argument(
        '--seed', '-S', type=int, required=True, help="Random seed"
    )
    parser.add_argument(
        '--solver', '-s', default='adam', help="Type of solver to use",
        dest='opt_str'
    )
    parser.add_argument(
        '--learning-rate', '-l', type=float, required=True, 
        help="Learning rate",
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100,
        help="Number of epochs to run.  Embeddings are saved after each epoch."
    )
    parser.add_argument(
        '--iters-per-epoch', '-I', type=int, default=100,
        help="Number of iterations per epoch"
    )
    parser.add_argument(
        '--update-density', '-u', type=float, default=1,
        help="proportion of samples to keep at each iteration (minibatching)"
    )
    parser.add_argument(
        '--sector-factor', '-g', type=int, default=1, 
        help='Sharding factor used to generate cooccurrence data files on disk' 
    )
    parser.add_argument(
        '--shard-factor', '-f', type=int, default=1, 
        help='Sharding factor used to generate minibatches from sectors' 
    )
    parser.add_argument(
        '--shard-times', '-H', type=int, default=1, 
        help='Number of update iterations before loading a new shard'
    )

    return parser
