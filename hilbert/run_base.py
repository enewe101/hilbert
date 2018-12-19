import os
import hilbert as h
import shared
from argparse import ArgumentParser


def init_workspace(embsolver, save_embeddings_dir):
    # Work out the path at which embeddings will be saved.
    if not os.path.exists(save_embeddings_dir):
        os.makedirs(save_embeddings_dir)

    # Write a description of this run within the embeddings save directory
    trace_path = os.path.join(save_embeddings_dir,'trace.txt')
    with open(trace_path, 'w') as trace_file:
        trace_file.write(embsolver.describe())


def save_embeddings(embsolver, save_embeddings_dir, count):
    # Save a copy of the embeddings.
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


def modify_args(args):
    # For convenience, paths are relative to dedicated subdirectories in the
    # hilbert data folder.
    args['save_embeddings_dir'] = os.path.join(
        shared.CONSTANTS.EMBEDDINGS_DIR, args['save_embeddings_dir'])
    args['bigram_path'] = os.path.join(
        shared.CONSTANTS.COOCCURRENCE_DIR, args['bigram_path'])
    if args['init_embeddings_path'] is not None:
        args['init_embeddings_path'] = os.path.join(
            shared.CONSTANTS.EMBEDDINGS_DIR, args['init_embeddings_path'])


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
        '--init', '-i', dest="init_embeddings_path", default=None,
        help="Name of embeddings subdirectory to use as initialization"
    )
    parser.add_argument(
        '--seed', '-S', type=int, required=True, help="Random seed"
    )
    parser.add_argument(
        '--solver', '-s', default='sgd', help="Type of solver to use",
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
    return parser
